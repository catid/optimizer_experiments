from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.sodamuseeq_ablation import RunSpec, write_final_bin_table_and_plots  # noqa: E402


LABELS = {
    "adamw": "AdamW",
    "soda_pmuoneq": "SODA+PMuonEq+Gram",
    "normuon_base": "NorMuon+BaseGram",
}


@dataclass(frozen=True)
class Trial:
    spec: RunSpec
    family: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.spec.name}_seed{self.seed}"


def adamw_specs() -> list[Trial]:
    specs: list[Trial] = []
    for lr in (0.002, 0.003, 0.004, 0.005):
        for wd in (0.01, 0.02, 0.05):
            spec = RunSpec(f"hpo_adamw_lr{lr:g}_wd{wd:g}", "adamw", lr=lr, weight_decay=wd)
            specs.append(Trial(spec, "adamw", 0))
    return specs


def soda_pmuoneq_specs() -> list[Trial]:
    specs: list[Trial] = []
    for lr in (0.01, 0.012, 0.014, 0.016, 0.018):
        for gamma in (0.0, 0.025, 0.05, 0.075, 0.1):
            spec = RunSpec(
                f"hpo_soda_pmuoneq_lr{lr:g}_g{gamma:g}",
                "sodamuseeq",
                lr=lr,
                weight_decay=0.0,
                pmuon_gamma=gamma,
                use_soda=True,
                use_amuse=False,
                use_pmuoneq=True,
                use_gram=True,
                use_normuon=False,
                soda_replaces_weight_decay=True,
                warmup_steps=500,
                soda_warmup_steps=500,
            )
            specs.append(Trial(spec, "soda_pmuoneq", 0))
    return specs


def normuon_base_specs() -> list[Trial]:
    specs: list[Trial] = []
    for lr in (0.002, 0.003, 0.004, 0.006):
        for wd in (0.0, 0.02, 0.05):
            for beta2 in (0.9, 0.95, 0.98):
                spec = RunSpec(
                    f"hpo_normuon_base_lr{lr:g}_wd{wd:g}_nb{beta2:g}",
                    "sodamuseeq",
                    lr=lr,
                    weight_decay=wd,
                    pmuon_gamma=0.0,
                    use_soda=False,
                    use_amuse=False,
                    use_pmuoneq=False,
                    use_gram=True,
                    use_normuon=True,
                    normuon_beta2=beta2,
                    soda_replaces_weight_decay=False,
                    warmup_steps=500,
                    soda_warmup_steps=0,
                )
                specs.append(Trial(spec, "normuon_base", 0))
    return specs


def hpo_trials() -> list[Trial]:
    return adamw_specs() + soda_pmuoneq_specs() + normuon_base_specs()


def _row_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row.get("best_val_loss", float("inf"))), -float(row.get("best_val_acc", 0.0)))


def _write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with (root / "all_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    ok.sort(key=_row_key)
    (root / "summary_sorted.json").write_text(json.dumps(ok, indent=2, sort_keys=True) + "\n")


def launch_trials(args, trials: list[Trial], subdir: str, steps: int, eval_bins: int, eval_test: bool) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(args.out_dir) / subdir
    root.mkdir(parents=True, exist_ok=True)
    (root / "trials.json").write_text(
        json.dumps(
            [
                {"family": trial.family, "seed": trial.seed, **trial.spec.__dict__}
                for trial in trials
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    gpus = [str(i) for i in range(torch.cuda.device_count())]
    if not gpus:
        raise RuntimeError("CUDA GPUs are required")
    queue = list(trials)
    active: dict[subprocess.Popen, tuple[Trial, Path, str, Any]] = {}
    finished: list[dict[str, Any]] = []

    def start(trial: Trial, gpu: str) -> None:
        run_dir = root / trial.name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "sodamuseeq_ablation.py"),
            "--trial",
            "--run-name",
            trial.name,
            "--out-dir",
            str(run_dir.resolve()),
            "--data-dir",
            str(Path(args.data_dir).resolve()),
            "--max-steps",
            str(steps),
            "--eval-bins",
            str(eval_bins),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--workers",
            str(args.workers),
            "--seed",
            str(trial.seed),
            "--split-seed",
            str(args.split_seed),
            "--val-size",
            str(args.val_size),
            "--train-eval-size",
            str(args.train_eval_size),
            "--embed-dim",
            str(args.embed_dim),
            "--depth",
            str(args.depth),
            "--num-heads",
            str(args.num_heads),
            "--warmup-steps",
            str(max(1, min(args.warmup_steps, max(1, steps // 4)))),
            "--soda-warmup-steps",
            str(max(0, min(args.soda_warmup_steps, max(0, steps // 4)))),
            *trial.spec.to_args(),
        ]
        if eval_test:
            cmd.append("--eval-test")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTHONUNBUFFERED", "1")
        log = (run_dir / "stdout.log").open("w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active[proc] = (trial, run_dir, gpu, log)
        print(f"started {subdir}:{trial.name} on gpu {gpu}", flush=True)

    free = gpus.copy()
    while queue or active:
        while queue and free:
            start(queue.pop(0), free.pop(0))
        time.sleep(2)
        for proc in list(active):
            code = proc.poll()
            if code is None:
                continue
            trial, run_dir, gpu, log = active.pop(proc)
            log.close()
            free.append(gpu)
            if code != 0:
                row = {"run_name": trial.name, "family": trial.family, "seed": trial.seed, "status": "failed", "returncode": code}
                print(f"failed {subdir}:{trial.name}; see {run_dir / 'stdout.log'}", flush=True)
            else:
                row = json.loads((run_dir / "summary.json").read_text())
                row["family"] = trial.family
                row["seed"] = trial.seed
                row["status"] = "ok"
                print(
                    f"finished {subdir}:{trial.name}: family={trial.family} "
                    f"val_loss={row['best_val_loss']:.4f} val_acc={row['best_val_acc']:.4f}",
                    flush=True,
                )
            finished.append(row)
            _write_rows(root, finished)
    _write_rows(root, finished)
    return root, finished


def _top_by_family(rows: list[dict[str, Any]], n: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for family in LABELS:
        family_rows = [row for row in rows if row.get("status") == "ok" and row.get("family") == family]
        family_rows.sort(key=_row_key)
        out[family] = family_rows[:n]
    return out


def _spec_from_row(row: dict[str, Any], name: str) -> RunSpec:
    return RunSpec(
        name,
        str(row["optimizer"]),
        lr=float(row["lr"]),
        weight_decay=float(row["weight_decay"]),
        pmuon_gamma=float(row.get("pmuon_gamma", 0.0)),
        pmuon_beta=float(row.get("pmuon_beta", 0.95)),
        use_soda=_as_bool(row.get("use_soda"), True),
        use_amuse=_as_bool(row.get("use_amuse"), False),
        use_pmuoneq=_as_bool(row.get("use_pmuoneq"), True),
        use_gram=_as_bool(row.get("use_gram"), True),
        use_mimuon=_as_bool(row.get("use_mimuon"), False),
        use_normuon=_as_bool(row.get("use_normuon"), False),
        mimuon_tau=float(row.get("mimuon_tau", 0.005)),
        normuon_beta2=float(row.get("normuon_beta2", 0.95)),
        beta1=float(row.get("beta1", 0.6)),
        rho=float(row.get("rho", 0.8)),
        warmup_steps=int(float(row.get("warmup_steps", 500))) or None,
        soda_warmup_steps=int(float(row.get("soda_warmup_steps", 0))) or None,
        soda_replaces_weight_decay=_as_bool(row.get("soda_replaces_weight_decay"), True),
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def top_trials(rows: list[dict[str, Any]], n: int, prefix: str, seeds: list[int] | None = None) -> list[Trial]:
    selected = _top_by_family(rows, n)
    trials: list[Trial] = []
    for family, family_rows in selected.items():
        for rank, row in enumerate(family_rows, 1):
            for seed in (seeds if seeds is not None else [int(row.get("seed", 0))]):
                spec = _spec_from_row(row, f"{prefix}_{family}_rank{rank}")
                trials.append(Trial(spec, family, seed))
    return trials


def _mean_std(rows: list[dict[str, Any]], key: str) -> str:
    vals = [float(row[key]) for row in rows if row.get("status") == "ok" and math.isfinite(float(row.get(key, float("nan"))))]
    if not vals:
        return "n/a"
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {statistics.pstdev(vals):.4f}"


def write_report(args, roots: dict[str, Path], rows_by_phase: dict[str, list[dict[str, Any]]]) -> Path:
    report = Path(args.out_dir) / "report.md"
    final_rows = [row for row in rows_by_phase.get("final", []) if row.get("status") == "ok"]
    long_rows = [row for row in rows_by_phase.get("long", []) if row.get("status") == "ok"]
    final_by_family = {family: [row for row in final_rows if row.get("family") == family] for family in LABELS}
    long_by_family = {family: [row for row in long_rows if row.get("family") == family] for family in LABELS}

    lines = [
        "# Focused Optimizer Confidence Comparison",
        "",
        "Compared only the requested arms: AdamW baseline, NorMuon+BaseGram, and SODA+PMuonEq+Gram.",
        "",
        "## Artifacts",
        "",
    ]
    for phase, root in roots.items():
        lines.append(f"- {phase}: `{root}`")
    lines.extend(
        [
            "",
            "## Final 10k Multi-Seed Summary",
            "",
            "| rank | family | label | seeds | val loss | val acc | test acc | steps/sec |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    ranked = []
    for family, rows in final_by_family.items():
        if rows:
            ranked.append((family, rows, statistics.fmean(float(r["best_val_loss"]) for r in rows)))
    ranked.sort(key=lambda item: item[2])
    for idx, (family, rows, _) in enumerate(ranked, 1):
        lines.append(
            f"| {idx} | `{family}` | {LABELS[family]} | {len(rows)} | "
            f"{_mean_std(rows, 'best_val_loss')} | {_mean_std(rows, 'best_val_acc')} | "
            f"{_mean_std(rows, 'test_acc')} | {_mean_std(rows, 'steps_per_sec')} |"
        )

    lines.extend(
        [
            "",
            "## Long 20k Seed-0 Check",
            "",
            "| rank | family | label | val loss | val acc | test acc | steps/sec |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    long_ranked = []
    for family, rows in long_by_family.items():
        if rows:
            row = min(rows, key=_row_key)
            long_ranked.append((family, row))
    long_ranked.sort(key=lambda item: _row_key(item[1]))
    for idx, (family, row) in enumerate(long_ranked, 1):
        lines.append(
            f"| {idx} | `{family}` | {LABELS[family]} | {float(row['best_val_loss']):.4f} | "
            f"{100 * float(row['best_val_acc']):.2f}% | {100 * float(row['test_acc']):.2f}% | "
            f"{float(row['steps_per_sec']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Configs",
            "",
            "| phase | family | run | lr | wd | pmuon_gamma | normuon_beta2 | seed | best val loss |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for phase in ("top3k", "final", "long"):
        for row in sorted([r for r in rows_by_phase.get(phase, []) if r.get("status") == "ok"], key=lambda r: (str(r.get("family")), int(r.get("seed", 0)), _row_key(r))):
            lines.append(
                f"| {phase} | `{row['family']}` | `{row['run_name']}` | {float(row['lr']):.5g} | "
                f"{float(row['weight_decay']):.5g} | {float(row.get('pmuon_gamma', 0.0)):.5g} | "
                f"{float(row.get('normuon_beta2', 0.95)):.5g} | {int(row.get('seed', 0))} | "
                f"{float(row['best_val_loss']):.4f} |"
            )

    if "final" in roots:
        plot_root = roots["final"] / "plots"
        lines.extend(
            [
                "",
                "## Final Plots",
                "",
                f"- Validation loss: `{plot_root / 'val_loss.png'}`",
                f"- Validation accuracy: `{plot_root / 'val_acc.png'}`",
                f"- Train interval loss: `{plot_root / 'train_interval_loss.png'}`",
                f"- Step speed bar chart: `{plot_root / 'step_speed_bar.png'}`",
            ]
        )
    report.write_text("\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/focused_optimizer_confidence")
    p.add_argument("--data-dir", default="data/cifar10")
    p.add_argument("--hpo-steps", type=int, default=1000)
    p.add_argument("--top-steps", type=int, default=3000)
    p.add_argument("--final-steps", type=int, default=10000)
    p.add_argument("--long-steps", type=int, default=20000)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--final-seeds", default="0,1,2")
    p.add_argument("--long-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=max(4, min(12, (os.cpu_count() or 16) // 3)))
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--train-eval-size", type=int, default=5000)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--soda-warmup-steps", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(part) for part in args.final_seeds.split(",") if part.strip()]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    hpo_root, hpo_rows = launch_trials(args, hpo_trials(), "hpo1k", args.hpo_steps, eval_bins=4, eval_test=False)
    top_root, top_rows = launch_trials(args, top_trials(hpo_rows, args.top_n, "top3k"), "top3k", args.top_steps, eval_bins=6, eval_test=False)
    final_trials = top_trials(top_rows, 1, "final10k", seeds=seeds)
    final_root, final_rows = launch_trials(args, final_trials, "final10k", args.final_steps, eval_bins=8, eval_test=True)
    write_final_bin_table_and_plots(final_root)
    long_trials = top_trials(top_rows, 1, "long20k", seeds=[args.long_seed])
    long_root, long_rows = launch_trials(args, long_trials, "long20k", args.long_steps, eval_bins=8, eval_test=True)
    write_final_bin_table_and_plots(long_root)
    report = write_report(
        args,
        {"hpo1k": hpo_root, "top3k": top_root, "final": final_root, "long": long_root},
        {"hpo1k": hpo_rows, "top3k": top_rows, "final": final_rows, "long": long_rows},
    )
    print(report)


if __name__ == "__main__":
    main()
