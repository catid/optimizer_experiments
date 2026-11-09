#!/usr/bin/env python3
"""Aggregate a MuonC x MuonGS result bundle into CSV / markdown tables.

Ranking rules (they matter because most grid points are single-seed):

* A configuration with *any* diverged or missing seed is reported as diverged
  and can never be "best".
* ``best_tables`` selects the best (lr, wd) of a cell on the selection seed
  only.  When other seeds of that configuration exist, the reported number is
  their mean, so the winner of a noisy single-seed grid is not also scored on
  the draw that made it win.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRIC = "final_full_val_loss"
SELECTION_SEED = 123
CLASSES = ("emb", "qkv", "attn_proj", "mlp_fc", "mlp_proj")
# Optimizer-side fields that only matter for some bases; shown when non-default.
EXTRA_DEFAULTS = {
    "emb": "matrix", "unprojected_decay": "", "fallback_mode": "adamc", "fallback_lr_mult": 1.0,
    "fallback_wd": 0.0, "fallback_decay": "standard", "row_gamma": 0.45, "pmuoneq_beta": 0.90,
    "normuon_beta2": 0.93, "soda": 0.003, "init_resid_scale": 1.0, "emulate_gs_theta0": 0.0,
    "ns_dtype": "", "init_scale": 1.0, "lr_profile": "", "radial_lr": 0.0,
}
PROTOCOL_DEFAULTS = {"batch_size": 16, "block_size": 256, "warmup_steps": 100, "final_lr_scale": 0.1, "decay_frac": 0.2}


def _g(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def cell_label(row: dict[str, Any]) -> str:
    if row["opt"] == "repo_muon":
        return "repo PlainMuon"
    if row["opt"] == "repo_adamw":
        return "repo AdamW"
    gs = float(row["gs"])
    proj = "GS" if gs == 1.0 else ("noGS" if gs == 0.0 else f"GS{gs:g}")
    if row.get("gs_mode", "both") != "both":
        proj += f"({row['gs_mode']})"
    target = float(row.get("norm_target", 0.0) or 0.0)
    decay = f"T{target:g}" if target > 0 else {"none": "0", "standard": "W", "corrected": "C"}[row["decay"]]
    label = f"{proj}+{decay}"
    if row.get("norm_mode", "free") != "free":
        label += "+fixednorm"
    extras = []
    base = row.get("base", "plain")
    if base != "plain":
        extras.append(base)
    for key, default in EXTRA_DEFAULTS.items():
        value = row.get(key, default)
        if key in ("row_gamma", "pmuoneq_beta") and "pmuoneq" not in base:
            continue
        if key == "normuon_beta2" and "normuon" not in base:
            continue
        if key == "soda" and "soda" not in base:
            continue
        if value != default:
            extras.append("lr_profile" if key == "lr_profile" else f"{key}={_g(value)}")
    return label + (" [" + ", ".join(extras) + "]" if extras else "")


def protocol_label(row: dict[str, Any]) -> str:
    label = f"{row['data']}-{row['schedule']}-{int(row['steps']) // 1000}k"
    for key, default in PROTOCOL_DEFAULTS.items():
        value = row.get(key, default)
        if key == "decay_frac" and row["schedule"] == "cosine":
            continue
        if value != default:
            label += f"-{key}={_g(value)}"
    return label


def load(bundle: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, missing = [], []
    for trial_dir in sorted((bundle / "trials").iterdir()):
        path = trial_dir / "summary.json"
        if not path.exists():
            if (trial_dir / "trial.json").exists():
                missing.append(trial_dir.name)
            continue
        row = json.loads(path.read_text())
        row["cell"] = cell_label(row)
        row["protocol"] = protocol_label(row)
        rows.append(row)
    return rows, missing


def effective_wd(row: dict[str, Any]) -> float:
    if row["opt"] == "muonc" and (row["decay"] == "none" or float(row.get("norm_target", 0.0) or 0.0) > 0):
        return 0.0
    return float(row["wd"])


def group_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse seeds: one entry per (protocol, cell, lr, wd)."""

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["protocol"], row["cell"], float(row["lr"]), effective_wd(row))].append(row)
    out = []
    for (protocol, cell, lr, wd), members in groups.items():
        seeds = [int(m["seed"]) for m in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"two trials share a seed inside {protocol} / {cell} / lr={lr} / wd={wd}: grouping key is too coarse")
        ok = [m for m in members if not m.get("diverged")]
        entry: dict[str, Any] = {
            "protocol": protocol, "cell": cell, "lr": lr, "wd": wd,
            "n": len(members), "n_diverged": len(members) - len(ok), "seeds": sorted(seeds),
        }
        if len(ok) == len(members):
            by_seed = {int(m["seed"]): float(m[METRIC]) for m in ok}
            vals = np.array(list(by_seed.values()))
            entry["loss"] = float(vals.mean())
            entry["loss_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            entry["loss_by_seed"] = by_seed
            entry["select"] = by_seed.get(SELECTION_SEED, float(vals.mean()))
            fresh = [v for seed, v in by_seed.items() if seed != SELECTION_SEED]
            entry["confirm"] = float(np.mean(fresh)) if fresh else float("nan")
            entry["confirm_std"] = float(np.std(fresh, ddof=1)) if len(fresh) > 1 else float("nan")
            entry["confirm_n"] = len(fresh)
            entry["acc"] = float(np.mean([m["final_full_val_acc"] for m in ok]))
            entry["train"] = float(np.mean([m["final_train_loss_avg100"] for m in ok]))
            for key in CLASSES:
                if f"final_norm_ratio/{key}" in ok[0]:
                    entry[f"norm/{key}"] = float(np.mean([m[f"final_norm_ratio/{key}"] for m in ok]))
                for prefix, out_key in (("angular_path", "path"), ("mean_radial_cos", "cos")):
                    if all(f"{prefix}/{key}" in m for m in ok):
                        entry[f"{out_key}/{key}"] = float(np.mean([m[f"{prefix}/{key}"] for m in ok]))
        else:
            entry["loss"] = entry["select"] = float("inf")
        out.append(entry)
    return out


def fmt(value: float, digits: int = 4) -> str:
    return "div" if not math.isfinite(value) else f"{value:.{digits}f}"


def grid_tables(stats: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for protocol in sorted({s["protocol"] for s in stats}):
        sub = [s for s in stats if s["protocol"] == protocol]
        lrs = sorted({s["lr"] for s in sub})
        lines.append(f"\n### {protocol}: {METRIC} by lr (rows: cell, wd; mean over seeds)\n")
        lines.append("| cell | wd | " + " | ".join(f"{lr:g}" for lr in lrs) + " |")
        lines.append("|---|---:|" + "---:|" * len(lrs))
        for cell, wd in sorted({(s["cell"], s["wd"]) for s in sub}):
            cells = []
            for lr in lrs:
                match = [s for s in sub if s["cell"] == cell and s["wd"] == wd and s["lr"] == lr]
                if not match:
                    cells.append("")
                    continue
                m = match[0]
                text = fmt(m["loss"])
                if m["n_diverged"]:
                    text = f"div {m['n_diverged']}/{m['n']}"
                elif m["n"] > 1:
                    text += f"±{m['loss_std']:.3f} (n{m['n']})"
                cells.append(text)
            lines.append(f"| {cell} | {wd:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_per_cell(stats: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for s in stats:
        if s["protocol"] != protocol or not math.isfinite(s["select"]):
            continue
        if s["cell"] not in best or s["select"] < best[s["cell"]]["select"]:
            best[s["cell"]] = s
    return sorted(best.values(), key=lambda s: s["confirm"] if math.isfinite(s["confirm"]) else s["select"])


def best_tables(stats: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for protocol in sorted({s["protocol"] for s in stats}):
        lines.append(f"\n### {protocol}: best (lr, wd) per cell, selected on seed {SELECTION_SEED}\n")
        lines.append("| cell | lr | wd | select | fresh seeds | acc | train | norm emb | qkv | mlp_fc | mlp_proj | path qkv | cos qkv |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in best_per_cell(stats, protocol):
            fresh = "" if not s["confirm_n"] else f"{s['confirm']:.4f}" + (f"±{s['confirm_std']:.3f}" if s["confirm_n"] > 1 else "") + f" (n{s['confirm_n']})"
            lines.append(
                f"| {s['cell']} | {s['lr']:g} | {s['wd']:g} | {s['select']:.4f} | {fresh} | {100 * s['acc']:.2f}% | {s['train']:.3f} | "
                + " | ".join(f"{s.get(f'norm/{k}', float('nan')):.2f}" for k in ("emb", "qkv", "mlp_fc", "mlp_proj"))
                + f" | {s.get('path/qkv', float('nan')):.1f} | {s.get('cos/qkv', float('nan')):+.3f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--protocol", action="append", default=[])
    parser.add_argument("--write", action="store_true", help="write summary.csv next to the trials")
    args = parser.parse_args()
    rows, missing = load(args.bundle)
    if args.stage:
        rows = [r for r in rows if r.get("stage") in args.stage]
    if args.protocol:
        rows = [r for r in rows if r["protocol"] in args.protocol]
    stats = group_stats(rows)
    print(f"{len(rows)} finished trials; {sum(bool(r.get('diverged')) for r in rows)} diverged; {len(missing)} started but unfinished")
    for name in missing[:10]:
        print("  unfinished:", name)
    print(best_tables(stats))
    print(grid_tables(stats))
    if args.write:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with (args.bundle / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
