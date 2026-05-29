#!/usr/bin/env python3
"""CIFAR-10 NorMuon aspect-scaling ablation for SodaPmuonEqNorMuon.

The supervisor launches one trial per visible GPU. The default trial set uses
the tuned standalone recipe plus two NorMuon variants requested by peer review:

* current row-wise NorMuon with the tuned aspect-ratio scale;
* row-wise NorMuon without the extra aspect-ratio scale;
* orientation-aware NorMuon without the extra aspect-ratio scale;
* orientation-aware NorMuon with the same aspect-ratio scale;
* tuned AdamW baseline for context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from timm.models import create_model
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
NORADAM_ROOT = REPO_ROOT / "workers" / "codex_noradam_confidence"
for path in (ROOT, NORADAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import models_vit5  # noqa: F401 registers ViT-5 models with timm
from soda_pmuoneq_normuon import SodaPmuonEqNorMuon, build_soda_pmuoneq_normuon_param_groups


@dataclass(frozen=True)
class Trial:
    name: str
    optimizer: str
    lr: float = 0.0
    weight_decay: float = 0.005
    matrix_lr: float = 8e-3
    adam_lr: float = 8e-4
    momentum: float = 0.95
    pmuoneq_beta: float = 0.90
    row_gamma: float = 0.35
    col_gamma: float = 0.05
    normuon_beta2: float = 0.93
    normuon_mode: str = "row"
    normuon_aspect_scale: bool = True
    seed: int = 34000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-json", type=Path)
    parser.add_argument("--data-path", type=Path, default=Path("/home/catid/attractor/data/cifar10"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cifar10_normuon_aspect_ablation")
    parser.add_argument("--model", default="vit5_tiny")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-bins", type=int, default=8)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=34000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--only", default="", help="Regex-free substring filter for trial names")
    parser.add_argument("--preset", choices=("fixed", "tune", "peer_final"), default="fixed")
    parser.add_argument("--sync-step-timing", action="store_true", help="Synchronize CUDA around every measured step")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def fixed_trials(seed: int) -> list[Trial]:
    return [
        Trial("adamw_lr0.0025_wd0.005", "adamw", lr=2.5e-3, weight_decay=0.005, seed=seed),
        Trial(
            "row_aspect_mlr0.008_rg0.35_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            normuon_mode="row",
            normuon_aspect_scale=True,
            seed=seed,
        ),
        Trial(
            "row_noaspect_mlr0.008_rg0.35_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            normuon_mode="row",
            normuon_aspect_scale=False,
            seed=seed,
        ),
        Trial(
            "orientation_noaspect_mlr0.008_rg0.35_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            normuon_mode="orientation",
            normuon_aspect_scale=False,
            seed=seed,
        ),
        Trial(
            "orientation_aspect_mlr0.008_rg0.35_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            normuon_mode="orientation",
            normuon_aspect_scale=True,
            seed=seed,
        ),
    ]


def tuning_trials(seed: int) -> list[Trial]:
    trials = [Trial("adamw_lr0.0025_wd0.005", "adamw", lr=2.5e-3, weight_decay=0.005, seed=seed)]
    for mode, aspect in [("row", True), ("orientation", True), ("row", False), ("orientation", False)]:
        for row_gamma, col_gamma, beta2 in [(0.30, 0.0, 0.95), (0.35, 0.05, 0.93), (0.40, 0.05, 0.93)]:
            aspect_tag = "aspect" if aspect else "noaspect"
            name = (
                f"{mode}_{aspect_tag}_mlr0.008_"
                f"rg{row_gamma:g}_cg{col_gamma:g}_nb{beta2:g}"
            ).replace(".", "p")
            trials.append(
                Trial(
                    name,
                    "soda_pmuoneq_normuon",
                    row_gamma=row_gamma,
                    col_gamma=col_gamma,
                    normuon_beta2=beta2,
                    normuon_mode=mode,
                    normuon_aspect_scale=aspect,
                    seed=seed,
                )
            )
    return trials


def peer_final_trials(seed: int) -> list[Trial]:
    return [
        Trial("adamw_lr0.0025_wd0.005", "adamw", lr=2.5e-3, weight_decay=0.005, seed=seed),
        Trial(
            "row_aspect_mlr0.008_rg0.35_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            normuon_mode="row",
            normuon_aspect_scale=True,
            seed=seed,
        ),
        Trial(
            "row_aspect_mlr0.008_rg0.4_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            row_gamma=0.40,
            col_gamma=0.05,
            normuon_beta2=0.93,
            normuon_mode="row",
            normuon_aspect_scale=True,
            seed=seed,
        ),
        Trial(
            "orientation_aspect_mlr0.008_rg0.3_cg0_nb0.95",
            "soda_pmuoneq_normuon",
            row_gamma=0.30,
            col_gamma=0.0,
            normuon_beta2=0.95,
            normuon_mode="orientation",
            normuon_aspect_scale=True,
            seed=seed,
        ),
        Trial(
            "orientation_aspect_mlr0.008_rg0.4_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            row_gamma=0.40,
            col_gamma=0.05,
            normuon_beta2=0.93,
            normuon_mode="orientation",
            normuon_aspect_scale=True,
            seed=seed,
        ),
        Trial(
            "row_noaspect_mlr0.008_rg0.3_cg0_nb0.95",
            "soda_pmuoneq_normuon",
            row_gamma=0.30,
            col_gamma=0.0,
            normuon_beta2=0.95,
            normuon_mode="row",
            normuon_aspect_scale=False,
            seed=seed,
        ),
        Trial(
            "orientation_noaspect_mlr0.008_rg0.4_cg0.05_nb0.93",
            "soda_pmuoneq_normuon",
            row_gamma=0.40,
            col_gamma=0.05,
            normuon_beta2=0.93,
            normuon_mode="orientation",
            normuon_aspect_scale=False,
            seed=seed,
        ),
    ]


def default_trials(seed: int, preset: str) -> list[Trial]:
    if preset == "fixed":
        return fixed_trials(seed)
    if preset == "tune":
        return tuning_trials(seed)
    if preset == "peer_final":
        return peer_final_trials(seed)
    raise ValueError(preset)


def selected_eval_steps(total_steps: int, bins: int) -> set[int]:
    return {max(1, round(total_steps * i / max(1, bins))) for i in range(1, max(1, bins) + 1)}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(args.data_path, train=True, transform=train_tf, download=True)
    val = datasets.CIFAR10(args.data_path, train=False, transform=val_tf, download=True)
    if 0 < args.train_subset < len(train):
        generator = torch.Generator().manual_seed(args.seed)
        train = Subset(train, torch.randperm(len(train), generator=generator)[: args.train_subset].tolist())
    if 0 < args.val_subset < len(val):
        val = Subset(val, list(range(args.val_subset)))
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return DataLoader(train, shuffle=True, drop_last=True, **kwargs), DataLoader(val, shuffle=False, **kwargs)


def matrix_filter(name: str, p: torch.nn.Parameter) -> bool:
    lower = name.lower()
    if p.ndim < 2:
        return False
    if any(token in lower for token in ("head", "norm", "bn", "embed", "cls_token", "reg_token", "pos_embed")):
        return False
    return True


def make_optimizer(model: nn.Module, trial: Trial, args: argparse.Namespace) -> torch.optim.Optimizer:
    if trial.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=trial.lr, weight_decay=trial.weight_decay, betas=(0.9, 0.999), fused=torch.cuda.is_available())
    if trial.optimizer != "soda_pmuoneq_normuon":
        raise ValueError(trial.optimizer)
    groups = build_soda_pmuoneq_normuon_param_groups(
        model.named_parameters(),
        matrix_lr=trial.matrix_lr,
        adam_lr=trial.adam_lr,
        matrix_weight_decay=0.0,
        adam_weight_decay=0.05,
        momentum=trial.momentum,
        pmuoneq_beta=trial.pmuoneq_beta,
        row_gamma=trial.row_gamma,
        col_gamma=trial.col_gamma,
        normuon_beta2=trial.normuon_beta2,
        normuon_mode=trial.normuon_mode,
        normuon_aspect_scale=trial.normuon_aspect_scale,
        matrix_filter=matrix_filter,
    )
    return SodaPmuonEqNorMuon(
        groups,
        matrix_lr=trial.matrix_lr,
        adam_lr=trial.adam_lr,
        matrix_weight_decay=0.0,
        adam_weight_decay=0.05,
        momentum=trial.momentum,
        pmuoneq_beta=trial.pmuoneq_beta,
        row_gamma=trial.row_gamma,
        col_gamma=trial.col_gamma,
        normuon_beta2=trial.normuon_beta2,
        normuon_mode=trial.normuon_mode,
        normuon_aspect_scale=trial.normuon_aspect_scale,
        warmup_steps=args.warmup_steps,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction="sum")
        loss_sum += float(loss.detach().cpu())
        correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total += int(targets.numel())
    model.train()
    return loss_sum / max(total, 1), correct / max(total, 1)


def run_worker(args: argparse.Namespace) -> None:
    if args.trial_json is None:
        raise SystemExit("--trial-json is required in worker mode")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    trial = Trial(**json.loads(args.trial_json.read_text()))
    args.seed = trial.seed
    set_seed(trial.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    run_dir = args.output_dir / trial.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    train_loader, val_loader = make_loaders(args)
    model = create_model(args.model, pretrained=False, num_classes=10, img_size=32, drop_path_rate=0.05)
    model.to(device=device, memory_format=torch.channels_last)
    optimizer = make_optimizer(model, trial, args)
    total_steps = args.epochs * len(train_loader)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    eval_steps = selected_eval_steps(total_steps, args.eval_bins)
    global_step = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    final_val_loss = float("nan")
    final_val_accuracy = float("nan")
    final_train_loss_interval = float("nan")
    total_examples = 0
    total_step_s = 0.0
    started = time.perf_counter()

    with metrics_path.open("w", encoding="utf-8") as handle:
        interval_examples = 0
        interval_loss = 0.0
        interval_step_s = 0.0
        interval_steps = 0
        interval_start = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            for batch_idx, (images, targets) in enumerate(train_loader):
                if global_step >= total_steps:
                    break
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                if args.sync_step_timing:
                    torch.cuda.synchronize()
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                if args.sync_step_timing:
                    torch.cuda.synchronize()
                step_s = time.perf_counter() - step_start
                batch = int(targets.numel())
                total_examples += batch
                total_step_s += step_s
                interval_examples += batch
                interval_loss += float(loss.detach().cpu()) * batch
                interval_step_s += step_s
                interval_steps += 1
                global_step += 1
                if global_step % args.log_every == 0:
                    handle.write(json.dumps({"event": "step", "step": global_step, "epoch": epoch, "batch": batch_idx, "train_loss": float(loss.detach().cpu())}) + "\n")
                    handle.flush()
                if global_step in eval_steps:
                    val_loss, val_accuracy = evaluate(model, val_loader, device)
                    final_val_loss = val_loss
                    final_val_accuracy = val_accuracy
                    final_train_loss_interval = interval_loss / max(interval_examples, 1)
                    best_val_loss = min(best_val_loss, val_loss)
                    best_val_accuracy = max(best_val_accuracy, val_accuracy)
                    row = {
                        "event": "bin",
                        "trial": trial.name,
                        "step": global_step,
                        "epoch": epoch + 1,
                        "progress": global_step / max(total_steps, 1),
                        "train_loss_interval": final_train_loss_interval,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "best_val_loss": best_val_loss,
                        "best_val_accuracy": best_val_accuracy,
                        "examples_per_s_interval": interval_examples / max(time.perf_counter() - interval_start, 1e-9),
                        "mean_step_s_interval": interval_step_s / max(interval_steps, 1),
                        "overall_examples_per_s": total_examples / max(total_step_s, 1e-9),
                        "elapsed_s": time.perf_counter() - started,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(json.dumps(row), flush=True)
                    interval_examples = 0
                    interval_loss = 0.0
                    interval_step_s = 0.0
                    interval_steps = 0
                    interval_start = time.perf_counter()
            if global_step >= total_steps:
                break
    final = {
        **asdict(trial),
        "display_lr": trial.lr if trial.optimizer == "adamw" else trial.matrix_lr,
        "steps": global_step,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "final_val_loss": final_val_loss,
        "final_val_accuracy": final_val_accuracy,
        "final_train_loss_interval": final_train_loss_interval,
        "overall_examples_per_s": total_examples / max(total_step_s, 1e-9),
        "mean_step_s": total_step_s / max(global_step, 1),
        "elapsed_s": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")


def summarize(output_dir: Path) -> None:
    summaries = []
    curves = []
    for path in sorted(output_dir.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text()))
    for path in sorted(output_dir.glob("*/metrics.jsonl")):
        trial = path.parent.name
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["trial"] = trial
                curves.append(row)
    if summaries:
        keys = sorted({key for row in summaries for key in row})
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(summaries)
    if curves:
        keys = sorted({key for row in curves for key in row})
        with (output_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(curves)
    ranked = sorted(summaries, key=lambda row: (row["best_val_loss"], -row["best_val_accuracy"]))
    lines = [
        "# CIFAR-10 NorMuon Aspect Ablation\n",
        "| trial | best val loss | final val loss | best val acc | final val acc | examples/s | mean step ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['name']} | {row['best_val_loss']:.4f} | {row.get('final_val_loss', float('nan')):.4f} | "
            f"{100*row['best_val_accuracy']:.2f}% | {100*row.get('final_val_accuracy', float('nan')):.2f}% | "
            f"{row['overall_examples_per_s']:.0f} | {1000*row['mean_step_s']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these training runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials = [trial for trial in default_trials(args.seed, args.preset) if not args.only or args.only in trial.name]
    pending = list(trials)
    running: list[tuple[subprocess.Popen, str]] = []
    gpu_count = torch.cuda.device_count()
    next_gpu = 0
    while pending or running:
        while pending and len(running) < gpu_count:
            trial = pending.pop(0)
            run_dir = args.output_dir / trial.name
            if args.skip_existing and (run_dir / "summary.json").exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            trial_json = run_dir / "trial.json"
            trial_json.write_text(json.dumps(asdict(trial), indent=2) + "\n", encoding="utf-8")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--trial-json",
                str(trial_json),
                "--data-path",
                str(args.data_path),
                "--output-dir",
                str(args.output_dir),
                "--model",
                args.model,
                "--epochs",
                str(args.epochs),
                "--max-steps",
                str(args.max_steps),
                "--eval-bins",
                str(args.eval_bins),
                "--train-subset",
                str(args.train_subset),
                "--val-subset",
                str(args.val_subset),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--warmup-steps",
                str(args.warmup_steps),
                "--log-every",
                str(args.log_every),
                "--preset",
                args.preset,
            ]
            if args.sync_step_timing:
                cmd.append("--sync-step-timing")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(next_gpu % gpu_count)
            next_gpu += 1
            (run_dir / "resolved_command.sh").write_text("CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"] + " " + " ".join(cmd) + "\n", encoding="utf-8")
            stdout = (run_dir / "stdout.log").open("w", encoding="utf-8")
            stderr = (run_dir / "stderr.log").open("w", encoding="utf-8")
            print(f"Launching {trial.name} on GPU {env['CUDA_VISIBLE_DEVICES']}", flush=True)
            running.append((subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, text=True), trial.name))
        time.sleep(1.0)
        still_running = []
        for proc, name in running:
            code = proc.poll()
            if code is None:
                still_running.append((proc, name))
            elif code != 0:
                raise SystemExit(f"Trial {name} failed with code {code}; see {args.output_dir / name}")
        running = still_running
    summarize(args.output_dir)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
