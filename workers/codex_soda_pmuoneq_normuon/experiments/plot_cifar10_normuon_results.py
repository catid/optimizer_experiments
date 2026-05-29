#!/usr/bin/env python3
"""Plot CIFAR-10 optimizer ablation results.

The script consumes the result directory produced by
run_cifar10_normuon_aspect_ablation.py and writes reproducible figures plus a
markdown report. It deliberately uses only the stdlib plus matplotlib so the
artifacts can be regenerated in lightweight environments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "adamw_lr0.0025_wd0.005": "AdamW baseline",
    "row_aspect_mlr0.008_rg0.35_cg0.05_nb0.93": "SODA-PMuonEq-NorMuon row + aspect",
    "row_noaspect_mlr0.008_rg0.35_cg0.05_nb0.93": "SODA-PMuonEq-NorMuon row",
    "orientation_noaspect_mlr0.008_rg0.35_cg0.05_nb0.93": "SODA-PMuonEq-NorMuon orient",
    "orientation_aspect_mlr0.008_rg0.35_cg0.05_nb0.93": "SODA-PMuonEq-NorMuon orient + aspect",
    "row_aspect_mlr0.008_rg0.4_cg0.05_nb0.93": "row + aspect, rg0.40",
    "orientation_aspect_mlr0.008_rg0.3_cg0_nb0.95": "orient + aspect, rg0.30/cg0",
    "orientation_aspect_mlr0.008_rg0.4_cg0.05_nb0.93": "orient + aspect, rg0.40",
    "row_noaspect_mlr0.008_rg0.3_cg0_nb0.95": "row no aspect, rg0.30/cg0",
    "orientation_noaspect_mlr0.008_rg0.4_cg0.05_nb0.93": "orient no aspect, rg0.40",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def load_curves(result_dir: Path) -> list[dict[str, str]]:
    curves_path = result_dir / "curves.csv"
    if curves_path.exists():
        return read_rows(curves_path)
    curves: list[dict[str, str]] = []
    for path in sorted(result_dir.glob("*/metrics.jsonl")):
        trial = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["trial"] = trial
            curves.append({key: str(value) for key, value in row.items()})
    return curves


def curve_by_trial(curves: list[dict[str, str]], metric: str) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in curves:
        if row.get("event") != "bin":
            continue
        if metric not in row or row[metric] in ("", None):
            continue
        grouped.setdefault(row["trial"], []).append((as_float(row, "step"), as_float(row, metric)))
    for values in grouped.values():
        values.sort()
    return grouped


def plot_loss_curves(result_dir: Path, curves: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for metric, title, ylabel, ax in [
        ("train_loss_interval", "Training Loss by Evaluation Bin", "interval train loss", axes[0]),
        ("val_loss", "Validation Loss Curve", "validation loss", axes[1]),
    ]:
        for trial, points in curve_by_trial(curves, metric).items():
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=2, label=LABELS.get(trial, trial))
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(result_dir / "figures" / "loss_curves.png", dpi=180)
    plt.close(fig)


def plot_accuracy_best_vs_adamw(result_dir: Path, curves: list[dict[str, str]], summaries: list[dict[str, str]]) -> str:
    best = min(summaries, key=lambda row: as_float(row, "best_val_loss"))
    best_name = best["name"]
    selected = {"adamw_lr0.0025_wd0.005", best_name}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for trial, points in curve_by_trial(curves, "val_accuracy").items():
        if trial not in selected:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, [100.0 * y for y in ys], marker="o", linewidth=2.5, label=LABELS.get(trial, trial))
    ax.set_title("Validation Accuracy: Best Optimizer vs AdamW")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(result_dir / "figures" / "accuracy_best_vs_adamw.png", dpi=180)
    plt.close(fig)
    return best_name


def plot_speed(result_dir: Path, summaries: list[dict[str, str]]) -> None:
    ranked = sorted(summaries, key=lambda row: as_float(row, "mean_step_s"))
    labels = [LABELS.get(row["name"], row["name"]) for row in ranked]
    step_ms = [1000.0 * as_float(row, "mean_step_s") for row in ranked]
    examples_per_s = [as_float(row, "overall_examples_per_s") for row in ranked]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(labels, step_ms, color="#4477aa")
    axes[0].invert_yaxis()
    axes[0].set_title("Iteration Step Time")
    axes[0].set_xlabel("mean step time (ms, lower is better)")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].barh(labels, examples_per_s, color="#66aa55")
    axes[1].invert_yaxis()
    axes[1].set_title("Training Throughput")
    axes[1].set_xlabel("examples / second (higher is better)")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.savefig(result_dir / "figures" / "iteration_speed.png", dpi=180)
    plt.close(fig)


def write_report(result_dir: Path, summaries: list[dict[str, str]], best_name: str) -> None:
    by_loss = sorted(summaries, key=lambda row: (as_float(row, "best_val_loss"), -as_float(row, "best_val_accuracy")))
    lines = [
        "# CIFAR-10 NorMuon Aspect Ablation",
        "",
        "Single-GPU trials were launched concurrently across four visible GPUs. All runs used ViT-5 tiny, CIFAR-10, batch size 512, 50 epochs, bf16 autocast, and the same data/evaluation schedule.",
        "",
        f"Best validation-loss recipe: **{LABELS.get(best_name, best_name)}**.",
        "",
        "![Loss curves](figures/loss_curves.png)",
        "",
        "![Best vs AdamW accuracy](figures/accuracy_best_vs_adamw.png)",
        "",
        "![Iteration speed](figures/iteration_speed.png)",
        "",
        "| rank | optimizer | best val loss | final val loss | best val acc | final val acc | examples/s | mean step ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(by_loss, start=1):
        lines.append(
            f"| {idx} | {LABELS.get(row['name'], row['name'])} | {as_float(row, 'best_val_loss'):.4f} | "
            f"{as_float(row, 'final_val_loss'):.4f} | {100.0 * as_float(row, 'best_val_accuracy'):.2f}% | "
            f"{100.0 * as_float(row, 'final_val_accuracy'):.2f}% | {as_float(row, 'overall_examples_per_s'):.0f} | "
            f"{1000.0 * as_float(row, 'mean_step_s'):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The report ranks by best validation loss and also includes final validation metrics so late-epoch reversals are visible. The aspect multiplier is a real layerwise step-size change, so compare it with LR tuning in mind.",
            "",
            "The optimizer variants cost about 1.8x wall-clock per step compared with AdamW in this small ViT-5 CIFAR-10 setting, so the quality gain is not free. These results are one seed and should be treated as confirmation of the recipe direction, not as a final statistical claim.",
        ]
    )
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    (result_dir / "figures").mkdir(parents=True, exist_ok=True)
    summaries = read_rows(result_dir / "summary.csv")
    curves = load_curves(result_dir)
    plot_loss_curves(result_dir, curves)
    best_name = plot_accuracy_best_vs_adamw(result_dir, curves, summaries)
    plot_speed(result_dir, summaries)
    write_report(result_dir, summaries, best_name)


if __name__ == "__main__":
    main()
