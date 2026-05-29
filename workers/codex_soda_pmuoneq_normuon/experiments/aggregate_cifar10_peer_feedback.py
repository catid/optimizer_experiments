#!/usr/bin/env python3
"""Aggregate peer-feedback CIFAR-10 optimizer runs across seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

LABELS = {
    "adamw_lr0.0025_wd0.005": "AdamW baseline",
    "row_aspect_mlr0.008_rg0.35_cg0.05_nb0.93": "row + aspect, rg0.35",
    "row_aspect_mlr0.008_rg0.4_cg0.05_nb0.93": "row + aspect, rg0.40",
    "orientation_aspect_mlr0.008_rg0.3_cg0_nb0.95": "orient + aspect, rg0.30/cg0",
    "orientation_aspect_mlr0.008_rg0.4_cg0.05_nb0.93": "orient + aspect, rg0.40",
    "row_noaspect_mlr0.008_rg0.3_cg0_nb0.95": "row no aspect, rg0.30/cg0",
    "orientation_noaspect_mlr0.008_rg0.4_cg0.05_nb0.93": "orient no aspect, rg0.40",
}

METRICS = (
    "best_val_loss",
    "final_val_loss",
    "best_val_accuracy",
    "final_val_accuracy",
    "overall_examples_per_s",
    "mean_step_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("result_dirs", type=Path, nargs="+")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def seed_from_dir(result_dir: Path) -> int:
    rows = read_rows(result_dir / "summary.csv")
    if not rows:
        raise ValueError(f"{result_dir} has an empty summary.csv")
    return int(float(rows[0]["seed"]))


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.fmean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_summaries(result_dirs: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_seed: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result_dir in result_dirs:
        seed = seed_from_dir(result_dir)
        for row in read_rows(result_dir / "summary.csv"):
            record: dict[str, object] = {
                "seed": seed,
                "name": row["name"],
                "label": LABELS.get(row["name"], row["name"]),
                "optimizer": row["optimizer"],
            }
            for metric in METRICS:
                record[metric] = as_float(row, metric)
            per_seed.append(record)
            grouped[row["name"]].append(record)

    aggregate: list[dict[str, object]] = []
    for name, rows in grouped.items():
        record = {
            "name": name,
            "label": LABELS.get(name, name),
            "seeds": len(rows),
        }
        for metric in METRICS:
            vals = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = mean(vals)
            record[f"{metric}_std"] = stdev(vals)
        aggregate.append(record)

    aggregate.sort(key=lambda row: (float(row["final_val_loss_mean"]), float(row["best_val_loss_mean"])))
    return per_seed, aggregate


def aggregate_curves(result_dirs: list[Path]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result_dir in result_dirs:
        path = result_dir / "curves.csv"
        if not path.exists():
            continue
        for row in read_rows(path):
            if row.get("event") != "bin":
                continue
            key = (row["trial"], int(float(row["step"])))
            for metric in ("train_loss_interval", "val_loss", "val_accuracy"):
                if row.get(metric) not in ("", None):
                    grouped[key][metric].append(as_float(row, metric))

    rows: list[dict[str, object]] = []
    for (name, step), metric_values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        record: dict[str, object] = {"name": name, "label": LABELS.get(name, name), "step": step}
        for metric, values in metric_values.items():
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values)
        rows.append(record)
    return rows


def plot_mean_curves(output_dir: Path, curve_rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in curve_rows:
        by_name[str(row["name"])].append(row)
    for name, rows in by_name.items():
        rows.sort(key=lambda row: int(row["step"]))
        xs = [int(row["step"]) for row in rows]
        ys = [float(row["val_loss_mean"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=LABELS.get(name, name))
    ax.set_title("Mean Validation Loss Across Seeds")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "figures" / "mean_validation_loss.png", dpi=180)
    plt.close(fig)


def plot_bars(output_dir: Path, aggregate: list[dict[str, object]]) -> None:
    specs = [
        ("final_val_loss_mean", "Final Validation Loss", "loss (lower is better)", "final_validation_loss.png"),
        ("final_val_accuracy_mean", "Final Validation Accuracy", "accuracy (higher is better)", "final_validation_accuracy.png"),
        ("mean_step_s_mean", "Mean Step Time", "step time ms (lower is better)", "mean_step_time.png"),
    ]
    for key, title, xlabel, filename in specs:
        ranked = sorted(aggregate, key=lambda row: float(row[key]), reverse="accuracy" in key)
        labels = [str(row["label"]) for row in ranked]
        values = [float(row[key]) for row in ranked]
        if key.endswith("_s_mean"):
            values = [1000.0 * value for value in values]
        if "accuracy" in key:
            values = [100.0 * value for value in values]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.barh(labels, values, color="#4477aa")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", alpha=0.25)
        fig.savefig(output_dir / "figures" / filename, dpi=180)
        plt.close(fig)


def write_report(output_dir: Path, aggregate: list[dict[str, object]], per_seed: list[dict[str, object]]) -> None:
    best_final = aggregate[0]
    best_transient = min(aggregate, key=lambda row: float(row["best_val_loss_mean"]))
    lines = [
        "# CIFAR-10 Peer-Feedback Aggregate",
        "",
        "ViT-5 tiny / CIFAR-10, 50 epochs, batch size 512, bf16 autocast, one single-GPU trial per visible GPU. The final comparison uses seeds 34000, 456, and 789.",
        "",
        f"Best mean final validation loss: **{best_final['label']}** ({float(best_final['final_val_loss_mean']):.4f} +/- {float(best_final['final_val_loss_std']):.4f}).",
        f"Best mean transient validation loss: **{best_transient['label']}** ({float(best_transient['best_val_loss_mean']):.4f} +/- {float(best_transient['best_val_loss_std']):.4f}).",
        "",
        "![Mean validation loss](figures/mean_validation_loss.png)",
        "",
        "![Final validation loss](figures/final_validation_loss.png)",
        "",
        "![Final validation accuracy](figures/final_validation_accuracy.png)",
        "",
        "![Mean step time](figures/mean_step_time.png)",
        "",
        "## Aggregate Table",
        "",
        "| rank | optimizer | final val loss | best val loss | final val acc | best val acc | examples/s | step ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(aggregate, start=1):
        lines.append(
            f"| {idx} | {row['label']} | "
            f"{float(row['final_val_loss_mean']):.4f} +/- {float(row['final_val_loss_std']):.4f} | "
            f"{float(row['best_val_loss_mean']):.4f} +/- {float(row['best_val_loss_std']):.4f} | "
            f"{100.0 * float(row['final_val_accuracy_mean']):.2f}% +/- {100.0 * float(row['final_val_accuracy_std']):.2f}% | "
            f"{100.0 * float(row['best_val_accuracy_mean']):.2f}% +/- {100.0 * float(row['best_val_accuracy_std']):.2f}% | "
            f"{float(row['overall_examples_per_s_mean']):.0f} | "
            f"{1000.0 * float(row['mean_step_s_mean']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Winners",
            "",
            "| seed | best final-loss optimizer | final val loss | best transient optimizer | best val loss |",
            "|---:|---|---:|---|---:|",
        ]
    )
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        by_seed[int(row["seed"])].append(row)
    for seed, rows in sorted(by_seed.items()):
        final_winner = min(rows, key=lambda row: float(row["final_val_loss"]))
        transient_winner = min(rows, key=lambda row: float(row["best_val_loss"]))
        lines.append(
            f"| {seed} | {final_winner['label']} | {float(final_winner['final_val_loss']):.4f} | "
            f"{transient_winner['label']} | {float(transient_winner['best_val_loss']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The peer-requested orientation plus aspect recipe was a strong early-training candidate, but the three-seed final-loss aggregate favors the row-wise aspect recipe. Orientation/no-aspect won one seed by final loss but also showed the largest late reversal on seed 789.",
            "",
            "All SODA-PMuonEq-NorMuon variants beat the tuned AdamW baseline on loss and accuracy. AdamW remains much faster per step, so the optimizer is a quality-first recipe on this small ViT-5 workload rather than a raw-throughput winner.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    per_seed, aggregate = aggregate_summaries(args.result_dirs)
    curve_rows = aggregate_curves(args.result_dirs)
    write_rows(
        output_dir / "per_seed_summary.csv",
        per_seed,
        ["seed", "name", "label", "optimizer", *METRICS],
    )
    aggregate_fields = ["name", "label", "seeds"]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_rows(output_dir / "aggregate_summary.csv", aggregate, aggregate_fields)
    write_rows(
        output_dir / "aggregate_curves.csv",
        curve_rows,
        [
            "name",
            "label",
            "step",
            "train_loss_interval_mean",
            "train_loss_interval_std",
            "val_loss_mean",
            "val_loss_std",
            "val_accuracy_mean",
            "val_accuracy_std",
        ],
    )
    plot_mean_curves(output_dir, curve_rows)
    plot_bars(output_dir, aggregate)
    write_report(output_dir, aggregate, per_seed)


if __name__ == "__main__":
    main()
