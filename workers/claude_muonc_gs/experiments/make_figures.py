#!/usr/bin/env python3
"""Render the figures of the MuonC x MuonGS write-up from a result bundle."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import plots  # noqa: E402
from plots import DECAY_COLOR, INK_2, REFERENCE, SURFACE, hidden_mean, load_metrics, marker_kw, save, style  # noqa: E402

BLUE, ORANGE, AQUA = DECAY_COLOR["corrected"], DECAY_COLOR["standard"], DECAY_COLOR["none"]


def exists(bundle: Path, *names: str) -> bool:
    return all((bundle / "trials" / name / "summary.json").exists() for name in names)


def fig_norms(bundle: Path, out: Path) -> None:
    unprojected = [
        ("sp-cosine8k_nogs_none_lr0.001_s123", "no decay", AQUA),
        ("sp-cosine8k_nogs_standard_lr0.001_wd0.2_s123", "standard decay, wd 0.2", ORANGE),
        ("sp-cosine8k_nogs_corrected_lr0.001_wd0.4_s123", "corrected decay, wd 0.4", BLUE),
    ]
    projected = [
        ("sp-cosine8k_gs_none_lr0.001_s123", "no decay", AQUA),
        ("sp-cosine8k_gs_standard_lr0.001_wd0.2_s123", "standard decay, wd 0.2", ORANGE),
        ("sp-cosine8k_gs_corrected_lr0.001_wd0.2_s123", "corrected decay, wd 0.2", BLUE),
    ]
    if not exists(bundle, *[n for n, _l, _c in unprojected + projected]):
        return
    predicted = {name: plots.predicted_norm(bundle, name) for name, _l, _c in projected}
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    for ax, title, series in ((axes[0], "No projection: the loss moves the norm", unprojected),
                              (axes[1], "GS projection: the schedule moves the norm", projected)):
        for name, label, color in series:
            rows = load_metrics(bundle / "trials" / name, "diag")
            if name in predicted:
                ax.plot(*predicted[name], color=plots.INK, linewidth=4.2, alpha=0.15)
            ax.plot([r["step"] for r in rows], [hidden_mean(r, "norm_ratio") for r in rows], color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[0].set_ylabel("hidden-matrix norm  |W| / |W at init|")
    axes[1].plot([], [], color=plots.INK, linewidth=4.2, alpha=0.15, label="closed-form prediction")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="center right")
    fig.suptitle("Weight norm over training, single-pass protocol, peak LR 1e-3", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig_norms.png")


def fig_effective_step(bundle: Path, out: Path) -> None:
    series = [
        ("sp-cosine8k_nogs_standard_lr0.001_wd0.2_s123", "unprojected, wd 0.2, LR 1e-3", ORANGE),
        ("sp-cosine8k_gs_none_lr0.0007_s123", "projected, LR 7e-4", AQUA),
        ("sp-cosine8k_iscale2_gs_none_lr0.001_s123", "projected, 2x init scale, LR 1e-3", BLUE),
    ]
    if not exists(bundle, *[n for n, _l, _c in series]):
        return
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for name, label, color in series:
        diag = load_metrics(bundle / "trials" / name, "diag")
        evals = load_metrics(bundle / "trials" / name, "eval")
        steps = [r["step"] for r in diag]
        axes[0].plot(steps, [1e3 * hidden_mean(r, "theta") for r in diag], color=color, label=label)
        scale = 2.0 if "iscale2" in name else 1.0
        axes[1].plot(steps, [scale * hidden_mean(r, "norm_ratio") for r in diag], color=color, label=label)
        axes[2].plot([r["step"] for r in evals], [r["val_loss"] for r in evals], color=color, label=label)
    axes[0].set_title("Relative step size")
    axes[0].set_ylabel("LR x |update| / |W|   (x 1e-3)")
    axes[1].set_title("Operating scale")
    axes[1].set_ylabel("|W| / |W at default init|")
    axes[2].set_title("Validation loss")
    axes[2].set_ylabel("loss (16 batches)")
    axes[2].set_ylim(4.2, 5.4)
    for ax in axes:
        ax.set_xlabel("step")
    axes[0].legend(loc="upper right")
    fig.suptitle("What the projection removes: the early anneal and the move to a larger scale", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig_effective_step.png")


def fig_finalists(
    stats: list[dict], protocol: str, out: Path, title: str, rows: list[tuple[str, str, float, float]], all_seeds: bool = False
) -> None:
    """Dot plot: fresh-seed mean (or the mean of every seed run) with the range of individual seeds."""

    found = []
    for label, cell, lr, wd in rows:
        match = [s for s in stats if s["protocol"] == protocol and s["cell"] == cell and s["lr"] == lr and s["wd"] == wd and math.isfinite(s["loss"])]
        if match:
            found.append((label, match[0]))
    if not found:
        return

    def chosen(entry: dict) -> list[float]:
        seeds = entry["loss_by_seed"]
        if all_seeds:
            return list(seeds.values())
        return [v for seed, v in seeds.items() if seed != analyze.SELECTION_SEED] or list(seeds.values())

    def fresh_mean(entry: dict) -> float:
        return float(np.mean(chosen(entry)))

    found.sort(key=lambda item: fresh_mean(item[1]))
    style()
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(found) + 1.5))
    for index, (label, entry) in enumerate(found):
        y = len(found) - 1 - index
        fresh = chosen(entry)
        mean = float(np.mean(fresh))
        ax.plot([min(fresh), max(fresh)], [y, y], color=BLUE, linewidth=2.2, solid_capstyle="round")
        ax.plot([mean], [y], linestyle="none", **marker_kw(BLUE))
        ax.annotate(f"{mean:.3f}" + (f"  (n={len(fresh)})" if len(fresh) > 1 else "  (1 seed)"), (max(fresh), y), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left", color=INK_2, fontsize=8.6)
    ax.set_yticks(range(len(found)))
    ax.set_yticklabels([label for label, _e in reversed(found)], color=plots.INK)
    ax.set_xlabel("validation loss (full pass); dot = mean of " + ("the seeds run" if all_seeds else "fresh seeds") + ", bar = seed range")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18, y=0.08)
    ax.set_title(title)
    fig.tight_layout()
    save(fig, out / f"fig_finalists_{protocol.split('-')[0]}{'' if protocol.endswith('8k') else '_' + protocol.split('-')[-1]}.png")


def fig_decay_law(bundle: Path, out: Path) -> None:
    """C vs W under the projection when the norm window is well inside the run."""

    pairs = [
        ("8K steps, LR 4e-3, wd 0.05 (window 2.5K steps)",
         "sp-cosine8k_gs_corrected_lr0.004_wd0.05_s123", "sp-cosine8k_gs_standard_lr0.004_wd0.05_s123"),
        ("32K steps, LR 2e-3, wd 0.025, 2x init (window 10K steps)",
         "sp-cosine32k_iscale2_gs_corrected_lr0.002_wd0.025_s123", "sp-cosine32k_iscale2_gs_standard_lr0.002_wd0.025_s123"),
    ]
    pairs = [p for p in pairs if exists(bundle, p[1], p[2])]
    if not pairs:
        return
    import json

    style()
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.9 * len(pairs), 3.8), squeeze=False)
    for ax, (title, corrected, standard) in zip(axes[0], pairs):
        for name, label, color in ((corrected, "corrected decay", BLUE), (standard, "standard decay", ORANGE)):
            rows = load_metrics(bundle / "trials" / name, "diag")
            loss = json.loads((bundle / "trials" / name / "summary.json").read_text())["final_full_val_loss"]
            ax.plot([r["step"] for r in rows], [hidden_mean(r, "norm_ratio") for r in rows], color=color, label=f"{label}  (final loss {loss:.3f})")
        ax.set_title(title, fontsize=9.6)
        ax.set_xlabel("step")
        ax.legend(loc="lower right")
    axes[0][0].set_ylabel("hidden-matrix norm  |W| / |W at init|")
    fig.suptitle("Under the projection, standard decay drags the scale down with the LR schedule", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig_decay_law.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    out = args.bundle / "plots"
    rows, _missing = analyze.load(args.bundle)
    stats = analyze.group_stats(rows)
    plots.lr_sweep(stats, "sp-cosine-8k", out / "fig_lr_sweep_sp.png", "Single-pass protocol (512M-token cache, cosine to 10%), 8K steps", ylim=(4.30, 4.72))
    plots.lr_sweep(stats, "std-wsd-8k", out / "fig_lr_sweep_std.png", "Repo-standard protocol (8M-token cache, about 4 passes, WSD), 8K steps", ylim=(4.74, 5.12))
    fig_norms(args.bundle, out)
    fig_effective_step(args.bundle, out)
    fig_decay_law(args.bundle, out)
    fig_finalists(stats, "sp-cosine-8k", out, "Single-pass protocol, 8K steps: finalists on fresh seeds", [
        ("Unprojected + corrected decay (wd 0.4)", "noGS+C", 1e-3, 0.4),
        ("Unprojected + standard decay (wd 0.2)", "noGS+W", 1e-3, 0.2),
        ("Half projection + corrected decay (wd 0.2)", "GS0.5+C", 1e-3, 0.2),
        ("Half projection, no decay", "GS0.5+0", 1e-3, 0.0),
        ("Projected, pinned at 2x init, replayed LR", "GS+0+fixednorm [init_scale=2, lr_profile]", 2e-3, 0.0),
        ("Projected + radial step (c=1)", "GS+0 [radial_lr=1]", 1e-3, 0.0),
        ("Projected + corrected decay holding 2x init", "GS+C [init_scale=2]", 1e-3, 0.0125),
        ("Projected, 2x init scale", "GS+0 [init_scale=2]", 1e-3, 0.0),
        ("Unprojected, no decay", "noGS+0", 1e-3, 0.0),
        ("Projected, embeddings exempt", "GS+0 [emb=matrix_nogs]", 1e-3, 0.0),
        ("Projected (MuonGS), default init", "GS+0", 7e-4, 0.0),
        ("Projected, fixed norm (MuonH-style)", "GS+0+fixednorm", 7e-4, 0.0),
        ("Projected + corrected decay, wd 0.05", "GS+C", 5e-4, 0.05),
    ])
    fig_finalists(stats, "std-wsd-8k", out, "Repo-standard protocol, 8K steps: finalists on fresh seeds", [
        ("Unprojected + standard decay (wd 0.4)", "noGS+W", 1e-3, 0.4),
        ("Projected (MuonGS)", "GS+0", 7e-4, 0.0),
        ("Projected + corrected decay (wd 0.0125)", "GS+C", 7e-4, 0.0125),
        ("Projected, fixed norm (MuonH-style)", "GS+0+fixednorm", 5e-4, 0.0),
        ("README baseline: repo PlainMuon, wd 0.05", "repo PlainMuon", 1.8e-3, 0.05),
    ])
    fig_finalists(stats, "sp-cosine-32k", out, "Single-pass protocol, 32K steps: each family at its best LR", [
        ("Half projection, no decay", "GS0.5+0", 7e-4, 0.0),
        ("Projected at 2x scale, fixed norm", "GS+0+fixednorm [init_scale=2]", 7e-4, 0.0),
        ("Projected at 2x scale, no decay", "GS+0 [init_scale=2]", 7e-4, 0.0),
        ("Projected at 2x scale, corrected decay", "GS+C [init_scale=2]", 7e-4, 0.0125),
        ("Unprojected, corrected decay", "noGS+C", 5e-4, 0.4),
        ("Half projection, corrected decay", "GS0.5+C", 5e-4, 0.2),
        ("Unprojected, no decay", "noGS+0", 7e-4, 0.0),
        ("Unprojected, standard decay", "noGS+W", 5e-4, 0.2),
        ("Projected, default init, no decay", "GS+0", 3.5e-4, 0.0),
        ("Projected, default init, corrected decay wd 0.05", "GS+C", 5e-4, 0.05),
    ], all_seeds=True)
    print("figures written to", out)


if __name__ == "__main__":
    main()
