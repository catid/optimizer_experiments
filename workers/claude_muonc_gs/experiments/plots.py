#!/usr/bin/env python3
"""Figures for the MuonC x MuonGS study (static PNGs for the README).

Encoding is fixed across every figure: colour is the decay law, panels split
"no projection" from "GS projection", and the fixed-norm reference is neutral
grey.  Values that are not directly labelled are in the tables of the write-up.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
# Validated (light surface, all-pairs): scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a"
DECAY_COLOR = {"corrected": "#2a78d6", "standard": "#eb6834", "none": "#1baf7a"}
DECAY_NAME = {"corrected": "corrected decay (C)", "standard": "standard decay (W)", "none": "no decay"}
REFERENCE = "#52514e"
HIDDEN = ("qkv", "attn_proj", "mlp_fc", "mlp_proj")


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans", "font.size": 9.5,
            "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
            "axes.edgecolor": GRID, "axes.linewidth": 0.8,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
            "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "legend.frameon": False, "legend.fontsize": 8.8,
            "lines.linewidth": 1.7, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
            "xtick.major.size": 0, "ytick.major.size": 0,
        }
    )


def marker_kw(color: str) -> dict[str, Any]:
    return dict(marker="o", markersize=6.2, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.4)


def load_metrics(trial_dir: Path, phase: str) -> list[dict[str, Any]]:
    rows = []
    for line in (trial_dir / "metrics.jsonl").read_text().splitlines():
        if f'"phase": "{phase}"' in line:
            rows.append(json.loads(line))
    return rows


def hidden_mean(row: dict[str, Any], prefix: str) -> float:
    values = [row[f"{prefix}/{label}"] for label in HIDDEN if f"{prefix}/{label}" in row]
    return float(np.mean(values)) if values else float("nan")


def end_label(ax, x: float, y: float, text: str, dy: float = 0.0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", ha="left", color=INK_2, fontsize=8.6)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def lr_sweep(stats: list[dict[str, Any]], protocol: str, path: Path, title: str, ylim: tuple[float, float] | None = None) -> None:
    """Loss vs LR; one panel per projection setting; best wd per (cell, lr)."""

    style()
    sub = [s for s in stats if s["protocol"] == protocol and math.isfinite(s["loss"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    panels = (("noGS", "No projection"), ("GS", "GS projection"))
    for ax, (prefix, name) in zip(axes, panels):
        for decay, short in (("none", "0"), ("standard", "W"), ("corrected", "C")):
            cell = f"{prefix}+{short}"
            by_lr: dict[float, dict[str, Any]] = {}
            for s in sub:
                if s["cell"] == cell and (s["lr"] not in by_lr or s["loss"] < by_lr[s["lr"]]["loss"]):
                    by_lr[s["lr"]] = s
            if not by_lr:
                continue
            lrs = sorted(by_lr)
            ax.plot(lrs, [by_lr[lr]["loss"] for lr in lrs], color=DECAY_COLOR[decay], label=DECAY_NAME[decay], **marker_kw(DECAY_COLOR[decay]))
        if prefix == "GS":
            fixed = sorted((s["lr"], s["loss"]) for s in sub if s["cell"] == "GS+0+fixednorm")
            if fixed:
                ax.plot(*zip(*fixed), color=REFERENCE, label="fixed norm (MuonH-style)", **marker_kw(REFERENCE))
        ax.set_xscale("log")
        ticks = [0.25e-3, 0.5e-3, 1e-3, 2e-3, 4e-3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{1e3 * tick:g}" for tick in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(name)
        ax.set_xlabel("peak learning rate (x 1e-3)")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel("validation loss (full pass)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title + "\nEach point is the best weight decay tried at that learning rate.", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, path)


def trajectories(
    bundle: Path, panels: list[tuple[str, list[tuple[str, str, str]]]], key: str, ylabel: str, path: Path, title: str,
    yscale: str = "linear", predicted: dict[str, tuple[list[int], list[float]]] | None = None,
) -> None:
    """Per-step diagnostics.  ``panels`` is [(panel title, [(trial name, label, color)])]."""

    style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 3.8), sharey=True, squeeze=False)
    for ax, (name, series) in zip(axes[0], panels):
        for trial, label, color in series:
            rows = load_metrics(bundle / "trials" / trial, "diag")
            xs = [r["step"] for r in rows]
            ys = [hidden_mean(r, key) for r in rows]
            if predicted and trial in predicted:
                px, py = predicted[trial]
                ax.plot(px, py, color=INK, linewidth=3.6, alpha=0.16, solid_capstyle="round")
            ax.plot(xs, ys, color=color, label=label)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_yscale(yscale)
        ax.legend(loc="best")
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def predicted_norm(bundle: Path, trial: str) -> tuple[list[int], list[float]]:
    """Closed-form norm recursion r^2 <- f^2 r^2 + (lr |N|)^2, mean over hidden matrices."""

    from run_muonc_gs_lm import lr_scale, trial_from_dict

    spec = trial_from_dict(json.loads((bundle / "trials" / trial / "trial.json").read_text()))
    rows = load_metrics(bundle / "trials" / trial, "diag")
    curves = []
    for label in HIDDEN:
        tau_at = {r["step"]: r[f"theta/{label}"] / r["lr"] * r[f"norm_ratio/{label}"] for r in rows if f"theta/{label}" in r}
        if not tau_at:
            continue
        tau = tau_at[min(tau_at)]
        r, curve = 1.0, []
        for step in range(1, spec.steps + 1):
            lr = spec.lr * lr_scale(spec, step)
            tau = tau_at.get(step, tau)
            factor = 1.0
            if spec.decay == "corrected":
                factor = 1.0 - spec.wd * lr * lr / spec.lr
            elif spec.decay == "standard":
                factor = 1.0 - spec.wd * lr
            r = math.sqrt((factor * r) ** 2 + (lr * tau) ** 2)
            curve.append(r)
        curves.append(curve)
    mean = np.mean(np.asarray(curves), axis=0)
    steps = list(range(1, spec.steps + 1))
    return steps[::20], [float(v) for v in mean[::20]]
