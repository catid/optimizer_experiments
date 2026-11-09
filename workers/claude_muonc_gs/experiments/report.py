#!/usr/bin/env python3
"""Write ``summary.md`` and ``summary.csv`` for a MuonC x MuonGS result bundle.

Every number in the write-up comes from the tables this script generates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze  # noqa: E402

HNET = "pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16"
HNET_FREE = HNET.replace("fallback_mode", "emb=matrix_nogs, fallback_mode")
SP, STD, LONG = "sp-cosine-8k", "std-wsd-8k", "sp-cosine-32k"
SP_CONST, SP_WSD = "sp-cosine-8k-final_lr_scale=1", "sp-wsd-8k"

# (label, protocol, cell, lr, wd)
CURATED: dict[str, list[tuple[str, str, str, float, float]]] = {
    "Single-pass protocol, 8K steps: one row per approach at its best (lr, wd)": [
        ("Unprojected, corrected decay", SP, "noGS+C", 1e-3, 0.4),
        ("Unprojected, standard decay (MuonW)", SP, "noGS+W", 1e-3, 0.2),
        ("Unprojected, no decay (Muon)", SP, "noGS+0", 1e-3, 0.0),
        ("Half projection, corrected decay", SP, "GS0.5+C", 1e-3, 0.2),
        ("Half projection, no decay", SP, "GS0.5+0", 1e-3, 0.0),
        ("Projected, no decay (MuonGS)", SP, "GS+0", 7e-4, 0.0),
        ("Projected, fixed norm (MuonH-style)", SP, "GS+0+fixednorm", 7e-4, 0.0),
        ("Projected, corrected decay wd 0.0125", SP, "GS+C", 7e-4, 0.0125),
        ("Projected, corrected decay wd 0.05", SP, "GS+C", 5e-4, 0.05),
        ("Projected, standard decay wd 0.05", SP, "GS+W", 7e-4, 0.05),
    ],
    "Single-pass protocol: what closes the projection's gap": [
        ("Projected, default init (reference)", SP, "GS+0", 7e-4, 0.0),
        ("+ embeddings exempt from projection", SP, "GS+0 [emb=matrix_nogs]", 1e-3, 0.0),
        ("+ replay unprojected run's effective LR (norm pinned)", SP, "GS+0+fixednorm [lr_profile]", 1e-3, 0.0),
        ("+ adaptive radial step, c=2", SP, "GS+0 [radial_lr=2]", 1e-3, 0.0),
        ("+ adaptive radial step, c=1", SP, "GS+0 [radial_lr=1]", 1e-3, 0.0),
        ("+ start at 2x scale", SP, "GS+0 [init_scale=2]", 1e-3, 0.0),
        ("+ start at 3x scale", SP, "GS+0 [init_scale=3]", 1.5e-3, 0.0),
        ("+ 2x scale, corrected decay holding it (wd 0.0125)", SP, "GS+C [init_scale=2]", 1e-3, 0.0125),
        ("+ 2x scale and replayed effective LR (norm pinned)", SP, "GS+0+fixednorm [init_scale=2, lr_profile]", 2e-3, 0.0),
        ("Half projection, corrected decay", SP, "GS0.5+C", 1e-3, 0.2),
        ("Unprojected, standard decay (target)", SP, "noGS+W", 1e-3, 0.2),
    ],
    "Repo-standard protocol (8M-token cache, about 4 passes), 8K steps": [
        ("Unprojected, standard decay wd 0.4", STD, "noGS+W", 1e-3, 0.4),
        ("Unprojected, corrected decay wd 0.4", STD, "noGS+C", 1e-3, 0.4),
        ("Projected, no decay (MuonGS)", STD, "GS+0", 7e-4, 0.0),
        ("Projected, corrected decay wd 0.0125", STD, "GS+C", 7e-4, 0.0125),
        ("Projected, fixed norm (MuonH-style)", STD, "GS+0+fixednorm", 5e-4, 0.0),
        ("Projected, embeddings exempt", STD, "GS+0 [emb=matrix_nogs]", 5e-4, 0.0),
        ("Projected + adaptive radial step c=1", STD, "GS+0 [radial_lr=1]", 7e-4, 0.0),
        ("Unprojected, no decay (Muon)", STD, "noGS+0", 2e-3, 0.0),
        ("This port as MuonW at the README settings (wd 0.05)", STD, "noGS+W", 1.8e-3, 0.05),
        ("Repo PlainMuon at the README settings (wd 0.05)", STD, "repo PlainMuon", 1.8e-3, 0.05),
        ("Repo AdamW at the README settings", STD, "repo AdamW", 5e-4, 0.05),
    ],
    "H-Net optimizer stack on the single-pass protocol (LR 1e-3)": [
        ("Unprojected, standard decay wd 0.05 (previous path)", SP, f"noGS+W [{HNET}]", 1e-3, 0.05),
        ("Unprojected, corrected decay wd 0.05", SP, f"noGS+C [{HNET}]", 1e-3, 0.05),
        ("Projected, corrected decay wd 0.05 (MuonC as configured)", SP, f"GS+C [{HNET}]", 1e-3, 0.05),
        ("Projected, standard decay wd 0.05", SP, f"GS+W [{HNET}]", 1e-3, 0.05),
        ("Projected, no decay", SP, f"GS+0 [{HNET}]", 1e-3, 0.0),
        ("Projected + C, embeddings exempt", SP, f"GS+C [{HNET_FREE}]", 1e-3, 0.05),
        ("Half projection, corrected decay wd 0.05", SP, f"GS0.5+C [{HNET}]", 1e-3, 0.05),
        ("Projected + C wd 0.0125 at 2x scale", SP, f"GS+C [{HNET}, init_scale=2]", 1e-3, 0.0125),
    ],
    "Projection on hidden matrices only (embedding/output matrix exempt), single-pass, 8K steps": [
        ("Plain base: unprojected, corrected decay (reference)", SP, "noGS+C", 1e-3, 0.4),
        ("Plain base: everything projected, no decay (reference)", SP, "GS+0", 7e-4, 0.0),
        ("Plain base: hidden projected, no decay", SP, "GS+0 [emb=matrix_nogs]", 1e-3, 0.0),
        ("Plain base: hidden projected, corrected decay wd 0.0125", SP, "GS+C [emb=matrix_nogs]", 1e-3, 0.0125),
        ("Plain base: hidden projected, corrected decay wd 0.05", SP, "GS+C [emb=matrix_nogs]", 7e-4, 0.05),
        ("Plain base: hidden projected, corrected decay wd 0.05", SP, "GS+C [emb=matrix_nogs]", 1e-3, 0.05),
        ("Plain base: hidden projected, standard decay wd 0.05", SP, "GS+W [emb=matrix_nogs]", 1e-3, 0.05),
        ("Plain base: hidden projected, corrected decay wd 0.2", SP, "GS+C [emb=matrix_nogs]", 1e-3, 0.2),
        ("Plain base: hidden projected, fixed norm", SP, "GS+0+fixednorm [emb=matrix_nogs]", 1e-3, 0.0),
        ("Plain base: hidden projected + C wd 0.05 + radial step c=1", SP, "GS+C [emb=matrix_nogs, radial_lr=1]", 1e-3, 0.05),
        ("H-Net stack: unprojected, standard decay wd 0.05 (reference)", SP, f"noGS+W [{HNET}]", 1e-3, 0.05),
        ("H-Net stack: everything projected, corrected decay wd 0.05 (reference)", SP, f"GS+C [{HNET}]", 1e-3, 0.05),
        ("H-Net stack: hidden projected, no decay", SP, f"GS+0 [{HNET_FREE}]", 1e-3, 0.0),
        ("H-Net stack: hidden projected, corrected decay wd 0.0125", SP, f"GS+C [{HNET_FREE}]", 1e-3, 0.0125),
        ("H-Net stack: hidden projected, corrected decay wd 0.05", SP, f"GS+C [{HNET_FREE}]", 7e-4, 0.05),
        ("H-Net stack: hidden projected, corrected decay wd 0.05", SP, f"GS+C [{HNET_FREE}]", 1e-3, 0.05),
        ("H-Net stack: hidden projected, standard decay wd 0.05", SP, f"GS+W [{HNET_FREE}]", 1e-3, 0.05),
        ("H-Net stack: hidden projected, corrected decay wd 0.2", SP, f"GS+C [{HNET_FREE}]", 1e-3, 0.2),
        ("H-Net stack: hidden projected, fixed norm", SP, f"GS+0+fixednorm [{HNET_FREE}]", 1e-3, 0.0),
        ("H-Net stack: half projection (all matrices), corrected decay wd 0.05", SP, f"GS0.5+C [{HNET}]", 1e-3, 0.05),
    ],
    "LR schedule and the free-norm accumulator, single-pass, 8K steps": [
        ("Unprojected MuonW wd 0.2: cosine to 10%", SP, "noGS+W", 1e-3, 0.2),
        ("Unprojected MuonW wd 0.2: constant, last 20% linear to 10%", SP_WSD, "noGS+W", 1e-3, 0.2),
        ("Unprojected MuonW wd 0.2: constant LR", SP_CONST, "noGS+W", 1e-3, 0.2),
        ("Projected + C wd 0.0125: cosine to 10%", SP, "GS+C", 7e-4, 0.0125),
        ("Projected + C wd 0.0125: constant, last 20% linear to 10%", SP_WSD, "GS+C", 7e-4, 0.0125),
        ("Projected + C wd 0.0125: constant LR", SP_CONST, "GS+C", 7e-4, 0.0125),
        ("Free-norm accumulator (projected, no decay, constant LR)", SP_CONST, "GS+0", 1e-3, 0.0),
        ("Free-norm accumulator (projected, no decay, constant LR)", SP_CONST, "GS+0", 2e-3, 0.0),
        ("Free-norm accumulator (projected, no decay, constant LR)", SP_CONST, "GS+0", 4e-3, 0.0),
        ("Free-norm accumulator (projected, no decay, constant LR)", SP_CONST, "GS+0", 8e-3, 0.0),
        ("Inverse-sqrt LR on a pinned norm", SP_CONST, "GS+0+fixednorm [emulate_gs_theta0=0.02]", 2e-3, 0.0),
        ("Inverse-sqrt LR on a pinned norm", SP_CONST, "GS+0+fixednorm [emulate_gs_theta0=0.04]", 4e-3, 0.0),
        ("Inverse-sqrt LR on a pinned norm", SP_CONST, "GS+0+fixednorm [emulate_gs_theta0=0.08]", 8e-3, 0.0),
        ("Free-norm accumulator + terminal decay", SP_WSD, "GS+0", 2e-3, 0.0),
        ("Free-norm accumulator + terminal decay", SP_WSD, "GS+0", 4e-3, 0.0),
        ("Free-norm accumulator + terminal decay", SP_WSD, "GS+0", 8e-3, 0.0),
        ("Inverse-sqrt LR on a pinned norm + terminal decay", SP_WSD, "GS+0+fixednorm [emulate_gs_theta0=0.04]", 4e-3, 0.0),
    ],
}

LONG_TITLE = "Long horizon: 32K steps, single-pass (every configuration run, best first)"


def find(stats: list[dict], protocol: str, cell: str, lr: float, wd: float) -> dict | None:
    for s in stats:
        if s["protocol"] == protocol and s["cell"] == cell and math.isclose(s["lr"], lr, rel_tol=1e-9) and math.isclose(s["wd"], wd, rel_tol=1e-9, abs_tol=1e-12):
            return s
    return None


def describe(entry: dict) -> tuple[str, str, str]:
    """(seed-123 loss, fresh-seed mean +- std (n), norm ratio)."""

    if not math.isfinite(entry["loss"]):
        return "diverged", "", ""
    seeds = entry["loss_by_seed"]
    select = f"{seeds[analyze.SELECTION_SEED]:.4f}" if analyze.SELECTION_SEED in seeds else ""
    fresh = [v for seed, v in seeds.items() if seed != analyze.SELECTION_SEED]
    if len(fresh) > 1:
        fresh_text = f"{np.mean(fresh):.4f} ± {np.std(fresh, ddof=1):.4f} (n={len(fresh)})"
    elif fresh:
        fresh_text = f"{fresh[0]:.4f} (n=1)"
    else:
        fresh_text = ""
    hidden = [entry[f"norm/{k}"] for k in ("qkv", "attn_proj", "mlp_fc", "mlp_proj") if f"norm/{k}" in entry]
    norm = f"{np.mean(hidden):.2f}" if hidden else ""
    if "norm/emb" in entry:
        norm += f" / {entry['norm/emb']:.2f}"
    return select, fresh_text, norm


def curated_tables(stats: list[dict]) -> str:
    lines: list[str] = []
    for title, rows in CURATED.items():
        body = []
        for label, protocol, cell, lr, wd in rows:
            entry = find(stats, protocol, cell, lr, wd)
            if entry is None:
                continue
            select, fresh, norm = describe(entry)
            body.append(f"| {label} | {lr:g} | {wd:g} | {select} | {fresh} | {norm} |")
        if not body:
            continue
        lines += [f"\n## {title}\n", "| Configuration | LR | wd | Seed 123 | Fresh seeds 124+ | Final norm ratio hidden / emb |", "|---|---:|---:|---:|---:|---:|", *body]
    long_rows = sorted((e for e in stats if e["protocol"] == LONG and math.isfinite(e["loss"])), key=lambda e: e["loss"])
    if long_rows:
        lines += [f"\n## {LONG_TITLE}\n", "| Cell | LR | wd | Seed 123 | Fresh seeds 124+ | Mean of all seeds | Final norm ratio hidden / emb |", "|---|---:|---:|---:|---:|---:|---:|"]
        for entry in long_rows:
            select, fresh, norm = describe(entry)
            lines.append(f"| {entry['cell']} | {entry['lr']:g} | {entry['wd']:g} | {select} | {fresh} | {entry['loss']:.4f} | {norm} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    rows, missing = analyze.load(args.bundle)
    stats = analyze.group_stats(rows)
    stages: dict[str, int] = {}
    for row in rows:
        stages[row.get("stage", "")] = stages.get(row.get("stage", ""), 0) + 1
    text = [
        "# MuonC x MuonGS on the GPT-2-tokenized 50M LM proxy: generated tables",
        "",
        f"{len(rows)} finished trials ({sum(bool(r.get('diverged')) for r in rows)} diverged, {len(missing)} unfinished). "
        "Metric: validation loss over 256 sequential batches (`final_full_val_loss`). "
        "Configurations were selected on seed 123; the fresh-seed column is the unbiased estimate.",
        "",
        "Trials per stage: " + ", ".join(f"`{k}` {v}" for k, v in sorted(stages.items())),
        curated_tables(stats),
        "\n## Every configuration, by learning rate (mean over all seeds run)",
        analyze.grid_tables(stats),
        "",
    ]
    (args.bundle / "summary.md").write_text("\n".join(text))
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (args.bundle / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    environment = {
        "gpu": rows[0].get("gpu_name"), "torch": rows[0].get("torch"), "trials": len(rows),
        "model": "TinyGPT 8 layers, width 512, 8 heads, context 256, GPT-2 vocab, tied embeddings, 51,045,888 parameters",
        "data": "HuggingFaceFW/fineweb-edu sample-10BT, gpt2 tokenizer + EOS, shuffle buffer 10000 seed 123; "
                "validation = first 524,288 tokens of the stream; std = next 8,000,000 tokens; sp = next 512,000,000 tokens",
        "batch": "16 x 256 tokens, BF16 autocast, FP32 parameters, clip_grad_norm 1.0",
    }
    (args.bundle / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print((args.bundle / "summary.md").read_text().split("## Every configuration")[0])


if __name__ == "__main__":
    main()
