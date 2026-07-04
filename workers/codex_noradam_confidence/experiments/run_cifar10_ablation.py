#!/usr/bin/env python3
"""Run CIFAR-10 optimizer ablations for ViT-5 + AnchorMuon.

The supervisor process launches one worker per visible GPU. Each worker trains a
single small ViT-5 model, writes JSONL curves, and exits. The default quick preset
tunes AdamW and several AnchorMuon ablations with the same model, data subset,
batch size, and epoch budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from timm.models import create_model
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import models_vit5  # noqa: F401  Registers vit5_micro/vit5_tiny with timm.
import optimizer as root_optimizer
from golden_soda_pmuoneq_normuon import GoldenMuon
from optim_anchormuon import AnchorMuon
from optim_factory import _anchor_param_groups
from optim_sfplus import SFPlusAnchorMuon


@dataclass
class TrialConfig:
    name: str
    optimizer: str
    lr: float
    seed: int | None = None
    lr_schedule: str = "cosine"
    weight_decay: float = 0.05
    soda: str = "matrix"
    pmuon_eq: bool = True
    use_gram: bool = True
    row_gamma: float = 0.20
    col_gamma: float = 0.0
    pmuon_beta: float = 0.95
    momentum: float = 0.95
    amuse: bool = True
    mimuon: bool = False
    mimuon_mix: float = 0.85
    normuon: bool = False
    normuon_beta: float = 0.95
    normuon_aspect_scale: bool = False
    root_grouping: str = "anchor"
    root_normuon_mode: str = "row"
    fallback_mode: str = "rms"
    fallback_lr_mult: float = 1.0
    fallback_beta1: float = 0.90
    fallback_beta2: float = 0.95
    fallback_weight_decay: float = 0.0
    sfplus_polyak: bool = False
    sfplus_c_warmup_enabled: bool = False
    sfplus_beta_anneal: bool = False
    sfplus_adamc_decay: bool = False
    sfplus_inner_momentum: bool = False
    sfplus_beta1: float = 0.90
    sfplus_beta1_max: float = 0.965
    sfplus_beta1_anneal_steps: int = 0
    sfplus_polyak_beta: float = 0.0
    sfplus_c_warmup: int = 0
    sfplus_r: float = 0.0
    sfplus_weight_lr_power: float = 2.0
    external_lr: bool = False
    ema_beta: float = 0.0
    ema_gamma: float = 0.99
    ema_warmup_frac: float = 0.30
    ema_rest_frac: float = 0.20
    muown_mag_lr_mult: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-json", type=Path)
    parser.add_argument("--two-stage", action="store_true",
                        help="Run HPO first, select best configs by family, then run fixed-step 8-bin finals")
    parser.add_argument("--final-from-hpo", type=Path,
                        help="Run only the fixed-step finals from an existing hpo/best_by_family.json")
    parser.add_argument(
        "--preset",
        default="quick",
        choices=[
            "smoke",
            "quick",
            "full",
            "sweep",
            "focused20",
            "confidence",
            "feedback",
            "best_cifar10",
            "root_normuon_ablation",
            "root_soda_ablation",
            "root_lr_schedule_sweep",
            "root_fallback_modes",
            "cifar5_compare",
            "sfplus_combo20",
            "component_ablation",
            "muown_ema_cifar10",
            "ema_muown_cifar10_hard",
        ],
    )
    parser.add_argument("--only", default="", help="Regex filter for trial names")
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Supervisor mode: skip trials whose output directory already has summary.json.",
    )
    parser.add_argument("--data-path", type=Path, default=Path("/home/catid/screen/repos/TinyRecursiveModels/data/cifar10"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "results" / "cifar10_anchormuon")
    parser.add_argument("--model", default="vit5_micro")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many optimizer steps instead of using all epoch steps")
    parser.add_argument("--eval-bins", type=int, default=8,
                        help="Number of fixed-step validation bins for final loss-curve comparison")
    parser.add_argument("--hpo-epochs", type=int, default=4)
    parser.add_argument("--hpo-max-steps", type=int, default=0)
    parser.add_argument("--final-epochs", type=int, default=12)
    parser.add_argument("--final-max-steps", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--val-subset", type=int, default=2000)
    parser.add_argument(
        "--val-source",
        choices=["test", "train_split"],
        default="train_split",
        help=(
            "Validation source. Use train_split for HPO/model selection and reserve "
            "CIFAR-10 train=False for final test evaluation via --eval-test."
        ),
    )
    parser.add_argument(
        "--train-val-size",
        type=int,
        default=5000,
        help="Number of CIFAR-10 train examples held out for validation when --val-source=train_split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=12345,
        help="Seed for deterministic train/validation split. Kept separate from trial seed.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the official CIFAR-10 test split once at the end of each selected final run.",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=0,
        help="Optional official test subset size for debugging. 0 means full test split.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=max(2, (os.cpu_count() or 8) // 4))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", default="",
                        help="Comma-separated seeds. Supervisor duplicates selected trials for each seed.")
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--lr-final-scale", type=float, default=0.1)
    parser.add_argument("--wsd-decay-frac", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--no-sync-step-timing",
        dest="sync_step_timing",
        action="store_false",
        help=(
            "Skip CUDA synchronization around each training step. Per-step timing "
            "then measures CPU launch time, but end-to-end interval throughput is less perturbed."
        ),
    )
    parser.set_defaults(sync_step_timing=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_experiment_args(args: argparse.Namespace) -> None:
    if args.val_source == "test" and args.eval_test:
        raise ValueError("--val-source=test cannot be combined with --eval-test; validation would overlap test")
    if not args.worker and args.val_source == "test":
        raise ValueError(
            "--val-source=test is not allowed for supervisor/HPO runs. "
            "Use --val-source=train_split and reserve the official test split for --eval-test."
        )


def trial_grid(preset: str) -> list[TrialConfig]:
    smoke = [
        TrialConfig("adamw_lr1e-3", "adamw", 1e-3),
        TrialConfig("anchormuon_full_lr4e-3_rg0p20", "anchormuon", 4e-3),
        TrialConfig("anchormuon_no_soda_lr4e-3", "anchormuon", 4e-3, soda="none"),
        TrialConfig("anchormuon_no_pmuoneq_lr4e-3", "anchormuon", 4e-3, pmuon_eq=False),
        TrialConfig("muown_lr1e-3_wd0", "muown", 1e-3, weight_decay=0.0, lr_schedule="wsd"),
        TrialConfig("ema_muown_lr1e-3_wd0_b0.3", "ema_muown", 1e-3, weight_decay=0.0, lr_schedule="wsd", ema_beta=0.3),
    ]
    if preset == "smoke":
        return smoke
    quick = [
        TrialConfig("adamw_lr5e-4", "adamw", 5e-4),
        TrialConfig("adamw_lr1e-3", "adamw", 1e-3),
        TrialConfig("adamw_lr2e-3", "adamw", 2e-3),
        TrialConfig("anchormuon_full_lr2e-3_rg0p20", "anchormuon", 2e-3),
        TrialConfig("anchormuon_full_lr4e-3_rg0p20", "anchormuon", 4e-3),
        TrialConfig("anchormuon_full_lr6e-3_rg0p20", "anchormuon", 6e-3),
        TrialConfig("anchormuon_no_soda_lr4e-3", "anchormuon", 4e-3, soda="none"),
        TrialConfig("anchormuon_no_pmuoneq_lr4e-3", "anchormuon", 4e-3, pmuon_eq=False),
        TrialConfig("anchormuon_mimuon_lr4e-3_mix0p85", "anchormuon", 4e-3, mimuon=True, mimuon_mix=0.85),
    ]
    if preset == "quick":
        return quick
    full = quick + [
        TrialConfig("anchormuon_full_lr4e-3_rg0p10", "anchormuon", 4e-3, row_gamma=0.10),
        TrialConfig("anchormuon_full_lr4e-3_rg0p30", "anchormuon", 4e-3, row_gamma=0.30),
        TrialConfig("anchormuon_full_lr4e-3_rg0p20_cg0p10", "anchormuon", 4e-3, row_gamma=0.20, col_gamma=0.10),
        TrialConfig("anchormuon_mimuon_lr6e-3_mix0p75", "anchormuon", 6e-3, mimuon=True, mimuon_mix=0.75),
        TrialConfig("anchormuon_no_soda_no_pmuoneq_lr4e-3", "anchormuon", 4e-3, soda="none", pmuon_eq=False),
    ]
    if preset == "full":
        return full
    focused20 = [
        TrialConfig("adamw_lr0.003_wd0.005", "adamw", 3e-3, weight_decay=0.005),
        TrialConfig(
            "base_mlr0.009_rg0.2_cg0.1_mom0.95_pb0.9",
            "anchormuon",
            9e-3,
            soda="all",
            row_gamma=0.20,
            col_gamma=0.10,
            pmuon_beta=0.90,
            momentum=0.95,
            amuse=False,
        ),
        TrialConfig(
            "normuon_mlr0.007_rg0.3_cg0.1_mom0.95_pb0.9_nb0.9",
            "anchormuon",
            7e-3,
            soda="all",
            row_gamma=0.30,
            col_gamma=0.10,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.90,
            amuse=False,
        ),
        TrialConfig(
            "mimuon_mlr0.007_rg0.3_cg0.1_mom0.95_pb0.9_mix0.85",
            "anchormuon",
            7e-3,
            soda="all",
            row_gamma=0.30,
            col_gamma=0.10,
            pmuon_beta=0.90,
            momentum=0.95,
            mimuon=True,
            mimuon_mix=0.85,
            amuse=False,
        ),
        TrialConfig(
            "mimuon_normuon_mlr0.009_rg0.3_cg0.1_mom0.95_pb0.9_mix0.85_nb0.95",
            "anchormuon",
            9e-3,
            soda="all",
            row_gamma=0.30,
            col_gamma=0.10,
            pmuon_beta=0.90,
            momentum=0.95,
            mimuon=True,
            mimuon_mix=0.85,
            normuon=True,
            normuon_beta=0.95,
            amuse=False,
        ),
    ]
    if preset == "focused20":
        return focused20
    best_cifar10 = [
        TrialConfig(
            "adamw_cosine_lr0.004_wd0.001",
            "adamw",
            4e-3,
            lr_schedule="cosine",
            weight_decay=0.001,
        ),
        TrialConfig(
            "normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93",
            "anchormuon",
            8e-3,
            soda="all",
            row_gamma=0.35,
            col_gamma=0.0,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            normuon_aspect_scale=False,
            amuse=False,
        ),
        TrialConfig(
            "golden_normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93",
            "golden",
            8e-3,
            soda="all",
            row_gamma=0.35,
            col_gamma=0.0,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            normuon_aspect_scale=False,
            amuse=False,
        ),
        TrialConfig(
            "normuon_aspect_mlr0.008_rg0.3_cg0.05_mom0.95_pb0.9_nb0.95",
            "anchormuon",
            8e-3,
            soda="all",
            row_gamma=0.30,
            col_gamma=0.05,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.95,
            normuon_aspect_scale=True,
            amuse=False,
        ),
    ]
    if preset == "best_cifar10":
        return best_cifar10
    root_normuon_ablation = [
        TrialConfig("adamw_cosine_lr0.004_wd0.001", "adamw", 4e-3, lr_schedule="cosine", weight_decay=0.001),
        TrialConfig(
            "root_row_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93",
            "root",
            8e-3,
            soda="all",
            row_gamma=0.35,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            amuse=False,
            root_grouping="anchor",
            root_normuon_mode="row",
        ),
        TrialConfig(
            "root_orient_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93",
            "root",
            8e-3,
            soda="all",
            row_gamma=0.35,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            amuse=False,
            root_grouping="anchor",
            root_normuon_mode="orientation",
        ),
        TrialConfig(
            "golden_orient_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93",
            "golden",
            8e-3,
            soda="all",
            row_gamma=0.35,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            amuse=False,
        ),
    ]
    if preset == "root_normuon_ablation":
        return root_normuon_ablation
    root_soda_ablation = [
        TrialConfig("adamw_cosine_lr0.004_wd0.001", "adamw", 4e-3, lr_schedule="cosine", weight_decay=0.001),
        TrialConfig(
            "root_row_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93",
            "root",
            8e-3,
            soda="all",
            row_gamma=0.35,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            amuse=False,
            root_grouping="anchor",
            root_normuon_mode="row",
        ),
        TrialConfig(
            "root_row_named_soda_mlr0.008_rg0.35_pb0.9_nb0.93",
            "root",
            8e-3,
            soda="all",
            row_gamma=0.35,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            amuse=False,
            root_grouping="named",
            root_normuon_mode="row",
        ),
        TrialConfig(
            "golden_orient_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93",
            "golden",
            8e-3,
            soda="all",
            row_gamma=0.35,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.93,
            amuse=False,
        ),
    ]
    if preset == "root_soda_ablation":
        return root_soda_ablation
    root_lr_schedule_sweep = [
        TrialConfig("adamw_cosine_lr0.004_wd0.001", "adamw", 4e-3, lr_schedule="cosine", weight_decay=0.001),
    ]
    for schedule in ["constant", "cosine", "linear", "wsd"]:
        for lr in [6e-3, 8e-3, 1e-2, 1.2e-2]:
            root_lr_schedule_sweep.append(TrialConfig(
                f"root_named_{schedule}_lr{lr:g}_rg0.35_pb0.9_nb0.93",
                "root",
                lr,
                lr_schedule=schedule,
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
            ))
    if preset == "root_lr_schedule_sweep":
        return root_lr_schedule_sweep
    if preset == "root_fallback_modes":
        trials = [
            TrialConfig(
                "fallback_rms_control_wsd_lr0.012_flr1_b2_0.95",
                "root",
                1.2e-2,
                lr_schedule="wsd",
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
                fallback_mode="rms",
                fallback_lr_mult=1.0,
                fallback_beta1=0.90,
                fallback_beta2=0.95,
            ),
        ]
        for mode in ["atan2", "adamc"]:
            for lr in [0.010, 0.012, 0.014]:
                for fallback_lr_mult in [0.5, 1.0, 2.0]:
                    for beta2 in [0.95, 0.99]:
                        trials.append(TrialConfig(
                            (
                                f"fallback_{mode}_wsd_lr{lr:g}"
                                f"_flr{fallback_lr_mult:g}_b2{beta2:g}"
                            ),
                            "root",
                            lr,
                            lr_schedule="wsd",
                            soda="all",
                            row_gamma=0.35,
                            pmuon_beta=0.90,
                            momentum=0.95,
                            normuon=True,
                            normuon_beta=0.93,
                            amuse=False,
                            root_grouping="named",
                            root_normuon_mode="row",
                            fallback_mode=mode,
                            fallback_lr_mult=fallback_lr_mult,
                            fallback_beta1=0.90,
                            fallback_beta2=beta2,
                        ))
        for wd in [0.1, 1.0]:
            trials.append(TrialConfig(
                f"fallback_adamc_wd{wd:g}_wsd_lr0.012_flr1_b2_0.95",
                "root",
                1.2e-2,
                lr_schedule="wsd",
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
                fallback_mode="adamc",
                fallback_lr_mult=1.0,
                fallback_beta1=0.90,
                fallback_beta2=0.95,
                fallback_weight_decay=wd,
            ))
        return trials
    if preset == "cifar5_compare":
        trials = [
            TrialConfig(
                "adamw_cosine_lr0.004_wd0.001",
                "adamw",
                4e-3,
                lr_schedule="cosine",
                weight_decay=0.001,
            ),
            TrialConfig(
                "previous_best_rms_wsd_lr0.012_flr1_b2_0.95",
                "root",
                1.2e-2,
                lr_schedule="wsd",
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
                fallback_mode="rms",
                fallback_lr_mult=1.0,
                fallback_beta1=0.90,
                fallback_beta2=0.95,
            ),
            TrialConfig(
                "atan2_best_wsd_lr0.014_flr0.5_b2_0.95",
                "root",
                1.4e-2,
                lr_schedule="wsd",
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
                fallback_mode="atan2",
                fallback_lr_mult=0.5,
                fallback_beta1=0.90,
                fallback_beta2=0.95,
            ),
        ]
        for lr in [0.006, 0.008, 0.010, 0.012, 0.014]:
            for wd in [0.0, 0.001, 0.005]:
                trials.append(TrialConfig(
                    f"plain_muon_wsd_lr{lr:g}_wd{wd:g}",
                    "anchormuon",
                    lr,
                    lr_schedule="wsd",
                    weight_decay=wd,
                    soda="none",
                    pmuon_eq=False,
                    use_gram=True,
                    row_gamma=0.0,
                    col_gamma=0.0,
                    pmuon_beta=0.90,
                    momentum=0.95,
                    amuse=False,
                    mimuon=False,
                    normuon=False,
                    external_lr=True,
                ))
        return trials
    if preset == "muown_ema_cifar10":
        trials = [
            TrialConfig("adamw_cosine_lr0.0035_wd0.001", "adamw", 3.5e-3, lr_schedule="cosine", weight_decay=0.001),
            TrialConfig("adamw_cosine_lr0.004_wd0.001", "adamw", 4e-3, lr_schedule="cosine", weight_decay=0.001),
            TrialConfig("adamw_cosine_lr0.0045_wd0.001", "adamw", 4.5e-3, lr_schedule="cosine", weight_decay=0.001),
        ]
        for lr in [0.012, 0.014, 0.016]:
            trials.append(TrialConfig(
                f"anchor_wsd_lr{lr:g}_flr0.5_atan2_rg0.35_pb0.9_nb0.93",
                "root",
                lr,
                lr_schedule="wsd",
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
                fallback_mode="atan2",
                fallback_lr_mult=0.5,
                fallback_beta1=0.90,
                fallback_beta2=0.95,
            ))
        for lr in [0.010, 0.014, 0.018]:
            for wd in [0.0, 0.001]:
                trials.append(TrialConfig(
                    f"muon_wsd_lr{lr:g}_wd{wd:g}",
                    "anchormuon",
                    lr,
                    lr_schedule="wsd",
                    weight_decay=wd,
                    soda="none",
                    pmuon_eq=False,
                    use_gram=True,
                    row_gamma=0.0,
                    col_gamma=0.0,
                    pmuon_beta=0.90,
                    momentum=0.95,
                    amuse=False,
                    mimuon=False,
                    normuon=False,
                    external_lr=True,
                ))
        for lr in [0.008, 0.012, 0.016, 0.020]:
            for wd in [0.0, 0.001, 0.005]:
                trials.append(TrialConfig(
                    f"muown_wsd_lr{lr:g}_wd{wd:g}",
                    "muown",
                    lr,
                    lr_schedule="wsd",
                    weight_decay=wd,
                    momentum=0.95,
                ))
        for lr in [0.010, 0.014]:
            for ema_beta in [0.1, 0.3]:
                trials.append(TrialConfig(
                    f"ema_muon_wsd_lr{lr:g}_wd0.001_b{ema_beta:g}_g0.99",
                    "ema_muon",
                    lr,
                    lr_schedule="wsd",
                    weight_decay=0.001,
                    momentum=0.95,
                    ema_beta=ema_beta,
                    ema_gamma=0.99,
                ))
        for lr in [0.012, 0.016]:
            for wd in [0.0, 0.001]:
                for ema_beta in [0.1, 0.3]:
                    trials.append(TrialConfig(
                        f"ema_muown_wsd_lr{lr:g}_wd{wd:g}_b{ema_beta:g}_g0.99",
                        "ema_muown",
                        lr,
                        lr_schedule="wsd",
                        weight_decay=wd,
                        momentum=0.95,
                        ema_beta=ema_beta,
                        ema_gamma=0.99,
                    ))
        return trials
    if preset == "ema_muown_cifar10_hard":
        trials = [
            TrialConfig(
                "adamw_cosine_lr0.004_wd0.001",
                "adamw",
                4e-3,
                lr_schedule="cosine",
                weight_decay=0.001,
            ),
            TrialConfig(
                "anchor_wsd_lr0.016_flr0.5_atan2_rg0.35_pb0.9_nb0.93",
                "root",
                1.6e-2,
                lr_schedule="wsd",
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
                fallback_mode="atan2",
                fallback_lr_mult=0.5,
                fallback_beta1=0.90,
                fallback_beta2=0.95,
            ),
        ]
        for lr in [0.008, 0.010, 0.012, 0.014]:
            for mag_mult in [0.5, 1.0]:
                trials.append(TrialConfig(
                    f"muown_wsd_lr{lr:g}_wd0_flr0.5_mag{mag_mult:g}",
                    "muown",
                    lr,
                    lr_schedule="wsd",
                    weight_decay=0.0,
                    momentum=0.95,
                    fallback_lr_mult=0.5,
                    muown_mag_lr_mult=mag_mult,
                ))
        for schedule in ["wsd", "cosine"]:
            for lr in [0.006, 0.008, 0.010, 0.012, 0.014]:
                for fallback_mult in [0.25, 0.5, 1.0]:
                    for mag_mult in [0.5, 1.0]:
                        for ema_beta in [0.03, 0.06, 0.10, 0.15]:
                            trials.append(TrialConfig(
                                (
                                    f"ema_muown_{schedule}_lr{lr:g}_wd0"
                                    f"_flr{fallback_mult:g}_mag{mag_mult:g}"
                                    f"_b{ema_beta:g}_g0.995"
                                ),
                                "ema_muown",
                                lr,
                                lr_schedule=schedule,
                                weight_decay=0.0,
                                momentum=0.95,
                                fallback_lr_mult=fallback_mult,
                                muown_mag_lr_mult=mag_mult,
                                ema_beta=ema_beta,
                                ema_gamma=0.995,
                                ema_warmup_frac=0.20,
                                ema_rest_frac=0.10,
                            ))
        for lr in [0.008, 0.010, 0.012]:
            for wd in [0.0001, 0.0003]:
                for ema_beta in [0.03, 0.06]:
                    trials.append(TrialConfig(
                        f"ema_muown_wsd_lr{lr:g}_wd{wd:g}_flr0.5_mag0.5_b{ema_beta:g}_g0.995",
                        "ema_muown",
                        lr,
                        lr_schedule="wsd",
                        weight_decay=wd,
                        momentum=0.95,
                        fallback_lr_mult=0.5,
                        muown_mag_lr_mult=0.5,
                        ema_beta=ema_beta,
                        ema_gamma=0.995,
                        ema_warmup_frac=0.20,
                        ema_rest_frac=0.10,
                    ))
        for lr in [0.010, 0.014]:
            for ema_beta in [0.03, 0.06, 0.10]:
                trials.append(TrialConfig(
                    f"ema_muon_wsd_lr{lr:g}_wd0.001_b{ema_beta:g}_g0.995",
                    "ema_muon",
                    lr,
                    lr_schedule="wsd",
                    weight_decay=0.001,
                    momentum=0.95,
                    ema_beta=ema_beta,
                    ema_gamma=0.995,
                    ema_warmup_frac=0.20,
                    ema_rest_frac=0.10,
                ))
        return trials
    if preset == "sfplus_combo20":
        trials = [
            TrialConfig(
                "adamw_cosine_lr0.004_wd0.001",
                "adamw",
                4e-3,
                lr_schedule="cosine",
                weight_decay=0.001,
            ),
            TrialConfig(
                "root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93",
                "root",
                1.2e-2,
                lr_schedule="wsd",
                soda="all",
                row_gamma=0.35,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                root_grouping="named",
                root_normuon_mode="row",
            ),
        ]
        # Exhaustive on/off combinations of five SF+ mechanisms:
        # P=Polyak LR, C=c_warmup, B=beta anneal, D=AdamC decay, M=inner momentum.
        # Polyak and non-Polyak variants require different base-LR scales because
        # Polyak supplies an online scalar while non-Polyak uses lr directly.
        toggles = ["P", "C", "B", "D", "M"]
        for mask in range(32):
            enabled = {letter: bool(mask & (1 << idx)) for idx, letter in enumerate(toggles)}
            code = "".join(letter if enabled[letter] else letter.lower() for letter in toggles)
            trials.append(TrialConfig(
                f"sfplus_{code}_lr{'3' if enabled['P'] else '0.008'}_wd2",
                "sfplus",
                3.0 if enabled["P"] else 8e-3,
                lr_schedule="constant",
                weight_decay=2.0,
                soda="none",
                row_gamma=0.35,
                col_gamma=0.0,
                pmuon_beta=0.90,
                momentum=0.95,
                normuon=True,
                normuon_beta=0.93,
                amuse=False,
                sfplus_polyak=enabled["P"],
                sfplus_c_warmup_enabled=enabled["C"],
                sfplus_beta_anneal=enabled["B"],
                sfplus_adamc_decay=enabled["D"],
                sfplus_inner_momentum=enabled["M"],
                sfplus_beta1=0.90,
                sfplus_beta1_max=0.965,
                sfplus_polyak_beta=0.0,
                sfplus_r=0.0,
                sfplus_weight_lr_power=2.0,
            ))
        return trials
    if preset == "component_ablation":
        trials = [
            TrialConfig("component_adamw_lr0.004_wd0.001", "adamw", 4e-3, lr_schedule="cosine", weight_decay=0.001),
            TrialConfig("component_adamw_lr0.003_wd0.005", "adamw", 3e-3, lr_schedule="cosine", weight_decay=0.005),
        ]

        def add_family(
            family: str,
            *,
            soda: str = "all",
            pmuon_eq: bool = True,
            use_gram: bool = True,
            normuon: bool = True,
            lr_values: list[float],
            row_gamma_values: list[float],
            pmuon_beta_values: list[float],
            normuon_beta_values: list[float],
        ) -> None:
            for lr in lr_values:
                for row_gamma in row_gamma_values:
                    for pmuon_beta in pmuon_beta_values:
                        for normuon_beta in normuon_beta_values:
                            trials.append(TrialConfig(
                                (
                                    f"component_{family}_lr{lr:g}_rg{row_gamma:g}"
                                    f"_pb{pmuon_beta:g}_nb{normuon_beta:g}"
                                ),
                                "anchormuon",
                                lr,
                                lr_schedule="wsd",
                                weight_decay=0.001,
                                soda=soda,
                                pmuon_eq=pmuon_eq,
                                use_gram=use_gram,
                                row_gamma=row_gamma,
                                col_gamma=0.0,
                                pmuon_beta=pmuon_beta,
                                momentum=0.95,
                                amuse=False,
                                normuon=normuon,
                                normuon_beta=normuon_beta,
                                normuon_aspect_scale=False,
                            ))

        # Progressive, comparable HPO around the current WSD winner. Each
        # component-removal family gets LR tuning plus the relevant local knob.
        add_family(
            "full",
            lr_values=[0.010, 0.012, 0.014],
            row_gamma_values=[0.35],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "full",
            lr_values=[0.012],
            row_gamma_values=[0.25, 0.45],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "full",
            lr_values=[0.012],
            row_gamma_values=[0.35],
            pmuon_beta_values=[0.95],
            normuon_beta_values=[0.90, 0.95],
        )
        add_family(
            "no_soda",
            soda="none",
            lr_values=[0.006, 0.008, 0.010, 0.012],
            row_gamma_values=[0.35],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "no_soda",
            soda="none",
            lr_values=[0.008],
            row_gamma_values=[0.25, 0.45],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "no_soda",
            soda="none",
            lr_values=[0.008],
            row_gamma_values=[0.35],
            pmuon_beta_values=[0.95],
            normuon_beta_values=[0.90, 0.95],
        )
        add_family(
            "no_pmuoneq",
            pmuon_eq=False,
            lr_values=[0.006, 0.008, 0.010, 0.012],
            row_gamma_values=[0.0],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "no_pmuoneq",
            pmuon_eq=False,
            lr_values=[0.008, 0.010],
            row_gamma_values=[0.0],
            pmuon_beta_values=[0.95],
            normuon_beta_values=[0.90, 0.95],
        )
        add_family(
            "no_gram",
            use_gram=False,
            lr_values=[0.002, 0.004, 0.006, 0.008],
            row_gamma_values=[0.35],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "no_gram",
            use_gram=False,
            lr_values=[0.004, 0.006],
            row_gamma_values=[0.25, 0.45],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "no_normuon",
            normuon=False,
            lr_values=[0.004, 0.006, 0.008, 0.010],
            row_gamma_values=[0.35],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        add_family(
            "no_normuon",
            normuon=False,
            lr_values=[0.006, 0.008],
            row_gamma_values=[0.25, 0.45],
            pmuon_beta_values=[0.90],
            normuon_beta_values=[0.93],
        )
        return trials
    feedback = [
        TrialConfig("adamw_cosine_lr0.004_wd0.001", "adamw", 4e-3, lr_schedule="cosine", weight_decay=0.001),
    ]
    for aspect in [False, True]:
        for row_gamma in [0.30, 0.35]:
            for col_gamma in [0.0, 0.05]:
                for normuon_beta in [0.93, 0.95]:
                    feedback.append(TrialConfig(
                        (
                            f"normuon{'_aspect' if aspect else ''}_mlr0.008"
                            f"_rg{row_gamma:g}_cg{col_gamma:g}"
                            f"_mom0.95_pb0.9_nb{normuon_beta:g}"
                        ),
                        "anchormuon",
                        8e-3,
                        soda="all",
                        row_gamma=row_gamma,
                        col_gamma=col_gamma,
                        pmuon_beta=0.90,
                        momentum=0.95,
                        normuon=True,
                        normuon_beta=normuon_beta,
                        normuon_aspect_scale=aspect,
                        amuse=False,
                    ))
    if preset == "feedback":
        return feedback
    confidence: list[TrialConfig] = []
    for schedule in ["cosine", "constant"]:
        for lr in [2e-3, 3e-3, 4e-3]:
            for wd in [0.001, 0.005, 0.01, 0.03, 0.05]:
                confidence.append(TrialConfig(
                    f"adamw_{schedule}_lr{lr:g}_wd{wd:g}",
                    "adamw",
                    lr,
                    lr_schedule=schedule,
                    weight_decay=wd,
                ))
    for lr in [6e-3, 7e-3, 8e-3]:
        for row_gamma in [0.20, 0.30]:
            for col_gamma in [0.0, 0.10]:
                for normuon_beta in [0.90, 0.95]:
                    confidence.append(TrialConfig(
                        (
                            f"normuon_mlr{lr:g}_rg{row_gamma:g}_cg{col_gamma:g}"
                            f"_mom0.95_pb0.9_nb{normuon_beta:g}"
                        ),
                        "anchormuon",
                        lr,
                        soda="all",
                        row_gamma=row_gamma,
                        col_gamma=col_gamma,
                        pmuon_beta=0.90,
                        momentum=0.95,
                        normuon=True,
                        normuon_beta=normuon_beta,
                        amuse=False,
                    ))
    if preset == "confidence":
        return confidence
    return full + [
        TrialConfig("adamw_lr3e-3", "adamw", 3e-3),
        TrialConfig("adamw_lr2e-3_wd0p03", "adamw", 2e-3, weight_decay=0.03),
        TrialConfig("adamw_lr2e-3_wd0p10", "adamw", 2e-3, weight_decay=0.10),
        TrialConfig("anchormuon_full_lr3e-3_rg0p20", "anchormuon", 3e-3),
        TrialConfig("anchormuon_full_lr8e-3_rg0p20", "anchormuon", 8e-3),
        TrialConfig("anchormuon_full_lr6e-3_rg0p10", "anchormuon", 6e-3, row_gamma=0.10),
        TrialConfig("anchormuon_full_lr6e-3_rg0p30", "anchormuon", 6e-3, row_gamma=0.30),
        TrialConfig("anchormuon_full_lr6e-3_rg0p20_cg0p10", "anchormuon", 6e-3, row_gamma=0.20, col_gamma=0.10),
        TrialConfig("anchormuon_full_lr6e-3_beta0p90", "anchormuon", 6e-3, pmuon_beta=0.90),
        TrialConfig("anchormuon_full_lr6e-3_beta0p98", "anchormuon", 6e-3, pmuon_beta=0.98),
        TrialConfig("anchormuon_soda_all_lr4e-3", "anchormuon", 4e-3, soda="all"),
        TrialConfig("anchormuon_soda_all_lr6e-3", "anchormuon", 6e-3, soda="all"),
        TrialConfig("anchormuon_soda_all_normuon_lr3e-3", "anchormuon", 3e-3, soda="all", normuon=True),
        TrialConfig("anchormuon_soda_all_normuon_lr4e-3", "anchormuon", 4e-3, soda="all", normuon=True),
        TrialConfig("anchormuon_soda_all_normuon_lr5e-3", "anchormuon", 5e-3, soda="all", normuon=True),
        TrialConfig("anchormuon_soda_all_normuon_lr4e-3_beta0p90", "anchormuon", 4e-3, soda="all", normuon=True, normuon_beta=0.90),
        TrialConfig("anchormuon_soda_all_mimuon_lr3e-3_mix0p85", "anchormuon", 3e-3, soda="all", mimuon=True, mimuon_mix=0.85),
        TrialConfig("anchormuon_soda_all_mimuon_lr4e-3_mix0p75", "anchormuon", 4e-3, soda="all", mimuon=True, mimuon_mix=0.75),
        TrialConfig("anchormuon_soda_all_mimuon_lr4e-3_mix0p85", "anchormuon", 4e-3, soda="all", mimuon=True, mimuon_mix=0.85),
        TrialConfig("anchormuon_soda_all_mimuon_lr4e-3_mix0p95", "anchormuon", 4e-3, soda="all", mimuon=True, mimuon_mix=0.95),
        TrialConfig("anchormuon_soda_all_mimuon_normuon_lr4e-3_mix0p85", "anchormuon", 4e-3, soda="all", mimuon=True, mimuon_mix=0.85, normuon=True),
        TrialConfig("anchormuon_no_soda_lr2e-3", "anchormuon", 2e-3, soda="none"),
        TrialConfig("anchormuon_no_soda_lr3e-3", "anchormuon", 3e-3, soda="none"),
        TrialConfig("anchormuon_no_soda_lr6e-3", "anchormuon", 6e-3, soda="none"),
        TrialConfig("anchormuon_no_soda_lr4e-3_rg0p10", "anchormuon", 4e-3, soda="none", row_gamma=0.10),
        TrialConfig("anchormuon_no_soda_lr4e-3_rg0p30", "anchormuon", 4e-3, soda="none", row_gamma=0.30),
        TrialConfig("anchormuon_no_soda_lr4e-3_cg0p10", "anchormuon", 4e-3, soda="none", col_gamma=0.10),
        TrialConfig("anchormuon_no_soda_lr4e-3_beta0p90", "anchormuon", 4e-3, soda="none", pmuon_beta=0.90),
        TrialConfig("anchormuon_no_soda_lr4e-3_beta0p98", "anchormuon", 4e-3, soda="none", pmuon_beta=0.98),
        TrialConfig("anchormuon_no_pmuoneq_lr2e-3", "anchormuon", 2e-3, pmuon_eq=False),
        TrialConfig("anchormuon_no_pmuoneq_lr6e-3", "anchormuon", 6e-3, pmuon_eq=False),
        TrialConfig("anchormuon_no_soda_no_pmuoneq_lr2e-3", "anchormuon", 2e-3, soda="none", pmuon_eq=False),
        TrialConfig("anchormuon_no_soda_no_pmuoneq_lr6e-3", "anchormuon", 6e-3, soda="none", pmuon_eq=False),
        TrialConfig("anchormuon_mimuon_lr4e-3_mix0p75", "anchormuon", 4e-3, mimuon=True, mimuon_mix=0.75),
        TrialConfig("anchormuon_mimuon_lr4e-3_mix0p95", "anchormuon", 4e-3, mimuon=True, mimuon_mix=0.95),
        TrialConfig("anchormuon_mimuon_lr6e-3_mix0p85", "anchormuon", 6e-3, mimuon=True, mimuon_mix=0.85),
    ]


def trial_family(cfg: TrialConfig) -> str:
    match = re.match(r"(?:final_)?component_(full|no_soda|no_pmuoneq|no_gram|no_normuon|adamw)", cfg.name)
    if match:
        return f"component_{match.group(1)}"
    if cfg.optimizer == "sfplus":
        match = re.search(r"sfplus_([PCBDMpcbdm]{5})", cfg.name)
        return f"sfplus_{match.group(1)}" if match else "sfplus"
    if cfg.optimizer == "root":
        if cfg.name.startswith("fallback_") or "fallback_rms" in cfg.name or cfg.fallback_mode != "rms":
            return f"root_fallback_{cfg.fallback_mode}"
    if cfg.name.startswith("plain_muon_") or "_plain_muon_" in cfg.name:
        return "plain_muon"
    if cfg.name.startswith("previous_best_rms_") or "_previous_best_rms_" in cfg.name:
        return "previous_best_rms"
    if cfg.name.startswith("atan2_best_") or "_atan2_best_" in cfg.name:
        return "atan2_best"
    if cfg.optimizer in {"muown", "ema_muon", "ema_muown"}:
        return cfg.optimizer
    if cfg.optimizer == "root":
        return f"root_{cfg.root_grouping}_{cfg.root_normuon_mode}_{cfg.soda}"
    if cfg.optimizer == "golden":
        return "golden_normuon"
    if cfg.optimizer == "adamw":
        return "adamw"
    if cfg.mimuon and cfg.normuon and cfg.normuon_aspect_scale:
        return "anchormuon_mimuon_normuon_aspect"
    if cfg.mimuon and cfg.normuon:
        return "anchormuon_mimuon_normuon"
    if cfg.mimuon:
        return "anchormuon_mimuon"
    if cfg.normuon and cfg.normuon_aspect_scale:
        return "anchormuon_normuon_aspect"
    if cfg.normuon:
        return "anchormuon_normuon"
    if cfg.soda == "all":
        return "anchormuon_soda_all"
    if cfg.soda == "none" and not cfg.pmuon_eq:
        return "anchormuon_no_soda_no_pmuoneq"
    if cfg.soda == "none":
        return "anchormuon_no_soda"
    if not cfg.pmuon_eq:
        return "anchormuon_no_pmuoneq"
    return "anchormuon_full"


def cifar10_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader | None, dict[str, int | str]]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_full = datasets.CIFAR10(args.data_path, train=True, transform=train_tf, download=True)
    train_eval_full = datasets.CIFAR10(args.data_path, train=True, transform=val_tf, download=True)
    test_full = datasets.CIFAR10(args.data_path, train=False, transform=val_tf, download=True)
    split_gen = torch.Generator().manual_seed(int(args.split_seed))
    if args.val_source == "train_split":
        if not 0 < int(args.train_val_size) < len(train_full):
            raise ValueError(f"--train-val-size must be in [1, {len(train_full) - 1}] for train_split")
        perm = torch.randperm(len(train_full), generator=split_gen).tolist()
        val_indices = perm[: int(args.train_val_size)]
        train_indices = perm[int(args.train_val_size):]
        if 0 < args.train_subset < len(train_indices):
            train_indices = train_indices[: args.train_subset]
        if 0 < args.val_subset < len(val_indices):
            val_indices = val_indices[: args.val_subset]
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(train_eval_full, val_indices)
        dataset_info: dict[str, int | str] = {
            "val_source": "train_split",
            "split_seed": int(args.split_seed),
            "train_val_size": int(args.train_val_size),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "test_examples": len(test_full),
        }
    else:
        train_ds = train_full
        val_ds = test_full
        if 0 < args.train_subset < len(train_ds):
            subset_gen = torch.Generator().manual_seed(int(args.split_seed))
            train_indices = torch.randperm(len(train_ds), generator=subset_gen)[: args.train_subset].tolist()
            train_ds = Subset(train_ds, train_indices)
        if 0 < args.val_subset < len(val_ds):
            val_ds = Subset(val_ds, list(range(args.val_subset)))
        dataset_info = {
            "val_source": "test",
            "split_seed": int(args.split_seed),
            "train_val_size": 0,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "test_examples": len(test_full),
        }
    test_ds = test_full
    if 0 < args.test_subset < len(test_ds):
        test_ds = Subset(test_ds, list(range(args.test_subset)))
        dataset_info["test_examples"] = len(test_ds)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs) if args.eval_test else None
    return train_loader, val_loader, test_loader, dataset_info


def ensure_cifar10_downloaded(data_path: Path) -> None:
    """Download/extract CIFAR-10 once before launching parallel GPU workers."""

    data_path.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(data_path, train=True, download=True)
    datasets.CIFAR10(data_path, train=False, download=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scheduled_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    schedule: str = "cosine",
    *,
    final_scale: float = 0.1,
    wsd_decay_frac: float = 0.2,
) -> float:
    if step < max(1, warmup_steps):
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if schedule == "constant":
        return base_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    floor = max(0.0, float(final_scale))
    if schedule == "cosine":
        return base_lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    if schedule == "linear":
        return base_lr * (floor + (1.0 - floor) * (1.0 - progress))
    if schedule == "wsd":
        decay_frac = min(1.0, max(1e-9, float(wsd_decay_frac)))
        stable_frac = 1.0 - decay_frac
        if progress <= stable_frac:
            return base_lr
        decay_progress = (progress - stable_frac) / decay_frac
        return base_lr * (floor + (1.0 - floor) * (1.0 - decay_progress))
    raise ValueError(f"unknown lr schedule {schedule!r}")


def apply_scheduled_lrs(optimizer: torch.optim.Optimizer, scheduled_base_lr: float, config_base_lr: float) -> None:
    """Apply the trainer schedule while preserving per-group LR ratios.

    Root AnchorMuon uses a separate fallback group with ``fallback_lr``. Writing
    the same absolute LR into every group erases that tuned ratio, so each group
    keeps its construction-time LR as ``_bench_base_lr`` and receives the common
    schedule factor.
    """

    if config_base_lr <= 0.0:
        for group in optimizer.param_groups:
            group["lr"] = scheduled_base_lr
        return
    for group in optimizer.param_groups:
        group_base_lr = float(group.setdefault("_bench_base_lr", float(group.get("lr", config_base_lr))))
        group["lr"] = scheduled_base_lr * group_base_lr / float(config_base_lr)


def trainer_owns_lr_schedule(cfg: TrialConfig) -> bool:
    if cfg.external_lr:
        return True
    if cfg.optimizer in {"adamw", "root", "muown", "ema_muon", "ema_muown"}:
        return True
    return cfg.lr_schedule != "constant"


class CifarMuown(torch.optim.Optimizer):
    """Muon with Muown's implicit row-magnitude parameterization.

    This is local to the CIFAR runner so the new paper idea can be compared
    against the existing AnchorMuon baselines without changing root
    ``optimizer.py``. Matrix-like parameters use the Muown update; vectors,
    biases, and tiny tensors use an AdamW fallback with the group weight decay.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        lr: float,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        min_matrix_dim: int = 2,
        fallback_lr_mult: float = 1.0,
        mag_lr_mult: float = 1.0,
    ) -> None:
        groups: list[dict[str, Any]] = []
        for group in params:
            copied = dict(group)
            copied["params"] = list(group["params"])
            copied.setdefault("lr", lr)
            copied.setdefault("momentum", momentum)
            copied.setdefault("betas", betas)
            copied.setdefault("eps", eps)
            copied.setdefault("min_matrix_dim", min_matrix_dim)
            copied.setdefault("fallback_lr_mult", fallback_lr_mult)
            copied.setdefault("mag_lr_mult", mag_lr_mult)
            copied.setdefault("_bench_base_lr", float(copied["lr"]))
            groups.append(copied)
        super().__init__(groups, {})
        self._orthogonalizer = root_optimizer.GramNewtonSchulz()

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CifarMuown does not support sparse gradients")
                if root_optimizer._is_matrix_like_parameter(
                    p,
                    min_matrix_dim=int(group.get("min_matrix_dim", 2)),
                ):
                    self._step_matrix_param(p, group)
                else:
                    self._step_adamw_param(p, group)
        return loss

    def _step_matrix_param(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        wd = float(group.get("weight_decay", 0.0))
        beta = float(group["momentum"])
        eps = float(group.get("eps", 1e-8))
        w = root_optimizer._matrix_view(p.detach()).to(torch.float32)
        grad_w = root_optimizer._matrix_view(p.grad.detach()).to(torch.float32)
        rows, cols = w.shape
        state = self.state[p]
        row_norm = w.norm(dim=1).clamp_min(eps)

        if "g_mag" not in state or tuple(state["g_mag"].shape) != (rows,):
            state["g_mag"] = row_norm.clone()
            state["r_norm"] = row_norm.clone()
            state["momentum_buffer"] = torch.zeros_like(w, dtype=torch.float32)
            state["mag_exp_avg"] = torch.zeros_like(row_norm, dtype=torch.float32)
            state["mag_exp_avg_sq"] = torch.zeros_like(row_norm, dtype=torch.float32)
            state["mag_step"] = 0

        g_mag = state["g_mag"]
        r_norm = state["r_norm"]
        momentum = state["momentum_buffer"]
        mag_exp_avg = state["mag_exp_avg"]
        mag_exp_avg_sq = state["mag_exp_avg_sq"]

        safe_g = g_mag.clamp_min(eps)
        safe_r = r_norm.clamp_min(eps)
        hidden_r = w * (safe_r / safe_g).unsqueeze(1)
        direction = hidden_r / safe_r.unsqueeze(1)
        radial = (grad_w * direction).sum(dim=1, keepdim=True)
        grad_g = radial.squeeze(1)
        grad_hidden_r = (safe_g / safe_r).unsqueeze(1) * (grad_w - radial * direction)

        momentum.lerp_(grad_hidden_r, 1.0 - beta)
        source = torch.lerp(grad_hidden_r, momentum, beta)
        update = self._orthogonalizer(source)
        update = update * (0.2 * math.sqrt(max(rows, cols)))
        hidden_r.add_(update, alpha=-lr)

        state["mag_step"] = int(state["mag_step"]) + 1
        mag_step = int(state["mag_step"])
        mag_lr = lr * float(group.get("mag_lr_mult", 1.0))
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        mag_exp_avg.mul_(float(beta1)).add_(grad_g, alpha=1.0 - float(beta1))
        mag_exp_avg_sq.mul_(float(beta2)).addcmul_(grad_g, grad_g, value=1.0 - float(beta2))
        m_hat = mag_exp_avg / max(1.0 - float(beta1) ** mag_step, 1e-16)
        v_hat = mag_exp_avg_sq / max(1.0 - float(beta2) ** mag_step, 1e-16)
        g_mag.add_(m_hat / v_hat.sqrt().clamp_min(eps), alpha=-mag_lr)
        g_mag.clamp_(min=eps)

        new_r_norm = hidden_r.norm(dim=1).clamp_min(eps)
        w_new = hidden_r * (g_mag / new_r_norm).unsqueeze(1)
        if wd:
            w_new.mul_(1.0 - lr * wd)
            new_r_norm = w_new.norm(dim=1).clamp_min(eps)
            g_mag.copy_(new_r_norm)
        r_norm.copy_(new_r_norm)
        p.copy_(w_new.reshape_as(p).to(p.dtype))

    def _step_adamw_param(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        lr = float(group["lr"]) * float(group.get("fallback_lr_mult", 1.0))
        wd = float(group.get("weight_decay", 0.0))
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-8))
        if wd:
            p.mul_(1.0 - lr * wd)
        grad = p.grad.detach().to(torch.float32)
        state = self.state[p]
        if "exp_avg" not in state or state["exp_avg"].shape != p.shape:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["step"] = int(state["step"]) + 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(float(beta1)).add_(grad, alpha=1.0 - float(beta1))
        exp_avg_sq.mul_(float(beta2)).addcmul_(grad, grad, value=1.0 - float(beta2))
        step = int(state["step"])
        m_hat = exp_avg / max(1.0 - float(beta1) ** step, 1e-16)
        v_hat = exp_avg_sq / max(1.0 - float(beta2) ** step, 1e-16)
        p.add_((m_hat / v_hat.sqrt().add_(eps)).to(p.dtype), alpha=-lr)


class EMANesterovOptimizer:
    """EMA-Nesterov lookahead wrapper for the CIFAR training loop.

    The runner calls ``zero_grad`` immediately before the forward pass. That is
    where this wrapper evaluates gradients at y_t = x_t + beta_t * ema_delta.
    ``step`` then leaves parameters at x_{t+1} and updates the EMA direction.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        beta: float,
        gamma: float,
        warmup_frac: float,
        rest_frac: float,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        for group in self.param_groups:
            group.setdefault("_bench_base_lr", float(group.get("lr", 1.0)))
        self.state: dict[torch.Tensor, dict[str, Any]] = {}
        self.total_steps = int(total_steps)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.warmup_steps = int(round(self.total_steps * float(warmup_frac)))
        self.rest_start = int(round(self.total_steps * max(0.0, 1.0 - float(rest_frac))))
        self.step_index = 0

    def _beta_t(self) -> float:
        if self.beta <= 0.0:
            return 0.0
        if self.step_index < self.warmup_steps:
            return 0.0
        if self.step_index >= self.rest_start:
            return 0.0
        return self.beta

    def _beta_t_for_group(self, group: dict[str, Any]) -> float:
        beta_t = self._beta_t()
        if not beta_t:
            return 0.0
        base_lr = float(group.get("_bench_base_lr", group.get("lr", 1.0)))
        if base_lr <= 0.0:
            return beta_t
        return beta_t * float(group.get("lr", base_lr)) / base_lr

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            beta_t = self._beta_t_for_group(group)
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state.setdefault(p, {})
                if state.pop("lookahead_active", False):
                    p.copy_(state["base_param"].to(p.dtype))
                base_param = state.get("base_param")
                if base_param is None or base_param.shape != p.shape:
                    base_param = state["base_param"] = torch.empty_like(p, dtype=torch.float32)
                base_param.copy_(p.detach().to(torch.float32))
                ema_delta = state.get("ema_delta")
                if ema_delta is None or ema_delta.shape != p.shape:
                    ema_delta = state["ema_delta"] = torch.zeros_like(p, dtype=torch.float32)
                if beta_t:
                    p.add_(ema_delta.to(p.dtype), alpha=beta_t)
                state["lookahead_active"] = True

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        # Gradients were computed at the lookahead weights. Restore the base
        # weights before applying the base optimizer so the update is
        # Nesterov-style x_{t+1} = x_t - lr * direction(grad(y_t)).
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state and state.get("lookahead_active", False):
                    p.copy_(state["base_param"].to(p.dtype))
        loss = self.base_optimizer.step(closure=closure)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or not state.get("lookahead_active", False):
                    continue
                base = state["base_param"]
                delta = p.detach().to(torch.float32) - base
                state["ema_delta"].mul_(self.gamma).add_(delta, alpha=1.0 - self.gamma)
                state["lookahead_active"] = False
        self.step_index += 1
        return loss

    def _flat_params(self) -> list[torch.Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    @staticmethod
    def _clone_state_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: EMANesterovOptimizer._clone_state_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [EMANesterovOptimizer._clone_state_value(val) for val in value]
        if isinstance(value, tuple):
            return tuple(EMANesterovOptimizer._clone_state_value(val) for val in value)
        return value

    def state_dict(self) -> dict[str, Any]:
        param_to_index = {p: index for index, p in enumerate(self._flat_params())}
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "state": {
                param_to_index[p]: self._clone_state_value(state)
                for p, state in self.state.items()
                if p in param_to_index
            },
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "beta": self.beta,
            "gamma": self.gamma,
            "warmup_steps": self.warmup_steps,
            "rest_start": self.rest_start,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups
        flat_params = self._flat_params()
        packed_state = state_dict.get("state", {})
        if packed_state and not all(isinstance(index, (int, str)) for index in packed_state):
            packed_items = enumerate(packed_state.values())
        else:
            packed_items = packed_state.items()
        self.state = {}
        for index, state in packed_items:
            int_index = int(index)
            if int_index < len(flat_params):
                self.state[flat_params[int_index]] = self._clone_state_value(state)
        self.step_index = int(state_dict.get("step_index", 0))

    def train(self) -> None:
        if hasattr(self.base_optimizer, "train"):
            self.base_optimizer.train()

    def eval(self) -> None:
        if hasattr(self.base_optimizer, "eval"):
            self.base_optimizer.eval()


class RootAnchorMuonOrientation(root_optimizer.AnchorMuon):
    """Trainer-side root optimizer adapter for the NorMuon orientation ablation.

    The root optimizer intentionally stays untouched. This subclass keeps the
    root Polar Express / PMuonEq / SODA implementation, but changes only the
    post-Gram NorMuon second-moment axis: rows for tall matrices and columns for
    wide matrices. That isolates difference #2 without changing difference #1.
    """

    def _ensure_orientation_state(
        self,
        p: torch.Tensor,
        rows: int,
        cols: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state[p]
        row_ema = state.get("pmuoneq_row_ema")
        if row_ema is None or row_ema.shape != (rows,) or row_ema.device != device:
            row_ema = state["pmuoneq_row_ema"] = torch.ones(rows, device=device, dtype=torch.float32)
        row_factor = state.get("pmuoneq_row_factor")
        if row_factor is None or row_factor.shape != (rows,) or row_factor.device != device:
            row_factor = state["pmuoneq_row_factor"] = torch.ones(rows, device=device, dtype=torch.float32)
        second_shape = (rows, 1) if rows >= cols else (1, cols)
        second = state.get("normuon_second_momentum")
        if second is None or second.shape != second_shape or second.device != device:
            second = state["normuon_second_momentum"] = torch.zeros(*second_shape, device=device, dtype=torch.float32)
        return row_ema, row_factor, second

    @staticmethod
    def _orientation_normuon(
        update: torch.Tensor,
        second_momentum: torch.Tensor,
        *,
        beta2: float,
        eps: float,
    ) -> torch.Tensor:
        dtype = update.dtype
        eps_t = torch.tensor(eps, dtype=dtype, device=update.device)
        old_norm = update.norm(dim=(-2, -1), keepdim=True)
        if update.shape[-2] >= update.shape[-1]:
            power = update.square().mean(dim=-1, keepdim=True)
        else:
            power = update.square().mean(dim=-2, keepdim=True)
        second_momentum.lerp_(power.to(second_momentum.dtype), 1.0 - beta2)
        out = update * torch.rsqrt(second_momentum.to(dtype).clamp_min(eps))
        return out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps_t))

    def _transform_matrix_bucket(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        group: dict,
    ) -> torch.Tensor:
        sources = torch.stack([entry[2].to(torch.float32) for entry in entries], dim=0)
        grads = torch.stack([entry[3].to(torch.float32) for entry in entries], dim=0)
        _batch, rows, cols = grads.shape

        row_buffers: list[torch.Tensor] = []
        row_factor_buffers: list[torch.Tensor] = []
        second_buffers: list[torch.Tensor] = []
        for p, _anchor, _source, _grad in entries:
            row_ema, row_factor, second = self._ensure_orientation_state(p, rows, cols, grads.device)
            row_buffers.append(row_ema)
            row_factor_buffers.append(row_factor)
            second_buffers.append(second)

        beta_p = float(group["pmuoneq_beta"])
        row_stack = torch.stack(row_buffers, dim=0)
        row_stack.mul_(beta_p).add_(grads.square().mean(dim=2), alpha=1.0 - beta_p)
        row_factor = root_optimizer._diag_inverse_power(
            row_stack,
            gamma=float(group["row_gamma"]),
            eps=float(group["pmuoneq_eps"]),
        )

        preconditioned = sources * row_factor[:, :, None]
        update = self._orthogonalizer(preconditioned)
        update = update * (0.2 * math.sqrt(max(update.size(-2), update.size(-1))))

        second_stack = torch.stack(second_buffers, dim=0)
        update = self._orientation_normuon(
            update,
            second_stack,
            beta2=float(group["normuon_beta2"]),
            eps=float(group["normuon_eps"]),
        )

        for idx, (p, _anchor, _source, _grad) in enumerate(entries):
            self.state[p]["pmuoneq_row_ema"].copy_(row_stack[idx])
            self.state[p]["pmuoneq_row_factor"].copy_(row_factor[idx])
            self.state[p]["normuon_second_momentum"].copy_(second_stack[idx])
        return update


def root_anchor_param_groups(model: nn.Module, cfg: TrialConfig) -> list[dict]:
    groups: list[dict] = []
    for group in _anchor_param_groups(model, cfg.weight_decay):
        cloned = dict(group)
        cloned["params"] = list(group["params"])
        cloned.update({
            "use_matrix_update": True,
            "lr": cfg.lr,
            "momentum": cfg.momentum,
            "pmuoneq_beta": cfg.pmuon_beta,
            "row_gamma": cfg.row_gamma,
            "pmuoneq_eps": 1e-6,
            "normuon_beta2": cfg.normuon_beta,
            "normuon_eps": 1e-10,
            "fallback_mode": cfg.fallback_mode,
            "betas": (cfg.fallback_beta1, cfg.fallback_beta2),
            "fallback_weight_decay": cfg.fallback_weight_decay,
            "eps": 1e-8,
        })
        if cfg.soda == "all":
            cloned["weight_decay"] = 0.0
        groups.append(cloned)
    return groups


def make_optimizer(model: nn.Module, cfg: TrialConfig, args: argparse.Namespace) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    if cfg.optimizer == "muown":
        return CifarMuown(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            momentum=cfg.momentum,
            betas=(0.9, 0.95),
            min_matrix_dim=int(getattr(args, "anchor_min_matrix_dim", 2)),
            fallback_lr_mult=cfg.fallback_lr_mult,
            mag_lr_mult=cfg.muown_mag_lr_mult,
        )
    if cfg.optimizer == "ema_muown":
        base = CifarMuown(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            momentum=cfg.momentum,
            betas=(0.9, 0.95),
            min_matrix_dim=int(getattr(args, "anchor_min_matrix_dim", 2)),
            fallback_lr_mult=cfg.fallback_lr_mult,
            mag_lr_mult=cfg.muown_mag_lr_mult,
        )
        return EMANesterovOptimizer(
            base,
            total_steps=int(getattr(args, "total_steps_for_optimizer", args.epochs)),
            beta=cfg.ema_beta,
            gamma=cfg.ema_gamma,
            warmup_frac=cfg.ema_warmup_frac,
            rest_frac=cfg.ema_rest_frac,
        )
    if cfg.optimizer == "ema_muon":
        base = AnchorMuon(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            warmup_steps=args.warmup_steps,
            use_external_lr=True,
            weight_decay=cfg.weight_decay,
            soda="none",
            pmuon_eq=False,
            use_gram=True,
            row_gamma=0.0,
            col_gamma=0.0,
            pmuon_beta=cfg.pmuon_beta,
            momentum=cfg.momentum,
            amuse=False,
            mimuon=False,
            normuon=False,
        )
        for group in base.param_groups:
            group.setdefault("_bench_base_lr", float(group.get("lr", cfg.lr)))
        return EMANesterovOptimizer(
            base,
            total_steps=int(getattr(args, "total_steps_for_optimizer", args.epochs)),
            beta=cfg.ema_beta,
            gamma=cfg.ema_gamma,
            warmup_frac=cfg.ema_warmup_frac,
            rest_frac=cfg.ema_rest_frac,
        )
    if cfg.optimizer == "anchormuon":
        return AnchorMuon(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            warmup_steps=args.warmup_steps,
            use_external_lr=trainer_owns_lr_schedule(cfg),
            weight_decay=cfg.weight_decay,
            soda=cfg.soda,
            pmuon_eq=cfg.pmuon_eq,
            use_gram=cfg.use_gram,
            row_gamma=cfg.row_gamma,
            col_gamma=cfg.col_gamma,
            pmuon_beta=cfg.pmuon_beta,
            momentum=cfg.momentum,
            amuse=cfg.amuse,
            mimuon=cfg.mimuon,
            mimuon_mix=cfg.mimuon_mix,
            normuon=cfg.normuon,
            normuon_beta=cfg.normuon_beta,
            normuon_aspect_scale=cfg.normuon_aspect_scale,
        )
    if cfg.optimizer == "sfplus":
        if not cfg.pmuon_eq or not cfg.normuon:
            raise ValueError("sfplus preset expects PMuonEq + NorMuon matrix direction")
        return SFPlusAnchorMuon(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            betas=(0.9, 0.95),
            sfplus_beta1=cfg.sfplus_beta1,
            sfplus_beta1_max=cfg.sfplus_beta1_max,
            sfplus_beta1_anneal_steps=cfg.sfplus_beta1_anneal_steps,
            sfplus_polyak_beta=cfg.sfplus_polyak_beta,
            sfplus_c_warmup=cfg.sfplus_c_warmup,
            sfplus_r=cfg.sfplus_r,
            sfplus_weight_lr_power=cfg.sfplus_weight_lr_power,
            sfplus_polyak=cfg.sfplus_polyak,
            sfplus_c_warmup_enabled=cfg.sfplus_c_warmup_enabled,
            sfplus_beta_anneal=cfg.sfplus_beta_anneal,
            sfplus_adamc_decay=cfg.sfplus_adamc_decay,
            sfplus_inner_momentum=cfg.sfplus_inner_momentum,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            pmuon_beta=cfg.pmuon_beta,
            row_gamma=cfg.row_gamma,
            col_gamma=cfg.col_gamma,
            normuon_beta=cfg.normuon_beta,
            ns_steps=int(getattr(args, "anchor_ns_steps", 5)),
            min_matrix_dim=int(getattr(args, "anchor_min_matrix_dim", 2)),
        )
    if cfg.optimizer == "root":
        if cfg.soda != "all":
            raise ValueError("root optimizer always uses all-parameter SODA; set soda='all'")
        if cfg.col_gamma != 0.0:
            raise ValueError("root optimizer is row-only PMuonEq; col_gamma must be 0.0")
        if cfg.root_grouping not in {"anchor", "named"}:
            raise ValueError("root_grouping must be 'anchor' or 'named'")
        if cfg.root_normuon_mode not in {"row", "orientation"}:
            raise ValueError("root_normuon_mode must be 'row' or 'orientation'")
        opt_cls = (
            RootAnchorMuonOrientation
            if cfg.root_normuon_mode == "orientation"
            else root_optimizer.AnchorMuon
        )
        params = root_anchor_param_groups(model, cfg) if cfg.root_grouping == "anchor" else model.named_parameters()
        return opt_cls(
            params,
            lr=cfg.lr,
            fallback_lr=cfg.lr * cfg.fallback_lr_mult,
            fallback_mode=cfg.fallback_mode,
            momentum=cfg.momentum,
            pmuoneq_beta=cfg.pmuon_beta,
            row_gamma=cfg.row_gamma,
            normuon_beta2=cfg.normuon_beta,
            fallback_betas=(cfg.fallback_beta1, cfg.fallback_beta2),
            fallback_weight_decay=cfg.fallback_weight_decay,
            soda_lambda_scale=1.0,
            min_matrix_dim=int(getattr(args, "anchor_min_matrix_dim", 2)),
        )
    if cfg.optimizer == "golden":
        if cfg.soda != "all":
            raise ValueError("golden optimizer always uses SODA on all parameter groups")
        if cfg.col_gamma != 0.0:
            raise ValueError("golden optimizer is row-only PMuonEq; col_gamma must be 0.0")
        if cfg.amuse or cfg.mimuon or cfg.normuon_aspect_scale or not cfg.pmuon_eq or not cfg.normuon:
            raise ValueError("golden optimizer only implements the direct no-AMUSE/no-MiMuon/no-aspect winning path")
        return GoldenMuon(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            warmup_steps=args.warmup_steps,
            use_external_lr=trainer_owns_lr_schedule(cfg),
            momentum=cfg.momentum,
            pmuoneq_beta=cfg.pmuon_beta,
            row_gamma=cfg.row_gamma,
            normuon_beta=cfg.normuon_beta,
            ns_steps=int(getattr(args, "anchor_ns_steps", 5)),
            min_matrix_dim=int(getattr(args, "anchor_min_matrix_dim", 2)),
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer}")


def selected_eval_steps(total_steps: int, bins: int) -> set[int]:
    bins = max(1, int(bins))
    steps = {max(1, round(total_steps * i / bins)) for i in range(1, bins + 1)}
    steps.add(total_steps)
    return steps


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


@torch.no_grad()
def evaluate(model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    was_training = model.training
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, targets)
        batch = int(targets.numel())
        loss_sum += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch
    if hasattr(optimizer, "train"):
        optimizer.train()
    if was_training:
        model.train()
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def final_epoch_row_or_raise(epoch_rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    if not epoch_rows:
        raise RuntimeError("No training epoch completed; check --epochs, --max-steps, and dataloader size")
    return dict(epoch_rows[-1])


def run_worker(args: argparse.Namespace) -> None:
    assert args.trial_json is not None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    cfg = TrialConfig(**json.loads(args.trial_json.read_text()))
    trial_dir = args.output_dir / cfg.name
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "metrics.jsonl"
    trial_seed = int(cfg.seed if cfg.seed is not None else args.seed)
    args.seed = trial_seed
    set_seed(trial_seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    train_loader, val_loader, test_loader, dataset_info = cifar10_loaders(args)
    total_steps = args.epochs * len(train_loader)
    if args.max_steps > 0:
        total_steps = min(total_steps, int(args.max_steps))
    args.total_steps_for_optimizer = total_steps
    if cfg.optimizer == "sfplus":
        if cfg.sfplus_c_warmup_enabled and cfg.sfplus_c_warmup <= 0:
            cfg.sfplus_c_warmup = int(args.warmup_steps)
        if cfg.sfplus_beta_anneal and cfg.sfplus_beta1_anneal_steps <= 0:
            cfg.sfplus_beta1_anneal_steps = int(total_steps)
    model = create_model(args.model, pretrained=False, num_classes=10, img_size=32, drop_path_rate=0.05)
    model.to(device=device, memory_format=torch.channels_last)
    optimizer = make_optimizer(model, cfg, args)
    criterion = nn.CrossEntropyLoss()
    eval_steps = selected_eval_steps(total_steps, args.eval_bins)
    global_step = 0
    total_examples_seen = 0
    total_train_seconds = 0.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epoch_rows: list[dict[str, float | int | str]] = []
    started = time.perf_counter()

    with metrics_path.open("w") as f:
        interval_start = time.perf_counter()
        interval_loss_sum = 0.0
        interval_examples = 0
        interval_step_ms_sum = 0.0
        interval_steps = 0
        for epoch in range(args.epochs):
            if hasattr(optimizer, "train"):
                optimizer.train()
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            train_start = time.perf_counter()
            for batch_idx, (images, targets) in enumerate(train_loader):
                if global_step >= total_steps:
                    break
                if trainer_owns_lr_schedule(cfg):
                    lr = scheduled_lr(
                        global_step,
                        total_steps,
                        cfg.lr,
                        args.warmup_steps,
                        cfg.lr_schedule,
                        final_scale=args.lr_final_scale,
                        wsd_decay_frac=args.wsd_decay_frac,
                    )
                    apply_scheduled_lrs(optimizer, lr, cfg.lr)
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                if args.sync_step_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images)
                    loss = criterion(logits, targets)
                loss.backward()
                if cfg.optimizer == "sfplus":
                    optimizer.step(function_value=float(loss.detach().cpu()))
                else:
                    optimizer.step()
                if args.sync_step_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_ms = (time.perf_counter() - step_start) * 1000.0
                batch = int(targets.numel())
                train_loss_sum += float(loss.item()) * batch
                train_correct += int((logits.argmax(dim=1) == targets).sum().item())
                train_total += batch
                total_examples_seen += batch
                interval_loss_sum += float(loss.item()) * batch
                interval_examples += batch
                interval_step_ms_sum += step_ms
                interval_steps += 1
                total_train_seconds += step_ms / 1000.0
                if global_step % args.log_every == 0:
                    row = {
                        "type": "step",
                        "trial": cfg.name,
                        "epoch": epoch,
                        "step": global_step,
                        "batch": batch_idx,
                        "train_loss": float(loss.item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "step_ms": step_ms,
                        "step_timing": "cuda_synchronized" if args.sync_step_timing else "cpu_launch_unsynchronized",
                    }
                    if hasattr(optimizer, "last_stats"):
                        row.update({f"opt_{k}": v for k, v in optimizer.last_stats.items()})
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                global_step += 1
                if global_step in eval_steps:
                    interval_seconds = time.perf_counter() - interval_start
                    val_loss, val_acc = evaluate(model, optimizer, val_loader, device)
                    best_val_acc = max(best_val_acc, val_acc)
                    best_val_loss = min(best_val_loss, val_loss)
                    row = {
                        "type": "bin",
                        "trial": cfg.name,
                        "epoch": epoch,
                        "step": global_step,
                        "progress": global_step / max(total_steps, 1),
                        "train_loss": interval_loss_sum / max(interval_examples, 1),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                        "best_val_loss": best_val_loss,
                        "interval_examples_per_sec": interval_examples / max(interval_seconds, 1e-9),
                        "interval_avg_step_ms": interval_step_ms_sum / max(interval_steps, 1),
                        "overall_examples_per_sec": total_examples_seen / max(total_train_seconds, 1e-9),
                        "elapsed_wall_sec": time.perf_counter() - started,
                        "max_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                        "step_timing": "cuda_synchronized" if args.sync_step_timing else "cpu_launch_unsynchronized",
                    }
                    if hasattr(optimizer, "last_stats"):
                        row.update({f"opt_{k}": v for k, v in optimizer.last_stats.items()})
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(json.dumps(row), flush=True)
                    interval_start = time.perf_counter()
                    interval_loss_sum = 0.0
                    interval_examples = 0
                    interval_step_ms_sum = 0.0
                    interval_steps = 0
            train_seconds = time.perf_counter() - train_start
            if train_total == 0:
                break
            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = 100.0 * train_correct / max(train_total, 1)
            val_loss, val_acc = evaluate(model, optimizer, val_loader, device)
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            row = {
                "type": "epoch",
                "trial": cfg.name,
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "examples_per_sec": train_total / max(train_seconds, 1e-9),
                "max_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "step_timing": "cuda_synchronized" if args.sync_step_timing else "cpu_launch_unsynchronized",
            }
            if hasattr(optimizer, "last_stats"):
                row.update({f"opt_{k}": v for k, v in optimizer.last_stats.items()})
            f.write(json.dumps(row) + "\n")
            f.flush()
            epoch_rows.append(row)
            print(json.dumps(row), flush=True)
            if global_step >= total_steps:
                break

    final = final_epoch_row_or_raise(epoch_rows)
    if args.eval_test:
        if test_loader is None:
            raise RuntimeError("--eval-test was requested but no test loader was constructed")
        test_loss, test_acc = evaluate(model, optimizer, test_loader, device)
        final.update({
            "test_loss": test_loss,
            "test_acc": test_acc,
        })
    final.update({
        "trial": cfg.name,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "lr_schedule": cfg.lr_schedule,
        "lr_final_scale": args.lr_final_scale,
        "wsd_decay_frac": args.wsd_decay_frac,
        "seed": trial_seed,
        "weight_decay": cfg.weight_decay,
        "soda": cfg.soda,
        "pmuon_eq": cfg.pmuon_eq,
        "use_gram": cfg.use_gram,
        "row_gamma": cfg.row_gamma,
        "col_gamma": cfg.col_gamma,
        "pmuon_beta": cfg.pmuon_beta,
        "momentum": cfg.momentum,
        "amuse": cfg.amuse,
        "mimuon": cfg.mimuon,
        "mimuon_mix": cfg.mimuon_mix,
        "normuon": cfg.normuon,
        "normuon_beta": cfg.normuon_beta,
        "normuon_aspect_scale": cfg.normuon_aspect_scale,
        "root_grouping": cfg.root_grouping,
        "root_normuon_mode": cfg.root_normuon_mode,
        "fallback_mode": cfg.fallback_mode,
        "fallback_lr_mult": cfg.fallback_lr_mult,
        "fallback_beta1": cfg.fallback_beta1,
        "fallback_beta2": cfg.fallback_beta2,
        "fallback_weight_decay": cfg.fallback_weight_decay,
        "sfplus_polyak": cfg.sfplus_polyak,
        "sfplus_c_warmup_enabled": cfg.sfplus_c_warmup_enabled,
        "sfplus_beta_anneal": cfg.sfplus_beta_anneal,
        "sfplus_adamc_decay": cfg.sfplus_adamc_decay,
        "sfplus_inner_momentum": cfg.sfplus_inner_momentum,
        "sfplus_beta1": cfg.sfplus_beta1,
        "sfplus_beta1_max": cfg.sfplus_beta1_max,
        "sfplus_beta1_anneal_steps": cfg.sfplus_beta1_anneal_steps,
        "sfplus_polyak_beta": cfg.sfplus_polyak_beta,
        "sfplus_c_warmup": cfg.sfplus_c_warmup,
        "sfplus_r": cfg.sfplus_r,
        "sfplus_weight_lr_power": cfg.sfplus_weight_lr_power,
        "external_lr": cfg.external_lr,
        "ema_beta": cfg.ema_beta,
        "ema_gamma": cfg.ema_gamma,
        "ema_warmup_frac": cfg.ema_warmup_frac,
        "ema_rest_frac": cfg.ema_rest_frac,
        "muown_mag_lr_mult": cfg.muown_mag_lr_mult,
        "avg_step_ms": 1000.0 * total_train_seconds / max(global_step, 1),
        "overall_examples_per_sec": total_examples_seen / max(total_train_seconds, 1e-9),
        "elapsed_sec": time.perf_counter() - started,
        "step_timing": "cuda_synchronized" if args.sync_step_timing else "cpu_launch_unsynchronized",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    })
    final.update(dataset_info)
    (trial_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n")


def launch_trials(args: argparse.Namespace, trials: list[TrialConfig]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to silently fall back to CPU")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No visible CUDA GPUs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir.resolve()
    data_path = args.data_path.resolve()
    ensure_cifar10_downloaded(data_path)
    pending = list(trials)
    running: list[tuple[subprocess.Popen, str, int]] = []
    available_gpus = list(range(gpu_count))
    while pending or running:
        while pending and available_gpus:
            cfg = pending.pop(0)
            if args.skip_completed and (output_dir / cfg.name / "summary.json").exists():
                print(f"[skip] completed trial={cfg.name}", flush=True)
                continue
            gpu = available_gpus.pop(0)
            trial_json = output_dir / f"{cfg.name}.trial.json"
            trial_json.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--trial-json", str(trial_json),
                "--data-path", str(data_path),
                "--output-dir", str(output_dir),
                "--model", args.model,
                "--epochs", str(args.epochs),
                "--max-steps", str(args.max_steps),
                "--eval-bins", str(args.eval_bins),
                "--train-subset", str(args.train_subset),
                "--val-subset", str(args.val_subset),
                "--val-source", str(args.val_source),
                "--train-val-size", str(args.train_val_size),
                "--split-seed", str(args.split_seed),
                "--test-subset", str(args.test_subset),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--seed", str(int(cfg.seed if cfg.seed is not None else args.seed)),
                "--warmup-steps", str(args.warmup_steps),
                "--lr-final-scale", str(args.lr_final_scale),
                "--wsd-decay-frac", str(args.wsd_decay_frac),
                "--log-every", str(args.log_every),
            ]
            if args.eval_test:
                cmd.append("--eval-test")
            if not args.sync_step_timing:
                cmd.append("--no-sync-step-timing")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[launch] gpu={gpu} trial={cfg.name}", flush=True)
            running.append((subprocess.Popen(cmd, env=env, cwd=ROOT), cfg.name, gpu))
        time.sleep(2.0)
        still_running: list[tuple[subprocess.Popen, str, int]] = []
        for proc, name, gpu in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, name, gpu))
            elif rc != 0:
                raise RuntimeError(f"trial {name} failed with exit code {rc}")
            else:
                available_gpus.append(gpu)
                print(f"[done] {name}", flush=True)
        running = still_running
        available_gpus = sorted(available_gpus)


def collect_epoch_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("*/metrics.jsonl")):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("type") in {"epoch", "bin"}:
                    rows.append({k: str(v) for k, v in item.items()})
    return rows


def add_speed_fields_from_metrics(summary_path: Path, row: dict) -> dict:
    if "avg_step_ms" in row and "overall_examples_per_sec" in row:
        return row
    metrics_path = summary_path.parent / "metrics.jsonl"
    if not metrics_path.exists():
        return row
    step_ms_sum = 0.0
    final_examples_per_sec = 0.0
    interval_count = 0.0
    with metrics_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") != "bin":
                continue
            train_loss = float(item.get("train_loss", 0.0))
            if not math.isfinite(train_loss):
                continue
            examples_per_sec = float(item.get("overall_examples_per_sec", 0.0))
            step_ms = float(item.get("interval_avg_step_ms", 0.0))
            progress = float(item.get("progress", 0.0))
            if step_ms > 0:
                interval_count += 1.0
                step_ms_sum += step_ms
            if examples_per_sec > 0 and progress >= 1.0:
                final_examples_per_sec = examples_per_sec
    if interval_count > 0 and "avg_step_ms" not in row:
        row["avg_step_ms"] = step_ms_sum / interval_count
    if final_examples_per_sec > 0 and "overall_examples_per_sec" not in row:
        row["overall_examples_per_sec"] = final_examples_per_sec
    return row


def finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def summarize(output_dir: Path, make_plots: bool) -> None:
    summaries = []
    for path in sorted(output_dir.glob("*/summary.json")):
        summaries.append(add_speed_fields_from_metrics(path, json.loads(path.read_text())))
    if not summaries:
        return
    fieldnames = sorted({k for row in summaries for k in row.keys()})
    with (output_dir / "all_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    finite_summaries = [
        row
        for row in summaries
        if math.isfinite(finite_float(row.get("best_val_acc"), -math.inf))
        and math.isfinite(finite_float(row.get("best_val_loss"), math.inf))
    ]
    selectable_summaries = finite_summaries or summaries
    best = max(
        selectable_summaries,
        key=lambda row: (
            finite_float(row.get("best_val_acc"), -math.inf),
            -finite_float(row.get("best_val_loss"), math.inf),
        ),
    )
    (output_dir / "best_run.json").write_text(json.dumps(best, indent=2) + "\n")
    by_family_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for row in selectable_summaries:
        cfg = TrialConfig(
            name=row["trial"],
            optimizer=row["optimizer"],
            lr=float(row["lr"]),
            seed=int(row["seed"]) if row.get("seed") is not None else None,
            lr_schedule=str(row.get("lr_schedule", "cosine")),
            weight_decay=float(row["weight_decay"]),
            soda=row["soda"],
            pmuon_eq=parse_bool(row["pmuon_eq"]),
            use_gram=parse_bool(row.get("use_gram", True)),
            row_gamma=float(row["row_gamma"]),
            col_gamma=float(row["col_gamma"]),
            pmuon_beta=float(row.get("pmuon_beta", 0.95)),
            momentum=float(row.get("momentum", 0.95)),
            amuse=parse_bool(row.get("amuse", True)),
            mimuon=parse_bool(row["mimuon"]),
            mimuon_mix=float(row["mimuon_mix"]),
            normuon=parse_bool(row.get("normuon", False)),
            normuon_beta=float(row.get("normuon_beta", 0.95)),
            normuon_aspect_scale=parse_bool(row.get("normuon_aspect_scale", False)),
            root_grouping=str(row.get("root_grouping", "anchor")),
            root_normuon_mode=str(row.get("root_normuon_mode", "row")),
            fallback_mode=str(row.get("fallback_mode", "rms")),
            fallback_lr_mult=float(row.get("fallback_lr_mult", 1.0)),
            fallback_beta1=float(row.get("fallback_beta1", 0.90)),
            fallback_beta2=float(row.get("fallback_beta2", 0.95)),
            fallback_weight_decay=float(row.get("fallback_weight_decay", 0.0)),
            sfplus_polyak=parse_bool(row.get("sfplus_polyak", False)),
            sfplus_c_warmup_enabled=parse_bool(row.get("sfplus_c_warmup_enabled", False)),
            sfplus_beta_anneal=parse_bool(row.get("sfplus_beta_anneal", False)),
            sfplus_adamc_decay=parse_bool(row.get("sfplus_adamc_decay", False)),
            sfplus_inner_momentum=parse_bool(row.get("sfplus_inner_momentum", False)),
            sfplus_beta1=float(row.get("sfplus_beta1", 0.90)),
            sfplus_beta1_max=float(row.get("sfplus_beta1_max", 0.965)),
            sfplus_beta1_anneal_steps=int(float(row.get("sfplus_beta1_anneal_steps", 0))),
            sfplus_polyak_beta=float(row.get("sfplus_polyak_beta", 0.0)),
            sfplus_c_warmup=int(float(row.get("sfplus_c_warmup", 0))),
            sfplus_r=float(row.get("sfplus_r", 0.0)),
            sfplus_weight_lr_power=float(row.get("sfplus_weight_lr_power", 2.0)),
            external_lr=parse_bool(row.get("external_lr", False)),
            ema_beta=float(row.get("ema_beta", 0.0)),
            ema_gamma=float(row.get("ema_gamma", 0.99)),
            ema_warmup_frac=float(row.get("ema_warmup_frac", 0.30)),
            ema_rest_frac=float(row.get("ema_rest_frac", 0.20)),
            muown_mag_lr_mult=float(row.get("muown_mag_lr_mult", 1.0)),
        )
        fam = trial_family(cfg)
        seedless_cfg = TrialConfig(**asdict(cfg))
        seedless_cfg.name = re.sub(r"_seed\d+$", "", seedless_cfg.name)
        seedless_cfg.seed = None
        key = json.dumps(asdict(seedless_cfg), sort_keys=True)
        family_candidates = by_family_candidates.setdefault(fam, {})
        candidate = family_candidates.setdefault(key, {"cfg": seedless_cfg, "rows": []})
        candidate["rows"].append(row)

    by_family: dict[str, dict] = {}
    for fam, candidates in by_family_candidates.items():
        best_candidate: dict[str, Any] | None = None
        best_score: tuple[float, float, int] | None = None
        for candidate in candidates.values():
            rows = candidate["rows"]
            acc_values = [finite_float(row["best_val_acc"], -math.inf) for row in rows]
            loss_values = [finite_float(row["best_val_loss"], math.inf) for row in rows]
            mean_acc = sum(acc_values) / len(acc_values)
            mean_loss = sum(loss_values) / len(loss_values)
            score = (mean_acc, -mean_loss, len(rows))
            if best_score is None or score > best_score:
                best_candidate = candidate
                best_score = score
        if best_candidate is None:
            continue
        rows = best_candidate["rows"]
        cfg = best_candidate["cfg"]
        acc_values = [finite_float(row["best_val_acc"], -math.inf) for row in rows]
        loss_values = [finite_float(row["best_val_loss"], math.inf) for row in rows]
        mean_acc = sum(acc_values) / len(acc_values)
        mean_loss = sum(loss_values) / len(loss_values)
        acc_std = math.sqrt(sum((value - mean_acc) ** 2 for value in acc_values) / len(acc_values))
        loss_std = math.sqrt(sum((value - mean_loss) ** 2 for value in loss_values) / len(loss_values))
        selected = dict(rows[0])
        selected.update(
            {
                "trial": cfg.name,
                "seed": "",
                "hpo_selected_by": "mean_over_seeds",
                "hpo_seed_count": len(rows),
                "hpo_best_val_acc_mean": mean_acc,
                "hpo_best_val_acc_std": acc_std,
                "hpo_best_val_loss_mean": mean_loss,
                "hpo_best_val_loss_std": loss_std,
            }
        )
        by_family[fam] = selected
    (output_dir / "best_by_family.json").write_text(json.dumps(by_family, indent=2) + "\n")
    if make_plots:
        try:
            import matplotlib.pyplot as plt

            def display_label(trial: str) -> str:
                """Human-facing plot label; raw trial IDs remain in CSV/JSON."""
                seed_suffix = ""
                seed_match = re.search(r"_seed(\d+)$", trial)
                if seed_match:
                    seed_suffix = f" seed {seed_match.group(1)}"
                    trial = trial[: seed_match.start()]
                if trial.startswith("final_"):
                    parts = trial.split("_", 2)
                    if len(parts) == 3:
                        trial = parts[2]
                if trial.startswith("adamw_"):
                    schedule = "cosine" if "cosine" in trial else "schedule"
                    return f"AdamW {schedule}{seed_suffix}"
                if trial.startswith("sfplus_"):
                    match = re.search(r"sfplus_([PCBDMpcbdm]{5})", trial)
                    code = match.group(1) if match else trial[len("sfplus_"):].split("_", 1)[0]
                    return f"SF+ {code}{seed_suffix}"
                if trial.startswith("fallback_"):
                    parts = trial.split("_")
                    mode = parts[1].upper() if len(parts) > 1 else "fallback"
                    return f"AnchorMuon {mode} fallback{seed_suffix}"
                if trial.startswith("plain_muon_"):
                    return f"Plain Muon{seed_suffix}"
                if trial.startswith("muon_"):
                    return f"Plain Muon{seed_suffix}"
                if trial.startswith("muown_"):
                    return f"Muown{seed_suffix}"
                if trial.startswith("ema_muon_"):
                    return f"EMA-Nesterov Muon{seed_suffix}"
                if trial.startswith("ema_muown_"):
                    return f"EMA-Nesterov Muown{seed_suffix}"
                if trial.startswith("previous_best_rms_"):
                    return f"Previous best RMS{seed_suffix}"
                if trial.startswith("atan2_best_"):
                    return f"AnchorMuon AdamATan2{seed_suffix}"
                component = re.match(r"(?:final_)?component_(full|no_soda|no_pmuoneq|no_gram|no_normuon|adamw)", trial)
                if component:
                    labels = {
                        "full": "AnchorMuon full",
                        "no_soda": "AnchorMuon -SODA",
                        "no_pmuoneq": "AnchorMuon -PMuonEq",
                        "no_gram": "AnchorMuon -GramNS",
                        "no_normuon": "AnchorMuon -NorMuon",
                        "adamw": "AdamW",
                    }
                    return f"{labels[component.group(1)]}{seed_suffix}"
                if trial.startswith("root_named_"):
                    rest = trial[len("root_named_"):]
                    schedule = rest.split("_", 1)[0]
                    return f"AnchorMuon {schedule}{seed_suffix}"
                if trial.startswith("root_"):
                    rest = trial[len("root_"):]
                    schedule = rest.split("_", 1)[0]
                    return f"AnchorMuon {schedule}{seed_suffix}"
                return f"{trial}{seed_suffix}"

            rows = collect_epoch_rows(output_dir)
            by_trial: dict[str, list[dict[str, float]]] = {}
            for row in rows:
                by_trial.setdefault(row["trial"], []).append({
                    "epoch": float(row["epoch"]),
                    "step": float(row["step"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "examples_per_sec": float(row.get("examples_per_sec", row.get("interval_examples_per_sec", 0.0))),
                })
            for metric in ["train_loss", "val_loss", "val_acc", "examples_per_sec"]:
                plt.figure(figsize=(9, 5))
                for trial, vals in by_trial.items():
                    vals = sorted(vals, key=lambda x: x["step"])
                    plt.plot(
                        [v["step"] for v in vals],
                        [v[metric] for v in vals],
                        marker="o",
                        label=display_label(trial),
                    )
                plt.xlabel("optimizer step")
                plt.ylabel(metric)
                plt.legend(fontsize=7)
                plt.tight_layout()
                plt.savefig(output_dir / f"{metric}.png", dpi=160)
                plt.close()
            for metric, path_name, title, reverse in [
                ("avg_step_ms", "step_time_ms_bar.png", "Average training step time (ms)", False),
                ("overall_examples_per_sec", "examples_per_sec_bar.png", "Overall training examples/s", True),
            ]:
                available = [
                    row
                    for row in summaries
                    if math.isfinite(finite_float(row.get(metric), math.nan))
                ]
                if not available:
                    continue
                available = sorted(
                    available,
                    key=lambda row: finite_float(row.get(metric), -math.inf if reverse else math.inf),
                    reverse=reverse,
                )
                height = max(4.0, 0.32 * len(available))
                plt.figure(figsize=(11, height))
                labels = [display_label(str(row["trial"])) for row in available]
                vals = [finite_float(row.get(metric), math.nan) for row in available]
                bars = plt.barh(labels, vals)
                plt.gca().invert_yaxis()
                plt.xlabel(title)
                if vals:
                    plt.xlim(0.0, max(vals) * 1.12)
                for bar, val in zip(bars, vals):
                    plt.text(
                        val,
                        bar.get_y() + bar.get_height() / 2,
                        f" {val:.2f}",
                        va="center",
                        fontsize=7,
                    )
                plt.tight_layout()
                plt.savefig(output_dir / path_name, dpi=180)
                plt.close()
        except Exception as exc:  # pragma: no cover - plotting is best effort.
            print(f"plotting skipped: {exc}")
    print("best run:", json.dumps(best, indent=2), flush=True)


def filtered_trials(args: argparse.Namespace) -> list[TrialConfig]:
    trials = trial_grid(args.preset)
    if args.only:
        pattern = re.compile(args.only)
        trials = [trial for trial in trials if pattern.search(trial.name)]
    if args.max_trials > 0:
        trials = trials[: args.max_trials]
    trials = expand_trials_by_seeds(trials, args.seeds)
    if not trials:
        raise ValueError("no trials selected")
    return trials


def expand_trials_by_seeds(trials: list[TrialConfig], seeds_csv: str) -> list[TrialConfig]:
    if not seeds_csv:
        return trials
    seeds = [int(part) for part in seeds_csv.split(",") if part.strip()]
    if not seeds:
        return trials
    expanded = []
    for trial in trials:
        for seed in seeds:
            seeded = TrialConfig(**asdict(trial))
            seeded.seed = seed
            seeded.name = f"{trial.name}_seed{seed}"
            expanded.append(seeded)
    return expanded


def summaries_to_trials(rows: Iterable[dict]) -> list[TrialConfig]:
    trials = []
    for family, row in rows:
        trials.append(TrialConfig(
            name=f"final_{family}_{row['trial']}",
            optimizer=row["optimizer"],
            lr=float(row["lr"]),
            seed=int(row["seed"]) if str(row.get("seed", "")).strip() else None,
            lr_schedule=str(row.get("lr_schedule", "cosine")),
            weight_decay=float(row["weight_decay"]),
            soda=row["soda"],
            pmuon_eq=parse_bool(row["pmuon_eq"]),
            use_gram=parse_bool(row.get("use_gram", True)),
            row_gamma=float(row["row_gamma"]),
            col_gamma=float(row["col_gamma"]),
            pmuon_beta=float(row.get("pmuon_beta", 0.95)),
            momentum=float(row.get("momentum", 0.95)),
            amuse=parse_bool(row.get("amuse", True)),
            mimuon=parse_bool(row["mimuon"]),
            mimuon_mix=float(row["mimuon_mix"]),
            normuon=parse_bool(row.get("normuon", False)),
            normuon_beta=float(row.get("normuon_beta", 0.95)),
            normuon_aspect_scale=parse_bool(row.get("normuon_aspect_scale", False)),
            root_grouping=str(row.get("root_grouping", "anchor")),
            root_normuon_mode=str(row.get("root_normuon_mode", "row")),
            fallback_mode=str(row.get("fallback_mode", "rms")),
            fallback_lr_mult=float(row.get("fallback_lr_mult", 1.0)),
            fallback_beta1=float(row.get("fallback_beta1", 0.90)),
            fallback_beta2=float(row.get("fallback_beta2", 0.95)),
            fallback_weight_decay=float(row.get("fallback_weight_decay", 0.0)),
            sfplus_polyak=parse_bool(row.get("sfplus_polyak", False)),
            sfplus_c_warmup_enabled=parse_bool(row.get("sfplus_c_warmup_enabled", False)),
            sfplus_beta_anneal=parse_bool(row.get("sfplus_beta_anneal", False)),
            sfplus_adamc_decay=parse_bool(row.get("sfplus_adamc_decay", False)),
            sfplus_inner_momentum=parse_bool(row.get("sfplus_inner_momentum", False)),
            sfplus_beta1=float(row.get("sfplus_beta1", 0.90)),
            sfplus_beta1_max=float(row.get("sfplus_beta1_max", 0.965)),
            sfplus_beta1_anneal_steps=int(float(row.get("sfplus_beta1_anneal_steps", 0))),
            sfplus_polyak_beta=float(row.get("sfplus_polyak_beta", 0.0)),
            sfplus_c_warmup=int(float(row.get("sfplus_c_warmup", 0))),
            sfplus_r=float(row.get("sfplus_r", 0.0)),
            sfplus_weight_lr_power=float(row.get("sfplus_weight_lr_power", 2.0)),
            external_lr=parse_bool(row.get("external_lr", False)),
            ema_beta=float(row.get("ema_beta", 0.0)),
            ema_gamma=float(row.get("ema_gamma", 0.99)),
            ema_warmup_frac=float(row.get("ema_warmup_frac", 0.30)),
            ema_rest_frac=float(row.get("ema_rest_frac", 0.20)),
            muown_mag_lr_mult=float(row.get("muown_mag_lr_mult", 1.0)),
        ))
    return trials


def main() -> None:
    args = parse_args()
    validate_experiment_args(args)
    if args.worker:
        run_worker(args)
        return
    if args.final_from_hpo is not None:
        by_family = json.loads(args.final_from_hpo.read_text())
        final_args = argparse.Namespace(**vars(args))
        final_args.epochs = args.final_epochs
        final_args.max_steps = args.final_max_steps
        final_trials = summaries_to_trials(sorted(by_family.items()))
        final_trials = expand_trials_by_seeds(final_trials, args.seeds)
        if args.only:
            pattern = re.compile(args.only)
            final_trials = [trial for trial in final_trials if pattern.search(trial.name)]
        launch_trials(final_args, final_trials)
        summarize(final_args.output_dir, make_plots=not args.no_plots)
        return
    if args.two_stage:
        base_output = args.output_dir
        hpo_args = argparse.Namespace(**vars(args))
        hpo_args.output_dir = base_output / "hpo"
        hpo_args.epochs = args.hpo_epochs
        hpo_args.max_steps = args.hpo_max_steps
        hpo_args.eval_test = False
        hpo_trials = filtered_trials(hpo_args)
        launch_trials(hpo_args, hpo_trials)
        summarize(hpo_args.output_dir, make_plots=not args.no_plots)
        by_family = json.loads((hpo_args.output_dir / "best_by_family.json").read_text())
        final_args = argparse.Namespace(**vars(args))
        final_args.output_dir = base_output / "final_bins"
        final_args.epochs = args.final_epochs
        final_args.max_steps = args.final_max_steps
        final_trials = summaries_to_trials(sorted(by_family.items()))
        final_trials = expand_trials_by_seeds(final_trials, args.seeds)
        if args.only:
            pattern = re.compile(args.only)
            final_trials = [trial for trial in final_trials if pattern.search(trial.name)]
        launch_trials(final_args, final_trials)
        summarize(final_args.output_dir, make_plots=not args.no_plots)
        return
    trials = filtered_trials(args)
    launch_trials(args, trials)
    summarize(args.output_dir, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
