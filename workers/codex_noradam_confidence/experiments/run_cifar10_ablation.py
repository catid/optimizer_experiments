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
from typing import Iterable

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
from golden_soda_pmuoneq_normuon import GoldenSodaPmuonEqNorMuon
from optim_anchormuon import AnchorMuon
from optim_factory import _anchor_param_groups


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
        ],
    )
    parser.add_argument("--only", default="", help="Regex filter for trial names")
    parser.add_argument("--max-trials", type=int, default=0)
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
        default="test",
        help=(
            "Validation source. Historical runs use CIFAR-10 train=False as validation. "
            "Use train_split for HPO and reserve train=False for final test evaluation."
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


def trial_grid(preset: str) -> list[TrialConfig]:
    smoke = [
        TrialConfig("adamw_lr1e-3", "adamw", 1e-3),
        TrialConfig("anchormuon_full_lr4e-3_rg0p20", "anchormuon", 4e-3),
        TrialConfig("anchormuon_no_soda_lr4e-3", "anchormuon", 4e-3, soda="none"),
        TrialConfig("anchormuon_no_pmuoneq_lr4e-3", "anchormuon", 4e-3, pmuon_eq=False),
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
            subset_gen = torch.Generator().manual_seed(int(args.seed))
            train_indices = torch.randperm(len(train_ds), generator=subset_gen)[: args.train_subset].tolist()
            train_ds = Subset(train_ds, train_indices)
        if 0 < args.val_subset < len(val_ds):
            val_ds = Subset(val_ds, list(range(args.val_subset)))
        dataset_info = {
            "val_source": "test",
            "split_seed": int(args.seed),
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


def adamw_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    schedule: str = "cosine",
) -> float:
    if step < max(1, warmup_steps):
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"unknown AdamW lr schedule {schedule!r}")
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


class RootSodaPmuonEqNorMuonOrientation(root_optimizer.SodaPmuonEqNorMuon):
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
            "base_lr": cfg.lr,
            "momentum": cfg.momentum,
            "pmuoneq_beta": cfg.pmuon_beta,
            "row_gamma": cfg.row_gamma,
            "pmuoneq_eps": 1e-6,
            "normuon_beta2": cfg.normuon_beta,
            "normuon_eps": 1e-10,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "use_external_lr": False,
        })
        if cfg.soda == "all":
            cloned["weight_decay"] = 0.0
        groups.append(cloned)
    return groups


def make_optimizer(model: nn.Module, cfg: TrialConfig, args: argparse.Namespace) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    if cfg.optimizer == "anchormuon":
        return AnchorMuon(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            warmup_steps=args.warmup_steps,
            use_external_lr=False,
            weight_decay=cfg.weight_decay,
            soda=cfg.soda,
            pmuon_eq=cfg.pmuon_eq,
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
            RootSodaPmuonEqNorMuonOrientation
            if cfg.root_normuon_mode == "orientation"
            else root_optimizer.SodaPmuonEqNorMuon
        )
        fallback_weight_decay = 0.0
        params = root_anchor_param_groups(model, cfg) if cfg.root_grouping == "anchor" else model.named_parameters()
        return opt_cls(
            params,
            matrix_lr=cfg.lr,
            fallback_lr=cfg.lr,
            warmup_steps=args.warmup_steps,
            momentum=cfg.momentum,
            pmuoneq_beta=cfg.pmuon_beta,
            row_gamma=cfg.row_gamma,
            normuon_beta2=cfg.normuon_beta,
            fallback_betas=(0.9, 0.95),
            fallback_weight_decay=fallback_weight_decay,
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
        return GoldenSodaPmuonEqNorMuon(
            _anchor_param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            warmup_steps=args.warmup_steps,
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
    model = create_model(args.model, pretrained=False, num_classes=10, img_size=32, drop_path_rate=0.05)
    model.to(device=device, memory_format=torch.channels_last)
    optimizer = make_optimizer(model, cfg, args)
    criterion = nn.CrossEntropyLoss()
    total_steps = args.epochs * len(train_loader)
    if args.max_steps > 0:
        total_steps = min(total_steps, int(args.max_steps))
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
                if cfg.optimizer == "adamw":
                    lr = adamw_lr(global_step, total_steps, cfg.lr, args.warmup_steps, cfg.lr_schedule)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
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

    final = dict(epoch_rows[-1])
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
        "seed": trial_seed,
        "weight_decay": cfg.weight_decay,
        "soda": cfg.soda,
        "pmuon_eq": cfg.pmuon_eq,
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
    running: list[tuple[subprocess.Popen, str]] = []
    next_gpu = 0
    while pending or running:
        while pending and len(running) < gpu_count:
            cfg = pending.pop(0)
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
                "--log-every", str(args.log_every),
            ]
            if args.eval_test:
                cmd.append("--eval-test")
            if not args.sync_step_timing:
                cmd.append("--no-sync-step-timing")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(next_gpu)
            print(f"[launch] gpu={next_gpu} trial={cfg.name}", flush=True)
            running.append((subprocess.Popen(cmd, env=env, cwd=ROOT), cfg.name))
            next_gpu = (next_gpu + 1) % gpu_count
        time.sleep(2.0)
        still_running: list[tuple[subprocess.Popen, str]] = []
        for proc, name in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, name))
            elif rc != 0:
                raise RuntimeError(f"trial {name} failed with exit code {rc}")
            else:
                print(f"[done] {name}", flush=True)
        running = still_running


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
    best = max(summaries, key=lambda row: (float(row["best_val_acc"]), -float(row["best_val_loss"])))
    (output_dir / "best_run.json").write_text(json.dumps(best, indent=2) + "\n")
    by_family: dict[str, dict] = {}
    for row in summaries:
        cfg = TrialConfig(
            name=row["trial"],
            optimizer=row["optimizer"],
            lr=float(row["lr"]),
            seed=int(row["seed"]) if row.get("seed") is not None else None,
            lr_schedule=str(row.get("lr_schedule", "cosine")),
            weight_decay=float(row["weight_decay"]),
            soda=row["soda"],
            pmuon_eq=parse_bool(row["pmuon_eq"]),
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
        )
        fam = trial_family(cfg)
        current = by_family.get(fam)
        if current is None or (
            float(row["best_val_acc"]), -float(row["best_val_loss"])
        ) > (
            float(current["best_val_acc"]), -float(current["best_val_loss"])
        ):
            by_family[fam] = row
    (output_dir / "best_by_family.json").write_text(json.dumps(by_family, indent=2) + "\n")
    if make_plots:
        try:
            import matplotlib.pyplot as plt

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
                    plt.plot([v["step"] for v in vals], [v[metric] for v in vals], marker="o", label=trial)
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
                available = [row for row in summaries if row.get(metric) not in (None, "")]
                if not available:
                    continue
                available = sorted(available, key=lambda row: float(row[metric]), reverse=reverse)
                height = max(4.0, 0.32 * len(available))
                plt.figure(figsize=(11, height))
                labels = [str(row["trial"]) for row in available]
                vals = [float(row[metric]) for row in available]
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
        ))
    return trials


def main() -> None:
    args = parse_args()
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
