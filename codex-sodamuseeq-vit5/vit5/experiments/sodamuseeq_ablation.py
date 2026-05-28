from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models_vit5 import Block, RMSNorm, vit_models  # noqa: E402
from optim_sodamuseeq import SodaMuseEq, make_sodamuseeq_param_groups  # noqa: E402


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
RUN_LABELS = {
    "adamw": "AdamW",
    "full": "SODA+AMUSE+PMuonEq+Gram",
    "full_corrected": "SODA+AMUSE+PMuonEq+Gram tuned beta/rho",
    "full_lr008": "SODA+AMUSE+PMuonEq+Gram tuned beta/rho lr0.008",
    "no_soda": "AMUSE+PMuonEq+Gram",
    "no_amuse": "SODA+PMuonEq+Gram",
    "no_pmuoneq": "SODA+AMUSE+Gram",
    "mimuon": "SODA+AMUSE+PMuonEq+MiMuon+Gram",
    "soda_pmuoneq": "SODA+PMuonEq+Gram",
    "normuon_base": "NorMuon+BaseGram",
    "soda_pmuoneq_mimuon": "SODA+PMuonEq+MiMuon+Gram",
    "soda_pmuoneq_normuon": "SODA+PMuonEq+Gram+NorMuon",
    "soda_pmuoneq_mimuon_normuon": "SODA+PMuonEq+MiMuon+Gram+NorMuon",
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    optimizer: str
    lr: float
    weight_decay: float = 0.05
    pmuon_gamma: float = 0.2
    pmuon_beta: float = 0.95
    use_soda: bool = True
    use_amuse: bool = True
    use_pmuoneq: bool = True
    use_gram: bool = True
    use_mimuon: bool = False
    use_normuon: bool = False
    mimuon_tau: float = 0.005
    normuon_beta2: float = 0.95
    beta1: float = 0.6
    rho: float = 0.8
    warmup_steps: int | None = None
    soda_warmup_steps: int | None = None
    soda_replaces_weight_decay: bool = True

    def to_args(self) -> list[str]:
        out = [
            "--run-name",
            self.name,
            "--optimizer",
            self.optimizer,
            "--lr",
            str(self.lr),
            "--weight-decay",
            str(self.weight_decay),
            "--pmuon-gamma",
            str(self.pmuon_gamma),
            "--pmuon-beta",
            str(self.pmuon_beta),
            "--mimuon-tau",
            str(self.mimuon_tau),
            "--normuon-beta2",
            str(self.normuon_beta2),
            "--beta1",
            str(self.beta1),
            "--rho",
            str(self.rho),
        ]
        if self.warmup_steps is not None:
            out.extend(["--warmup-steps", str(self.warmup_steps)])
        if self.soda_warmup_steps is not None:
            out.extend(["--soda-warmup-steps", str(self.soda_warmup_steps)])
        for flag, value in (
            ("--use-soda", self.use_soda),
            ("--use-amuse", self.use_amuse),
            ("--use-pmuoneq", self.use_pmuoneq),
            ("--use-gram", self.use_gram),
            ("--use-mimuon", self.use_mimuon),
            ("--use-normuon", self.use_normuon),
        ):
            out.append(flag if value else flag.replace("--use-", "--no-"))
        out.append("--soda-replaces-weight-decay" if self.soda_replaces_weight_decay else "--soda-keeps-weight-decay")
        return out


class Cifar10WithTransform(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]
        return self.transform(image), target

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    data_root = Path(args.data_dir)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    base_train = datasets.CIFAR10(data_root, train=True, download=True)
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=eval_transform)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(base_train))
    val_idx = perm[: args.val_size].tolist()
    train_idx = perm[args.val_size :].tolist()
    train_eval_idx = train_idx[: min(args.train_eval_size, len(train_idx))]
    train = Cifar10WithTransform(base_train, train_idx, train_transform)
    train_eval = Cifar10WithTransform(base_train, train_eval_idx, eval_transform)
    val = Cifar10WithTransform(base_train, val_idx, eval_transform)
    workers = int(args.workers)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    eval_workers = max(1, workers // 2)
    train_eval_loader = DataLoader(train_eval, batch_size=args.eval_batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True, persistent_workers=workers > 0)
    val_loader = DataLoader(val, batch_size=args.eval_batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True, persistent_workers=workers > 0)
    test_loader = DataLoader(test, batch_size=args.eval_batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True, persistent_workers=workers > 0)
    return train_loader, train_eval_loader, val_loader, test_loader


def create_tiny_vit5(args):
    return vit_models(
        img_size=32,
        patch_size=args.patch_size,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=False,
        num_registers=4,
        flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6),
        block_layers=Block,
        rope=True,
        qk_norm=True,
        layer_scale=True,
    )


def create_optimizer(args, model):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if args.optimizer != "sodamuseeq":
        raise ValueError(f"unknown optimizer {args.optimizer!r}")
    groups = make_sodamuseeq_param_groups(model, lr=args.lr, weight_decay=args.weight_decay)
    return SodaMuseEq(
        groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        beta1=args.beta1,
        beta2=args.beta2,
        rho=args.rho,
        warmup_steps=args.warmup_steps,
        use_soda=args.use_soda,
        soda_warmup_steps=args.soda_warmup_steps,
        soda_replaces_weight_decay=args.soda_replaces_weight_decay,
        use_amuse=args.use_amuse,
        use_pmuoneq=args.use_pmuoneq,
        pmuon_beta=args.pmuon_beta,
        pmuon_gamma=args.pmuon_gamma,
        use_gram=args.use_gram,
        use_mimuon=args.use_mimuon,
        mimuon_tau=args.mimuon_tau,
        use_normuon=args.use_normuon,
        normuon_beta2=args.normuon_beta2,
        batch_project=args.batch_project,
        stats_interval=args.stats_interval,
    )


@torch.no_grad()
def evaluate(model, loader, device, optimizer=None, amp=True):
    if optimizer is not None and hasattr(optimizer, "eval"):
        optimizer.eval()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    if optimizer is not None and hasattr(optimizer, "train"):
        optimizer.train()
    return {"loss": total_loss / max(1, total), "acc": total_correct / max(1, total)}


def train_trial(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    train_loader, train_eval_loader, val_loader, test_loader = build_loaders(args)
    model = create_tiny_vit5(args).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = create_optimizer(args, model)
    if hasattr(optimizer, "train"):
        optimizer.train()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    max_steps = int(args.max_steps)
    eval_steps = sorted(set([max(1, round(max_steps * i / args.eval_bins)) for i in range(1, args.eval_bins + 1)]))
    train_iter = iter(train_loader)
    metrics_path = out_dir / "metrics.jsonl"
    step = 0
    interval_loss = 0.0
    interval_correct = 0
    interval_examples = 0
    interval_start = time.perf_counter()
    fwd_bwd_time = 0.0
    opt_time = 0.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    start = time.perf_counter()

    def write(record):
        with metrics_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        step += 1
        fwd_bwd_time += t1 - t0
        opt_time += t2 - t1
        interval_loss += float(loss.detach().item()) * int(y.numel())
        interval_correct += int((logits.detach().argmax(dim=-1) == y).sum().item())
        interval_examples += int(y.numel())
        if step in eval_steps:
            train_elapsed = max(1e-9, time.perf_counter() - interval_start)
            train_eval = evaluate(model, train_eval_loader, device, optimizer=optimizer, amp=args.amp)
            val = evaluate(model, val_loader, device, optimizer=optimizer, amp=args.amp)
            best_val_loss = min(best_val_loss, val["loss"])
            best_val_acc = max(best_val_acc, val["acc"])
            record = {
                "step": step,
                "bin": eval_steps.index(step) + 1,
                "train_interval_loss": interval_loss / max(1, interval_examples),
                "train_interval_acc": interval_correct / max(1, interval_examples),
                "train_eval_loss": train_eval["loss"],
                "train_eval_acc": train_eval["acc"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "examples_per_sec": interval_examples / train_elapsed,
                "step_ms": 1000.0 * train_elapsed / max(1, interval_examples / args.batch_size),
                "fwd_bwd_ms": 1000.0 * fwd_bwd_time / max(1, interval_examples / args.batch_size),
                "optimizer_ms": 1000.0 * opt_time / max(1, interval_examples / args.batch_size),
                "max_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                **{k: float(v) for k, v in getattr(optimizer, "get_last_stats", lambda: {})().items() if isinstance(v, (int, float)) and math.isfinite(float(v))},
            }
            write(record)
            print(json.dumps({"run": args.run_name, **record}, sort_keys=True), flush=True)
            interval_loss = 0.0
            interval_correct = 0
            interval_examples = 0
            interval_start = time.perf_counter()
            fwd_bwd_time = 0.0
            opt_time = 0.0
            model.train()
            if hasattr(optimizer, "train"):
                optimizer.train()

    final_val = evaluate(model, val_loader, device, optimizer=optimizer, amp=args.amp)
    final_test = evaluate(model, test_loader, device, optimizer=optimizer, amp=args.amp) if args.eval_test else {"loss": float("nan"), "acc": float("nan")}
    wall = time.perf_counter() - start
    summary = {
        "run_name": args.run_name,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pmuon_gamma": args.pmuon_gamma,
        "pmuon_beta": args.pmuon_beta,
        "beta1": args.beta1,
        "rho": args.rho,
        "use_soda": args.use_soda,
        "use_amuse": args.use_amuse,
        "use_pmuoneq": args.use_pmuoneq,
        "use_gram": args.use_gram,
        "use_mimuon": args.use_mimuon,
        "use_normuon": args.use_normuon,
        "mimuon_tau": args.mimuon_tau,
        "normuon_beta2": args.normuon_beta2,
        "soda_replaces_weight_decay": args.soda_replaces_weight_decay,
        "warmup_steps": args.warmup_steps,
        "soda_warmup_steps": args.soda_warmup_steps,
        "steps": max_steps,
        "seed": args.seed,
        "param_count": param_count,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_loss": final_val["loss"],
        "final_val_acc": final_val["acc"],
        "test_loss": final_test["loss"],
        "test_acc": final_test["acc"],
        "wall_sec": wall,
        "steps_per_sec": max_steps / max(wall, 1e-9),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def hpo_specs(args) -> list[RunSpec]:
    if args.sweep_kind == "amuse_tune":
        return amuse_tune_specs(args)
    if args.sweep_kind == "normuon_tune":
        return normuon_tune_specs(args)
    specs: list[RunSpec] = []
    for lr in (1e-3, 2e-3, 3e-3, 5e-3):
        for wd in (0.02, 0.05, 0.1):
            specs.append(RunSpec(f"hpo_adamw_lr{lr:g}_wd{wd:g}", "adamw", lr=lr, weight_decay=wd))
    variants = {
        "full": {},
        "no_soda": {"use_soda": False},
        "no_amuse": {"use_amuse": False},
        "no_pmuoneq": {"use_pmuoneq": False},
        "mimuon": {"use_mimuon": True},
    }
    for variant, overrides in variants.items():
        gammas = (0.0,) if overrides.get("use_pmuoneq") is False else (0.05, 0.1, 0.2)
        for lr in (4e-3, 7e-3, 1e-2, 1.4e-2):
            for wd in (0.03, 0.05, 0.1):
                for gamma in gammas:
                    specs.append(
                        RunSpec(
                            f"hpo_{variant}_lr{lr:g}_wd{wd:g}_g{gamma:g}",
                            "sodamuseeq",
                            lr=lr,
                            weight_decay=wd,
                            pmuon_gamma=gamma,
                            **overrides,
                        )
                    )
    return specs


def amuse_tune_specs(args) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for lr in (2e-3, 3e-3, 5e-3):
        for wd in (0.02, 0.05):
            specs.append(RunSpec(f"hpo_adamw_lr{lr:g}_wd{wd:g}", "adamw", lr=lr, weight_decay=wd))

    # Corrected pure-SODA control: no AMUSE, no ordinary weight decay.
    for lr in (0.008, 0.01, 0.012, 0.014):
        for gamma in (0.025, 0.05, 0.1):
            specs.append(
                RunSpec(
                    f"hpo_soda_pmuoneq_lr{lr:g}_g{gamma:g}",
                    "sodamuseeq",
                    lr=lr,
                    weight_decay=0.0,
                    pmuon_gamma=gamma,
                    use_soda=True,
                    use_amuse=False,
                    use_pmuoneq=True,
                    soda_replaces_weight_decay=True,
                    warmup_steps=500,
                    soda_warmup_steps=500,
                )
            )

    # AMUSE stability grid: stronger averaging, lower LR, longer warmup, pure SODA.
    for lr in (0.004, 0.006, 0.008, 0.01):
        for beta1 in (0.8, 0.9, 0.95):
            for rho in (0.9, 0.95):
                for gamma in (0.0, 0.025, 0.05):
                    specs.append(
                        RunSpec(
                            f"hpo_amuse_lr{lr:g}_b{beta1:g}_r{rho:g}_g{gamma:g}",
                            "sodamuseeq",
                            lr=lr,
                            weight_decay=0.0,
                            pmuon_gamma=gamma,
                            use_soda=True,
                            use_amuse=True,
                            use_pmuoneq=True,
                            beta1=beta1,
                            rho=rho,
                            soda_replaces_weight_decay=True,
                            warmup_steps=500,
                            soda_warmup_steps=500,
                        )
                    )
    return specs


def normuon_tune_specs(args) -> list[RunSpec]:
    specs: list[RunSpec] = []
    # Tuned AdamW reference from the corrected CIFAR runs.
    specs.append(RunSpec("hpo_adamw_lr0.003_wd0.02", "adamw", lr=3e-3, weight_decay=0.02))

    variants = {
        "soda_pmuoneq": {"use_mimuon": False, "use_normuon": False},
        "soda_pmuoneq_mimuon": {"use_mimuon": True, "use_normuon": False},
        "soda_pmuoneq_normuon": {"use_mimuon": False, "use_normuon": True},
        "soda_pmuoneq_mimuon_normuon": {"use_mimuon": True, "use_normuon": True},
    }
    for variant, overrides in variants.items():
        for lr in (0.01, 0.012, 0.014, 0.018):
            for gamma in (0.0, 0.025, 0.05, 0.1):
                for tau in ((0.0, 0.005, 0.02) if overrides["use_mimuon"] else (0.005,)):
                    for beta2 in ((0.9, 0.95, 0.98) if overrides["use_normuon"] else (0.95,)):
                        specs.append(
                            RunSpec(
                                f"hpo_{variant}_lr{lr:g}_g{gamma:g}_tau{tau:g}_nb{beta2:g}",
                                "sodamuseeq",
                                lr=lr,
                                weight_decay=0.0,
                                pmuon_gamma=gamma,
                                use_soda=True,
                                use_amuse=False,
                                use_pmuoneq=True,
                                soda_replaces_weight_decay=True,
                                warmup_steps=500,
                                soda_warmup_steps=500,
                                mimuon_tau=tau,
                                normuon_beta2=beta2,
                                **overrides,
                            )
                        )
    return specs


def launch_sweep(args, specs: list[RunSpec], subdir: str, steps: int, eval_bins: int, eval_test: bool = False):
    root = Path(args.sweep_out_dir) / subdir
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_specs.json").write_text(json.dumps([spec.__dict__ for spec in specs], indent=2, sort_keys=True) + "\n")
    gpus = [str(i) for i in range(torch.cuda.device_count())]
    if not gpus:
        raise RuntimeError("CUDA GPUs are required")
    queue = list(specs)
    active: dict[subprocess.Popen, tuple[RunSpec, Path, str, Any]] = {}
    finished: list[dict[str, Any]] = []

    def start(spec: RunSpec, gpu: str):
        out_dir = root / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--trial",
            "--out-dir",
            str(out_dir.resolve()),
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
            str(args.seed),
            "--embed-dim",
            str(args.embed_dim),
            "--depth",
            str(args.depth),
            "--num-heads",
            str(args.num_heads),
            "--warmup-steps",
            str(max(1, min(args.warmup_steps, steps // 4))),
            "--soda-warmup-steps",
            str(max(0, min(args.soda_warmup_steps, steps // 4))),
            *spec.to_args(),
        ]
        if eval_test:
            cmd.append("--eval-test")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTHONUNBUFFERED", "1")
        log = (out_dir / "stdout.log").open("w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active[proc] = (spec, out_dir, gpu, log)
        print(f"started {spec.name} on gpu {gpu}", flush=True)

    available = gpus.copy()
    while queue or active:
        while queue and available:
            start(queue.pop(0), available.pop(0))
        time.sleep(2)
        for proc in list(active):
            code = proc.poll()
            if code is None:
                continue
            spec, out_dir, gpu, log = active.pop(proc)
            log.close()
            available.append(gpu)
            if code != 0:
                row = {"run_name": spec.name, "status": "failed", "returncode": code}
                print(f"failed {spec.name}; see {out_dir / 'stdout.log'}", flush=True)
            else:
                row = json.loads((out_dir / "summary.json").read_text())
                row["status"] = "ok"
                print(f"finished {spec.name}: val_loss={row['best_val_loss']:.4f} val_acc={row['best_val_acc']:.4f}", flush=True)
            finished.append(row)
            write_summary(root, finished)
    write_summary(root, finished)
    return root, finished


def write_summary(root: Path, rows: list[dict[str, Any]]):
    csv_path = root / "all_runs.csv"
    keys = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: (float(r.get("best_val_loss", float("inf"))), -float(r.get("best_val_acc", 0.0))))
    (root / "summary_sorted.json").write_text(json.dumps(ok, indent=2, sort_keys=True) + "\n")


def final_specs_from_hpo(hpo_rows: list[dict[str, Any]]) -> list[RunSpec]:
    ok = [r for r in hpo_rows if r.get("status") == "ok"]
    best_by_variant: dict[str, dict[str, Any]] = {}
    for row in ok:
        if not row.get("use_gram", True):
            continue
        if row["optimizer"] == "adamw":
            key = "adamw"
        elif not row.get("use_amuse") and row.get("use_mimuon") and row.get("use_normuon"):
            key = "soda_pmuoneq_mimuon_normuon"
        elif not row.get("use_amuse") and row.get("use_mimuon"):
            key = "soda_pmuoneq_mimuon"
        elif not row.get("use_amuse") and row.get("use_normuon"):
            key = "soda_pmuoneq_normuon"
        elif not row.get("use_amuse"):
            key = "soda_pmuoneq"
        elif row.get("use_mimuon"):
            key = "mimuon"
        elif not row.get("use_soda"):
            key = "no_soda"
        elif not row.get("use_amuse"):
            key = "no_amuse"
        elif not row.get("use_pmuoneq"):
            key = "no_pmuoneq"
        else:
            key = "full"
        current = best_by_variant.get(key)
        if current is None or float(row["best_val_loss"]) < float(current["best_val_loss"]):
            best_by_variant[key] = row
    final = []
    for key, row in sorted(best_by_variant.items()):
        final.append(
            RunSpec(
                f"final_{key}",
                str(row["optimizer"]),
                lr=float(row["lr"]),
                weight_decay=float(row["weight_decay"]),
                pmuon_gamma=float(row.get("pmuon_gamma", 0.2)),
                use_soda=bool(row.get("use_soda", True)),
                use_amuse=bool(row.get("use_amuse", True)),
                use_pmuoneq=bool(row.get("use_pmuoneq", True)),
                use_gram=bool(row.get("use_gram", True)),
                use_mimuon=bool(row.get("use_mimuon", False)),
                use_normuon=bool(row.get("use_normuon", False)),
                beta1=float(row.get("beta1", 0.6)),
                rho=float(row.get("rho", 0.8)),
                warmup_steps=int(float(row.get("warmup_steps", 0))) or None,
                soda_warmup_steps=int(float(row.get("soda_warmup_steps", 0))) or None,
                soda_replaces_weight_decay=_as_bool(row.get("soda_replaces_weight_decay"), True),
                mimuon_tau=float(row.get("mimuon_tau", 0.005)),
                normuon_beta2=float(row.get("normuon_beta2", 0.95)),
            )
        )
    return final


def run_pipeline(args):
    hpo_root, hpo_rows = launch_sweep(args, hpo_specs(args), "hpo", args.hpo_steps, eval_bins=4, eval_test=False)
    final_specs = final_specs_from_hpo(hpo_rows)
    final_root, final_rows = launch_sweep(args, final_specs, "final", args.final_steps, eval_bins=8, eval_test=True)
    write_final_bin_table_and_plots(final_root)
    report = write_pipeline_report(Path(args.sweep_out_dir), hpo_root, final_root, final_rows)
    print(report)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _curve(rows: list[dict[str, Any]], key: str, digits: int = 3) -> str:
    return "`" + ", ".join(_fmt_float(row.get(key), digits) for row in rows) + "`"


def _mean_value(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / max(1, len(vals))


def write_pipeline_report(
    sweep_root: Path,
    hpo_root: Path,
    final_root: Path,
    final_rows: list[dict[str, Any]] | None = None,
) -> Path:
    if final_rows is None:
        final_rows = _read_csv_rows(final_root / "all_runs.csv")
    report = sweep_root / "report.md"
    hpo_rows = _read_csv_rows(hpo_root / "all_runs.csv")
    hpo_ok = [r for r in hpo_rows if r.get("status") == "ok"]
    hpo_steps = max((int(float(r.get("steps", 0))) for r in hpo_ok), default=0)
    ok = [r for r in final_rows if r.get("status") == "ok" and _as_bool(r.get("use_gram"), True)]
    ok.sort(key=lambda r: (float(r.get("best_val_loss", float("inf"))), -float(r.get("best_val_acc", 0.0))))
    final_steps = max((int(float(r.get("steps", 0))) for r in ok), default=0)
    bin_rows_for_scope = _read_csv_rows(final_root / "final_bins.csv")
    final_bins = max((int(float(r.get("bin", 0))) for r in bin_rows_for_scope), default=0)
    lines = [
        "# SodaMuseEq ViT-5 Ablation Study",
        "",
        f"HPO root: `{hpo_root}`",
        f"Final root: `{final_root}`",
        "",
        "## Scope",
        "",
        f"- HPO completed trials: {len(hpo_ok)}.",
        f"- HPO budget: {hpo_steps or 'n/a'} steps per trial.",
        f"- Final budget: {final_steps or 'n/a'} steps per selected variant, {final_bins or 'n/a'} evaluation bins, CIFAR-10 test evaluation.",
        "- Model: compact ViT-5-style CIFAR-10 model with 192-dim embeddings, 6 blocks, 3 heads.",
        "- Parallelism: one trial per visible GPU during sweeps.",
        "",
        "| rank | run | label | optimizer | val loss | val acc | test loss | test acc | wall sec | steps/sec |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ok, 1):
        label = RUN_LABELS.get(str(row["run_name"]).removeprefix("final_"), str(row["run_name"]))
        lines.append(
            f"| {idx} | `{row['run_name']}` | {label} | `{row['optimizer']}` | {float(row['best_val_loss']):.4f} | "
            f"{100 * float(row['best_val_acc']):.2f}% | {float(row['test_loss']):.4f} | "
            f"{100 * float(row['test_acc']):.2f}% | {float(row['wall_sec']):.1f} | {float(row['steps_per_sec']):.3f} |"
        )
    bin_rows = _read_csv_rows(final_root / "final_bins.csv")
    if bin_rows:
        by_name: dict[str, list[dict[str, str]]] = {}
        for row in bin_rows:
            by_name.setdefault(row["run_name"], []).append(row)
        lines.extend(
            [
                "",
                "## Eight-Bin Curves",
                "",
                "Each curve lists bins 1-8. Interval speed excludes validation time.",
                "",
                "| run | label | steps | train interval loss | train eval loss | val loss | val acc | avg examples/s | avg step ms | avg opt ms |",
                "|---|---|---|---|---|---|---|---:|---:|---:|",
            ]
        )
        ordered_names = [row["run_name"] for row in ok if row.get("run_name") in by_name]
        for name in ordered_names:
            series = sorted(by_name[name], key=lambda r: int(float(r["bin"])))
            steps = "`" + ", ".join(str(int(float(row["step"]))) for row in series) + "`"
            val_acc = "`" + ", ".join(f"{100.0 * float(row['val_acc']):.1f}" for row in series) + "`"
            label = RUN_LABELS.get(name.removeprefix("final_"), name)
            lines.append(
                f"| `{name}` | {label} | {steps} | {_curve(series, 'train_interval_loss')} | "
                f"{_curve(series, 'train_eval_loss')} | {_curve(series, 'val_loss')} | {val_acc} | "
                f"{_mean_value(series, 'examples_per_sec'):.0f} | {_mean_value(series, 'step_ms'):.2f} | "
                f"{_mean_value(series, 'optimizer_ms'):.2f} |"
            )
        lines.extend(
            [
                "",
                "## Plots",
                "",
                f"- Validation loss: `{final_root / 'plots' / 'val_loss.png'}`",
                f"- Train interval loss: `{final_root / 'plots' / 'train_interval_loss.png'}`",
                f"- Clean train-subset loss: `{final_root / 'plots' / 'train_eval_loss.png'}`",
                f"- Validation accuracy: `{final_root / 'plots' / 'val_acc.png'}`",
                f"- Step time: `{final_root / 'plots' / 'step_ms.png'}`",
                f"- Examples/sec: `{final_root / 'plots' / 'examples_per_sec.png'}`",
                f"- Sorted speed bar chart: `{final_root / 'plots' / 'step_speed_bar.png'}`",
            ]
        )
    report.write_text("\n".join(lines) + "\n")
    return report


def write_final_bin_table_and_plots(final_root: Path):
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(final_root.glob("*/metrics.jsonl")):
        run_name = metrics_path.parent.name
        with metrics_path.open() as f:
            for line in f:
                row = json.loads(line)
                row["run_name"] = run_name
                rows.append(row)
    if not rows:
        return
    csv_path = final_root / "final_bins.csv"
    keys = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = final_root / "plots"
    plot_dir.mkdir(exist_ok=True)
    names = sorted({row["run_name"] for row in rows})
    names = [name for name in names if name != "final_no_gram"]
    for key, ylabel, fname in (
        ("val_loss", "validation loss", "val_loss.png"),
        ("train_eval_loss", "clean train-subset loss", "train_eval_loss.png"),
        ("train_interval_loss", "augmented train interval loss", "train_interval_loss.png"),
        ("val_acc", "validation accuracy", "val_acc.png"),
        ("step_ms", "training step ms", "step_ms.png"),
        ("examples_per_sec", "training examples/sec", "examples_per_sec.png"),
    ):
        plt.figure(figsize=(10, 6))
        for name in names:
            series = sorted([row for row in rows if row["run_name"] == name], key=lambda r: int(r["bin"]))
            plt.plot([int(r["bin"]) for r in series], [float(r[key]) for r in series], marker="o", label=name)
        plt.xlabel("evaluation bin")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=150)
        plt.close()
    speed_rows = []
    for name in names:
        series = [row for row in rows if row["run_name"] == name]
        if series:
            speed_rows.append((name, _mean_value(series, "step_ms"), _mean_value(series, "examples_per_sec")))
    speed_rows.sort(key=lambda item: item[1])
    if speed_rows:
        labels = [RUN_LABELS.get(name.removeprefix("final_"), name) for name, _, _ in speed_rows]
        step_ms = [value for _, value, _ in speed_rows]
        plt.figure(figsize=(12, 5))
        bars = plt.bar(labels, step_ms)
        plt.ylabel("average training step ms")
        plt.title("Iteration Speed By Optimizer")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, step_ms, strict=True):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / "step_speed_bar.png", dpi=150)
        plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--run-name", default="trial")
    p.add_argument("--out-dir", default="runs/sodamuseeq_trial")
    p.add_argument("--sweep-out-dir", default="runs/sodamuseeq_ablation")
    p.add_argument("--data-dir", default="../../data/cifar10")
    p.add_argument("--optimizer", choices=["adamw", "sodamuseeq"], default="sodamuseeq")
    p.add_argument("--lr", type=float, default=9e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--beta1", type=float, default=0.6)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--pmuon-gamma", type=float, default=0.2)
    p.add_argument("--pmuon-beta", type=float, default=0.95)
    p.add_argument("--mimuon-tau", type=float, default=0.005)
    p.add_argument("--use-soda", action="store_true", default=True)
    p.add_argument("--no-soda", action="store_false", dest="use_soda")
    p.add_argument("--soda-replaces-weight-decay", action="store_true", default=True)
    p.add_argument("--soda-keeps-weight-decay", action="store_false", dest="soda_replaces_weight_decay")
    p.add_argument("--use-amuse", action="store_true", default=True)
    p.add_argument("--no-amuse", action="store_false", dest="use_amuse")
    p.add_argument("--use-pmuoneq", action="store_true", default=True)
    p.add_argument("--no-pmuoneq", action="store_false", dest="use_pmuoneq")
    p.add_argument("--use-gram", action="store_true", default=True)
    p.add_argument("--no-gram", action="store_false", dest="use_gram")
    p.add_argument("--use-mimuon", action="store_true", default=False)
    p.add_argument("--no-mimuon", action="store_false", dest="use_mimuon")
    p.add_argument("--use-normuon", action="store_true", default=False)
    p.add_argument("--no-normuon", action="store_false", dest="use_normuon")
    p.add_argument("--normuon-beta2", type=float, default=0.95)
    p.add_argument("--batch-project", action="store_true", default=True)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--soda-warmup-steps", type=int, default=50)
    p.add_argument("--stats-interval", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--hpo-steps", type=int, default=200)
    p.add_argument("--final-steps", type=int, default=800)
    p.add_argument("--eval-bins", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 8) // 4)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--train-eval-size", type=int, default=5000)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--eval-test", action="store_true", default=False)
    p.add_argument("--sweep-kind", choices=["default", "amuse_tune", "normuon_tune"], default="default")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.trial:
        train_trial(args)
    elif args.sweep:
        run_pipeline(args)
    elif args.report_only:
        root = Path(args.sweep_out_dir)
        print(write_pipeline_report(root, root / "hpo", root / "final"))
    else:
        raise SystemExit("Pass --trial, --sweep, or --report-only")
