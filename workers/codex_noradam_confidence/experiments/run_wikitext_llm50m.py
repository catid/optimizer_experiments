#!/usr/bin/env python3
"""Run a 50M-parameter byte-level LM optimizer comparison.

The benchmark uses real text encoded directly as UTF-8 bytes. This avoids
tokenizer downloads while still measuring an actual next-token language model
workload with embeddings, attention, MLPs, and tied output head.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, EXPERIMENT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import optimizer as root_optimizer
from run_synthetic_llm50m import AdamAtan2, PlainMuon, TinyGPT, lr_scale, make_decay_groups, split_muon_groups


@dataclass
class TrialConfig:
    name: str
    family: str
    lr: float
    weight_decay: float = 0.05
    row_gamma: float = 0.25
    pmuoneq_beta: float = 0.90
    normuon_beta2: float = 0.93
    fallback_lr_mult: float = 0.5
    fallback_mode: str = "atan2"
    soda_lambda_scale: float = 1.0
    soda_lambda_power: float = 1.0
    seed: int = 123
    steps: int = 800
    eval_every: int = 100
    warmup_steps: int = 20
    final_lr_scale: float = 0.1
    wsd_decay_frac: float = 0.2


class ByteTokenStream:
    def __init__(self, path: Path, device: torch.device) -> None:
        arr = np.load(path)
        if arr.dtype != np.uint8:
            raise ValueError(f"{path} must contain uint8 byte tokens, got {arr.dtype}")
        self.tokens = torch.from_numpy(arr.astype(np.int64, copy=False)).to(device)

    def batch(self, *, batch_size: int, block_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - block_size - 1
        if max_start <= 0:
            raise ValueError("token stream is shorter than block_size")
        idx = torch.randint(0, max_start, (batch_size,), device=self.tokens.device, generator=generator)
        offsets = torch.arange(block_size + 1, device=self.tokens.device)
        chunk = self.tokens[idx[:, None] + offsets[None, :]]
        return chunk[:, :-1], chunk[:, 1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("workers/codex_noradam_confidence/results/wikitext103_llm50m_20260529"))
    parser.add_argument("--preset", choices=["smoke", "main", "anchor_deep", "anchor_harder", "fineweb_long", "fineweb_anchor_hpo"], default="anchor_deep")
    parser.add_argument("--dataset-source", choices=["wikitext", "hf_text"], default="wikitext")
    parser.add_argument("--wiki-config", default="wikitext-103-raw-v1")
    parser.add_argument("--hf-dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--hf-config", default="sample-10BT")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-val-split", default="")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--cache-dir", type=Path, default=Path("workers/codex_noradam_confidence/data/wikitext_bytes"))
    parser.add_argument("--max-train-bytes", type=int, default=32_000_000)
    parser.add_argument("--max-val-bytes", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=10)
    parser.add_argument("--n-embd", type=int, default=640)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hpo-steps", type=int, default=800)
    parser.add_argument("--final-steps", type=int, default=800)
    parser.add_argument("--eval-bins", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-hpo-trials", type=int, default=0)
    parser.add_argument("--final-only", action="store_true")
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def dataset_label(args: argparse.Namespace) -> str:
    if args.dataset_source == "wikitext":
        return f"wikitext/{args.wiki_config}"
    config = args.hf_config or "default"
    return f"{args.hf_dataset}/{config}:{args.hf_split}"


def _bytes_from_dataset(config: str, split: str, max_bytes: int) -> np.ndarray:
    from datasets import load_dataset

    ds = load_dataset("wikitext", config, split=split)
    chunks: list[bytes] = []
    total = 0
    for row in ds:
        text = row["text"]
        if not text:
            text = ""
        data = (text + "\n").encode("utf-8", errors="replace")
        chunks.append(data)
        total += len(data)
        if total >= max_bytes:
            break
    blob = b"".join(chunks)[:max_bytes]
    return np.frombuffer(blob, dtype=np.uint8).copy()


def prepare_wikitext_cache(args: argparse.Namespace) -> tuple[Path, Path]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    safe_config = args.wiki_config.replace("/", "_")
    train_path = args.cache_dir / f"{safe_config}_train_{args.max_train_bytes}.uint8.npy"
    val_path = args.cache_dir / f"{safe_config}_validation_{args.max_val_bytes}.uint8.npy"
    if not train_path.exists():
        train = _bytes_from_dataset(args.wiki_config, "train", args.max_train_bytes)
        np.save(train_path, train)
    if not val_path.exists():
        val = _bytes_from_dataset(args.wiki_config, "validation", args.max_val_bytes)
        np.save(val_path, val)
    return train_path, val_path


def _row_text(row: dict[str, Any], text_field: str) -> str:
    value = row.get(text_field, "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _consume_text_bytes(rows: Any, *, text_field: str, max_bytes: int) -> np.ndarray:
    out = bytearray()
    for row in rows:
        data = (_row_text(row, text_field) + "\n").encode("utf-8", errors="replace")
        remaining = max_bytes - len(out)
        if remaining <= 0:
            break
        out.extend(data[:remaining])
        if len(out) >= max_bytes:
            break
    if not out:
        raise RuntimeError("dataset stream produced no text bytes")
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()


def prepare_hf_text_cache(args: argparse.Namespace) -> tuple[Path, Path]:
    from datasets import load_dataset

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(f"{args.hf_dataset}_{args.hf_config}_{args.hf_split}_{args.hf_text_field}_shuf{args.hf_shuffle_buffer}_seed{args.seed}")
    train_path = args.cache_dir / f"{safe}_train_{args.max_train_bytes}.uint8.npy"
    val_path = args.cache_dir / f"{safe}_validation_{args.max_val_bytes}.uint8.npy"
    if train_path.exists() and val_path.exists():
        return train_path, val_path

    name = args.hf_config or None
    if args.hf_val_split:
        val_ds = load_dataset(args.hf_dataset, name=name, split=args.hf_val_split, streaming=True)
        train_ds = load_dataset(args.hf_dataset, name=name, split=args.hf_split, streaming=True)
        if args.hf_shuffle_buffer > 0:
            train_ds = train_ds.shuffle(buffer_size=args.hf_shuffle_buffer, seed=args.seed)
        val = _consume_text_bytes(iter(val_ds), text_field=args.hf_text_field, max_bytes=args.max_val_bytes)
        train = _consume_text_bytes(iter(train_ds), text_field=args.hf_text_field, max_bytes=args.max_train_bytes)
    else:
        ds = load_dataset(args.hf_dataset, name=name, split=args.hf_split, streaming=True)
        if args.hf_shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=args.hf_shuffle_buffer, seed=args.seed)
        iterator = iter(ds)
        val = _consume_text_bytes(iterator, text_field=args.hf_text_field, max_bytes=args.max_val_bytes)
        train = _consume_text_bytes(iterator, text_field=args.hf_text_field, max_bytes=args.max_train_bytes)

    np.save(val_path, val)
    np.save(train_path, train)
    return train_path, val_path


def prepare_text_cache(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.dataset_source == "wikitext":
        return prepare_wikitext_cache(args)
    return prepare_hf_text_cache(args)


def make_optimizer(model: nn.Module, trial: TrialConfig) -> torch.optim.Optimizer:
    if trial.family == "adamw":
        return torch.optim.AdamW(
            make_decay_groups(model, trial.weight_decay),
            lr=trial.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    if trial.family == "adamatan2":
        return AdamAtan2(make_decay_groups(model, trial.weight_decay), lr=trial.lr, betas=(0.9, 0.95))
    if trial.family == "muon":
        matrix, fallback = split_muon_groups(model, trial.weight_decay)
        return PlainMuon(matrix, fallback, lr=trial.lr, weight_decay=trial.weight_decay)
    if trial.family == "anchormuon":
        return root_optimizer.AnchorMuon(
            model,
            lr=trial.lr,
            fallback_lr=trial.lr * trial.fallback_lr_mult,
            fallback_mode=trial.fallback_mode,
            row_gamma=trial.row_gamma,
            pmuoneq_beta=trial.pmuoneq_beta,
            normuon_beta2=trial.normuon_beta2,
            soda_lambda_scale=trial.soda_lambda_scale,
            soda_lambda_power=trial.soda_lambda_power,
        )
    raise ValueError(f"unknown optimizer family {trial.family}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    stream: ByteTokenStream,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for _ in range(batches):
        x, y = stream.batch(batch_size=batch_size, block_size=block_size, generator=generator)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.detach()))
        correct += int((logits.argmax(dim=-1) == y).sum().detach())
        total += int(y.numel())
    model.train()
    return float(np.mean(losses)), correct / max(total, 1)


def run_trial(args: argparse.Namespace, trial: TrialConfig) -> dict[str, Any]:
    torch.manual_seed(trial.seed)
    np.random.seed(trial.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    train_path, val_path = prepare_text_cache(args)

    model = TinyGPT(
        vocab_size=256,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, trial)
    for group in optimizer.param_groups:
        group["_bench_base_lr"] = float(group.get("lr", trial.lr))
    train_stream = ByteTokenStream(train_path, device)
    val_stream = ByteTokenStream(val_path, device)
    train_gen = torch.Generator(device=device).manual_seed(trial.seed + 10)
    val_gen = torch.Generator(device=device).manual_seed(trial.seed + 20)

    phase = "hpo" if not trial.name.startswith("final_") else "final"
    trial_dir = args.output_dir / phase / trial.name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(json.dumps(asdict(trial), indent=2) + "\n")

    eval_steps = set(range(trial.eval_every, trial.steps + 1, trial.eval_every))
    eval_steps.add(trial.steps)
    step_times: list[float] = []
    last_loss = float("nan")
    start = time.perf_counter()
    metrics_path = trial_dir / "metrics.jsonl"
    with metrics_path.open("w") as f:
        for step in range(1, trial.steps + 1):
            scale = lr_scale(step, trial.steps, trial.warmup_steps, trial.final_lr_scale, trial.wsd_decay_frac)
            for group in optimizer.param_groups:
                group["lr"] = float(group.get("_bench_base_lr", trial.lr)) * scale
            x, y = train_stream.batch(batch_size=args.batch_size, block_size=args.block_size, generator=train_gen)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            last_loss = float(loss.detach())
            if step % args.log_every == 0 or step in eval_steps:
                f.write(json.dumps({
                    "step": step,
                    "phase": "train",
                    "train_loss": last_loss,
                    "lr": trial.lr * scale,
                    "step_time_ms": elapsed * 1000.0,
                    "tokens_per_sec": args.batch_size * args.block_size / max(elapsed, 1e-9),
                }) + "\n")
            if step in eval_steps:
                val_loss, val_acc = evaluate(
                    model,
                    val_stream,
                    batch_size=args.batch_size,
                    block_size=args.block_size,
                    batches=args.eval_batches,
                    generator=val_gen,
                )
                recent = step_times[-max(1, min(len(step_times), trial.eval_every)):]
                mean_recent = float(np.mean(recent))
                f.write(json.dumps({
                    "step": step,
                    "phase": "eval",
                    "train_loss": last_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": trial.lr * scale,
                    "mean_step_time_ms": mean_recent * 1000.0,
                    "tokens_per_sec": args.batch_size * args.block_size / max(mean_recent, 1e-9),
                }) + "\n")
                f.flush()

    eval_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if '"phase": "eval"' in line]
    final_eval = eval_rows[-1]
    summary = {
        **asdict(trial),
        "dataset": dataset_label(args),
        "train_bytes": int(np.load(train_path, mmap_mode="r").shape[0]),
        "val_bytes": int(np.load(val_path, mmap_mode="r").shape[0]),
        "param_count": param_count,
        "final_train_loss": last_loss,
        "final_val_loss": final_eval["val_loss"],
        "best_val_loss": min(row["val_loss"] for row in eval_rows),
        "final_val_acc": final_eval["val_acc"],
        "best_val_acc": max(row["val_acc"] for row in eval_rows),
        "mean_step_time_ms": float(np.mean(step_times)) * 1000.0,
        "median_step_time_ms": float(np.median(step_times)) * 1000.0,
        "tokens_per_sec": args.batch_size * args.block_size / max(float(np.mean(step_times)), 1e-9),
        "wall_time_sec": time.perf_counter() - start,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def hpo_trials(args: argparse.Namespace) -> list[TrialConfig]:
    if args.preset == "smoke":
        return [
            TrialConfig("smoke_adamw", "adamw", 5e-4, steps=4, eval_every=2),
            TrialConfig("smoke_anchor", "anchormuon", 1e-3, steps=4, eval_every=2),
        ]

    trials: list[TrialConfig] = []
    if args.preset == "fineweb_long":
        trials = [
            TrialConfig(
                "fineweb_anchor_lr0.0015_rg0.55_soda0.01_pb0.9_nb0.93_flr0.5_atan2",
                "anchormuon",
                0.0015,
                row_gamma=0.55,
                pmuoneq_beta=0.90,
                normuon_beta2=0.93,
                fallback_lr_mult=0.5,
                fallback_mode="atan2",
                soda_lambda_scale=0.01,
                steps=args.hpo_steps,
            ),
            TrialConfig("fineweb_muon_lr0.0012", "muon", 0.0012, steps=args.hpo_steps),
            TrialConfig("fineweb_adamatan2_lr0.0003", "adamatan2", 0.0003, steps=args.hpo_steps),
            TrialConfig("fineweb_adamw_lr0.0003", "adamw", 0.0003, steps=args.hpo_steps),
        ]
        eval_every = max(1, args.hpo_steps // max(args.eval_bins, 1))
        warmup = min(args.warmup_steps, max(1, args.hpo_steps // 4))
        return [replace(trial, eval_every=eval_every, warmup_steps=warmup) for trial in trials]

    if args.preset == "fineweb_anchor_hpo":
        by_name: dict[str, TrialConfig] = {}

        def add(trial: TrialConfig) -> None:
            by_name.setdefault(trial.name, trial)

        # FineWeb-specific reference baselines around the previous long-run
        # winners. This keeps the AnchorMuon HPO honest without spending most
        # trials on Adam variants.
        for lr in (2e-4, 3e-4, 4e-4, 5e-4):
            add(TrialConfig(f"fineweb_adamw_lr{lr:g}", "adamw", lr, steps=args.hpo_steps))
            add(TrialConfig(f"fineweb_adamatan2_lr{lr:g}", "adamatan2", lr, steps=args.hpo_steps))
        for lr in (9e-4, 1.0e-3, 1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3, 1.5e-3):
            add(TrialConfig(f"fineweb_muon_lr{lr:g}", "muon", lr, steps=args.hpo_steps))

        # Main AnchorMuon matrix-path grid. The prior fixed-settings run lost
        # to Muon with lr=0.0015, row_gamma=0.55, soda=0.01, so this focuses
        # around lower/lateral LR, wider row-gamma, and much weaker SODA.
        for lr in (0.0010, 0.0012, 0.00135, 0.0015, 0.00165, 0.0018):
            for row_gamma in (0.25, 0.45, 0.55, 0.65):
                for soda_lambda_scale in (0.001, 0.003, 0.01, 0.03):
                    add(
                        TrialConfig(
                            f"fineweb_anchor_lr{lr:g}_rg{row_gamma:g}_soda{soda_lambda_scale:g}_pb0.9_nb0.93_flr0.5_atan2",
                            "anchormuon",
                            lr,
                            row_gamma=row_gamma,
                            pmuoneq_beta=0.90,
                            normuon_beta2=0.93,
                            fallback_lr_mult=0.5,
                            fallback_mode="atan2",
                            soda_lambda_scale=soda_lambda_scale,
                            steps=args.hpo_steps,
                        )
                    )

        # Check whether PMuonEq row scaling is helping at all on FineWeb.
        for lr in (0.0011, 0.0012, 0.00135, 0.0015):
            for soda_lambda_scale in (0.001, 0.003, 0.01):
                add(
                    TrialConfig(
                        f"fineweb_anchor_lr{lr:g}_rg0_soda{soda_lambda_scale:g}_pb0.9_nb0.93_flr0.5_atan2",
                        "anchormuon",
                        lr,
                        row_gamma=0.0,
                        pmuoneq_beta=0.90,
                        normuon_beta2=0.93,
                        fallback_lr_mult=0.5,
                        fallback_mode="atan2",
                        soda_lambda_scale=soda_lambda_scale,
                        steps=args.hpo_steps,
                    )
                )

        # Fallback parameters are a small fraction of the model but can move LM
        # loss through embeddings/head/scales. Sweep them near the strongest
        # matrix settings instead of assuming the WikiText fallback ratio.
        for lr in (0.0012, 0.00135, 0.0015):
            for row_gamma in (0.45, 0.55):
                for fallback_lr_mult in (0.25, 0.5, 0.75, 1.0):
                    for fallback_mode in ("atan2", "rms", "adamc"):
                        add(
                            TrialConfig(
                                f"fineweb_anchor_lr{lr:g}_rg{row_gamma:g}_soda0.003_pb0.9_nb0.93_flr{fallback_lr_mult:g}_{fallback_mode}",
                                "anchormuon",
                                lr,
                                row_gamma=row_gamma,
                                pmuoneq_beta=0.90,
                                normuon_beta2=0.93,
                                fallback_lr_mult=fallback_lr_mult,
                                fallback_mode=fallback_mode,
                                soda_lambda_scale=0.003,
                                steps=args.hpo_steps,
                            )
                        )

        # Matrix-state beta sweep around likely FineWeb winners.
        for lr in (0.0012, 0.00135, 0.0015):
            for row_gamma in (0.45, 0.55):
                for pmuoneq_beta in (0.85, 0.90, 0.95, 0.98):
                    for normuon_beta2 in (0.90, 0.93, 0.95):
                        add(
                            TrialConfig(
                                f"fineweb_anchor_lr{lr:g}_rg{row_gamma:g}_soda0.003_pb{pmuoneq_beta:g}_nb{normuon_beta2:g}_flr0.5_atan2",
                                "anchormuon",
                                lr,
                                row_gamma=row_gamma,
                                pmuoneq_beta=pmuoneq_beta,
                                normuon_beta2=normuon_beta2,
                                fallback_lr_mult=0.5,
                                fallback_mode="atan2",
                                soda_lambda_scale=0.003,
                                steps=args.hpo_steps,
                            )
                        )

        trials = list(by_name.values())
        if args.max_hpo_trials:
            trials = trials[: args.max_hpo_trials]
        eval_every = max(1, args.hpo_steps // max(args.eval_bins, 1))
        warmup = min(args.warmup_steps, max(1, args.hpo_steps // 4))
        return [replace(trial, eval_every=eval_every, warmup_steps=warmup) for trial in trials]

    if args.preset == "anchor_harder":
        by_name: dict[str, TrialConfig] = {}

        def add(trial: TrialConfig) -> None:
            by_name.setdefault(trial.name, trial)

        # Refined baselines around the previous winners. Adam variants won at
        # the low edge before; Muon was close to AnchorMuon, so give it a
        # denser local LR search.
        for lr in (1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 8e-4, 1e-3):
            add(TrialConfig(f"adamw_lr{lr:g}", "adamw", lr, steps=args.hpo_steps))
            add(TrialConfig(f"adamatan2_lr{lr:g}", "adamatan2", lr, steps=args.hpo_steps))
        for lr in (1e-3, 1.2e-3, 1.4e-3, 1.5e-3, 1.6e-3, 1.75e-3, 2e-3, 2.25e-3):
            add(TrialConfig(f"muon_lr{lr:g}", "muon", lr, steps=args.hpo_steps))

        # Dense local AnchorMuon grid around the corrected WikiText winner:
        # lr=0.00175, row_gamma=0.45, soda=0.1, fallback=AdamAtan2@0.5x.
        for lr in (0.0015, 0.00165, 0.00175, 0.0019, 0.00205, 0.0022, 0.00235):
            for row_gamma in (0.35, 0.45, 0.55):
                for soda_lambda_scale in (0.01, 0.03, 0.07, 0.1, 0.2):
                    add(
                        TrialConfig(
                            f"anchor_hard_lr{lr:g}_rg{row_gamma:g}_soda{soda_lambda_scale:g}_pb0.9_nb0.93_flr0.5_atan2",
                            "anchormuon",
                            lr,
                            row_gamma=row_gamma,
                            pmuoneq_beta=0.90,
                            normuon_beta2=0.93,
                            fallback_lr_mult=0.5,
                            fallback_mode="atan2",
                            soda_lambda_scale=soda_lambda_scale,
                            steps=args.hpo_steps,
                        )
                    )

        # Check row-gamma extremes only near the best LR/SODA region.
        for lr in (0.00165, 0.00175, 0.0019):
            for row_gamma in (0.25, 0.65):
                for soda_lambda_scale in (0.03, 0.07, 0.1):
                    add(
                        TrialConfig(
                            f"anchor_hard_lr{lr:g}_rg{row_gamma:g}_soda{soda_lambda_scale:g}_pb0.9_nb0.93_flr0.5_atan2",
                            "anchormuon",
                            lr,
                            row_gamma=row_gamma,
                            pmuoneq_beta=0.90,
                            normuon_beta2=0.93,
                            fallback_lr_mult=0.5,
                            fallback_mode="atan2",
                            soda_lambda_scale=soda_lambda_scale,
                            steps=args.hpo_steps,
                        )
                    )

        # Fallback-path sweep at the matrix settings that looked strongest.
        for lr in (0.00165, 0.00175, 0.0019):
            for fallback_lr_mult in (0.25, 0.5, 0.75, 1.0):
                for fallback_mode in ("atan2", "rms", "adamc"):
                    add(
                        TrialConfig(
                            f"anchor_hard_lr{lr:g}_rg0.45_soda0.1_pb0.9_nb0.93_flr{fallback_lr_mult:g}_{fallback_mode}",
                            "anchormuon",
                            lr,
                            row_gamma=0.45,
                            pmuoneq_beta=0.90,
                            normuon_beta2=0.93,
                            fallback_lr_mult=fallback_lr_mult,
                            fallback_mode=fallback_mode,
                            soda_lambda_scale=0.1,
                            steps=args.hpo_steps,
                        )
                    )

        # Matrix-path beta sweep around the same local optimum.
        for lr in (0.00165, 0.00175, 0.0019):
            for pmuoneq_beta in (0.85, 0.90, 0.95, 0.98):
                for normuon_beta2 in (0.90, 0.93, 0.95):
                    add(
                        TrialConfig(
                            f"anchor_hard_lr{lr:g}_rg0.45_soda0.1_pb{pmuoneq_beta:g}_nb{normuon_beta2:g}_flr0.5_atan2",
                            "anchormuon",
                            lr,
                            row_gamma=0.45,
                            pmuoneq_beta=pmuoneq_beta,
                            normuon_beta2=normuon_beta2,
                            fallback_lr_mult=0.5,
                            fallback_mode="atan2",
                            soda_lambda_scale=0.1,
                            steps=args.hpo_steps,
                        )
                    )

        trials = list(by_name.values())
        if args.max_hpo_trials:
            trials = trials[: args.max_hpo_trials]
        eval_every = max(1, args.hpo_steps // max(args.eval_bins, 1))
        warmup = min(args.warmup_steps, max(1, args.hpo_steps // 4))
        return [replace(trial, eval_every=eval_every, warmup_steps=warmup) for trial in trials]

    # Tuned baselines, kept modest because the user asked to spend extra effort
    # on AnchorMuon rather than only retesting Adam.
    for lr in (3e-4, 5e-4, 8e-4, 1e-3):
        trials.append(TrialConfig(f"adamw_lr{lr:g}", "adamw", lr, steps=args.hpo_steps))
        trials.append(TrialConfig(f"adamatan2_lr{lr:g}", "adamatan2", lr, steps=args.hpo_steps))
    for lr in (5e-4, 1e-3, 1.5e-3, 2e-3):
        trials.append(TrialConfig(f"muon_lr{lr:g}", "muon", lr, steps=args.hpo_steps))

    # Broader AnchorMuon search over the knobs exposed by root optimizer.py.
    for lr in (4e-4, 6e-4, 8e-4, 1e-3, 1.25e-3, 1.5e-3):
        for row_gamma in (0.0, 0.15, 0.25, 0.35):
            trials.append(
                TrialConfig(
                    f"anchor_lr{lr:g}_rg{row_gamma:g}_pb0.9_nb0.93_flr0.5_atan2",
                    "anchormuon",
                    lr,
                    row_gamma=row_gamma,
                    pmuoneq_beta=0.90,
                    normuon_beta2=0.93,
                    fallback_lr_mult=0.5,
                    fallback_mode="atan2",
                    steps=args.hpo_steps,
                )
            )
    for lr in (6e-4, 8e-4, 1e-3):
        for pmuoneq_beta in (0.85, 0.95):
            for normuon_beta2 in (0.90, 0.95):
                trials.append(
                    TrialConfig(
                        f"anchor_lr{lr:g}_rg0.25_pb{pmuoneq_beta:g}_nb{normuon_beta2:g}_flr0.5_atan2",
                        "anchormuon",
                        lr,
                        row_gamma=0.25,
                        pmuoneq_beta=pmuoneq_beta,
                        normuon_beta2=normuon_beta2,
                        fallback_lr_mult=0.5,
                        fallback_mode="atan2",
                        steps=args.hpo_steps,
                    )
                )
    for lr in (0.00175, 0.002, 0.0025, 0.003):
        for row_gamma in (0.25, 0.35, 0.45):
            for soda_lambda_scale in (0.1, 0.3, 1.0):
                trials.append(
                    TrialConfig(
                        f"anchor_lr{lr:g}_rg{row_gamma:g}_soda{soda_lambda_scale:g}_pb0.9_nb0.93_flr0.5_atan2",
                        "anchormuon",
                        lr,
                        row_gamma=row_gamma,
                        pmuoneq_beta=0.90,
                        normuon_beta2=0.93,
                        fallback_lr_mult=0.5,
                        fallback_mode="atan2",
                        soda_lambda_scale=soda_lambda_scale,
                        steps=args.hpo_steps,
                    )
                )
    for lr in (6e-4, 8e-4, 1e-3):
        for fallback_lr_mult in (0.25, 1.0):
            for fallback_mode in ("atan2", "rms"):
                trials.append(
                    TrialConfig(
                        f"anchor_lr{lr:g}_rg0.25_pb0.9_nb0.93_flr{fallback_lr_mult:g}_{fallback_mode}",
                        "anchormuon",
                        lr,
                        row_gamma=0.25,
                        pmuoneq_beta=0.90,
                        normuon_beta2=0.93,
                        fallback_lr_mult=fallback_lr_mult,
                        fallback_mode=fallback_mode,
                        steps=args.hpo_steps,
                    )
                )

    if args.max_hpo_trials:
        trials = trials[: args.max_hpo_trials]
    eval_every = max(1, args.hpo_steps // max(args.eval_bins, 1))
    warmup = min(args.warmup_steps, max(1, args.hpo_steps // 4))
    return [replace(trial, eval_every=eval_every, warmup_steps=warmup) for trial in trials]


def final_trials_from_hpo(args: argparse.Namespace, hpo_rows: list[dict[str, Any]]) -> list[TrialConfig]:
    best: dict[str, dict[str, Any]] = {}
    for row in hpo_rows:
        family = str(row["family"])
        if family not in best or float(row["best_val_loss"]) < float(best[family]["best_val_loss"]):
            best[family] = row
    eval_every = max(1, args.final_steps // max(args.eval_bins, 1))
    warmup = min(args.warmup_steps, max(1, args.final_steps // 4))
    trials: list[TrialConfig] = []
    for family in ("anchormuon", "adamw", "adamatan2", "muon"):
        if family not in best:
            continue
        row = best[family]
        trials.append(
            TrialConfig(
                name=f"final_{row['name']}",
                family=family,
                lr=float(row["lr"]),
                weight_decay=float(row.get("weight_decay", 0.05)),
                row_gamma=float(row.get("row_gamma", 0.25)),
                pmuoneq_beta=float(row.get("pmuoneq_beta", 0.90)),
                normuon_beta2=float(row.get("normuon_beta2", 0.93)),
                fallback_lr_mult=float(row.get("fallback_lr_mult", 0.5)),
                fallback_mode=str(row.get("fallback_mode", "atan2")),
                soda_lambda_scale=float(row.get("soda_lambda_scale", 1.0)),
                soda_lambda_power=float(row.get("soda_lambda_power", 1.0)),
                seed=args.seed,
                steps=args.final_steps,
                eval_every=eval_every,
                warmup_steps=warmup,
            )
        )
    return trials


def final_trials_from_preset(args: argparse.Namespace) -> list[TrialConfig]:
    eval_every = max(1, args.final_steps // max(args.eval_bins, 1))
    warmup = min(args.warmup_steps, max(1, args.final_steps // 4))
    trials: list[TrialConfig] = []
    for trial in hpo_trials(args):
        trials.append(
            replace(
                trial,
                name=f"final_{trial.name}",
                steps=args.final_steps,
                eval_every=eval_every,
                warmup_steps=warmup,
            )
        )
    return trials


def launch_trials(args: argparse.Namespace, trials: list[TrialConfig]) -> list[dict[str, Any]]:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("CUDA GPU is required for this benchmark")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = list(trials)
    running: list[tuple[subprocess.Popen[str], Path, int, str]] = []
    available_gpus = list(range(gpu_count))
    summaries: list[dict[str, Any]] = []
    while pending or running:
        still: list[tuple[subprocess.Popen[str], Path, int, str]] = []
        for proc, trial_file, gpu, name in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, trial_file, gpu, name))
                continue
            available_gpus.append(gpu)
            if rc != 0:
                raise RuntimeError(f"trial {name} failed on gpu {gpu}; see {args.output_dir / (name + '.worker.log')}")
            trial = TrialConfig(**json.loads(trial_file.read_text()))
            phase = "final" if trial.name.startswith("final_") else "hpo"
            summaries.append(json.loads((args.output_dir / phase / trial.name / "summary.json").read_text()))
            print(f"finished gpu={gpu} {name}", flush=True)
        running = still

        available_gpus = sorted(set(available_gpus))
        while pending and available_gpus:
            gpu = available_gpus.pop(0)
            trial = pending.pop(0)
            trial_file = args.output_dir / f"{trial.name}.trial.json"
            trial_file.write_text(json.dumps(asdict(trial), indent=2) + "\n")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--trial-json",
                str(trial_file),
                "--output-dir",
                str(args.output_dir),
                "--dataset-source",
                str(args.dataset_source),
                "--wiki-config",
                str(args.wiki_config),
                "--hf-dataset",
                str(args.hf_dataset),
                "--hf-config",
                str(args.hf_config),
                "--hf-split",
                str(args.hf_split),
                "--hf-val-split",
                str(args.hf_val_split),
                "--hf-text-field",
                str(args.hf_text_field),
                "--hf-shuffle-buffer",
                str(args.hf_shuffle_buffer),
                "--cache-dir",
                str(args.cache_dir),
                "--max-train-bytes",
                str(args.max_train_bytes),
                "--max-val-bytes",
                str(args.max_val_bytes),
                "--block-size",
                str(args.block_size),
                "--n-layer",
                str(args.n_layer),
                "--n-head",
                str(args.n_head),
                "--n-embd",
                str(args.n_embd),
                "--dropout",
                str(args.dropout),
                "--batch-size",
                str(args.batch_size),
                "--hpo-steps",
                str(args.hpo_steps),
                "--final-steps",
                str(args.final_steps),
                "--eval-bins",
                str(args.eval_bins),
                "--eval-batches",
                str(args.eval_batches),
                "--warmup-steps",
                str(args.warmup_steps),
                "--log-every",
                str(args.log_every),
            ]
            log_path = args.output_dir / f"{trial.name}.worker.log"
            log = log_path.open("w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            running.append((proc, trial_file, gpu, trial.name))
            print(f"launched gpu={gpu} {trial.name}", flush=True)
        if running:
            time.sleep(1.0)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_eval_curve(trial_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (trial_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("phase") == "eval":
            rows.append(row)
    return rows


def make_plots(args: argparse.Namespace, final_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = {"anchormuon": "AnchorMuon", "adamw": "AdamW", "adamatan2": "AdamW-Atan2", "muon": "Muon"}
    for metric, ylabel, out_name in [
        ("val_loss", "Validation loss", "val_loss_curve.png"),
        ("val_acc", "Byte accuracy", "val_acc_curve.png"),
        ("train_loss", "Training loss", "train_loss_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for row in final_rows:
            curve = load_eval_curve(args.output_dir / "final" / str(row["name"]))
            xs = [r["step"] for r in curve]
            ys = [r["train_loss"] if metric == "train_loss" else r[metric] for r in curve]
            ax.plot(xs, ys, marker="o", label=labels.get(str(row["family"]), str(row["family"])))
        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / out_name, dpi=180)
        plt.close(fig)

    sorted_rows = sorted(final_rows, key=lambda r: float(r["mean_step_time_ms"]))
    for metric, ylabel, out_name, scale in [
        ("mean_step_time_ms", "Mean step time (ms)", "step_time_ms_bar.png", 1.0),
        ("tokens_per_sec", "Throughput (k tokens/s)", "tokens_per_sec_bar.png", 1e-3),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar([labels.get(str(r["family"]), str(r["family"])) for r in sorted_rows], [float(r[metric]) * scale for r in sorted_rows])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / out_name, dpi=180)
        plt.close(fig)


def write_summary(args: argparse.Namespace, hpo_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> None:
    labels = {"anchormuon": "AnchorMuon", "adamw": "AdamW", "adamatan2": "AdamW-Atan2", "muon": "Muon"}
    final_sorted = sorted(final_rows, key=lambda r: float(r["final_val_loss"]))
    lines = [
        "# Byte-Level 50M LLM Optimizer Comparison",
        "",
        "This benchmark uses real text encoded as UTF-8 bytes. It is byte-level rather than BPE-tokenized, so the numbers should not be compared to standard word/BPE perplexities. It is still a real text next-byte language-model optimizer comparison.",
        "",
        "## Setup",
        "",
        f"- Dataset: `{dataset_label(args)}`",
        f"- Train bytes cached: {int(final_rows[0]['train_bytes']) if final_rows else args.max_train_bytes:,}",
        f"- Validation bytes cached: {int(final_rows[0]['val_bytes']) if final_rows else args.max_val_bytes:,}",
        f"- Model: decoder-only GPT, layers={args.n_layer}, width={args.n_embd}, heads={args.n_head}, context={args.block_size}, byte vocab=256",
        f"- Trainable parameters: {int(final_rows[0]['param_count']) if final_rows else 'n/a'}",
        f"- Batch: {args.batch_size} sequences x {args.block_size} bytes",
        f"- HPO: {'skipped; fixed preset configs replayed directly' if args.final_only else f'{args.hpo_steps} steps per candidate, selected by best validation loss'}",
        f"- Final replay: {args.final_steps} steps per selected optimizer",
        f"- Validation estimate: {args.eval_batches} random batches per evaluation point",
        f"- GPUs: {torch.cuda.device_count()} visible, one trial per GPU",
        "",
        "## Final Results",
        "",
        "| Rank | Optimizer | Selected config | Final val loss | Best val loss | Final byte acc | Step time | Throughput |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(final_sorted, start=1):
        config = f"lr={float(row['lr']):g}, wd={float(row.get('weight_decay', 0.0)):g}"
        if row["family"] == "anchormuon":
            config += (
                f", row_gamma={float(row['row_gamma']):g}, pmuon_beta={float(row['pmuoneq_beta']):g}, "
                f"normuon_beta2={float(row['normuon_beta2']):g}, fallback={row['fallback_mode']}@{float(row['fallback_lr_mult']):g}x, "
                f"soda={float(row.get('soda_lambda_scale', 1.0)):g}"
            )
        lines.append(
            f"| {rank} | {labels.get(str(row['family']), row['family'])} | {config} | "
            f"{float(row['final_val_loss']):.4f} | {float(row['best_val_loss']):.4f} | "
            f"{100.0 * float(row['final_val_acc']):.2f}% | {float(row['mean_step_time_ms']):.2f} ms | "
            f"{float(row['tokens_per_sec']) / 1000.0:.1f}k byte/s |"
        )
    lines += [
        "",
        "## Plots",
        "",
        "![Validation loss](plots/val_loss_curve.png)",
        "",
        "![Validation accuracy](plots/val_acc_curve.png)",
        "",
        "![Training loss](plots/train_loss_curve.png)",
        "",
        "![Step time](plots/step_time_ms_bar.png)",
        "",
        "![Token throughput](plots/tokens_per_sec_bar.png)",
        "",
    ]
    if hpo_rows:
        lines += [
            "## HPO Candidates",
            "",
            "| Family | Trial | LR | Best val loss | Final val loss | Step time |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in sorted(hpo_rows, key=lambda r: (str(r["family"]), float(r["best_val_loss"]))):
            lines.append(
                f"| {row['family']} | `{row['name']}` | {float(row['lr']):g} | "
                f"{float(row['best_val_loss']):.4f} | {float(row['final_val_loss']):.4f} | "
                f"{float(row['mean_step_time_ms']):.2f} ms |"
            )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if args.worker:
        trial = TrialConfig(**json.loads(args.trial_json.read_text()))
        run_trial(args, trial)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required; refusing to fall back to CPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = prepare_text_cache(args)
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "train_cache": str(train_path),
        "val_cache": str(val_path),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    if args.final_only:
        hpo_rows: list[dict[str, Any]] = []
        final_rows = launch_trials(args, final_trials_from_preset(args))
    else:
        hpo_rows = launch_trials(args, hpo_trials(args))
        write_csv(args.output_dir / "hpo_summary.csv", hpo_rows)
        final_rows = launch_trials(args, final_trials_from_hpo(args, hpo_rows))
    write_csv(args.output_dir / "final_summary.csv", final_rows)
    make_plots(args, final_rows)
    write_summary(args, hpo_rows, final_rows)


if __name__ == "__main__":
    main()
