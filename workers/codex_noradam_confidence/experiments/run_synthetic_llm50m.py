#!/usr/bin/env python3
"""Run a bounded 50M-parameter synthetic language-model optimizer comparison.

This is a local, download-free benchmark. It trains a decoder-only transformer
on a deterministic repeated-motif token stream and validates on a held-out
stream sampled from the same motif bank. The benchmark is not a substitute for
OpenWebText/FineWeb pretraining, but it exercises dense LLM-style embeddings,
attention, MLP matrices, output softmax, next-token loss, accuracy, and
optimizer step cost.
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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optimizer as root_optimizer


@dataclass
class TrialConfig:
    name: str
    family: str
    lr: float
    weight_decay: float = 0.05
    row_gamma: float = 0.35
    fallback_lr_mult: float = 0.5
    fallback_mode: str = "atan2"
    seed: int = 123
    steps: int = 200
    eval_every: int = 50
    warmup_steps: int = 20
    final_lr_scale: float = 0.1
    wsd_decay_frac: float = 0.2


class SyntheticMotifStream:
    def __init__(
        self,
        *,
        vocab_size: int,
        motif_count: int,
        motif_len: int,
        total_tokens: int,
        motif_seed: int,
        seed: int,
        device: torch.device,
    ) -> None:
        motif_rng = np.random.default_rng(motif_seed)
        rng = np.random.default_rng(seed)
        motifs = motif_rng.integers(4, vocab_size, size=(motif_count, motif_len), dtype=np.int64)
        motifs[:, 0] = motif_rng.integers(4, min(vocab_size, 512), size=motif_count, dtype=np.int64)
        ranks = np.arange(1, motif_count + 1, dtype=np.float64)
        probs = 1.0 / np.power(ranks, 1.07)
        probs /= probs.sum()
        num_motifs = int(math.ceil(total_tokens / motif_len)) + 2
        choices = rng.choice(motif_count, size=num_motifs, p=probs)
        stream = motifs[choices].reshape(-1)[: total_tokens + 1].copy()
        # Add a small deterministic position code so the model cannot solve the
        # task by unigram frequency alone.
        pos = np.arange(stream.size, dtype=np.int64)
        stream = (stream + (pos % motif_len)) % vocab_size
        stream = np.maximum(stream, 4)
        self.tokens = torch.from_numpy(stream.astype(np.int64)).to(device)

    def batch(self, *, batch_size: int, block_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - block_size - 1
        idx = torch.randint(0, max_start, (batch_size,), device=self.tokens.device, generator=generator)
        offsets = torch.arange(block_size + 1, device=self.tokens.device)
        chunk = self.tokens[idx[:, None] + offsets[None, :]]
        return chunk[:, :-1], chunk[:, 1:]


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = idx.shape
        pos = torch.arange(seqlen, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class AdamAtan2(torch.optim.Optimizer):
    def __init__(
        self,
        params: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1.0 - lr * wd)
                g = p.grad.detach().to(torch.float32)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                step = int(state["step"])
                mhat = exp_avg / max(1.0 - beta1**step, 1e-16)
                vhat = exp_avg_sq / max(1.0 - beta2**step, 1e-16)
                update = torch.atan2(mhat, vhat.sqrt().add_(eps))
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss


class PlainMuon(torch.optim.Optimizer):
    def __init__(
        self,
        matrix_params: list[nn.Parameter],
        fallback_groups: list[dict[str, Any]],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
    ) -> None:
        groups: list[dict[str, Any]] = []
        if matrix_params:
            groups.append(
                {
                    "params": matrix_params,
                    "use_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                }
            )
        for group in fallback_groups:
            copied = dict(group)
            copied["use_muon"] = False
            copied["lr"] = lr
            copied.setdefault("betas", (0.9, 0.95))
            copied.setdefault("eps", 1e-8)
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
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        wd = float(group.get("weight_decay", 0.0))
        beta = float(group["momentum"])
        for p in group["params"]:
            if p.grad is None:
                continue
            if wd:
                p.mul_(1.0 - lr * wd)
            g = p.grad.detach().to(torch.float32)
            state = self.state[p]
            momentum = state.get("momentum_buffer")
            if momentum is None or momentum.shape != p.shape:
                momentum = state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            momentum.lerp_(g, 1.0 - beta)
            source = torch.lerp(g, momentum, beta)
            matrix = root_optimizer._matrix_view(source)
            update = self._orthogonalizer(matrix)
            update = update * (0.2 * math.sqrt(max(update.shape[-2], update.shape[-1])))
            p.add_(update.reshape_as(p).to(p.dtype), alpha=-lr)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group.get("weight_decay", 0.0))
        for p in group["params"]:
            if p.grad is None:
                continue
            if wd:
                p.mul_(1.0 - lr * wd)
            g = p.grad.detach().to(torch.float32)
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            step = int(state["step"])
            denom = (exp_avg_sq / max(1.0 - beta2**step, 1e-16)).sqrt().add_(eps)
            update = (exp_avg / max(1.0 - beta1**step, 1e-16)) / denom
            p.add_(update.to(p.dtype), alpha=-lr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("workers/codex_noradam_confidence/results/synthetic_llm50m_20260529"))
    parser.add_argument("--preset", choices=["smoke", "main"], default="main")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=10)
    parser.add_argument("--n-embd", type=int, default=640)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hpo-steps", type=int, default=180)
    parser.add_argument("--final-steps", type=int, default=600)
    parser.add_argument("--eval-bins", type=int, default=6)
    parser.add_argument("--train-tokens", type=int, default=8_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--motif-count", type=int, default=2048)
    parser.add_argument("--motif-len", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-hpo-trials", type=int, default=0)
    parser.add_argument("--final-only", type=Path, default=None)
    return parser.parse_args()


def make_decay_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for _name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def split_muon_groups(model: nn.Module, weight_decay: float) -> tuple[list[nn.Parameter], list[dict[str, Any]]]:
    matrix: list[nn.Parameter] = []
    fallback_decay: list[nn.Parameter] = []
    fallback_nodecay: list[nn.Parameter] = []
    seen: set[int] = set()
    for _name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.is_floating_point() and len([dim for dim in p.shape if dim > 1]) >= 2:
            matrix.append(p)
        elif p.ndim >= 2:
            fallback_decay.append(p)
        else:
            fallback_nodecay.append(p)
    fallback_groups: list[dict[str, Any]] = []
    if fallback_decay:
        fallback_groups.append({"params": fallback_decay, "weight_decay": weight_decay})
    if fallback_nodecay:
        fallback_groups.append({"params": fallback_nodecay, "weight_decay": 0.0})
    return matrix, fallback_groups


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
            normuon_beta2=0.93,
        )
    raise ValueError(f"unknown optimizer family {trial.family}")


def lr_scale(step: int, total_steps: int, warmup_steps: int, final_lr_scale: float, decay_frac: float) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    decay_steps = max(1, int(total_steps * decay_frac))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return 1.0
    progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))
    return 1.0 + progress * (final_lr_scale - 1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    stream: SyntheticMotifStream,
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
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().detach())
        total += int(y.numel())
    model.train()
    return float(np.mean(losses)), correct / max(total, 1)


def run_trial(args: argparse.Namespace, trial: TrialConfig) -> dict[str, Any]:
    torch.manual_seed(trial.seed)
    np.random.seed(trial.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    model = TinyGPT(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, trial)
    train_stream = SyntheticMotifStream(
        vocab_size=args.vocab_size,
        motif_count=args.motif_count,
        motif_len=args.motif_len,
        total_tokens=args.train_tokens,
        motif_seed=777,
        seed=777,
        device=device,
    )
    val_stream = SyntheticMotifStream(
        vocab_size=args.vocab_size,
        motif_count=args.motif_count,
        motif_len=args.motif_len,
        total_tokens=args.val_tokens,
        motif_seed=777,
        seed=888,
        device=device,
    )
    train_gen = torch.Generator(device=device).manual_seed(trial.seed + 10)
    val_gen = torch.Generator(device=device).manual_seed(trial.seed + 20)

    trial_dir = args.output_dir / ("hpo" if trial.steps < args.final_steps else "final") / trial.name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(json.dumps(asdict(trial), indent=2) + "\n")

    metrics_path = trial_dir / "metrics.jsonl"
    eval_steps = set(range(trial.eval_every, trial.steps + 1, trial.eval_every))
    eval_steps.add(trial.steps)
    step_times: list[float] = []
    start = time.perf_counter()
    last_loss = float("nan")
    with metrics_path.open("w") as f:
        for step in range(1, trial.steps + 1):
            scale = lr_scale(step, trial.steps, trial.warmup_steps, trial.final_lr_scale, trial.wsd_decay_frac)
            for group in optimizer.param_groups:
                group["lr"] = trial.lr * scale
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
                row = {
                    "step": step,
                    "phase": "train",
                    "train_loss": last_loss,
                    "lr": trial.lr * scale,
                    "step_time_ms": elapsed * 1000.0,
                    "tokens_per_sec": args.batch_size * args.block_size / max(elapsed, 1e-9),
                }
                f.write(json.dumps(row) + "\n")
            if step in eval_steps:
                val_loss, val_acc = evaluate(
                    model,
                    val_stream,
                    batch_size=args.batch_size,
                    block_size=args.block_size,
                    batches=8,
                    generator=val_gen,
                )
                row = {
                    "step": step,
                    "phase": "eval",
                    "train_loss": last_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": trial.lr * scale,
                    "mean_step_time_ms": float(np.mean(step_times[-max(1, min(len(step_times), trial.eval_every)) :])) * 1000.0,
                    "tokens_per_sec": args.batch_size * args.block_size / max(float(np.mean(step_times[-max(1, min(len(step_times), trial.eval_every)) :])), 1e-9),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()

    total_time = time.perf_counter() - start
    eval_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if '"phase": "eval"' in line]
    final_eval = eval_rows[-1]
    summary = {
        **asdict(trial),
        "param_count": param_count,
        "final_train_loss": last_loss,
        "final_val_loss": final_eval["val_loss"],
        "best_val_loss": min(row["val_loss"] for row in eval_rows),
        "final_val_acc": final_eval["val_acc"],
        "best_val_acc": max(row["val_acc"] for row in eval_rows),
        "mean_step_time_ms": float(np.mean(step_times)) * 1000.0,
        "median_step_time_ms": float(np.median(step_times)) * 1000.0,
        "tokens_per_sec": args.batch_size * args.block_size / max(float(np.mean(step_times)), 1e-9),
        "wall_time_sec": total_time,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def hpo_trials(args: argparse.Namespace) -> list[TrialConfig]:
    if args.preset == "smoke":
        return [
            TrialConfig("smoke_adamw", "adamw", 1e-3, steps=4, eval_every=2),
            TrialConfig("smoke_anchor", "anchormuon", 4e-3, steps=4, eval_every=2),
        ]
    trials: list[TrialConfig] = []
    for lr in (5.0e-4, 1.0e-3, 1.5e-3, 2.0e-3):
        trials.append(TrialConfig(f"adamw_lr{lr:g}", "adamw", lr, steps=args.hpo_steps))
        trials.append(TrialConfig(f"adamatan2_lr{lr:g}", "adamatan2", lr, steps=args.hpo_steps))
    for lr in (1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3):
        trials.append(TrialConfig(f"muon_lr{lr:g}", "muon", lr, weight_decay=0.05, steps=args.hpo_steps))
    for lr in (1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3):
        for row_gamma in (0.25, 0.35):
            trials.append(
                TrialConfig(
                    f"anchormuon_lr{lr:g}_rg{row_gamma:g}_flr0.5",
                    "anchormuon",
                    lr,
                    row_gamma=row_gamma,
                    fallback_lr_mult=0.5,
                    steps=args.hpo_steps,
                )
            )
    if args.max_hpo_trials:
        trials = trials[: args.max_hpo_trials]
    eval_every = max(1, args.hpo_steps // max(args.eval_bins, 1))
    return [replace(trial, eval_every=eval_every, warmup_steps=min(args.warmup_steps, max(1, args.hpo_steps // 4))) for trial in trials]


def final_trials_from_hpo(args: argparse.Namespace, hpo_rows: list[dict[str, Any]]) -> list[TrialConfig]:
    best: dict[str, dict[str, Any]] = {}
    for row in hpo_rows:
        family = str(row["family"])
        if family not in best or float(row["best_val_loss"]) < float(best[family]["best_val_loss"]):
            best[family] = row
    trials: list[TrialConfig] = []
    eval_every = max(1, args.final_steps // max(args.eval_bins, 1))
    for family in ("anchormuon", "adamw", "adamatan2", "muon"):
        row = best[family]
        cfg = TrialConfig(
            name=f"final_{row['name']}",
            family=family,
            lr=float(row["lr"]),
            weight_decay=float(row.get("weight_decay", 0.05)),
            row_gamma=float(row.get("row_gamma", 0.35)),
            fallback_lr_mult=float(row.get("fallback_lr_mult", 0.5)),
            fallback_mode=str(row.get("fallback_mode", "atan2")),
            seed=args.seed,
            steps=args.final_steps,
            eval_every=eval_every,
            warmup_steps=min(args.warmup_steps, max(1, args.final_steps // 4)),
        )
        trials.append(cfg)
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
                raise RuntimeError(f"trial {name} on gpu {gpu} failed with exit code {rc}; see {args.output_dir / (name + '.worker.log')}")
            trial = TrialConfig(**json.loads(trial_file.read_text()))
            trial_dir = args.output_dir / ("hpo" if trial.steps < args.final_steps else "final") / trial.name
            summaries.append(json.loads((trial_dir / "summary.json").read_text()))
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
                "--vocab-size",
                str(args.vocab_size),
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
                "--train-tokens",
                str(args.train_tokens),
                "--val-tokens",
                str(args.val_tokens),
                "--motif-count",
                str(args.motif_count),
                "--motif-len",
                str(args.motif_len),
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

    final_dir = args.output_dir / "final"
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    label = {
        "anchormuon": "AnchorMuon",
        "adamw": "AdamW",
        "adamatan2": "AdamW-Atan2",
        "muon": "Muon",
    }
    for metric, ylabel, out_name in [
        ("val_loss", "Validation loss", "val_loss_curve.png"),
        ("val_acc", "Validation next-token accuracy", "val_acc_curve.png"),
        ("train_loss", "Training loss", "train_loss_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for row in final_rows:
            trial_dir = final_dir / str(row["name"])
            curve = load_eval_curve(trial_dir)
            xs = [r["step"] for r in curve]
            if metric == "train_loss":
                ys = [r["train_loss"] for r in curve]
            else:
                ys = [r[metric] for r in curve]
            ax.plot(xs, ys, marker="o", label=label.get(str(row["family"]), str(row["family"])))
        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / out_name, dpi=180)
        plt.close(fig)

    sorted_rows = sorted(final_rows, key=lambda r: float(r["mean_step_time_ms"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([label.get(str(r["family"]), str(r["family"])) for r in sorted_rows], [float(r["mean_step_time_ms"]) for r in sorted_rows])
    ax.set_ylabel("Mean step time (ms)")
    ax.set_title("Iteration Speed")
    fig.tight_layout()
    fig.savefig(plots_dir / "step_time_ms_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([label.get(str(r["family"]), str(r["family"])) for r in sorted_rows], [float(r["tokens_per_sec"]) / 1e3 for r in sorted_rows])
    ax.set_ylabel("Throughput (k tokens/s)")
    ax.set_title("Token Throughput")
    fig.tight_layout()
    fig.savefig(plots_dir / "tokens_per_sec_bar.png", dpi=180)
    plt.close(fig)


def write_summary(args: argparse.Namespace, hpo_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> None:
    final_sorted = sorted(final_rows, key=lambda r: float(r["final_val_loss"]))
    lines = [
        "# Synthetic 50M LLM Optimizer Comparison",
        "",
        "This is a download-free proxy benchmark: a decoder-only transformer trains on a deterministic repeated-motif token stream and validates on a held-out stream from the same motif bank. It measures next-token loss, next-token accuracy, and synchronized iteration speed. It is useful for optimizer smoke/proxy behavior, not a claim about natural-language pretraining.",
        "",
        "## Setup",
        "",
        f"- Model: decoder-only GPT, layers={args.n_layer}, width={args.n_embd}, heads={args.n_head}, context={args.block_size}, vocab={args.vocab_size}",
        f"- Trainable parameters: {int(final_rows[0]['param_count']) if final_rows else 'n/a'}",
        f"- Batch: {args.batch_size} sequences x {args.block_size} tokens",
        f"- HPO: {args.hpo_steps} steps per candidate, selected by best validation loss",
        f"- Final replay: {args.final_steps} steps per selected optimizer",
        f"- GPUs: {torch.cuda.device_count()} visible, one trial per GPU",
        "",
        "## Final Results",
        "",
        "| Rank | Optimizer | Selected config | Final val loss | Best val loss | Final val acc | Step time | Throughput |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    label = {
        "anchormuon": "AnchorMuon",
        "adamw": "AdamW",
        "adamatan2": "AdamW-Atan2",
        "muon": "Muon",
    }
    for rank, row in enumerate(final_sorted, start=1):
        config = f"lr={float(row['lr']):g}, wd={float(row.get('weight_decay', 0.0)):g}"
        if row["family"] == "anchormuon":
            config += f", row_gamma={float(row['row_gamma']):g}, fallback_lr_mult={float(row['fallback_lr_mult']):g}"
        lines.append(
            f"| {rank} | {label.get(str(row['family']), str(row['family']))} | {config} | "
            f"{float(row['final_val_loss']):.4f} | {float(row['best_val_loss']):.4f} | "
            f"{100.0 * float(row['final_val_acc']):.2f}% | {float(row['mean_step_time_ms']):.2f} ms | "
            f"{float(row['tokens_per_sec']) / 1000.0:.1f}k tok/s |"
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
    if args.worker:
        trial = TrialConfig(**json.loads(args.trial_json.read_text()))
        run_trial(args, trial)
        return

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required; refusing to fall back to CPU")
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    if args.final_only:
        hpo_rows = list(csv.DictReader(args.final_only.open()))
    else:
        hpo_rows = launch_trials(args, hpo_trials(args))
        write_csv(args.output_dir / "hpo_summary.csv", hpo_rows)
    if args.preset == "smoke":
        final_rows: list[dict[str, Any]] = []
    else:
        finals = final_trials_from_hpo(args, hpo_rows)
        final_rows = launch_trials(args, finals)
        write_csv(args.output_dir / "final_summary.csv", final_rows)
        make_plots(args, final_rows)
        write_summary(args, hpo_rows, final_rows)


if __name__ == "__main__":
    main()
