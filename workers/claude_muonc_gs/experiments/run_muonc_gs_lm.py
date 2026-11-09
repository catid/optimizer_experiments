#!/usr/bin/env python3
"""MuonC x MuonGS combination study on the repo's GPT-2-tokenized 50M LM proxy.

The model, data cache, batch sampling, evaluation and training loop mirror
``workers/codex_noradam_confidence/experiments/run_wikitext_llm50m.py`` as it
was run for the README's 8K-replay tables (``--eval-mode sequential
--eval-batches 16 --final-eval-batches 256``), so the ``std`` protocol here is
comparable with those tables.  Note that 256 sequential batches are 1,048,576
tokens over a 524,288-token validation cache: the "full" loss is two passes
with shifted block alignment, exactly as in the reference.  What this runner
adds:

* optimizer variants built from ``muonc.MuonC`` (projection on/off/partial,
  no / standard / corrected decay, fixed-norm reference, AnchorMuon stack),
* extra LR schedules (hnet-style cosine, MuonGS-paper linear-to-zero),
* a single-pass data regime (512M-token cache) next to the 8M-token cache,
* per-matrix-class weight-norm, radial-cosine and angular-step diagnostics,
* a multi-GPU launcher that packs several trials per GPU.

Usage:
    python run_muonc_gs_lm.py --stage s0_parity --out <bundle> --gpus 0-7 --jobs-per-gpu 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WORKER_DIR = HERE.parent
REPO_ROOT = WORKER_DIR.parents[1]
CODEX_EXPERIMENTS = REPO_ROOT / "workers" / "codex_noradam_confidence" / "experiments"
for import_root in (REPO_ROOT, CODEX_EXPERIMENTS, WORKER_DIR, HERE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_DATA_DIR = REPO_ROOT / "workers" / "codex_noradam_confidence" / "data" / "fineweb_gpt2_tokens"
VOCAB_SIZE = 50257
TRAIN_TOKENS = {"std": 8_000_000, "sp": 512_000_000}
VAL_TOKENS = 524_288
MATRIX_CLASSES = ("emb", "pos", "qkv", "attn_proj", "mlp_fc", "mlp_proj")


@dataclass
class Trial:
    name: str
    stage: str = ""
    # ---- protocol
    data: str = "std"  # "std": 8M-token cache (multi-pass); "sp": 512M-token cache (single-pass regime)
    steps: int = 8000
    seed: int = 123
    batch_size: int = 16
    block_size: int = 256
    schedule: str = "wsd"  # wsd | cosine | linear
    warmup_steps: int = 100
    final_lr_scale: float = 0.1
    decay_frac: float = 0.2  # wsd: trailing linear-decay fraction; linear: 1 - constant fraction
    # ---- optimizer
    opt: str = "muonc"  # muonc | repo_muon | repo_adamw
    lr: float = 1.8e-3
    wd: float = 0.05
    decay: str = "corrected"  # none | standard | corrected   (matrix path)
    gs: float = 1.0  # fraction of the radial component removed
    gs_mode: str = "both"
    norm_mode: str = "free"  # free | fixed
    base: str = "plain"  # "plain" or "+"-joined subset of {pmuoneq, normuon, soda}
    row_gamma: float = 0.45
    pmuoneq_beta: float = 0.90
    normuon_beta2: float = 0.93
    soda: float = 0.003
    fallback_mode: str = "adamc"  # "adamc" is the plain Adam direction, as in the repo's PlainMuon
    fallback_lr_mult: float = 1.0
    fallback_wd: float = 0.0  # decay for >=2-D tensors routed to the fallback (emb="fallback")
    fallback_decay: str = "standard"
    emb: str = "matrix"  # matrix | matrix_nogs | fallback
    unprojected_decay: str = ""  # decay law for _no_gs matrices; "" follows `decay`
    norm_target: float = 0.0  # kappa > 0: per-matrix corrected decay toward kappa*|W_0| (needs decay="none")
    init_resid_scale: float = 1.0  # multiply attn.proj / mlp.proj initial weights (depth-scaled init test)
    emulate_gs_theta0: float = 0.0  # > 0: matrix LR *= 1/sqrt(1 + theta0^2 * sum_{u<t} s_u^2)
    ns_dtype: str = ""  # "" = optimizer default (FP16 on CUDA); "bf16" as in the hnet trainer
    init_scale: float = 1.0  # multiply every matrix-like initial weight, embeddings included
    lr_profile: str = ""  # reference trial: matrix LR *= 1 / (its mean |W_t|/|W_0|), i.e. its effective LR
    radial_lr: float = 0.0  # c > 0: adaptive radial step on dL/dlog|W| (needs the full projection)
    # ---- logging
    eval_every: int = 500
    eval_batches: int = 16
    final_eval_batches: int = 256
    diag_every: int = 100


def trial_from_dict(payload: dict[str, Any]) -> Trial:
    known = {f.name for f in fields(Trial)}
    return Trial(**{key: value for key, value in payload.items() if key in known})


# --------------------------------------------------------------------- data
class Stream:
    """Token stream with the same sampling semantics as the repo's TokenStream.

    Tokens stay int32 on the GPU (the repo's class stores int64); the gathered
    batches are identical for the same generator state.
    """

    def __init__(self, path: Path, device: torch.device) -> None:
        arr = np.load(path, mmap_mode="r")
        if int(arr.max()) >= VOCAB_SIZE:
            raise ValueError(f"{path} contains token ids outside the GPT-2 vocabulary")
        self.tokens = torch.from_numpy(np.ascontiguousarray(arr).astype(np.int32, copy=False)).to(device)

    def batch(self, *, batch_size: int, block_size: int, generator: torch.Generator):
        max_start_exclusive = self.tokens.numel() - block_size
        idx = torch.randint(0, max_start_exclusive, (batch_size,), device=self.tokens.device, generator=generator)
        offsets = torch.arange(block_size + 1, device=self.tokens.device)
        chunk = self.tokens[idx[:, None] + offsets[None, :]].long()
        return chunk[:, :-1], chunk[:, 1:]

    def sequential_batch(self, *, batch_index: int, batch_size: int, block_size: int):
        max_start_exclusive = self.tokens.numel() - block_size
        start = (batch_index * batch_size * block_size) % max_start_exclusive
        idx = (start + torch.arange(batch_size, device=self.tokens.device) * block_size) % max_start_exclusive
        offsets = torch.arange(block_size + 1, device=self.tokens.device)
        chunk = self.tokens[idx[:, None] + offsets[None, :]].long()
        return chunk[:, :-1], chunk[:, 1:]


def cache_paths(data_dir: Path, data: str) -> tuple[Path, Path]:
    train = sorted(data_dir.glob(f"*_train_{TRAIN_TOKENS[data]}.tok.npy"))
    val = sorted(data_dir.glob(f"*_validation_{VAL_TOKENS}.tok.npy"))
    if len(train) != 1 or len(val) != 1:
        raise FileNotFoundError(f"expected exactly one train/val cache for data={data!r} in {data_dir}")
    return train[0], val[0]


# ----------------------------------------------------------------- schedule
def lr_scale(trial: Trial, step: int) -> float:
    """Schedule multiplier for 1-indexed ``step``."""

    total, warmup, final = trial.steps, trial.warmup_steps, trial.final_lr_scale
    if warmup > 0 and step <= warmup:
        return step / warmup
    if trial.schedule == "wsd":  # identical to the repo's lr_scale
        decay_steps = max(1, int(total * trial.decay_frac))
        decay_start = total - decay_steps
        if step <= decay_start:
            return 1.0
        progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))
        return 1.0 + progress * (final - 1.0)
    if trial.schedule == "cosine":  # hnet trainer: warmup, then cosine to `final`
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if trial.schedule == "linear":  # MuonGS paper: constant, then linear to `final`
        decay_steps = max(1, int(total * trial.decay_frac))
        decay_start = total - decay_steps
        if step <= decay_start:
            return 1.0
        return 1.0 + min(1.0, (step - decay_start) / decay_steps) * (final - 1.0)
    raise ValueError(f"unknown schedule {trial.schedule!r}")


# ---------------------------------------------------------------- optimizer
def matrix_class(name: str) -> str | None:
    if name == "tok_emb.weight":
        return "emb"
    if name == "pos_emb.weight":
        return "pos"
    for suffix, label in (
        ("attn.qkv.weight", "qkv"),
        ("attn.proj.weight", "attn_proj"),
        ("mlp.fc.weight", "mlp_fc"),
        ("mlp.proj.weight", "mlp_proj"),
    ):
        if name.endswith(suffix):
            return label
    return None


def build_groups(model: torch.nn.Module, trial: Trial) -> list[dict[str, Any]]:
    """Same routing as the repo's ``split_muon_groups`` unless ``emb`` says otherwise."""

    matrix: list[tuple[str, torch.nn.Parameter]] = []
    fallback_decay: list[tuple[str, torch.nn.Parameter]] = []
    fallback_nodecay: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        is_embedding = name in ("tok_emb.weight", "pos_emb.weight")
        matrix_like = parameter.is_floating_point() and len([d for d in parameter.shape if d > 1]) >= 2
        if matrix_like and not (is_embedding and trial.emb == "fallback"):
            if is_embedding and trial.emb == "matrix_nogs":
                parameter._no_gs = True
            matrix.append((name, parameter))
        elif parameter.ndim >= 2:
            fallback_decay.append((name, parameter))
        else:
            fallback_nodecay.append((name, parameter))
    fallback_lr = trial.lr * trial.fallback_lr_mult
    groups: list[dict[str, Any]] = [
        {
            "params": [p for _n, p in matrix],
            "param_names": [n for n, _p in matrix],
            "use_matrix_update": True,
            "lr": trial.lr,
        }
    ]
    if fallback_decay:
        groups.append(
            {
                "params": [p for _n, p in fallback_decay],
                "param_names": [n for n, _p in fallback_decay],
                "use_matrix_update": False,
                "lr": fallback_lr,
                "fallback_weight_decay": trial.fallback_wd,
            }
        )
    if fallback_nodecay:
        groups.append(
            {
                "params": [p for _n, p in fallback_nodecay],
                "param_names": [n for n, _p in fallback_nodecay],
                "use_matrix_update": False,
                "lr": fallback_lr,
                "fallback_weight_decay": 0.0,
            }
        )
    return groups


def make_optimizer(model: torch.nn.Module, trial: Trial) -> torch.optim.Optimizer:
    if trial.opt in {"repo_muon", "repo_adamw"}:
        import run_wikitext_llm50m as codex_runner

        family = "muon" if trial.opt == "repo_muon" else "adamw"
        return codex_runner.make_optimizer(
            model, codex_runner.TrialConfig(name=trial.name, family=family, lr=trial.lr, weight_decay=trial.wd)
        )
    if trial.opt != "muonc":
        raise ValueError(f"unknown optimizer {trial.opt!r}")
    from muonc_research import MuonCResearch

    parts = set() if trial.base == "plain" else set(trial.base.split("+"))
    unknown = parts - {"pmuoneq", "normuon", "soda"}
    if unknown:
        raise ValueError(f"unknown base components {sorted(unknown)}")
    use_soda = "soda" in parts and trial.soda > 0.0
    if trial.norm_target > 0.0 and trial.decay != "none":
        raise ValueError("norm_target replaces weight decay on the matrix path; use decay='none'")
    if trial.ns_dtype not in {"", "bf16"}:
        raise ValueError(f"unknown ns_dtype {trial.ns_dtype!r}")
    return MuonCResearch(
        build_groups(model, trial),
        lr=trial.lr,
        fallback_lr=trial.lr * trial.fallback_lr_mult,
        fallback_mode=trial.fallback_mode,
        momentum=0.95,
        matrix_weight_decay=0.0 if trial.decay == "none" else trial.wd,
        matrix_decay_law="standard" if trial.decay == "none" else trial.decay,
        fallback_decay_law=trial.fallback_decay,
        unprojected_decay_law=trial.unprojected_decay or None,
        use_soda=use_soda,
        soda_lambda_scale=trial.soda if use_soda else 1.0,
        use_pmuoneq="pmuoneq" in parts,
        row_gamma=trial.row_gamma,
        pmuoneq_beta=trial.pmuoneq_beta,
        use_normuon="normuon" in parts,
        normuon_beta2=trial.normuon_beta2,
        gs_strength=trial.gs,
        gs_mode=trial.gs_mode,
        norm_mode=trial.norm_mode,
        norm_target=trial.norm_target if trial.norm_target > 0.0 else None,
        radial_lr=trial.radial_lr,
        ns_compute_dtype=torch.bfloat16 if trial.ns_dtype == "bf16" else None,
    )


# -------------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model, stream: Stream, *, batch_size: int, block_size: int, batches: int) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for batch_index in range(batches):
        x, y = stream.sequential_batch(batch_index=batch_index, batch_size=batch_size, block_size=block_size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.detach()))
        correct += int((logits.argmax(dim=-1) == y).sum().detach())
        total += int(y.numel())
    model.train()
    return float(np.mean(losses)), correct / max(total, 1)


def run_trial(trial: Trial, out_dir: Path, data_dir: Path) -> dict[str, Any]:
    from run_synthetic_llm50m import TinyGPT

    torch.manual_seed(trial.seed)
    np.random.seed(trial.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model = TinyGPT(
        vocab_size=VOCAB_SIZE, block_size=trial.block_size, n_layer=8, n_head=8, n_embd=512, dropout=0.0
    ).to(device)
    if trial.init_resid_scale != 1.0 or trial.init_scale != 1.0:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if matrix_class(name) is not None:
                    parameter.mul_(trial.init_scale)
                if name.endswith(("attn.proj.weight", "mlp.proj.weight")):
                    parameter.mul_(trial.init_resid_scale)
    profile: tuple[np.ndarray, np.ndarray] | None = None
    if trial.lr_profile:
        reference = out_dir / "trials" / trial.lr_profile / "metrics.jsonl"
        rows = [json.loads(line) for line in reference.read_text().splitlines() if '"phase": "diag"' in line]
        profile = (
            np.array([row["step"] for row in rows], dtype=np.float64),
            np.array([np.mean([row[f"norm_ratio/{label}"] for label in MATRIX_CLASSES]) for row in rows]),
        )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, trial)
    for group in optimizer.param_groups:
        group["_bench_base_lr"] = float(group.get("lr", trial.lr))
    train_path, val_path = cache_paths(data_dir, trial.data)
    train_stream = Stream(train_path, device)
    val_stream = Stream(val_path, device)
    train_gen = torch.Generator(device=device).manual_seed(trial.seed + 10)

    tracked: dict[str, list[tuple[str, torch.nn.Parameter, float]]] = {label: [] for label in MATRIX_CLASSES}
    for name, parameter in model.named_parameters():
        label = matrix_class(name)
        if label is not None:
            tracked[label].append((name, parameter, float(parameter.detach().float().norm())))
    has_diag = hasattr(optimizer, "pop_diag")
    # Tangential relative path length sum_t lr_t |N_t,tangent| / |W_t|.  The ratio is
    # sampled on diagnostic steps and held in between; the LR is exact every step.
    angular_path = {label: 0.0 for label in MATRIX_CLASSES}
    last_ratio: dict[str, float] = {}
    cos_sum = {label: 0.0 for label in MATRIX_CLASSES}
    cos_abs_sum = {label: 0.0 for label in MATRIX_CLASSES}
    diag_count = 0
    cum_scale_sq = 0.0

    trial_dir = out_dir / "trials" / trial.name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(json.dumps(asdict(trial), indent=2) + "\n")
    eval_steps = set(range(trial.eval_every, trial.steps + 1, trial.eval_every)) | {trial.steps}
    step_times: list[float] = []
    recent_losses: list[float] = []
    eval_rows: list[dict[str, Any]] = []
    diverged = False
    start = time.perf_counter()
    metrics_path = trial_dir / "metrics.jsonl"
    with metrics_path.open("w") as handle:
        for step in range(1, trial.steps + 1):
            scale = lr_scale(trial, step)
            matrix_scale = scale
            if trial.emulate_gs_theta0 > 0.0:
                matrix_scale = scale / math.sqrt(1.0 + trial.emulate_gs_theta0**2 * cum_scale_sq)
                cum_scale_sq += scale * scale
            if profile is not None:
                matrix_scale = matrix_scale / float(np.interp(step - 1, profile[0], profile[1]))
            for group in optimizer.param_groups:
                is_matrix = bool(group.get("use_matrix_update", group.get("use_muon", False)))
                group["lr"] = float(group["_bench_base_lr"]) * (matrix_scale if is_matrix else scale)
            applied_lr = float(optimizer.param_groups[0]["lr"])
            x, y = train_stream.batch(batch_size=trial.batch_size, block_size=trial.block_size, generator=train_gen)
            do_diag = step == 1 or step % trial.diag_every == 0 or step == trial.steps
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if do_diag and has_diag:
                optimizer.diag_next = True
            optimizer.step()
            torch.cuda.synchronize()
            if not do_diag:  # diagnostic steps pay for extra host syncs; keep them out of the timing
                step_times.append(time.perf_counter() - t0)
            last_loss = float(loss.detach())
            recent_losses.append(last_loss)
            if len(recent_losses) > 100:
                recent_losses.pop(0)
            if not math.isfinite(last_loss):
                diverged = True
                handle.write(json.dumps({"step": step, "phase": "diverged"}) + "\n")
                break
            if do_diag:
                row: dict[str, Any] = {"step": step, "phase": "diag", "lr": applied_lr}
                opt_diag = optimizer.pop_diag() if has_diag else {}
                for label, items in tracked.items():
                    ratios = [float(p.detach().float().norm()) / r0 for _n, p, r0 in items]
                    row[f"norm_ratio/{label}"] = float(np.mean(ratios))
                    stats = [opt_diag[n] for n, _p, _r in items if n in opt_diag]
                    if stats:
                        cos = float(np.mean([s["radial_cos"] for s in stats]))
                        # theta: relative size of the applied step, lr * |N_applied| / |W|.
                        theta = float(np.mean([s["update_norm"] / max(s["weight_norm"], 1e-30) for s in stats]))
                        row[f"radial_cos/{label}"] = cos
                        if "radial_control" in stats[0]:
                            row[f"radial_control/{label}"] = float(np.mean([s["radial_control"] for s in stats]))
                        row[f"theta/{label}"] = theta * applied_lr
                        last_ratio[label] = float(
                            np.mean([s.get("tangent_norm", s["update_norm"]) / max(s["weight_norm"], 1e-30) for s in stats])
                        )
                        cos_sum[label] += cos
                        cos_abs_sum[label] += abs(cos)
                if opt_diag:
                    diag_count += 1
                handle.write(json.dumps(row) + "\n")
            for label, ratio in last_ratio.items():
                angular_path[label] += ratio * applied_lr
            if step % 500 == 0 or step in eval_steps:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "phase": "train",
                            "train_loss": last_loss,
                            "train_loss_avg100": float(np.mean(recent_losses)),
                            "lr": applied_lr,
                        }
                    )
                    + "\n"
                )
            if step in eval_steps:
                val_loss, val_acc = evaluate(
                    model, val_stream, batch_size=trial.batch_size, block_size=trial.block_size, batches=trial.eval_batches
                )
                eval_rows.append({"step": step, "val_loss": val_loss, "val_acc": val_acc})
                handle.write(
                    json.dumps({"step": step, "phase": "eval", "val_loss": val_loss, "val_acc": val_acc, "lr": applied_lr})
                    + "\n"
                )
                handle.flush()

    summary: dict[str, Any] = {**asdict(trial), "param_count": param_count, "diverged": diverged}
    if not diverged:
        full_loss, full_acc = evaluate(
            model, val_stream, batch_size=trial.batch_size, block_size=trial.block_size, batches=trial.final_eval_batches
        )
        summary.update(
            {
                "final_train_loss_avg100": float(np.mean(recent_losses)),
                "final_val_loss": eval_rows[-1]["val_loss"],
                "best_val_loss": min(row["val_loss"] for row in eval_rows),
                "final_val_acc": eval_rows[-1]["val_acc"],
                "final_full_val_loss": full_loss,
                "final_full_val_acc": full_acc,
            }
        )
        for label, items in tracked.items():
            summary[f"final_norm_ratio/{label}"] = float(
                np.mean([float(p.detach().float().norm()) / r0 for _n, p, r0 in items])
            )
            if diag_count and label in last_ratio:
                summary[f"angular_path/{label}"] = angular_path[label]
                summary[f"mean_radial_cos/{label}"] = cos_sum[label] / diag_count
                summary[f"mean_abs_radial_cos/{label}"] = cos_abs_sum[label] / diag_count
    times = np.asarray(step_times[10:] or step_times)
    summary.update(
        {
            "mean_step_time_ms": float(times.mean() * 1000.0),
            "median_step_time_ms": float(np.median(times) * 1000.0),
            "wall_time_sec": time.perf_counter() - start,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    )
    (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ----------------------------------------------------------------- launcher
def parse_gpus(spec: str) -> list[int]:
    gpus: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            gpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            gpus.append(int(part))
    return gpus


def launch(trials: list[Trial], out_dir: Path, data_dir: Path, gpus: list[int], jobs_per_gpu: int) -> None:
    names = [t.name for t in trials]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate trial names: {dupes[:5]}")
    (out_dir / "trials").mkdir(parents=True, exist_ok=True)
    pending = []
    for trial in trials:
        trial_dir = out_dir / "trials" / trial.name
        if not (trial_dir / "summary.json").exists():
            pending.append(trial)
            continue
        # A finished trial is only reused if it was run with exactly this configuration.
        previous = {**asdict(trial_from_dict({"name": trial.name})), **json.loads((trial_dir / "trial.json").read_text())}
        wanted = asdict(trial)
        differing = sorted(k for k in wanted if k != "stage" and wanted[k] != previous.get(k))
        if differing:
            raise ValueError(f"{trial.name} already exists with different settings for {differing}")
    print(f"{len(trials)} trials, {len(trials) - len(pending)} already complete, {len(pending)} to run", flush=True)
    free = [gpu for _ in range(jobs_per_gpu) for gpu in gpus]
    running: list[tuple[subprocess.Popen, Trial, int, Any]] = []
    done = 0
    total = len(pending)
    started = time.time()
    while pending or running:
        while pending and free:
            trial = pending.pop(0)
            gpu = free.pop(0)
            trial_dir = out_dir / "trials" / trial.name
            trial_dir.mkdir(parents=True, exist_ok=True)
            spec = trial_dir / "trial.json"
            spec.write_text(json.dumps(asdict(trial), indent=2) + "\n")
            log = (trial_dir / "worker.log").open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", "--trial-json", str(spec),
                 "--out", str(out_dir), "--data-dir", str(data_dir)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            running.append((proc, trial, gpu, log))
        time.sleep(2.0)
        still: list[tuple[subprocess.Popen, Trial, int, Any]] = []
        for proc, trial, gpu, log in running:
            code = proc.poll()
            if code is None:
                still.append((proc, trial, gpu, log))
                continue
            log.close()
            free.append(gpu)
            done += 1
            summary_path = out_dir / "trials" / trial.name / "summary.json"
            if code == 0 and summary_path.exists():
                s = json.loads(summary_path.read_text())
                tail = "DIVERGED" if s.get("diverged") else f"full_val={s['final_full_val_loss']:.4f}"
            else:
                tail = f"FAILED exit={code}"
            print(f"[{done}/{total}] {trial.name}: {tail} ({time.time() - started:.0f}s)", flush=True)
        running = still
    print("all trials finished", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-json", type=Path)
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--jobs-per-gpu", type=int, default=4)
    parser.add_argument("--list", action="store_true", help="print the trial names of the stage and exit")
    args = parser.parse_args()
    if args.worker:
        trial = trial_from_dict(json.loads(args.trial_json.read_text()))
        run_trial(trial, args.out, args.data_dir)
        return
    import grids

    trials: list[Trial] = []
    for stage in args.stage:
        trials.extend(grids.stage_trials(stage))
    if args.list:
        for trial in trials:
            print(trial.name)
        print(len(trials), "trials")
        return
    launch(trials, args.out, args.data_dir, parse_gpus(args.gpus), args.jobs_per_gpu)


if __name__ == "__main__":
    main()
