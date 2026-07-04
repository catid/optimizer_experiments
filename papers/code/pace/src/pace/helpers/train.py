"""Hydra training entry point (``pace-train``) for PACE / AdamW / EMA on a tokenized
SFT dataset. The optimizer is chosen by config; see the README for usage.
"""

import json
import math
import os
import random
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from pace.helpers.data import collate_fn, load_prepared
from pace.helpers.model import load_model
from pace.optimizer import PACE
from pace.helpers.schedules import build_scheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _Logger:
    """Thin optional wrapper around Weights & Biases."""

    def __init__(self, cfg, run_name):
        self.enabled = bool(cfg.wandb.get("enabled", False))
        if self.enabled:
            import wandb
            self._wandb = wandb
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get("entity"),
                       name=run_name, group=cfg.wandb.get("group"),
                       config=OmegaConf.to_container(cfg, resolve=True), reinit=True)

    def log(self, d, step=None):
        if self.enabled:
            self._wandb.log(d, step=step)

    def finish(self):
        if self.enabled:
            self._wandb.finish(quiet=True)


def make_run_name(cfg: DictConfig) -> str:
    opt = cfg.optimizer
    eff_bs = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    parts = [opt.name, f"lr{opt.lr}", f"bs{eff_bs}"]
    if opt.use_ema_eval and opt.get("ema_kappa") is not None:
        parts.append(f"k{opt.ema_kappa}")
    if opt.lambda_pullback > 0:
        parts.append(f"lp{opt.lambda_pullback}")
    parts.append(f"sched-{cfg.schedule.name}")
    parts.append(f"s{cfg.seed}")
    return "_".join(parts)


def build_optimizer(model, opt_cfg):
    """AdamW fast-path when no EMA/pullback; otherwise PACE."""
    use_ema_eval = bool(opt_cfg.use_ema_eval)
    if opt_cfg.name == "adamw" and not use_ema_eval and opt_cfg.lambda_pullback == 0:
        opt = torch.optim.AdamW(
            model.parameters(), lr=opt_cfg.lr, betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps, weight_decay=opt_cfg.weight_decay,
        )
        print("  Optimizer: torch.optim.AdamW")
        return opt
    opt = PACE(
        model.parameters(), lr=opt_cfg.lr, betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps, weight_decay=opt_cfg.weight_decay,
        lambda_pullback=opt_cfg.lambda_pullback,
        clamp_pullback=opt_cfg.get("clamp_pullback", True),
        beta_ema=opt_cfg.beta_ema, use_ema_eval=use_ema_eval,
        ema_kappa=opt_cfg.get("ema_kappa"), ema_rho=opt_cfg.get("ema_rho", 0.0),
        ema_gamma=opt_cfg.get("ema_gamma", 1.0),
        ema_update_freq=opt_cfg.get("ema_update_freq", 1),
        log_stats=opt_cfg.get("log_stats", True),
    )
    opt.train()
    print(f"  Optimizer: PACE (c={opt_cfg.lambda_pullback}, kappa={opt_cfg.get('ema_kappa')}, "
          f"use_ema_eval={use_ema_eval})")
    return opt


@torch.no_grad()
def _eval_loss(model, dataloader, max_samples, ctx):
    total_loss, total_tokens, n_samples = 0.0, 0, 0
    for batch in dataloader:
        if max_samples is not None:
            remaining = max_samples - n_samples
            if remaining <= 0:
                break
            if batch["input_ids"].size(0) > remaining:
                batch = {k: v[:remaining] for k, v in batch.items()}
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        with ctx:
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
        n_lab = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_lab
        total_tokens += n_lab
        n_samples += ids.size(0)
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, optimizer, dataloader, max_samples, ctx, use_ema_eval):
    model.eval()
    vanilla = _eval_loss(model, dataloader, max_samples, ctx)
    ema = None
    if use_ema_eval:
        optimizer.eval()
        ema = _eval_loss(model, dataloader, max_samples, ctx)
        optimizer.train()
    model.train()
    return {"val_loss": ema if use_ema_eval else vanilla,
            "vanilla_val_loss": vanilla, "ema_val_loss": ema}


def _opt_stats(optimizer):
    out = {}
    if hasattr(optimizer, "get_step_stats"):
        for k, v in optimizer.get_step_stats().items():
            out[k] = float(v)
    return out


def run_training(cfg: DictConfig) -> float:
    """Train for a composed config and return the best validation loss.

    This is the importable core; ``main`` is the Hydra CLI wrapper around it.
    """
    set_seed(cfg.seed)
    run_name = make_run_name(cfg)
    print("=" * 70)
    print(run_name)
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    logger = _Logger(cfg, run_name)

    # Data
    train_ds, val_ds = load_prepared(cfg.dataset.path, cfg.dataset.get("max_samples"))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Model + optimizer + schedule
    model = load_model(cfg.model.name, DEVICE)
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    opt_cfg = cfg.optimizer
    use_ema_eval = bool(opt_cfg.use_ema_eval)
    optimizer = build_optimizer(model, opt_cfg)

    ctx = torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE)
    grad_accum = cfg.training.gradient_accumulation_steps
    grad_clip = cfg.training.grad_clip
    eval_interval = cfg.training.eval_interval
    eval_samples = cfg.training.get("eval_samples")
    epochs = cfg.training.epochs
    max_steps = cfg.training.get("max_steps")  # optional cap (smoke tests)

    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    warmup_steps = cfg.schedule.get("warmup_steps", 0)
    scheduler = build_scheduler(optimizer, cfg.schedule.name, total_steps,
                                warmup_steps=warmup_steps,
                                decay_frac=cfg.schedule.get("decay_frac", 0.2))
    print(f"  Steps/epoch: {steps_per_epoch} | total_steps: {total_steps} | "
          f"schedule: {cfg.schedule.name} (warmup {warmup_steps})")

    eval_logs, step_logs = [], []
    best_val = float("inf")
    global_step = 0
    t_start = time.time()
    model.train()

    def do_eval(epoch, train_loss, tag=""):
        nonlocal best_val
        res = evaluate(model, optimizer, val_loader, eval_samples, ctx, use_ema_eval)
        best_val = min(best_val, res["val_loss"])
        entry = {"step": global_step, "epoch": epoch, "train_loss": train_loss,
                 "val_loss": res["val_loss"], "vanilla_val_loss": res["vanilla_val_loss"],
                 "ema_val_loss": res["ema_val_loss"], "best_val_loss": best_val,
                 "elapsed_sec": time.time() - t_start, "tag": tag}
        entry.update(_opt_stats(optimizer))
        eval_logs.append(entry)
        logger.log({"val/loss": res["val_loss"], "val/vanilla": res["vanilla_val_loss"]},
                   step=global_step)
        msg = [f"step {global_step}/{total_steps}", f"train={train_loss:.4f}",
               f"val={res['val_loss']:.4f}"]
        if res["ema_val_loss"] is not None:
            msg.append(f"vanilla={res['vanilla_val_loss']:.4f}")
        print("  " + " | ".join(msg), flush=True)

    stop = False
    for epoch in range(epochs):
        if stop:
            break
        micro_step, accum_loss = 0, 0.0
        for batch in train_loader:
            micro_step += 1
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            with ctx:
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
                loss = out.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()
            if not math.isfinite(accum_loss):
                print(f"  FATAL: non-finite loss at micro_step {micro_step}", flush=True)
                logger.finish()
                sys.exit(1)
            if micro_step % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip if grad_clip > 0 else float("inf"))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_loss, accum_loss = accum_loss, 0.0
                step_logs.append({"step": global_step, "train_loss": train_loss,
                                  "grad_norm": grad_norm.item(),
                                  "lr": optimizer.param_groups[0]["lr"],
                                  "elapsed_sec": time.time() - t_start})
                log_dict = {"train/loss": train_loss, "train/grad_norm": grad_norm.item(),
                            "train/lr": optimizer.param_groups[0]["lr"]}
                log_dict.update({f"train/{k}": v for k, v in _opt_stats(optimizer).items()})
                logger.log(log_dict, step=global_step)
                if global_step % eval_interval == 0 or global_step == 1:
                    do_eval(epoch + 1, train_loss)
                if max_steps and global_step >= max_steps:
                    stop = True
                    break

    # Final eval on the full val set (unless eval_samples caps it)
    final_eval_n = eval_samples if eval_samples else len(val_ds)
    res = evaluate(model, optimizer, val_loader, final_eval_n, ctx, use_ema_eval)
    best_val = min(best_val, res["val_loss"])
    total_time = time.time() - t_start
    final = {"step": global_step, "epoch": epochs,
             "train_loss": step_logs[-1]["train_loss"] if step_logs else 0.0,
             "val_loss": res["val_loss"], "vanilla_val_loss": res["vanilla_val_loss"],
             "ema_val_loss": res["ema_val_loss"], "best_val_loss": best_val,
             "elapsed_sec": total_time, "tag": "final"}
    final.update(_opt_stats(optimizer))
    eval_logs.append(final)

    summary = {"best_val_loss": best_val,
               "final_train_loss": step_logs[-1]["train_loss"] if step_logs else 0.0,
               "final_val_loss": res["val_loss"], "total_sec": total_time,
               "total_hours": total_time / 3600, "total_steps": global_step}

    try:  # the Hydra run dir when launched as a CLI; else PACE_RUN_DIR (or cwd)
        out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
        out_dir = os.environ.get("PACE_RUN_DIR", "runs")
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"{run_name}.json")
    with open(results_path, "w") as f:
        json.dump({"config": OmegaConf.to_container(cfg, resolve=True),
                   "summary": summary, "eval_logs": eval_logs, "step_logs": step_logs},
                  f, indent=2)
    print(f"\nDONE: best_val_loss={best_val:.4f} | results: {results_path}", flush=True)
    logger.finish()
    return best_val


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    return run_training(cfg)


if __name__ == "__main__":
    main()
