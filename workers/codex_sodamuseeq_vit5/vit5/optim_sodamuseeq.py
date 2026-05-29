"""SodaMuseEq optimizer for ViT-5 experiments.

This file is intentionally self-contained so it can be copied into other
projects. The optimizer combines:

- SODA anchor correction, optional via ``use_soda``.
- AMUSE schedule-free fast/eval iterate logic, optional via ``use_amuse``.
- PMuonEq row/column EMA equilibration before projection, optional via
  ``use_pmuoneq``.
- Gram/Newton-Schulz projection, optional via ``use_gram``.
- MiMuon-style polar-vs-momentum branch, optional via ``use_mimuon``.
- NorMuon row second-moment normalization after projection, optional via
  ``use_normuon``.

Recommended name: ``SodaMuseEq``.
Short alias: ``EquiMuse``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn


DAO_POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (7.89258287442441, -20.386750231831207, 13.554909290386428),
    (3.9114848681354314, -2.546436052530984, 0.4269351938498667),
    (3.7606579556974233, -2.5122235116543104, 0.4323966512045484),
    (3.160399673686287, -2.1496735790007567, 0.3996018288984032),
    (2.1910971618617306, -1.4419638886187698, 0.3281513704122412),
)


def _is_matrix_like(p: nn.Parameter) -> bool:
    return p.requires_grad and p.is_floating_point() and p.ndim in (2, 4)


def _is_aux_name(name: str) -> bool:
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return True
    if any(token in lowered for token in ("norm", "ln_", "layernorm", "rmsnorm")):
        return True
    if any(token in lowered for token in ("pos_embed", "cls_token", "reg_token", "wte", "tok_embeddings", "word_embeddings")):
        return True
    return lowered.endswith("head.weight") or lowered.endswith("head.bias")


def make_sodamuseeq_param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    aux_lr: float | None = None,
    aux_weight_decay: float | None = None,
    include_head: bool = False,
    include_embeddings: bool = False,
) -> list[dict[str, Any]]:
    matrix: list[nn.Parameter] = []
    aux: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        p._sodamuseeq_param_name = name  # type: ignore[attr-defined]
        force_aux = _is_aux_name(name)
        if include_head and "head." in name:
            force_aux = False
        if include_embeddings and any(token in name.lower() for token in ("pos_embed", "cls_token", "reg_token")):
            force_aux = False
        if _is_matrix_like(p) and not force_aux:
            matrix.append(p)
        else:
            aux.append(p)
    groups: list[dict[str, Any]] = []
    if matrix:
        groups.append({"params": matrix, "lr": lr, "weight_decay": weight_decay, "use_muon": True})
    if aux:
        groups.append(
            {
                "params": aux,
                "lr": lr if aux_lr is None else aux_lr,
                "weight_decay": weight_decay if aux_weight_decay is None else aux_weight_decay,
                "use_muon": False,
            }
        )
    return groups


class GramNewtonSchulzProjector:
    def __init__(
        self,
        coefficients: Sequence[Sequence[float]] = DAO_POLAR_EXPRESS_COEFFS,
        eps: float = 1e-7,
        projection_dtype: torch.dtype = torch.bfloat16,
        use_gram: bool = True,
    ):
        self.coefficients = tuple(tuple(float(v) for v in c) for c in coefficients)
        self.eps = float(eps)
        self.projection_dtype = projection_dtype
        self.use_gram = bool(use_gram)
        self.reset_iterations = {2}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("projector expects a matrix or batch of matrices")
        original_shape = x.shape
        y = x
        if y.ndim == 2:
            y = y.unsqueeze(0)
        elif y.ndim > 3:
            y = y.reshape(-1, *y.shape[-2:])
        original_dtype = y.dtype
        y = y.float()
        transposed = y.size(-2) > y.size(-1)
        if transposed:
            y = y.mT
        y = y / (y.norm(dim=(-2, -1), keepdim=True) + self.eps)
        y = y.to(self.projection_dtype)
        if self.use_gram and max(y.shape[-2:]) > min(y.shape[-2:]):
            y = self._gram(y)
        else:
            y = self._standard(y)
        if transposed:
            y = y.mT
        return y.to(original_dtype).reshape(original_shape)

    def _gram(self, x: torch.Tensor) -> torch.Tensor:
        r = x @ x.mT
        eye = torch.eye(r.size(-1), device=x.device, dtype=x.dtype).unsqueeze(0).expand(r.size(0), -1, -1).contiguous()
        q = None
        for idx, (a, b, c) in enumerate(self.coefficients):
            if idx in self.reset_iterations and idx != 0:
                x = q @ x
                r = x @ x.mT
                q = None
            z = torch.baddbmm(r, r, r, beta=b, alpha=c)
            if idx == 0 or idx in self.reset_iterations:
                q = z + a * eye
            else:
                q = torch.baddbmm(q, q, z, beta=a)
            if idx < len(self.coefficients) - 1 and idx + 1 not in self.reset_iterations:
                rz = torch.baddbmm(r, r, z, beta=a)
                r = torch.baddbmm(rz, z, rz, beta=a)
        return q @ x

    def _standard(self, x: torch.Tensor) -> torch.Tensor:
        for a, b, c in self.coefficients:
            xx = x @ x.mT
            poly = torch.baddbmm(xx, xx, xx, beta=b, alpha=c)
            x = torch.baddbmm(x, poly, x, beta=a)
        return x


def _row_stats(u: torch.Tensor) -> tuple[float, float]:
    if u.ndim != 2:
        return 0.0, 0.0
    row = u.detach().float().square().mean(dim=1).sqrt()
    mean = row.mean().clamp_min(1e-12)
    median = row.median().clamp_min(1e-12)
    return float((row.std(unbiased=False) / mean).item()), float((row < 0.1 * median).float().mean().item())


def _stiefel_defect(u: torch.Tensor) -> float:
    if u.ndim != 2 or min(u.shape) == 0:
        return 0.0
    x = u.detach().float()
    rows, cols = x.shape
    if rows >= cols:
        gram = x.T @ x
        eye = torch.eye(cols, device=x.device, dtype=x.dtype)
        return float(((gram - eye).norm() / math.sqrt(cols)).item())
    gram = x @ x.T
    eye = torch.eye(rows, device=x.device, dtype=x.dtype)
    return float(((gram - eye).norm() / math.sqrt(rows)).item())


def _normalize_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.square().mean().sqrt().clamp_min(eps)


@torch.no_grad()
def _pmuoneq(entries: list[dict[str, Any]], *, beta: float, row_gamma: float, col_gamma: float, sides: str, eps: float) -> dict[str, float]:
    stats = {"pmuoneq_factor_count": 0.0, "pmuoneq_left_count": 0.0, "pmuoneq_right_count": 0.0}
    if sides == "none":
        return stats
    for entry in entries:
        g = entry["grad_matrix"].float()
        work = entry["matrix"].float()
        state = entry["state"]
        rows, cols = g.shape
        use_left = sides in {"left", "both"} or (sides == "min" and rows <= cols)
        use_right = sides in {"right", "both"} or (sides == "min" and cols <= rows)
        if use_left and row_gamma != 0.0:
            stat = g.square().mean(dim=1)
            ema = state.get("pmuoneq_left_ema")
            if ema is None or ema.shape != stat.shape or ema.device != stat.device:
                ema = state["pmuoneq_left_ema"] = torch.zeros_like(stat, dtype=torch.float32)
            ema.mul_(beta).add_(stat, alpha=1.0 - beta)
            scale = _normalize_scale(ema.clamp_min(eps).pow(-row_gamma), eps).to(work.dtype)
            work = scale[:, None] * work
            stats["pmuoneq_factor_count"] += 1.0
            stats["pmuoneq_left_count"] += 1.0
        if use_right and col_gamma != 0.0:
            stat = g.square().mean(dim=0)
            ema = state.get("pmuoneq_right_ema")
            if ema is None or ema.shape != stat.shape or ema.device != stat.device:
                ema = state["pmuoneq_right_ema"] = torch.zeros_like(stat, dtype=torch.float32)
            ema.mul_(beta).add_(stat, alpha=1.0 - beta)
            scale = _normalize_scale(ema.clamp_min(eps).pow(-col_gamma), eps).to(work.dtype)
            work = work * scale[None, :]
            stats["pmuoneq_factor_count"] += 1.0
            stats["pmuoneq_right_count"] += 1.0
        entry["matrix"] = work
    return stats


def _normuon_second_shape(rows: int, cols: int, mode: str) -> tuple[int, int]:
    if mode == "row":
        return rows, 1
    if mode == "orientation":
        return (rows, 1) if rows >= cols else (1, cols)
    raise ValueError(f"unknown NorMuon mode {mode!r}")


@torch.no_grad()
def _normuon_update(
    update: torch.Tensor,
    state: dict[str, Any],
    *,
    beta2: float,
    eps: float,
    mode: str,
    aspect_scale: bool,
) -> torch.Tensor:
    """NorMuon normalization applied after the Gram/polar update."""
    if update.ndim != 2:
        return update
    work = update.float()
    rows, cols = work.shape
    norm_before = work.norm(dim=(-2, -1), keepdim=True)
    second_shape = _normuon_second_shape(rows, cols, mode)
    reduce_dim = -1 if second_shape[1] == 1 else -2
    row_power = work.square().mean(dim=reduce_dim, keepdim=True)
    second = state.get("normuon_second_moment")
    if second is None or second.shape != row_power.shape or second.device != row_power.device:
        second = state["normuon_second_moment"] = torch.zeros_like(row_power, dtype=torch.float32)
    second.mul_(beta2).add_(row_power, alpha=1.0 - beta2)
    work = work * second.clamp_min(eps).rsqrt().to(work.dtype)
    norm_after = work.norm(dim=(-2, -1), keepdim=True)
    work = work * (norm_before / norm_after.clamp_min(eps))
    if aspect_scale:
        work = work * math.sqrt(max(1.0, rows / cols))
    return work.to(update.dtype)


class SodaMuseEq(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 9e-3,
        weight_decay: float = 0.05,
        momentum: float = 0.95,
        beta1: float = 0.6,
        beta2: float = 0.999,
        rho: float = 0.8,
        warmup_steps: int = 200,
        eps: float = 1e-10,
        use_soda: bool = True,
        soda_warmup_steps: int = 200,
        soda_anchor: str = "warmup",
        soda_replaces_weight_decay: bool = True,
        use_amuse: bool = True,
        use_pmuoneq: bool = True,
        pmuon_beta: float = 0.95,
        pmuon_gamma: float = 0.2,
        pmuon_row_gamma: float | None = None,
        pmuon_col_gamma: float | None = None,
        pmuon_sides: str = "both",
        pmuon_eps: float = 1e-6,
        use_gram: bool = True,
        use_mimuon: bool = False,
        mimuon_tau: float = 0.005,
        use_normuon: bool = False,
        normuon_beta2: float = 0.95,
        normuon_eps: float = 1e-10,
        normuon_mode: str = "row",
        normuon_aspect_scale: bool = False,
        batch_project: bool = True,
        stats_interval: int = 100,
        projection_dtype: torch.dtype = torch.bfloat16,
    ):
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if soda_anchor not in {"initial", "warmup"}:
            raise ValueError("soda_anchor must be 'initial' or 'warmup'")
        if pmuon_sides not in {"none", "left", "right", "both", "min"}:
            raise ValueError(f"unknown pmuon_sides={pmuon_sides!r}")
        if normuon_mode not in {"row", "orientation"}:
            raise ValueError(f"unknown normuon_mode={normuon_mode!r}")

        self.beta1_init = float(beta1)
        self.rho = float(rho)
        self.warmup_steps = int(warmup_steps)
        self.use_soda = bool(use_soda)
        self.soda_warmup_steps = int(soda_warmup_steps)
        self.soda_anchor = soda_anchor
        self.soda_replaces_weight_decay = bool(soda_replaces_weight_decay)
        self.use_amuse = bool(use_amuse)
        self.use_pmuoneq = bool(use_pmuoneq)
        self.pmuon_beta = float(pmuon_beta)
        self.pmuon_row_gamma = float(pmuon_gamma if pmuon_row_gamma is None else pmuon_row_gamma)
        self.pmuon_col_gamma = float(pmuon_gamma if pmuon_col_gamma is None else pmuon_col_gamma)
        self.pmuon_sides = pmuon_sides
        self.pmuon_eps = float(pmuon_eps)
        self.use_gram = bool(use_gram)
        self.use_mimuon = bool(use_mimuon)
        self.mimuon_tau = float(mimuon_tau)
        self.use_normuon = bool(use_normuon)
        self.normuon_beta2 = float(normuon_beta2)
        self.normuon_eps = float(normuon_eps)
        self.normuon_mode = normuon_mode
        self.normuon_aspect_scale = bool(normuon_aspect_scale)
        self.batch_project = bool(batch_project)
        self.stats_interval = int(stats_interval)
        self.projector = GramNewtonSchulzProjector(use_gram=use_gram, projection_dtype=projection_dtype)
        self.train_mode = True
        self.global_step = 0
        self.last_stats: dict[str, float] = {}

        groups = self._normalize_groups(params, lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2, eps=eps)
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2, eps=eps, use_muon=True))
        for group in self.param_groups:
            group.setdefault("base_lr", float(group["lr"]))
            group["k"] = int(group.get("k", 0))
            group["weight_sum"] = float(group.get("weight_sum", 0.0))
            group["beta1"] = self.beta1_init
            group["ckp1"] = 1.0
            group["raw_weight_decay"] = float(group.get("weight_decay", 0.0))
            if group.get("use_muon", False):
                group["params"] = sorted(group["params"], key=lambda p: tuple(p.shape), reverse=True)
        if self.soda_anchor == "initial":
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["soda_z0"] = p.detach().float().clone()

    @staticmethod
    def _normalize_groups(params, *, lr: float, weight_decay: float, momentum: float, beta2: float, eps: float) -> list[dict[str, Any]]:
        raw = list(params)
        if not raw:
            raise ValueError("empty optimizer parameter list")
        groups: list[dict[str, Any]] = []
        if isinstance(raw[0], dict):
            for item in raw:
                group = dict(item)
                group["params"] = list(group["params"])
                group.setdefault("use_muon", True)
                groups.append(group)
        else:
            params_list = []
            for item in raw:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], nn.Parameter):
                    item[1]._sodamuseeq_param_name = str(item[0])  # type: ignore[attr-defined]
                    params_list.append(item[1])
                elif isinstance(item, nn.Parameter):
                    params_list.append(item)
                else:
                    raise TypeError(f"invalid parameter entry {type(item)!r}")
            matrix = [p for p in params_list if _is_matrix_like(p)]
            mids = {id(p) for p in matrix}
            aux = [p for p in params_list if id(p) not in mids]
            if matrix:
                groups.append({"params": matrix, "use_muon": True})
            if aux:
                groups.append({"params": aux, "use_muon": False})
        for group in groups:
            group.setdefault("lr", float(lr))
            group.setdefault("base_lr", float(group["lr"]))
            group.setdefault("weight_decay", float(weight_decay))
            group.setdefault("momentum", float(momentum))
            group.setdefault("beta2", float(beta2))
            group.setdefault("eps", float(eps))
        return groups

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def state_dict(self):
        state = super().state_dict()
        state["sodamuseeq_extra"] = {
            "global_step": int(self.global_step),
            "train_mode": bool(self.train_mode),
            "last_stats": dict(self.last_stats),
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop("sodamuseeq_extra", {})
        super().load_state_dict(state_dict)
        inferred_step = max((int(group.get("k", 0)) for group in self.param_groups), default=0)
        self.global_step = int(extra.get("global_step", inferred_step))
        self.train_mode = bool(extra.get("train_mode", True))
        self.last_stats = dict(extra.get("last_stats", {}))

    def set_base_lr(self, lr: float):
        for group in self.param_groups:
            group["base_lr"] = float(lr)
            group["lr"] = float(lr)

    def _z(self, p: nn.Parameter) -> torch.Tensor:
        z = self.state[p].get("z")
        if z is None:
            z = self.state[p]["z"] = p.detach().float().clone(memory_format=torch.preserve_format)
        return z

    def _beta1(self, group: dict[str, Any], t: int, ckp1: float) -> float:
        if not self.use_amuse:
            return 0.0
        if t <= self.warmup_steps:
            if t == self.warmup_steps:
                group["c_warmup"] = ckp1
            return self.beta1_init
        c_warmup = float(group.get("c_warmup", 1.0 / self.warmup_steps))
        denom = max(c_warmup * (1.0 - ckp1), 1e-24)
        s_t = (ckp1 * (1.0 - c_warmup)) / denom
        return float(min(max(1.0 - (s_t**self.rho) * (1.0 - self.beta1_init), 1e-6), 1.0 - 1e-8))

    @torch.no_grad()
    def train(self):
        if self.train_mode or not self.use_amuse:
            self.train_mode = True
            return self
        for group in self.param_groups:
            beta1 = float(group.get("beta1", self.beta1_init))
            for p in group["params"]:
                z = self.state[p].get("z")
                if z is not None:
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - beta1)
        self.train_mode = True
        return self

    @torch.no_grad()
    def eval(self):
        if not self.train_mode or not self.use_amuse:
            self.train_mode = False
            return self
        for group in self.param_groups:
            beta1 = float(group.get("beta1", self.beta1_init))
            if beta1 <= 0.0:
                continue
            for p in group["params"]:
                z = self.state[p].get("z")
                if z is not None:
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - 1.0 / beta1)
        self.train_mode = False
        return self

    def _project(self, entries: list[dict[str, Any]]) -> list[torch.Tensor]:
        if not entries:
            return []
        if not self.batch_project:
            return [self.projector(e["matrix"]) for e in entries]
        out: list[torch.Tensor | None] = [None] * len(entries)
        buckets: dict[tuple[torch.Size, torch.device], list[int]] = {}
        for i, e in enumerate(entries):
            buckets.setdefault((e["matrix"].shape, e["matrix"].device), []).append(i)
        for idxs in buckets.values():
            mats = torch.stack([entries[i]["matrix"] for i in idxs])
            projected = self.projector(mats)
            for local, global_idx in enumerate(idxs):
                out[global_idx] = projected[local]
        return [x for x in out if x is not None]

    def _soda_coeff(self, p: nn.Parameter) -> float:
        for group in self.param_groups:
            if any(id(q) == id(p) for q in group["params"]):
                beta1 = float(group.get("beta1", 0.0))
                ckp1 = float(group.get("ckp1", 1.0))
                return 1.0 - beta1 + beta1 * ckp1
        return 1.0

    def _effective_weight_decay(self, group: dict[str, Any]) -> float:
        if self.use_soda and self.soda_replaces_weight_decay:
            return 0.0
        return float(group.get("weight_decay", 0.0))

    def _capture_soda(self, enabled: bool):
        active = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        prev: dict[nn.Parameter, torch.Tensor] = {}
        if enabled:
            for p in active:
                state = self.state[p]
                if "soda_z0" not in state:
                    state["soda_z0"] = p.detach().float().clone()
                z = state.get("z")
                prev[p] = z.detach().float().clone() if isinstance(z, torch.Tensor) else p.detach().float().clone()
        return active, prev

    @torch.no_grad()
    def _apply_soda(self, enabled: bool, active: list[nn.Parameter], prev: dict[nn.Parameter, torch.Tensor]) -> dict[str, float]:
        if not enabled:
            return {"soda_lambda": 0.0, "soda_applied_count": 0.0}
        k = self.global_step - self.soda_warmup_steps - 1
        lam = 1.0 / float(k + 2)
        applied = 0
        for p in active:
            if p not in prev:
                continue
            state = self.state[p]
            z0 = state["soda_z0"].to(device=p.device)
            corr = z0 - prev[p]
            z = state.get("z")
            if isinstance(z, torch.Tensor):
                z.add_(corr.to(z.dtype), alpha=lam)
                p.add_(corr.to(p.dtype), alpha=lam * self._soda_coeff(p))
            else:
                p.add_(corr.to(p.dtype), alpha=lam)
            applied += 1
        return {"soda_lambda": float(lam), "soda_applied_count": float(applied)}

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            self.train()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1
        collect_stats = self.stats_interval > 0 and (self.global_step == 1 or self.global_step % self.stats_interval == 0)
        soda_enabled = self.use_soda and self.global_step > self.soda_warmup_steps
        soda_active, soda_prev = self._capture_soda(soda_enabled)

        accum = {
            "matrix_count": 0.0,
            "fallback_count": 0.0,
            "mimuon_active_frac": 0.0,
            "normuon_second_moment_mean": 0.0,
            "normuon_applied_count": 0.0,
            "row_update_cv": 0.0,
            "row_dead_frac": 0.0,
            "stiefel_defect": 0.0,
            "pmuoneq_factor_count": 0.0,
            "pmuoneq_left_count": 0.0,
            "pmuoneq_right_count": 0.0,
        }
        betas: list[float] = []
        lrs: list[float] = []

        for group in self.param_groups:
            t = int(group["k"]) + 1
            lr = float(group["base_lr"]) * min(1.0, t / self.warmup_steps)
            group["lr"] = lr
            if self.use_amuse:
                weight = lr * lr
                future_sum = float(group.get("weight_sum", 0.0)) + weight
                ckp1 = weight / future_sum if future_sum > 0.0 else 1.0
                group["weight_sum"] = future_sum
            else:
                ckp1 = 1.0
            group["ckp1"] = ckp1
            beta1 = self._beta1(group, t, ckp1)
            group["beta1"] = beta1
            betas.append(beta1)
            lrs.append(lr)

            if group.get("use_muon", False):
                entries: list[dict[str, Any]] = []
                momentum = float(group["momentum"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    if g.ndim == 4:
                        matrix = g.reshape(g.shape[0], -1)
                    elif g.ndim == 2:
                        matrix = g
                    else:
                        continue
                    state = self.state[p]
                    z = self._z(p)
                    if self.use_amuse and beta1 > 0.0:
                        p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - 1.0 / beta1)
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g, dtype=torch.float32)
                    buf.mul_(momentum).add_(g, alpha=1.0 - momentum)
                    update = torch.lerp(g, buf, momentum)
                    entries.append(
                        {
                            "param": p,
                            "z": z,
                            "grad_matrix": matrix,
                            "matrix": update.reshape(matrix.shape) if g.ndim == 4 else update,
                            "state": state,
                            "shape": tuple(g.shape),
                        }
                    )

                if self.use_pmuoneq:
                    ps = _pmuoneq(
                        entries,
                        beta=self.pmuon_beta,
                        row_gamma=self.pmuon_row_gamma,
                        col_gamma=self.pmuon_col_gamma,
                        sides=self.pmuon_sides,
                        eps=self.pmuon_eps,
                    )
                    for k, v in ps.items():
                        accum[k] += v

                projected = self._project(entries) if self.use_gram else [e["matrix"] for e in entries]
                for entry, proj in zip(entries, projected, strict=True):
                    p = entry["param"]
                    z = entry["z"]
                    wd = self._effective_weight_decay(group)
                    if wd:
                        z.mul_(1.0 - lr * wd)
                    scale = 0.2 * math.sqrt(max(proj.shape[-2], proj.shape[-1]))
                    norm_proj = proj
                    if self.use_normuon:
                        norm_proj = _normuon_update(
                            proj,
                            entry["state"],
                            beta2=self.normuon_beta2,
                            eps=self.normuon_eps,
                            mode=self.normuon_mode,
                            aspect_scale=self.normuon_aspect_scale,
                        )
                        if collect_stats:
                            second = entry["state"].get("normuon_second_moment")
                            if isinstance(second, torch.Tensor):
                                accum["normuon_second_moment_mean"] += float(second.mean().item())
                            accum["normuon_applied_count"] += 1.0
                    if self.use_mimuon:
                        active = entry["matrix"].float().norm() >= self.mimuon_tau
                        final_matrix = torch.where(active, norm_proj * scale, entry["matrix"].float())
                        if collect_stats:
                            accum["mimuon_active_frac"] += float(active.float().item())
                    else:
                        final_matrix = norm_proj * scale
                    update_full = final_matrix.reshape(entry["shape"]).float()
                    z.add_(update_full, alpha=-lr)
                    if self.use_amuse and beta1 > 0.0:
                        p.lerp_(z.to(device=p.device, dtype=p.dtype), ckp1)
                        p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - beta1)
                    else:
                        p.copy_(z.to(device=p.device, dtype=p.dtype))
                    accum["matrix_count"] += 1.0
                    if collect_stats:
                        row_cv, dead = _row_stats(proj)
                        accum["row_update_cv"] += row_cv
                        accum["row_dead_frac"] += dead
                        accum["stiefel_defect"] += _stiefel_defect(proj)
            else:
                beta2 = float(group["beta2"])
                eps = float(group["eps"])
                wd = self._effective_weight_decay(group)
                bc2 = 1.0 - beta2**t
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    state = self.state[p]
                    z = self._z(p)
                    if self.use_amuse and beta1 > 0.0:
                        p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - 1.0 / beta1)
                    v = state.get("exp_avg_sq")
                    if v is None:
                        v = state["exp_avg_sq"] = torch.zeros_like(g, dtype=torch.float32)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    update = g / ((v / bc2).sqrt() + eps)
                    if wd:
                        update = update.add(z.to(update.device), alpha=wd)
                    z.add_(update, alpha=-lr)
                    if self.use_amuse and beta1 > 0.0:
                        p.lerp_(z.to(device=p.device, dtype=p.dtype), ckp1)
                        p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - beta1)
                    else:
                        p.copy_(z.to(device=p.device, dtype=p.dtype))
                    accum["fallback_count"] += 1.0
            group["k"] = t

        soda_stats = self._apply_soda(soda_enabled, soda_active, soda_prev)
        count = max(1.0, accum["matrix_count"])
        self.last_stats = {
            "sodamuseeq_step": float(self.global_step),
            "amuse_beta": float(sum(betas) / max(1, len(betas))),
            "amuse_lr": float(sum(lrs) / max(1, len(lrs))),
            "matrix_count": accum["matrix_count"],
            "fallback_count": accum["fallback_count"],
            "mimuon_active_frac": accum["mimuon_active_frac"] / count if collect_stats else float("nan"),
            "normuon_second_moment_mean": accum["normuon_second_moment_mean"] / max(1.0, accum["normuon_applied_count"]) if collect_stats else float("nan"),
            "normuon_applied_count": accum["normuon_applied_count"] if collect_stats else float("nan"),
            "row_update_cv": accum["row_update_cv"] / count if collect_stats else float("nan"),
            "row_dead_frac": accum["row_dead_frac"] / count if collect_stats else float("nan"),
            "stiefel_defect": accum["stiefel_defect"] / count if collect_stats else float("nan"),
            "pmuoneq_factor_count": accum["pmuoneq_factor_count"],
            "pmuoneq_left_count": accum["pmuoneq_left_count"],
            "pmuoneq_right_count": accum["pmuoneq_right_count"],
            "soda_replaces_weight_decay": float(self.use_soda and self.soda_replaces_weight_decay),
            **soda_stats,
        }
        return loss

    def get_last_stats(self) -> dict[str, float]:
        return dict(self.last_stats)


EquiMuse = SodaMuseEq

__all__ = ["SodaMuseEq", "EquiMuse", "GramNewtonSchulzProjector", "make_sodamuseeq_param_groups"]
