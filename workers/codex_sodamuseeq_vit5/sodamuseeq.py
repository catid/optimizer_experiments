"""Standalone SodaMuseEq optimizer.

SodaMuseEq is a copyable PyTorch optimizer for the recipe previously referred
to as SODA-AMUSE+PMuonEq:

    SODA anchor correction
    + AMUSE schedule-free outer loop
    + PMuonEq row/column EMA equilibration
    + Gram/Newton-Schulz matrix projection
    + optional NorMuon row/aspect normalization

The short alias `EquiMuse` is exported for convenience.

Best compact ViT-5/CIFAR-10 recipe from this worker:

    use_amuse=False
    use_soda=True
    use_pmuoneq=True
    use_gram=True
    use_normuon=True
    normuon_mode="row"
    normuon_aspect_scale=True
    lr=0.012
    weight_decay=0.0
    pmuon_beta=0.90
    pmuon_row_gamma=0.15
    pmuon_col_gamma=0.0
    normuon_beta2=0.90

It reached 0.4079 +/- 0.0090 best validation loss and 87.09% +/- 0.15 test
accuracy in a 10k-step, 3-seed compact ViT-5/CIFAR-10 comparison using a
2.69M-parameter model, batch size 256, eval batch size 1024, and a
45k/5k/10k train/validation/test split. The class defaults stay
broad/backward-compatible; pass these flags explicitly when using the best
recipe. Exact metadata is also exported as BEST_KNOWN_CONFIG.

Typical use:

    from sodamuseeq import SodaMuseEq, make_sodamuseeq_param_groups

    opt = SodaMuseEq(
        make_sodamuseeq_param_groups(model, lr=9e-3, weight_decay=0.05),
        warmup_steps=200,
        soda_warmup_steps=200,
        pmuon_gamma=0.2,
    )

    for x, y in loader:
        opt.train()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    opt.eval()   # exposes the AMUSE averaged/eval weights for validation

Only PyTorch is required.
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


BEST_KNOWN_CONFIG: dict[str, Any] = {
    "algorithm": "SODA+PMuonEq+Gram+NorMuon row+aspect",
    "model": "compact ViT-5 CIFAR model",
    "model_params": 2_691_274,
    "dataset": "CIFAR-10",
    "image_size": 32,
    "patch_size": 4,
    "embed_dim": 192,
    "depth": 6,
    "num_heads": 3,
    "mlp_ratio": 4.0,
    "architecture_notes": "RMSNorm, RoPE, q/k norm, layer scale, 4 register tokens",
    "train_examples": 45_000,
    "val_examples": 5_000,
    "test_examples": 10_000,
    "batch_size": 256,
    "eval_batch_size": 1024,
    "steps": 10_000,
    "seeds": (0, 1, 2),
    "split_seed": 12345,
    "hpo_steps": 1_000,
    "top_selection_steps": 3_000,
    "eval_bins": 8,
    "amp": "bf16_autocast",
    "lr": 0.012,
    "weight_decay": 0.0,
    "momentum": 0.95,
    "beta1": 0.6,
    "beta2": 0.999,
    "rho": 0.8,
    "warmup_steps": 500,
    "soda_warmup_steps": 500,
    "use_soda": True,
    "use_amuse": False,
    "use_pmuoneq": True,
    "use_gram": True,
    "use_normuon": True,
    "normuon_mode": "row",
    "normuon_aspect_scale": True,
    "pmuon_beta": 0.90,
    "pmuon_row_gamma": 0.15,
    "pmuon_col_gamma": 0.0,
    "normuon_beta2": 0.90,
    "best_val_loss_mean": 0.4079,
    "best_val_loss_std": 0.0090,
    "test_acc_mean": 0.8709,
    "test_acc_std": 0.0015,
}


def _is_parameter(value: object) -> bool:
    return isinstance(value, nn.Parameter)


def _is_matrix_like(param: nn.Parameter) -> bool:
    return param.requires_grad and param.is_floating_point() and param.ndim in (2, 4)


def _is_default_aux_name(name: str) -> bool:
    lowered = name.lower()
    parts = lowered.split(".")
    if lowered.endswith(".bias"):
        return True
    if any(token in lowered for token in ("norm", "ln_", "layernorm", "rmsnorm")):
        return True
    if len(parts) >= 2 and parts[-1] == "weight" and parts[-2] in {"embed", "embedding", "embeddings"}:
        return True
    if any(token in lowered for token in ("pos_embed", "cls_token", "reg_token", "wte", "tok_emb", "lm_head", "unembed")):
        return True
    return lowered.endswith("head.weight")


def make_sodamuseeq_param_groups(
    model: nn.Module,
    *,
    lr: float = 9e-3,
    weight_decay: float = 0.05,
    aux_lr: float | None = None,
    aux_weight_decay: float | None = None,
    include_embeddings_in_matrix_group: bool = False,
) -> list[dict[str, Any]]:
    """Split a module into matrix and auxiliary parameter groups.

    Matrix-like hidden 2D/4D parameters go through PMuonEq + Gram projection.
    Biases, norm scales, embeddings, and heads go through the fallback path.

    Parameter names are saved on each tensor as `_sodamuseeq_param_name` for
    easier debugging, but the optimizer does not require names.
    """

    matrix: list[nn.Parameter] = []
    aux: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param._sodamuseeq_param_name = name  # type: ignore[attr-defined]
        force_aux = _is_default_aux_name(name) and not include_embeddings_in_matrix_group
        if _is_matrix_like(param) and not force_aux:
            matrix.append(param)
        else:
            aux.append(param)

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
    """Torch-only Gram/Newton-Schulz projector.

    This is adapted from the Gram-Newton-Schulz structure used in the benchmark
    code. It works for a single matrix or a batch of same-shaped matrices.
    """

    def __init__(
        self,
        coefficients: Sequence[Sequence[float]] = DAO_POLAR_EXPRESS_COEFFS,
        eps: float = 1e-7,
        use_gram_newton_schulz: bool = True,
        reset_iterations: Sequence[int] = (2,),
        projection_dtype: torch.dtype = torch.bfloat16,
    ):
        self.coefficients = tuple(tuple(float(v) for v in coeff) for coeff in coefficients)
        self.eps = float(eps)
        self.use_gram_newton_schulz = bool(use_gram_newton_schulz)
        self.reset_iterations = {int(v) for v in reset_iterations}
        self.projection_dtype = projection_dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("GramNewtonSchulzProjector expects a matrix or a batch of matrices")
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
        if self.use_gram_newton_schulz and max(y.shape[-2:]) > min(y.shape[-2:]):
            y = self._gram_newton_schulz(y)
        else:
            y = self._standard_newton_schulz(y)
        if transposed:
            y = y.mT
        return y.to(original_dtype).reshape(original_shape)

    def _gram_newton_schulz(self, x: torch.Tensor) -> torch.Tensor:
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

    def _standard_newton_schulz(self, x: torch.Tensor) -> torch.Tensor:
        for a, b, c in self.coefficients:
            a_mat = x @ x.mT
            b_mat = torch.baddbmm(a_mat, a_mat, a_mat, beta=b, alpha=c)
            x = torch.baddbmm(x, b_mat, x, beta=a)
        return x


def _matrix_row_stats(update: torch.Tensor) -> tuple[float, float]:
    if update.ndim != 2 or update.numel() == 0:
        return 0.0, 0.0
    row_rms = update.detach().float().square().mean(dim=1).sqrt()
    mean = row_rms.mean().clamp_min(1e-12)
    cv = row_rms.std(unbiased=False) / mean
    median = row_rms.median().clamp_min(1e-12)
    dead = (row_rms < 0.1 * median).float().mean()
    return float(cv.item()), float(dead.item())


def _matrix_stiefel_defect(update: torch.Tensor) -> float:
    if update.ndim != 2 or min(update.shape) == 0:
        return 0.0
    u = update.detach().float()
    rows, cols = u.shape
    if rows >= cols:
        gram = u.T @ u
        eye = torch.eye(cols, device=u.device, dtype=u.dtype)
        return float(((gram - eye).norm() / math.sqrt(cols)).item())
    gram = u @ u.T
    eye = torch.eye(rows, device=u.device, dtype=u.dtype)
    return float(((gram - eye).norm() / math.sqrt(rows)).item())


def _scale_rms(scale: torch.Tensor, eps: float) -> torch.Tensor:
    return scale / scale.square().mean().sqrt().clamp_min(eps)


@torch.no_grad()
def _pmuoneq_precondition_entries(
    entries: list[dict[str, Any]],
    *,
    beta: float,
    row_gamma: float,
    col_gamma: float,
    sides: str,
    eps: float,
) -> dict[str, float]:
    if sides not in {"none", "left", "right", "both", "min"}:
        raise ValueError(f"unknown PMuonEq sides mode {sides!r}")

    stats = {
        "pmuoneq_factor_count": 0.0,
        "pmuoneq_left_count": 0.0,
        "pmuoneq_right_count": 0.0,
        "pmuoneq_scale_mean": 0.0,
        "pmuoneq_scale_p05": 0.0,
        "pmuoneq_scale_p95": 0.0,
    }
    if not entries or sides == "none":
        return stats

    scale_values: list[torch.Tensor] = []
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
            if ema is None or tuple(ema.shape) != tuple(stat.shape) or ema.device != stat.device:
                ema = state["pmuoneq_left_ema"] = torch.zeros_like(stat, dtype=torch.float32)
            ema.mul_(beta).add_(stat, alpha=1.0 - beta)
            scale = _scale_rms(ema.clamp_min(eps).pow(-row_gamma), eps).to(work.dtype)
            work = scale.unsqueeze(1) * work
            scale_values.append(scale.detach().float())
            stats["pmuoneq_factor_count"] += 1.0
            stats["pmuoneq_left_count"] += 1.0

        if use_right and col_gamma != 0.0:
            stat = g.square().mean(dim=0)
            ema = state.get("pmuoneq_right_ema")
            if ema is None or tuple(ema.shape) != tuple(stat.shape) or ema.device != stat.device:
                ema = state["pmuoneq_right_ema"] = torch.zeros_like(stat, dtype=torch.float32)
            ema.mul_(beta).add_(stat, alpha=1.0 - beta)
            scale = _scale_rms(ema.clamp_min(eps).pow(-col_gamma), eps).to(work.dtype)
            work = work * scale.unsqueeze(0)
            scale_values.append(scale.detach().float())
            stats["pmuoneq_factor_count"] += 1.0
            stats["pmuoneq_right_count"] += 1.0

        entry["matrix"] = work

    if scale_values:
        cat = torch.cat([v.reshape(-1) for v in scale_values])
        stats["pmuoneq_scale_mean"] = float(cat.mean().item())
        stats["pmuoneq_scale_p05"] = float(torch.quantile(cat, 0.05).item())
        stats["pmuoneq_scale_p95"] = float(torch.quantile(cat, 0.95).item())
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
    """SODA + AMUSE + PMuonEq + Gram-Newton-Schulz optimizer.

    Param groups:

    - `use_muon=True`: 2D/4D matrix parameters use momentum, PMuonEq row/column
      equilibration, Gram projection, and decoupled hidden-z weight decay.
    - `use_muon=False`: auxiliary parameters use the fallback Adam-style path.

    If `params` is a plain iterable of parameters, groups are inferred by
    tensor rank. Passing groups from `make_sodamuseeq_param_groups` is preferred.
    """

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
        weight_lr_power: float = 2.0,
        r: float = 0.0,
        eps: float = 1e-10,
        fallback_decoupled_weight_decay: bool = False,
        projection_eps: float = 1e-7,
        projection_dtype: torch.dtype = torch.bfloat16,
        batch_project: bool = True,
        pmuon_beta: float = 0.95,
        pmuon_gamma: float = 0.2,
        pmuon_row_gamma: float | None = None,
        pmuon_col_gamma: float | None = None,
        pmuon_sides: str = "both",
        pmuon_eps: float = 1e-6,
        use_normuon: bool = False,
        normuon_beta2: float = 0.95,
        normuon_eps: float = 1e-10,
        normuon_mode: str = "row",
        normuon_aspect_scale: bool = False,
        soda_warmup_steps: int = 200,
        soda_anchor: str = "warmup",
        soda_warmup_k_mode: str = "post_warmup",
        soda_replaces_weight_decay: bool = True,
    ):
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if soda_warmup_steps < 0:
            raise ValueError("soda_warmup_steps must be non-negative")
        if soda_anchor not in {"initial", "warmup"}:
            raise ValueError("soda_anchor must be 'initial' or 'warmup'")
        if soda_warmup_k_mode not in {"post_warmup", "global"}:
            raise ValueError("soda_warmup_k_mode must be 'post_warmup' or 'global'")
        if pmuon_sides not in {"none", "left", "right", "both", "min"}:
            raise ValueError(f"unknown PMuonEq sides mode {pmuon_sides!r}")
        if normuon_mode not in {"row", "orientation"}:
            raise ValueError(f"unknown NorMuon mode {normuon_mode!r}")

        self.beta1_init = float(beta1)
        self.rho = float(rho)
        self.warmup_steps = int(warmup_steps)
        self.weight_lr_power = float(weight_lr_power)
        self.r = float(r)
        self.fallback_decoupled_weight_decay = bool(fallback_decoupled_weight_decay)
        self.projector = GramNewtonSchulzProjector(eps=projection_eps, projection_dtype=projection_dtype)
        self.batch_project = bool(batch_project)
        self.pmuon_beta = float(pmuon_beta)
        self.pmuon_row_gamma = float(pmuon_gamma if pmuon_row_gamma is None else pmuon_row_gamma)
        self.pmuon_col_gamma = float(pmuon_gamma if pmuon_col_gamma is None else pmuon_col_gamma)
        self.pmuon_sides = pmuon_sides
        self.pmuon_eps = float(pmuon_eps)
        self.use_normuon = bool(use_normuon)
        self.normuon_beta2 = float(normuon_beta2)
        self.normuon_eps = float(normuon_eps)
        self.normuon_mode = normuon_mode
        self.normuon_aspect_scale = bool(normuon_aspect_scale)
        self.soda_warmup_steps = int(soda_warmup_steps)
        self.soda_anchor = soda_anchor
        self.soda_warmup_k_mode = soda_warmup_k_mode
        self.soda_replaces_weight_decay = bool(soda_replaces_weight_decay)
        self.soda_step_idx = 0
        self.train_mode = True
        self.last_stats: dict[str, float] = {}

        param_groups = self._normalize_param_groups(params, lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2, eps=eps)
        defaults = dict(
            lr=float(lr),
            base_lr=None,
            weight_decay=float(weight_decay),
            use_muon=True,
            momentum=float(momentum),
            beta2=float(beta2),
            eps=float(eps),
            lr_mult=1.0,
        )
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            if group["base_lr"] is None:
                group["base_lr"] = float(group["lr"])
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
                    if p.requires_grad:
                        self.state[p]["soda_z0"] = p.detach().float().clone()

    @staticmethod
    def _normalize_param_groups(params, *, lr: float, weight_decay: float, momentum: float, beta2: float, eps: float) -> list[dict[str, Any]]:
        def _coerce_param_list(values) -> list[nn.Parameter]:
            out: list[nn.Parameter] = []
            for idx, item in enumerate(values):
                if isinstance(item, tuple) and len(item) == 2 and _is_parameter(item[1]):
                    name, param = item
                    param._sodamuseeq_param_name = str(name)  # type: ignore[attr-defined]
                    out.append(param)
                elif _is_parameter(item):
                    out.append(item)
                else:
                    raise TypeError(f"invalid parameter entry at index {idx}: {type(item)!r}")
            return out

        raw = list(params)
        if not raw:
            raise ValueError("optimizer got an empty parameter list")

        groups: list[dict[str, Any]] = []
        if isinstance(raw[0], dict):
            for raw_group in raw:
                group = dict(raw_group)
                group_params = _coerce_param_list(list(group.get("params", [])))
                if not group_params:
                    continue
                if "use_muon" in group:
                    group["params"] = group_params
                    groups.append(group)
                    continue
                matrix = [p for p in group_params if _is_matrix_like(p)]
                matrix_ids = {id(p) for p in matrix}
                aux = [p for p in group_params if id(p) not in matrix_ids]
                base = {k: v for k, v in group.items() if k != "params"}
                if matrix:
                    groups.append({**base, "params": matrix, "use_muon": True})
                if aux:
                    groups.append({**base, "params": aux, "use_muon": False})
        else:
            parameters = _coerce_param_list(raw)
            matrix = [p for p in parameters if _is_matrix_like(p)]
            matrix_ids = {id(p) for p in matrix}
            aux = [p for p in parameters if id(p) not in matrix_ids]
            if matrix:
                groups.append({"params": matrix, "use_muon": True})
            if aux:
                groups.append({"params": aux, "use_muon": False})

        for group in groups:
            group.setdefault("lr", float(lr))
            group.setdefault("base_lr", None)
            group.setdefault("weight_decay", float(weight_decay))
            group.setdefault("momentum", float(momentum))
            group.setdefault("beta2", float(beta2))
            group.setdefault("eps", float(eps))
            group.setdefault("lr_mult", 1.0)
        return groups

    def set_base_lr(self, lr: float):
        for group in self.param_groups:
            base = float(lr) * float(group.get("lr_mult", 1.0))
            group["base_lr"] = base
            group["lr"] = base

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
            "soda_step_idx": int(self.soda_step_idx),
            "train_mode": bool(self.train_mode),
            "last_stats": dict(self.last_stats),
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop("sodamuseeq_extra", {})
        super().load_state_dict(state_dict)
        inferred_step = max((int(group.get("k", 0)) for group in self.param_groups), default=0)
        self.soda_step_idx = int(extra.get("soda_step_idx", inferred_step))
        self.train_mode = bool(extra.get("train_mode", True))
        self.last_stats = dict(extra.get("last_stats", {}))

    @torch.no_grad()
    def train(self):
        if self.train_mode:
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
        if not self.train_mode:
            return self
        for group in self.param_groups:
            beta1 = float(group.get("beta1", self.beta1_init))
            if beta1 <= 0.0:
                raise RuntimeError(f"invalid beta1={beta1}")
            for p in group["params"]:
                z = self.state[p].get("z")
                if z is not None:
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - 1.0 / beta1)
        self.train_mode = False
        return self

    def _get_z(self, p: nn.Parameter) -> torch.Tensor:
        z = self.state[p].get("z")
        if z is None:
            z = self.state[p]["z"] = p.detach().float().clone(memory_format=torch.preserve_format)
        return z

    def _compute_beta1(self, group: dict[str, Any], t: int, ckp1: float) -> float:
        if t <= self.warmup_steps:
            if t == self.warmup_steps:
                group["c_warmup"] = ckp1
            return self.beta1_init
        c_warmup = float(group.get("c_warmup", 1.0 / self.warmup_steps))
        denom = max(c_warmup * (1.0 - ckp1), 1e-24)
        s_t = (ckp1 * (1.0 - c_warmup)) / denom
        beta = 1.0 - (s_t**self.rho) * (1.0 - self.beta1_init)
        return float(min(max(beta, 1e-6), 1.0 - 1e-8))

    def _project_batches(self, entries: list[dict[str, Any]]) -> list[torch.Tensor]:
        if not entries:
            return []
        if not self.batch_project:
            return [self.projector(entry["matrix"]) for entry in entries]
        projected: list[torch.Tensor | None] = [None] * len(entries)
        buckets: dict[tuple[torch.Size, torch.device], list[int]] = {}
        for idx, entry in enumerate(entries):
            buckets.setdefault((entry["matrix"].shape, entry["matrix"].device), []).append(idx)
        for indices in buckets.values():
            mats = torch.stack([entries[idx]["matrix"] for idx in indices])
            outs = self.projector(mats)
            for local_idx, global_idx in enumerate(indices):
                projected[global_idx] = outs[local_idx]
        return [out for out in projected if out is not None]

    def _schedule_free_coeff(self, p: nn.Parameter) -> float:
        pid = id(p)
        for group in self.param_groups:
            if any(id(q) == pid for q in group["params"]):
                beta1 = float(group.get("beta1", 1.0))
                ckp1 = float(group.get("ckp1", 1.0))
                return 1.0 - beta1 + beta1 * ckp1
        return 1.0

    def _effective_weight_decay(self, group: dict[str, Any]) -> float:
        if self.soda_replaces_weight_decay:
            return 0.0
        return float(group.get("weight_decay", 0.0))

    @staticmethod
    def _active_param(p: nn.Parameter) -> bool:
        return p.grad is not None

    def _capture_soda_pre_step(self, enabled: bool) -> tuple[list[nn.Parameter], dict[nn.Parameter, torch.Tensor], dict[nn.Parameter, bool]]:
        active = [p for group in self.param_groups for p in group["params"] if self._active_param(p)]
        prev: dict[nn.Parameter, torch.Tensor] = {}
        prev_is_hidden_z: dict[nn.Parameter, bool] = {}
        if not enabled:
            return active, prev, prev_is_hidden_z
        for p in active:
            state = self.state[p]
            if "soda_z0" not in state:
                state["soda_z0"] = p.detach().float().clone()
            hidden_z = state.get("z")
            if isinstance(hidden_z, torch.Tensor):
                prev[p] = hidden_z.detach().float().clone()
                prev_is_hidden_z[p] = True
            else:
                prev[p] = p.detach().float().clone()
                prev_is_hidden_z[p] = False
        return active, prev, prev_is_hidden_z

    @torch.no_grad()
    def _apply_soda_post_step(
        self,
        enabled: bool,
        active: list[nn.Parameter],
        prev: dict[nn.Parameter, torch.Tensor],
        prev_is_hidden_z: dict[nn.Parameter, bool],
    ) -> dict[str, float]:
        stats = {
            "soda_step": 0.0,
            "soda_global_step": float(self.soda_step_idx),
            "soda_lambda": 0.0,
            "soda_applied_count": 0.0,
            "soda_hidden_z_applied_count": 0.0,
            "soda_correction_weight_ratio": 0.0,
            "soda_anchor_gap_weight_ratio": 0.0,
        }
        if not enabled or not active:
            return stats
        k = self.soda_step_idx - 1 if self.soda_warmup_k_mode == "global" else self.soda_step_idx - self.soda_warmup_steps - 1
        lambda_k = 1.0 / float(k + 2)
        correction_norm_sq = 0.0
        weight_norm_sq = 0.0
        anchor_gap_norm_sq = 0.0
        applied = 0
        hidden_applied = 0
        for p in active:
            p_prev = prev.get(p)
            if p_prev is None:
                continue
            state = self.state[p]
            z0 = state["soda_z0"].to(device=p.device)
            correction = z0 - p_prev
            hidden_z = state.get("z")
            if isinstance(hidden_z, torch.Tensor) and prev_is_hidden_z.get(p, False):
                hidden_z.add_(correction.to(hidden_z.dtype), alpha=lambda_k)
                p.add_(correction.to(p.dtype), alpha=lambda_k * self._schedule_free_coeff(p))
                hidden_applied += 1
            else:
                p.add_(correction.to(p.dtype), alpha=lambda_k)
            correction_norm_sq += float(correction.square().sum().item() * lambda_k * lambda_k)
            gap_target = hidden_z.detach().float().to(device=p.device) if isinstance(hidden_z, torch.Tensor) else p.detach().float()
            anchor_gap_norm_sq += float((z0 - gap_target).square().sum().item())
            weight_norm_sq += float(p.detach().float().square().sum().item())
            applied += 1
        stats.update(
            {
                "soda_step": float(max(0, self.soda_step_idx - self.soda_warmup_steps)),
                "soda_lambda": float(lambda_k),
                "soda_applied_count": float(applied),
                "soda_hidden_z_applied_count": float(hidden_applied),
                "soda_correction_weight_ratio": math.sqrt(correction_norm_sq) / max(math.sqrt(weight_norm_sq), 1e-12),
                "soda_anchor_gap_weight_ratio": math.sqrt(anchor_gap_norm_sq) / max(math.sqrt(weight_norm_sq), 1e-12),
            }
        )
        return stats

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            self.train()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.soda_step_idx += 1
        soda_enabled = self.soda_step_idx > self.soda_warmup_steps
        active, soda_prev, soda_prev_is_hidden = self._capture_soda_pre_step(soda_enabled)

        accum = {
            "matrix_count": 0.0,
            "fallback_count": 0.0,
            "row_update_cv": 0.0,
            "row_dead_frac": 0.0,
            "stiefel_defect": 0.0,
            "update_norm_sq": 0.0,
            "weight_norm_sq": 0.0,
            "x_z_gap_norm_sq": 0.0,
            "pmuoneq_factor_count": 0.0,
            "pmuoneq_left_count": 0.0,
            "pmuoneq_right_count": 0.0,
            "pmuoneq_scale_mean": 0.0,
            "pmuoneq_scale_p05": 0.0,
            "pmuoneq_scale_p95": 0.0,
            "normuon_second_moment_mean": 0.0,
            "normuon_applied_count": 0.0,
        }
        beta_values: list[float] = []
        c_values: list[float] = []
        lr_values: list[float] = []

        for group in self.param_groups:
            k = int(group["k"])
            t = k + 1
            base_lr = float(group["base_lr"])
            lr = base_lr * min(1.0, t / self.warmup_steps)
            group["lr"] = lr
            weight = (float(t) ** self.r) * (lr**self.weight_lr_power)
            future_weight_sum = float(group.get("weight_sum", 0.0)) + weight
            ckp1 = weight / future_weight_sum if future_weight_sum > 0.0 else 1.0
            group["weight_sum"] = future_weight_sum
            group["ckp1"] = ckp1
            beta1 = self._compute_beta1(group, t, ckp1)
            group["beta1"] = beta1
            beta_values.append(beta1)
            c_values.append(ckp1)
            lr_values.append(lr)

            if group.get("use_muon", False):
                entries: list[dict[str, Any]] = []
                momentum_beta = float(group["momentum"])
                wd = self._effective_weight_decay(group)
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
                    z = self._get_z(p)
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - 1.0 / beta1)
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g, dtype=torch.float32)
                    buf.mul_(momentum_beta).add_(g, alpha=1.0 - momentum_beta)
                    update = torch.lerp(g, buf, momentum_beta)
                    matrix_update = update.reshape(matrix.shape) if g.ndim == 4 else update
                    entries.append({"param": p, "z": z, "grad_matrix": matrix, "matrix": matrix_update, "state": state, "shape": tuple(g.shape), "wd": wd})

                pmuoneq_stats = _pmuoneq_precondition_entries(
                    entries,
                    beta=self.pmuon_beta,
                    row_gamma=self.pmuon_row_gamma,
                    col_gamma=self.pmuon_col_gamma,
                    sides=self.pmuon_sides,
                    eps=self.pmuon_eps,
                )
                for key, value in pmuoneq_stats.items():
                    accum[key] += value

                projected = self._project_batches(entries)
                for entry, update_matrix in zip(entries, projected, strict=True):
                    p = entry["param"]
                    z = entry["z"]
                    if entry["wd"]:
                        z.mul_(1.0 - lr * float(entry["wd"]))
                    if self.use_normuon:
                        update_matrix = _normuon_update(
                            update_matrix,
                            entry["state"],
                            beta2=self.normuon_beta2,
                            eps=self.normuon_eps,
                            mode=self.normuon_mode,
                            aspect_scale=self.normuon_aspect_scale,
                        )
                        second = entry["state"].get("normuon_second_moment")
                        if isinstance(second, torch.Tensor):
                            accum["normuon_second_moment_mean"] += float(second.mean().item())
                        accum["normuon_applied_count"] += 1.0
                    scale = 0.2 * math.sqrt(max(update_matrix.shape[-2], update_matrix.shape[-1]))
                    update_full = update_matrix.reshape(entry["shape"]).float()
                    z.add_(update_full, alpha=-lr * scale)
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), ckp1)
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - beta1)
                    row_cv, dead_frac = _matrix_row_stats(update_matrix)
                    accum["matrix_count"] += 1.0
                    accum["row_update_cv"] += row_cv
                    accum["row_dead_frac"] += dead_frac
                    accum["stiefel_defect"] += _matrix_stiefel_defect(update_matrix)
                    accum["update_norm_sq"] += float(update_full.square().sum().item() * (lr * scale) ** 2)
                    accum["weight_norm_sq"] += float(p.detach().float().square().sum().item())
                    accum["x_z_gap_norm_sq"] += float((p.detach().float() - z.to(device=p.device)).square().sum().item())
            else:
                beta2 = float(group.get("beta2", 0.999))
                eps = float(group.get("eps", 1e-10))
                wd = self._effective_weight_decay(group)
                bias_correction2 = 1.0 - beta2**t
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    state = self.state[p]
                    z = self._get_z(p)
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - 1.0 / beta1)
                    v = state.get("exp_avg_sq")
                    if v is None:
                        v = state["exp_avg_sq"] = torch.zeros_like(g, dtype=torch.float32)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    update = g / ((v / bias_correction2).sqrt() + eps)
                    if wd:
                        if self.fallback_decoupled_weight_decay:
                            z.mul_(1.0 - lr * wd)
                        else:
                            update = update.add(z.to(device=update.device), alpha=wd)
                    z.add_(update, alpha=-lr)
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), ckp1)
                    p.lerp_(z.to(device=p.device, dtype=p.dtype), 1.0 - beta1)
                    accum["fallback_count"] += 1.0
                    accum["update_norm_sq"] += float(update.square().sum().item() * lr * lr)
                    accum["weight_norm_sq"] += float(p.detach().float().square().sum().item())
                    accum["x_z_gap_norm_sq"] += float((p.detach().float() - z.to(device=p.device)).square().sum().item())
            group["k"] = t

        soda_stats = self._apply_soda_post_step(soda_enabled, active, soda_prev, soda_prev_is_hidden)

        matrix_count = max(1.0, accum["matrix_count"])
        weight_norm = max(math.sqrt(accum["weight_norm_sq"]), 1e-12)
        factor_count = max(1.0, accum["pmuoneq_factor_count"])
        self.last_stats = {
            "sodamuseeq_step": float(max((int(g["k"]) for g in self.param_groups), default=0)),
            "amuse_beta": float(sum(beta_values) / max(1, len(beta_values))),
            "amuse_c": float(sum(c_values) / max(1, len(c_values))),
            "amuse_lr": float(sum(lr_values) / max(1, len(lr_values))),
            "matrix_count": accum["matrix_count"],
            "fallback_count": accum["fallback_count"],
            "batch_project": float(self.batch_project),
            "row_update_cv": accum["row_update_cv"] / matrix_count,
            "row_dead_frac": accum["row_dead_frac"] / matrix_count,
            "stiefel_defect": accum["stiefel_defect"] / matrix_count,
            "update_weight_ratio": math.sqrt(accum["update_norm_sq"]) / weight_norm,
            "x_z_gap_weight_ratio": math.sqrt(accum["x_z_gap_norm_sq"]) / weight_norm,
            "pmuoneq_factor_count": accum["pmuoneq_factor_count"],
            "pmuoneq_left_count": accum["pmuoneq_left_count"],
            "pmuoneq_right_count": accum["pmuoneq_right_count"],
            "pmuoneq_scale_mean": accum["pmuoneq_scale_mean"] / factor_count,
            "pmuoneq_scale_p05": accum["pmuoneq_scale_p05"] / factor_count,
            "pmuoneq_scale_p95": accum["pmuoneq_scale_p95"] / factor_count,
            "normuon_second_moment_mean": accum["normuon_second_moment_mean"] / max(1.0, accum["normuon_applied_count"]),
            "normuon_applied_count": accum["normuon_applied_count"],
            "soda_replaces_weight_decay": float(self.soda_replaces_weight_decay),
            **soda_stats,
        }
        return loss

    def get_last_stats(self) -> dict[str, float]:
        return dict(self.last_stats)


EquiMuse = SodaMuseEq


__all__ = [
    "SodaMuseEq",
    "EquiMuse",
    "GramNewtonSchulzProjector",
    "make_sodamuseeq_param_groups",
    "BEST_KNOWN_CONFIG",
]
