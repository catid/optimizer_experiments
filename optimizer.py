"""Configurable SODA + PMuonEq + GramNS + NorMuon optimizer.

This root-level file is the handoff optimizer for the next experiment round.
It implements the direct, no-AMUSE optimizer family that all worker results
converged toward:

    SODA anchor pull
    + PMuonEq row/column diagonal gradient-power preconditioning
    + Gram Newton-Schulz matrix direction
    + NorMuon post-projection normalization
    + RMS/AdamW-style fallback for non-matrix parameters

Default policy is the conservative language-model candidate selected from the
cross-worker comparison:

    row-only PMuonEq, no aspect multiplier

The known disagreement options are configurable:

    col_gamma > 0
        Enables column PMuonEq. This helped some ViT/CIFAR proxy runs.

    normuon_aspect_scale=True
        Enables the post-NorMuon sqrt(max(1, rows / cols)) multiplier. This won
        some proxy runs but lost the cleanest held-out official-test protocol.

    normuon_mode="orientation"
        Uses row statistics for tall matrices and column statistics for wide
        matrices. Default "row" always uses output-row statistics.

    ns_variant="polar_express" or "classic_muon"
        Selects the Gram Newton-Schulz coefficient family.

Default numeric starting point:

    matrix_lr = 8e-3
    fallback_lr = matrix_lr
    row_gamma = 0.35
    col_gamma = 0.0
    normuon_beta2 = 0.93
    warmup_steps = 80

Preferred use:

    optimizer = SodaPmuonEqNorMuon(model)

Passing a ``torch.nn.Module`` or ``model.named_parameters()`` lets the optimizer
keep embeddings, tied heads, norms, biases, vectors, and tiny tensors on the
fallback path. Passing raw ``model.parameters()`` is supported, but names are not
available, so all matrix-like tensors are routed by shape only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import torch

__version__ = "0.3.0"

try:
    from torch.optim.optimizer import ParamsT
except ImportError:  # pragma: no cover
    ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


POLAR_EXPRESS_UNSCALED: tuple[tuple[float, float, float], ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
POLAR_EXPRESS_SAFETY_FACTOR = 1.05
POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = tuple(
    (
        a / POLAR_EXPRESS_SAFETY_FACTOR,
        b / POLAR_EXPRESS_SAFETY_FACTOR**3,
        c / POLAR_EXPRESS_SAFETY_FACTOR**5,
    )
    for a, b, c in POLAR_EXPRESS_UNSCALED
)
CLASSIC_MUON_COEFFICIENT = (3.4445, -4.7750, 2.0315)


def _is_default_fallback_name(name: str) -> bool:
    lower = name.lower().replace("/", ".")
    parts = [part for part in lower.split(".") if part]
    if not parts:
        return False
    leaf = parts[-1]
    if leaf == "bias" or lower.endswith(".bias"):
        return True

    norm_parts = {
        "norm",
        "ln",
        "bn",
        "rmsnorm",
        "layernorm",
        "batchnorm",
        "groupnorm",
        "final_norm",
    }
    if any(part in norm_parts or part.endswith("norm") for part in parts):
        return True

    embed_parts = {
        "embed",
        "embedding",
        "embeddings",
        "token_embed",
        "token_embedding",
        "tok_embeddings",
        "word_embedding",
        "word_embeddings",
        "wte",
        "wpe",
        "pos_embed",
        "position_embeddings",
        "cls_token",
        "reg_token",
    }
    if any(part in embed_parts for part in parts):
        return True

    head_tokens = ("lm_head", "classifier_head", "unembed", "output_projection")
    if any(token in lower for token in head_tokens):
        return True
    return lower == "head.weight" or (lower.endswith(".head.weight") and "attn" not in lower)


def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("matrix path requires tensors with ndim >= 2")
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def _normuon_second_shape(rows: int, cols: int, mode: str) -> tuple[int, int]:
    if mode == "row":
        return (rows, 1)
    if mode == "orientation":
        return (rows, 1) if rows >= cols else (1, cols)
    raise ValueError(f"unknown normuon_mode {mode!r}")


def _normalized_inverse_power(values: torch.Tensor, *, gamma: float, eps: float) -> torch.Tensor:
    if float(gamma) == 0.0:
        return torch.ones_like(values, dtype=torch.float32)
    x = values.float()
    scale = x.mean(dim=-1, keepdim=True).clamp_min(1.0) if x.ndim > 1 else x.mean().clamp_min(1.0)
    lam = (x + eps * scale).clamp_min(eps * scale)
    out = lam.pow(-float(gamma))
    norm = out.norm(dim=-1, keepdim=True).clamp_min(eps) if out.ndim > 1 else out.norm().clamp_min(eps)
    return out * (math.sqrt(out.size(-1)) / norm)


class GramNewtonSchulz:
    """Batched Gram Newton-Schulz polar approximation."""

    def __init__(
        self,
        *,
        variant: str = "polar_express",
        steps: int = 5,
        epsilon: float = 1e-7,
        compute_dtype: torch.dtype | None = None,
        reset_iterations: Iterable[int] = (2,),
    ) -> None:
        if variant not in {"polar_express", "classic_muon"}:
            raise ValueError("ns_variant must be 'polar_express' or 'classic_muon'")
        if int(steps) <= 0:
            raise ValueError("ns_steps must be positive")
        self.variant = variant
        self.steps = int(steps)
        self.epsilon = float(epsilon)
        self.compute_dtype = compute_dtype
        self.reset_iterations = set(int(i) for i in reset_iterations)
        if self.variant == "classic_muon":
            self.coefficients = (CLASSIC_MUON_COEFFICIENT,) * self.steps
        else:
            coeffs = POLAR_EXPRESS_COEFFICIENTS[: self.steps]
            if len(coeffs) < self.steps:
                coeffs = coeffs + (POLAR_EXPRESS_COEFFICIENTS[-1],) * (self.steps - len(coeffs))
            self.coefficients = coeffs

    def _compute_dtype_for(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        return torch.float16 if x.is_cuda else torch.float32

    @torch.no_grad()
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.ndim < 2:
            raise ValueError("GramNewtonSchulz expects a tensor with ndim >= 2")

        original_shape = matrix.shape
        work = matrix
        if work.ndim == 2:
            work = work.unsqueeze(0)
        elif work.ndim > 3:
            work = work.reshape(-1, *work.shape[-2:])

        original_dtype = work.dtype
        work = work.float()
        transposed = work.size(-2) > work.size(-1)
        if transposed:
            work = work.mT

        work = work / (work.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
        work = work.to(self._compute_dtype_for(work))

        if max(work.shape[-2:]) > min(work.shape[-2:]) and self.variant == "polar_express":
            work = self._gram_recurrence(work)
        else:
            work = self._standard_recurrence(work)

        if transposed:
            work = work.mT
        return work.to(original_dtype).reshape(original_shape)

    def _standard_recurrence(self, work: torch.Tensor) -> torch.Tensor:
        for a, b, c in self.coefficients:
            gram = work @ work.mT
            poly = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
            work = torch.baddbmm(work, poly, work, beta=a)
        return work

    def _gram_recurrence(self, work: torch.Tensor) -> torch.Tensor:
        gram = work @ work.mT
        eye = torch.eye(gram.size(-1), device=work.device, dtype=work.dtype).expand(gram.size(0), -1, -1).contiguous()
        q: torch.Tensor | None = None

        for i, (a, b, c) in enumerate(self.coefficients):
            if i in self.reset_iterations and i != 0:
                if q is None:
                    raise RuntimeError("Gram Newton-Schulz reset reached without an inverse estimate")
                work = q @ work
                gram = work @ work.mT
                q = None

            z = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
            if i == 0 or i in self.reset_iterations:
                q = z + a * eye
            else:
                if q is None:
                    raise RuntimeError("Gram Newton-Schulz inverse estimate was not initialized")
                q = torch.baddbmm(q, q, z, beta=a)

            if i < len(self.coefficients) - 1 and i + 1 not in self.reset_iterations:
                rz = torch.baddbmm(gram, gram, z, beta=a)
                gram = torch.baddbmm(rz, z, rz, beta=a)

        if q is None:
            raise RuntimeError("Gram Newton-Schulz finished without an inverse estimate")
        return q @ work


@torch.no_grad()
def _normuon_normalize(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta2: float,
    eps: float,
    mode: str,
    aspect_scale: bool,
) -> torch.Tensor:
    rows, cols = update.shape[-2], update.shape[-1]
    expected = _normuon_second_shape(rows, cols, mode)
    if tuple(second_momentum.shape[-2:]) != expected:
        raise ValueError(f"NorMuon second-moment shape {tuple(second_momentum.shape[-2:])} != {expected}")

    dtype = update.dtype
    eps_t = torch.tensor(eps, dtype=dtype, device=update.device)
    old_norm = update.norm(dim=(-2, -1), keepdim=True)
    reduce_dim = -1 if expected[-1] == 1 else -2
    try:
        power = update.square().mean(dim=reduce_dim, keepdim=True, dtype=dtype)
    except TypeError:
        power = update.square().mean(dim=reduce_dim, keepdim=True).to(dtype)
    second_momentum.lerp_(power, 1.0 - float(beta2))
    out = update * torch.rsqrt(second_momentum + eps_t)
    out = out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps_t))
    if aspect_scale:
        out = out * math.sqrt(max(1.0, rows / cols))
    return out


def build_param_groups(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    *,
    matrix_lr: float = 8e-3,
    fallback_lr: float | None = None,
    matrix_weight_decay: float = 0.0,
    fallback_weight_decay: float = 0.05,
    min_matrix_dim: int = 2,
    force_matrix: Callable[[str, torch.Tensor], bool] | None = None,
    force_fallback: Callable[[str, torch.Tensor], bool] | None = None,
) -> list[dict[str, Any]]:
    """Build safe optimizer groups from named parameters.

    The helper deduplicates tied/shared parameters and keeps common embeddings,
    heads, norms, biases, vectors, and tiny matrices off the spectral path.
    If any alias of a tied tensor looks like an embedding/head/norm/bias, the
    shared tensor is conservatively routed to the fallback path.
    """

    fallback_lr = matrix_lr if fallback_lr is None else fallback_lr
    matrix_params: list[torch.Tensor] = []
    matrix_names: list[str] = []
    matrix_aliases: list[list[str]] = []
    fallback_params: list[torch.Tensor] = []
    fallback_names: list[str] = []
    fallback_aliases: list[list[str]] = []
    unique: dict[int, tuple[torch.Tensor, list[str]]] = {}

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        ident = id(param)
        if ident not in unique:
            unique[ident] = (param, [name])
        else:
            unique[ident][1].append(name)

    for _ident, (param, names) in unique.items():
        rows = int(param.shape[0]) if param.ndim >= 1 else 1
        cols = int(param.numel() // max(rows, 1))
        eligible = (
            param.is_floating_point()
            and param.ndim >= 2
            and min(rows, cols) >= int(min_matrix_dim)
            and not any(_is_default_fallback_name(name) for name in names)
        )
        if force_matrix is not None and any(force_matrix(name, param) for name in names):
            eligible = True
        if force_fallback is not None and any(force_fallback(name, param) for name in names):
            eligible = False

        display_name = "|".join(names)
        if eligible:
            matrix_params.append(param)
            matrix_names.append(display_name)
            matrix_aliases.append(names)
        else:
            fallback_params.append(param)
            fallback_names.append(display_name)
            fallback_aliases.append(names)

    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "param_names": matrix_names,
                "param_aliases": matrix_aliases,
                "use_matrix_update": True,
                "lr": float(matrix_lr),
                "base_lr": float(matrix_lr),
                "weight_decay": float(matrix_weight_decay),
            }
        )
    if fallback_params:
        groups.append(
            {
                "params": fallback_params,
                "param_names": fallback_names,
                "param_aliases": fallback_aliases,
                "use_matrix_update": False,
                "lr": float(fallback_lr),
                "base_lr": float(fallback_lr),
                "weight_decay": float(fallback_weight_decay),
            }
        )
    return groups


build_optimizer_param_groups = build_param_groups


class SodaPmuonEqNorMuon(torch.optim.Optimizer):
    """Direct SODA + PMuonEq + GramNS + NorMuon optimizer.

    Defaults are the conservative cross-worker favorite:
    row-only PMuonEq, no aspect multiplier, no AMUSE/SF, no MiMuon.
    """

    def __init__(
        self,
        params: ParamsT | Iterable[tuple[str, torch.Tensor]] | torch.nn.Module,
        *,
        matrix_lr: float = 8e-3,
        fallback_lr: float | None = None,
        momentum: float = 0.95,
        beta2: float = 0.999,
        eps: float = 1e-10,
        matrix_weight_decay: float = 0.0,
        fallback_weight_decay: float = 0.05,
        warmup_steps: int = 80,
        soda_lambda_scale: float = 1.0,
        soda_lambda_power: float = 1.0,
        soda_on_fallback: bool = True,
        soda_replaces_weight_decay: bool | None = None,
        soda_replaces_matrix_weight_decay: bool = True,
        soda_replaces_fallback_weight_decay: bool = False,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.35,
        col_gamma: float = 0.0,
        pmuoneq_eps: float = 1e-6,
        normuon_beta2: float = 0.93,
        normuon_eps: float = 1e-10,
        normuon_mode: str = "row",
        normuon_aspect_scale: bool = False,
        ns_variant: str = "polar_express",
        ns_steps: int = 5,
        ns_epsilon: float = 1e-7,
        ns_compute_dtype: torch.dtype | None = None,
        ns_reset_iterations: Iterable[int] = (2,),
        use_external_lr: bool = False,
        min_matrix_dim: int = 2,
        matrix_filter: Callable[[str, torch.Tensor], bool] | None = None,
        fallback_filter: Callable[[str, torch.Tensor], bool] | None = None,
    ) -> None:
        if isinstance(params, torch.nn.Module):
            params = params.named_parameters()
        fallback_lr = matrix_lr if fallback_lr is None else fallback_lr
        if float(matrix_lr) < 0.0 or float(fallback_lr) < 0.0:
            raise ValueError("learning rates must be non-negative")
        if int(warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        if not (0.0 <= float(momentum) < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if not (0.0 <= float(beta2) < 1.0):
            raise ValueError("beta2 must be in [0, 1)")
        if not (0.0 <= float(pmuoneq_beta) < 1.0):
            raise ValueError("pmuoneq_beta must be in [0, 1)")
        if float(row_gamma) < 0.0 or float(col_gamma) < 0.0:
            raise ValueError("PMuonEq gamma values must be non-negative")
        if not (0.0 <= float(normuon_beta2) < 1.0):
            raise ValueError("normuon_beta2 must be in [0, 1)")
        if float(eps) <= 0.0 or float(pmuoneq_eps) <= 0.0 or float(normuon_eps) <= 0.0 or float(ns_epsilon) <= 0.0:
            raise ValueError("epsilon values must be positive")
        if float(matrix_weight_decay) < 0.0 or float(fallback_weight_decay) < 0.0:
            raise ValueError("weight decay values must be non-negative")
        if int(min_matrix_dim) < 1:
            raise ValueError("min_matrix_dim must be >= 1")
        if float(soda_lambda_scale) < 0.0:
            raise ValueError("soda_lambda_scale must be non-negative")
        if float(soda_lambda_power) <= 0.0:
            raise ValueError("soda_lambda_power must be positive")
        if normuon_mode not in {"row", "orientation"}:
            raise ValueError("normuon_mode must be 'row' or 'orientation'")
        if soda_replaces_weight_decay is not None:
            soda_replaces_matrix_weight_decay = bool(soda_replaces_weight_decay)
            soda_replaces_fallback_weight_decay = bool(soda_replaces_weight_decay)

        prepared = self._prepare_param_groups(
            params,
            matrix_lr=matrix_lr,
            fallback_lr=fallback_lr,
            matrix_weight_decay=matrix_weight_decay,
            fallback_weight_decay=fallback_weight_decay,
            momentum=momentum,
            beta2=beta2,
            eps=eps,
            pmuoneq_beta=pmuoneq_beta,
            row_gamma=row_gamma,
            col_gamma=col_gamma,
            pmuoneq_eps=pmuoneq_eps,
            normuon_beta2=normuon_beta2,
            normuon_eps=normuon_eps,
            normuon_mode=normuon_mode,
            normuon_aspect_scale=normuon_aspect_scale,
            soda_on_fallback=soda_on_fallback,
            soda_replaces_matrix_weight_decay=soda_replaces_matrix_weight_decay,
            soda_replaces_fallback_weight_decay=soda_replaces_fallback_weight_decay,
            use_external_lr=use_external_lr,
            min_matrix_dim=min_matrix_dim,
            matrix_filter=matrix_filter,
            fallback_filter=fallback_filter,
        )
        super().__init__(prepared, defaults={})

        self.soda_lambda_scale = float(soda_lambda_scale)
        self.soda_lambda_power = float(soda_lambda_power)
        self.projector = GramNewtonSchulz(
            variant=ns_variant,
            steps=ns_steps,
            epsilon=ns_epsilon,
            compute_dtype=ns_compute_dtype,
            reset_iterations=ns_reset_iterations,
        )
        self.last_stats: dict[str, float] = {}

        for group in self.param_groups:
            group.setdefault("step", 0)
            group.setdefault("warmup_steps", int(warmup_steps))
            group.setdefault("base_lr", float(group["lr"]))

    @classmethod
    def from_model(cls, model: torch.nn.Module, **kwargs: Any) -> "SodaPmuonEqNorMuon":
        """Construct from ``model.named_parameters()`` with safe grouping."""

        return cls(model.named_parameters(), **kwargs)

    @staticmethod
    def _prepare_param_groups(
        params: ParamsT | Iterable[tuple[str, torch.Tensor]],
        *,
        matrix_lr: float,
        fallback_lr: float,
        matrix_weight_decay: float,
        fallback_weight_decay: float,
        momentum: float,
        beta2: float,
        eps: float,
        pmuoneq_beta: float,
        row_gamma: float,
        col_gamma: float,
        pmuoneq_eps: float,
        normuon_beta2: float,
        normuon_eps: float,
        normuon_mode: str,
        normuon_aspect_scale: bool,
        soda_on_fallback: bool,
        soda_replaces_matrix_weight_decay: bool,
        soda_replaces_fallback_weight_decay: bool,
        use_external_lr: bool,
        min_matrix_dim: int,
        matrix_filter: Callable[[str, torch.Tensor], bool] | None,
        fallback_filter: Callable[[str, torch.Tensor], bool] | None,
    ) -> list[dict[str, Any]]:
        items = list(params)
        if not items:
            raise ValueError("optimizer got an empty parameter list")

        if (
            isinstance(items[0], tuple)
            and len(items[0]) == 2
            and isinstance(items[0][0], str)
            and isinstance(items[0][1], torch.Tensor)
        ):
            items = build_param_groups(
                items,  # type: ignore[arg-type]
                matrix_lr=matrix_lr,
                fallback_lr=fallback_lr,
                matrix_weight_decay=matrix_weight_decay,
                fallback_weight_decay=fallback_weight_decay,
                min_matrix_dim=min_matrix_dim,
                force_matrix=matrix_filter,
                force_fallback=fallback_filter,
            )

        if isinstance(items[0], dict):
            groups = [dict(group) for group in items]  # type: ignore[arg-type]
            for group in groups:
                group.setdefault("use_matrix_update", group.get("use_muon", False))
                is_matrix = bool(group["use_matrix_update"])
                group.setdefault("lr", matrix_lr if is_matrix else fallback_lr)
                group.setdefault("base_lr", group["lr"])
                group.setdefault("weight_decay", matrix_weight_decay if is_matrix else fallback_weight_decay)
                group.setdefault("momentum", momentum)
                group.setdefault("beta2", beta2)
                group.setdefault("eps", eps)
                group.setdefault("pmuoneq_beta", pmuoneq_beta)
                group.setdefault("row_gamma", row_gamma)
                group.setdefault("col_gamma", col_gamma)
                group.setdefault("pmuoneq_eps", pmuoneq_eps)
                group.setdefault("normuon_beta2", normuon_beta2)
                group.setdefault("normuon_eps", normuon_eps)
                group.setdefault("normuon_mode", normuon_mode)
                group.setdefault("normuon_aspect_scale", normuon_aspect_scale)
                group.setdefault("soda_enabled", is_matrix or soda_on_fallback)
                group.setdefault(
                    "soda_replaces_weight_decay",
                    soda_replaces_matrix_weight_decay if is_matrix else soda_replaces_fallback_weight_decay,
                )
                group.setdefault("use_external_lr", use_external_lr)
                group.setdefault("min_matrix_dim", min_matrix_dim)
            return groups

        matrix_params: list[torch.Tensor] = []
        fallback_params: list[torch.Tensor] = []
        for param in items:  # type: ignore[assignment]
            if not isinstance(param, torch.Tensor):
                raise TypeError("params must be tensors or optimizer parameter-group dictionaries")
            if param.requires_grad and param.ndim >= 2:
                rows = int(param.shape[0])
                cols = int(param.numel() // max(rows, 1))
                if min(rows, cols) >= int(min_matrix_dim):
                    matrix_params.append(param)
                    continue
            fallback_params.append(param)

        matrix_ids = {id(param) for param in matrix_params}
        groups = build_param_groups(
            [(f"param_{i}", p) for i, p in enumerate(matrix_params + fallback_params)],
            matrix_lr=matrix_lr,
            fallback_lr=fallback_lr,
            matrix_weight_decay=matrix_weight_decay,
            fallback_weight_decay=fallback_weight_decay,
            min_matrix_dim=min_matrix_dim,
            force_matrix=lambda name, param: id(param) in matrix_ids,
        )
        for group in groups:
            is_matrix = bool(group["use_matrix_update"])
            group.setdefault("momentum", momentum)
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)
            group.setdefault("pmuoneq_beta", pmuoneq_beta)
            group.setdefault("row_gamma", row_gamma)
            group.setdefault("col_gamma", col_gamma)
            group.setdefault("pmuoneq_eps", pmuoneq_eps)
            group.setdefault("normuon_beta2", normuon_beta2)
            group.setdefault("normuon_eps", normuon_eps)
            group.setdefault("normuon_mode", normuon_mode)
            group.setdefault("normuon_aspect_scale", normuon_aspect_scale)
            group.setdefault("soda_enabled", is_matrix or soda_on_fallback)
            group.setdefault(
                "soda_replaces_weight_decay",
                soda_replaces_matrix_weight_decay if is_matrix else soda_replaces_fallback_weight_decay,
            )
            group.setdefault("use_external_lr", use_external_lr)
            group.setdefault("min_matrix_dim", min_matrix_dim)
        return groups

    def train(self) -> "SodaPmuonEqNorMuon":
        return self

    def eval(self) -> "SodaPmuonEqNorMuon":
        return self

    def _group_lr(self, group: dict[str, Any]) -> tuple[int, float]:
        step = int(group.get("step", 0)) + 1
        if bool(group.get("use_external_lr", False)):
            lr = float(group["lr"])
        else:
            warmup = max(1, int(group.get("warmup_steps", 1)))
            lr = float(group.get("base_lr", group["lr"])) * min(1.0, step / warmup)
            group["lr"] = lr
        group["step"] = step
        return step, lr

    def _soda_weight(self, step: int) -> float:
        if self.soda_lambda_scale == 0.0:
            return 0.0
        return min(1.0, self.soda_lambda_scale / float(step + 1) ** self.soda_lambda_power)

    def _anchor(self, param: torch.Tensor) -> torch.Tensor:
        state = self.state[param]
        anchor = state.get("soda_anchor")
        if anchor is None or anchor.shape != param.shape or anchor.device != param.device:
            anchor = state["soda_anchor"] = torch.clone(param.detach(), memory_format=torch.preserve_format)
        return anchor

    def _use_matrix_for_param(self, group: dict[str, Any], param: torch.Tensor) -> bool:
        explicit = group.get("use_matrix_update", None)
        if explicit is not None:
            return bool(explicit)
        if param.ndim < 2:
            return False
        rows = int(param.shape[0])
        cols = int(param.numel() // max(rows, 1))
        return min(rows, cols) >= int(group.get("min_matrix_dim", 2))

    def _ensure_matrix_state(
        self,
        param: torch.Tensor,
        rows: int,
        cols: int,
        normuon_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state[param]
        momentum = state.get("momentum_buffer")
        if momentum is None or momentum.shape != param.shape or momentum.device != param.device:
            momentum = state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
        row_ema = state.get("pmuoneq_row_ema")
        if row_ema is None or row_ema.shape != (rows,) or row_ema.device != param.device:
            row_ema = state["pmuoneq_row_ema"] = torch.ones(rows, device=param.device, dtype=torch.float32)
        col_ema = state.get("pmuoneq_col_ema")
        if col_ema is None or col_ema.shape != (cols,) or col_ema.device != param.device:
            col_ema = state["pmuoneq_col_ema"] = torch.ones(cols, device=param.device, dtype=torch.float32)
        second_shape = _normuon_second_shape(rows, cols, normuon_mode)
        second = state.get("normuon_second")
        if second is None or second.shape != second_shape or second.device != param.device:
            second = state["normuon_second"] = torch.zeros(*second_shape, device=param.device, dtype=torch.float32)
        return momentum, row_ema, col_ema, second

    def _matrix_source(self, param: torch.Tensor, grad: torch.Tensor, momentum_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state[param]
        momentum = state.get("momentum_buffer")
        if momentum is None or momentum.shape != param.shape or momentum.device != param.device:
            momentum = state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
        grad32 = grad.detach().to(torch.float32)
        momentum.lerp_(grad32, 1.0 - momentum_beta)
        source = torch.lerp(grad32, momentum, momentum_beta)
        return _matrix_view(source), _matrix_view(grad32)

    def _matrix_bucket_update(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        group: dict[str, Any],
    ) -> torch.Tensor:
        sources = torch.stack([entry[1].float() for entry in entries], dim=0)
        grads = torch.stack([entry[2].float() for entry in entries], dim=0)
        batch, rows, cols = grads.shape
        normuon_mode = str(group.get("normuon_mode", "row"))

        row_buffers: list[torch.Tensor] = []
        col_buffers: list[torch.Tensor] = []
        second_buffers: list[torch.Tensor] = []
        for param, _source, _grad in entries:
            _momentum, row_ema, col_ema, second = self._ensure_matrix_state(param, rows, cols, normuon_mode)
            row_buffers.append(row_ema)
            col_buffers.append(col_ema)
            second_buffers.append(second)

        row_stack = torch.stack(row_buffers, dim=0)
        col_stack = torch.stack(col_buffers, dim=0)
        g2 = grads.square()
        pmuon_beta = float(group["pmuoneq_beta"])
        row_stack.mul_(pmuon_beta).add_(g2.mean(dim=2), alpha=1.0 - pmuon_beta)
        col_stack.mul_(pmuon_beta).add_(g2.mean(dim=1), alpha=1.0 - pmuon_beta)

        row_factor = _normalized_inverse_power(
            row_stack,
            gamma=float(group["row_gamma"]),
            eps=float(group["pmuoneq_eps"]),
        ).to(sources.dtype)
        col_factor = _normalized_inverse_power(
            col_stack,
            gamma=float(group["col_gamma"]),
            eps=float(group["pmuoneq_eps"]),
        ).to(sources.dtype)
        preconditioned = sources * row_factor[:, :, None] * col_factor[:, None, :]

        update = self.projector(preconditioned)
        update = update * (0.2 * math.sqrt(max(rows, cols)))

        second_stack = torch.stack(second_buffers, dim=0)
        update = _normuon_normalize(
            update,
            second_stack,
            beta2=float(group["normuon_beta2"]),
            eps=float(group["normuon_eps"]),
            mode=normuon_mode,
            aspect_scale=bool(group.get("normuon_aspect_scale", False)),
        )

        for idx, (param, _source, _grad) in enumerate(entries):
            state = self.state[param]
            state["pmuoneq_row_ema"].copy_(row_stack[idx])
            state["pmuoneq_col_ema"].copy_(col_stack[idx])
            state["normuon_second"].copy_(second_stack[idx])

        if batch <= 0:
            raise RuntimeError("empty matrix update bucket")
        return update

    def _step_matrix_group(self, group: dict[str, Any], *, lr: float, step: int, soda_weight: float) -> int:
        entries_by_shape: dict[tuple[torch.device, tuple[int, int]], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        momentum_beta = float(group["momentum"])
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.ndim < 2:
                self._step_fallback_param(param, group, lr=lr, step=step, soda_weight=soda_weight)
                continue
            source, grad_matrix = self._matrix_source(param, param.grad, momentum_beta)
            entries_by_shape[(source.device, tuple(source.shape))].append((param, source, grad_matrix))

        matrix_count = 0
        wd = float(group.get("weight_decay", 0.0))
        soda_enabled = bool(group.get("soda_enabled", True))
        replace_wd = bool(group.get("soda_replaces_weight_decay", True))
        for entries in entries_by_shape.values():
            updates = self._matrix_bucket_update(entries, group)
            for (param, _source, _grad), update in zip(entries, updates.unbind(0), strict=True):
                if wd and not (soda_enabled and replace_wd):
                    param.mul_(1.0 - lr * wd)
                if soda_enabled and soda_weight:
                    param.lerp_(self._anchor(param), soda_weight)
                param.add_(update.reshape_as(param).to(param.dtype), alpha=-lr)
                matrix_count += 1
        return matrix_count

    def _step_fallback_param(self, param: torch.Tensor, group: dict[str, Any], *, lr: float, step: int, soda_weight: float) -> int:
        grad = param.grad
        if grad is None:
            return 0
        state = self.state[param]
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None or exp_avg_sq.shape != param.shape or exp_avg_sq.device != param.device:
            exp_avg_sq = state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
        beta2 = float(group["beta2"])
        eps = float(group["eps"])
        grad32 = grad.detach().float()
        exp_avg_sq.mul_(beta2).addcmul_(grad32, grad32, value=1.0 - beta2)
        denom = (exp_avg_sq / max(1.0 - beta2**step, 1e-16)).sqrt().add_(eps)
        update = grad32 / denom

        soda_enabled = bool(group.get("soda_enabled", True))
        replace_wd = bool(group.get("soda_replaces_weight_decay", True))
        wd = float(group.get("weight_decay", 0.0))
        if wd and not (soda_enabled and replace_wd):
            update = update + param.detach().float() * wd
        if soda_enabled and soda_weight:
            param.lerp_(self._anchor(param), soda_weight)
        param.add_(update.to(param.dtype), alpha=-lr)
        return 1

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        matrix_count = 0
        fallback_count = 0
        last_lr = 0.0
        last_soda = 0.0
        for group in self.param_groups:
            step, lr = self._group_lr(group)
            last_lr = lr
            soda_weight = self._soda_weight(step)
            last_soda = soda_weight
            if bool(group.get("use_matrix_update", False)):
                matrix_count += self._step_matrix_group(group, lr=lr, step=step, soda_weight=soda_weight)
            else:
                for param in group["params"]:
                    fallback_count += self._step_fallback_param(param, group, lr=lr, step=step, soda_weight=soda_weight)

        self.last_stats = {
            "matrix_count": float(matrix_count),
            "fallback_count": float(fallback_count),
            "lr": float(last_lr),
            "soda_weight": float(last_soda),
        }
        return loss


FavoriteOptimizer = SodaPmuonEqNorMuon
GoldenSodaPmuonEqNorMuon = SodaPmuonEqNorMuon
GoldenOptimizer = SodaPmuonEqNorMuon
GoldenMuon = SodaPmuonEqNorMuon
Optimizer = SodaPmuonEqNorMuon
build_soda_pmuoneq_normuon_param_groups = build_param_groups
build_golden_soda_pmuoneq_normuon_param_groups = build_param_groups


__all__ = [
    "__version__",
    "FavoriteOptimizer",
    "GoldenSodaPmuonEqNorMuon",
    "GoldenOptimizer",
    "GoldenMuon",
    "GramNewtonSchulz",
    "Optimizer",
    "SodaPmuonEqNorMuon",
    "build_golden_soda_pmuoneq_normuon_param_groups",
    "build_optimizer_param_groups",
    "build_param_groups",
    "build_soda_pmuoneq_normuon_param_groups",
]
