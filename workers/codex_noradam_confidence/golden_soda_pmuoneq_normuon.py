"""Golden direct SODA + PMuonEq + GramNS + NorMuon optimizer.

This file intentionally implements only the current winning optimizer family
from this worker folder:

    SODA + row-only PMuonEq + Gram Newton-Schulz + NorMuon

There is no AMUSE/schedule-free outer loop, no MiMuon branch, no full PMuon, no
post-NorMuon aspect multiplier, and no ablation switches. The goal is to provide
a small copyable optimizer for the next experiment round.

Default hyperparameters match the best proper-split CIFAR-10 run in this folder:

    lr=8e-3, warmup_steps=80, momentum=0.95, pmuoneq_beta=0.90,
    row_gamma=0.35, normuon_beta=0.93, ns_steps=5

All optimizer state is kept in FP32. Matrix-like parameters use the spectral
path. Fallback parameters use the same RMS-style second-moment path as the
winning AnchorMuon configuration. SODA anchors all parameters and replaces
ordinary weight decay in this golden version.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "GoldenSodaPmuonEqNorMuon",
    "GoldenMuon",
    "build_golden_param_groups",
    "golden_gram_newton_schulz",
]


def _as_float(x: Tensor) -> Tensor:
    return x.detach().float()


def _matrix_view(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    if x.ndim < 2:
        raise ValueError("matrix view requires a tensor with ndim >= 2")
    shape = tuple(x.shape)
    return x.reshape(x.shape[0], -1), shape


def _restore_matrix_view(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return x.reshape(shape)


def _state_tensor(
    state: dict[str, Any],
    key: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    fill: float,
) -> Tensor:
    value = state.get(key)
    if isinstance(value, Tensor) and tuple(value.shape) == shape:
        if value.device != device or value.dtype != torch.float32:
            value = value.to(device=device, dtype=torch.float32)
            state[key] = value
        return value
    value = torch.full(shape, fill, device=device, dtype=torch.float32)
    state[key] = value
    return value


@torch.no_grad()
def golden_gram_newton_schulz(update: Tensor, *, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Muon quintic Newton-Schulz polar/zero-power approximation."""

    if update.ndim not in {2, 3}:
        raise ValueError("golden_gram_newton_schulz expects a matrix or batch of matrices")
    x = update.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x.float()


def _normalized_inverse_power(values: Tensor, *, gamma: float, eps: float) -> Tensor:
    powered = values.float().clamp_min(float(eps)).pow(-float(gamma))
    return powered * (math.sqrt(powered.numel()) / powered.norm().clamp_min(float(eps)))


def _is_head_name(name: str) -> bool:
    lowered = name.lower()
    if "lm_head" in lowered or "unembed" in lowered or "output_projection" in lowered:
        return True
    if lowered.endswith("head.weight") and "attn" not in lowered and "attention" not in lowered:
        return True
    return False


def _is_embedding_name(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "embed",
            "embedding",
            "wte",
            "tok_embeddings",
            "word_embeddings",
        )
    )


def _should_use_matrix_path(name: str, param: Tensor, *, min_matrix_dim: int) -> bool:
    if param.ndim < 2 or not param.is_floating_point():
        return False
    lowered = name.lower()
    if name.endswith(".bias") or "norm" in lowered or "ln_" in lowered or "layernorm" in lowered:
        return False
    if _is_embedding_name(name) or _is_head_name(name):
        return False
    rows = int(param.shape[0])
    cols = int(param.numel() // max(rows, 1))
    return min(rows, cols) >= int(min_matrix_dim)


def build_golden_param_groups(
    named_parameters: Iterable[tuple[str, Tensor]],
    *,
    lr: float = 8e-3,
    min_matrix_dim: int = 2,
) -> list[dict[str, Any]]:
    """Build safe parameter groups for the golden optimizer.

    The helper deduplicates tied/shared parameters by object id. It keeps
    embeddings, LM heads/unembeddings, biases, norm weights, vectors, and scalars
    in the fallback path. Transformer attention and MLP matrices remain eligible
    for the spectral matrix path.

    Passing raw ``model.parameters()`` directly to the optimizer is still
    supported, but named grouping is preferred for language-modeling runs.
    """

    matrix_params: list[Tensor] = []
    fallback_params: list[Tensor] = []
    seen: set[int] = set()
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        if _should_use_matrix_path(name, param, min_matrix_dim=min_matrix_dim):
            matrix_params.append(param)
        else:
            fallback_params.append(param)

    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append({"params": matrix_params, "use_matrix": True, "lr": float(lr)})
    if fallback_params:
        groups.append({"params": fallback_params, "use_matrix": False, "lr": float(lr)})
    return groups


class GoldenSodaPmuonEqNorMuon(torch.optim.Optimizer):
    """Direct golden optimizer: SODA + row-PMuonEq + GramNS + NorMuon.

    ``train()`` and ``eval()`` are no-op compatibility methods. Unlike AMUSE or
    schedule-free optimizers, this class stores the train/eval model parameters
    in the same tensor values.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 8e-3,
        warmup_steps: int = 80,
        momentum: float = 0.95,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.35,
        normuon_beta: float = 0.93,
        beta2: float = 0.95,
        eps: float = 1e-8,
        pmuon_eps: float = 1e-6,
        normuon_eps: float = 1e-10,
        ns_steps: int = 5,
        min_matrix_dim: int = 2,
        use_external_lr: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if not (0.0 <= pmuoneq_beta < 1.0):
            raise ValueError("pmuoneq_beta must be in [0, 1)")
        if not (0.0 <= normuon_beta < 1.0):
            raise ValueError("normuon_beta must be in [0, 1)")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("beta2 must be in [0, 1)")

        defaults = dict(
            lr=float(lr),
            base_lr=float(lr),
            warmup_steps=int(warmup_steps),
            momentum=float(momentum),
            pmuoneq_beta=float(pmuoneq_beta),
            row_gamma=float(row_gamma),
            normuon_beta=float(normuon_beta),
            beta2=float(beta2),
            eps=float(eps),
            pmuon_eps=float(pmuon_eps),
            normuon_eps=float(normuon_eps),
            ns_steps=int(ns_steps),
            min_matrix_dim=int(min_matrix_dim),
            use_external_lr=bool(use_external_lr),
            use_matrix=None,
            step=0,
        )
        super().__init__(params, defaults)
        self.last_stats: dict[str, float] = {}

    def train(self) -> "GoldenSodaPmuonEqNorMuon":
        return self

    def eval(self) -> "GoldenSodaPmuonEqNorMuon":
        return self

    def _group_lr(self, group: dict[str, Any]) -> tuple[int, float]:
        step = int(group.get("step", 0)) + 1
        if bool(group.get("use_external_lr", False)):
            lr = float(group["lr"])
        else:
            warmup = max(1, int(group["warmup_steps"]))
            lr = float(group.get("base_lr", group["lr"])) * min(1.0, float(step) / float(warmup))
            group["lr"] = lr
        group["step"] = step
        return step, lr

    def _use_matrix_for_param(self, group: dict[str, Any], param: Tensor) -> bool:
        explicit = group.get("use_matrix", None)
        if explicit is not None:
            return bool(explicit)
        if param.ndim < 2:
            return False
        rows = int(param.shape[0])
        cols = int(param.numel() // max(rows, 1))
        return min(rows, cols) >= int(group["min_matrix_dim"])

    def _ensure_common_state(self, param: Tensor, state: dict[str, Any]) -> tuple[Tensor, Tensor]:
        if "z" not in state or tuple(state["z"].shape) != tuple(param.shape) or state["z"].device != param.device:
            state["z"] = _as_float(param).clone(memory_format=torch.preserve_format)
        if (
            "soda_init" not in state
            or tuple(state["soda_init"].shape) != tuple(param.shape)
            or state["soda_init"].device != param.device
        ):
            state["soda_init"] = _as_float(param).clone(memory_format=torch.preserve_format)
        return state["z"], state["soda_init"]

    def _ensure_matrix_state(self, param: Tensor, state: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        matrix, _ = _matrix_view(param)
        rows, cols = matrix.shape
        momentum = _state_tensor(state, "momentum", shape=(rows, cols), device=param.device, fill=0.0)
        row_ema = _state_tensor(state, "row_ema", shape=(rows,), device=param.device, fill=1.0)
        norm_shape = (rows, 1) if rows >= cols else (1, cols)
        normuon_second = _state_tensor(
            state,
            "normuon_second_moment",
            shape=norm_shape,
            device=param.device,
            fill=0.0,
        )
        return momentum, row_ema, normuon_second

    def _ensure_fallback_state(self, param: Tensor, state: dict[str, Any]) -> Tensor:
        return _state_tensor(state, "exp_avg_sq", shape=tuple(param.shape), device=param.device, fill=0.0)

    @torch.no_grad()
    def _matrix_update(self, grad: Tensor, param: Tensor, group: dict[str, Any], state: dict[str, Any]) -> Tensor:
        grad_matrix, original_shape = _matrix_view(grad)
        momentum, row_ema, normuon_second = self._ensure_matrix_state(param, state)

        momentum_beta = float(group["momentum"])
        momentum.lerp_(grad_matrix, 1.0 - momentum_beta)
        nesterov = torch.lerp(grad_matrix, momentum, momentum_beta)

        beta = float(group["pmuoneq_beta"])
        row_ema.mul_(beta).add_(grad_matrix.float().square().mean(dim=1), alpha=1.0 - beta)
        row_scale = _normalized_inverse_power(
            row_ema,
            gamma=float(group["row_gamma"]),
            eps=float(group["pmuon_eps"]),
        ).to(nesterov.dtype)
        preconditioned = row_scale[:, None] * nesterov.float()

        update = golden_gram_newton_schulz(
            preconditioned,
            steps=int(group["ns_steps"]),
            eps=float(group["pmuon_eps"]),
        )

        old_norm = update.norm(dim=(-2, -1), keepdim=True).clamp_min(float(group["normuon_eps"]))
        if update.shape[0] >= update.shape[1]:
            mean_square = update.square().mean(dim=1, keepdim=True)
        else:
            mean_square = update.square().mean(dim=0, keepdim=True)
        normuon_second.lerp_(mean_square.to(normuon_second.dtype), 1.0 - float(group["normuon_beta"]))
        update = update * torch.rsqrt(normuon_second.to(update.dtype).clamp_min(float(group["normuon_eps"])))
        new_norm = update.norm(dim=(-2, -1), keepdim=True).clamp_min(float(group["normuon_eps"]))
        update = update * (old_norm / new_norm)

        update = update * (0.2 * math.sqrt(max(update.shape[-2], update.shape[-1])))
        return _restore_matrix_view(update, original_shape)

    @torch.no_grad()
    def _fallback_update(self, grad: Tensor, group: dict[str, Any], state: dict[str, Any], step: int, param: Tensor) -> Tensor:
        exp_avg_sq = self._ensure_fallback_state(param, state)
        beta2 = float(group["beta2"])
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = (exp_avg_sq / max(1.0 - beta2**step, float(group["eps"]))).sqrt().add_(float(group["eps"]))
        return grad / denom

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_params = 0
        matrix_params = 0
        fallback_params = 0
        lr_last = 0.0

        for group in self.param_groups:
            step, lr = self._group_lr(group)
            lr_last = lr
            for param in group["params"]:
                if param.grad is None:
                    continue
                total_params += 1
                grad = _as_float(param.grad)
                state = self.state[param]
                state["step"] = step
                z, init = self._ensure_common_state(param, state)
                old_base = z.clone(memory_format=torch.preserve_format)

                if self._use_matrix_for_param(group, param):
                    matrix_params += 1
                    update = self._matrix_update(grad, param, group, state)
                else:
                    fallback_params += 1
                    update = self._fallback_update(grad, group, state, step, param)

                z.add_(update.to(device=z.device, dtype=torch.float32), alpha=-lr)
                z.add_(init.to(device=z.device, dtype=torch.float32) - old_base, alpha=1.0 / float(step + 1))
                param.copy_(z.to(device=param.device, dtype=param.dtype))

        self.last_stats = {
            "step": float(max((int(g.get("step", 0)) for g in self.param_groups), default=0)),
            "lr": float(lr_last),
            "params": float(total_params),
            "matrix_params": float(matrix_params),
            "fallback_params": float(fallback_params),
        }
        return loss


GoldenMuon = GoldenSodaPmuonEqNorMuon
