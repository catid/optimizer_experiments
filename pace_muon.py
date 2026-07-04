"""Standalone PACE + Muon optimizer.

This file is intentionally independent of the experiment runners so other
agents can copy or import it into new training jobs.  It implements the variant
that worked best in the local FineWeb GPT-2-tokenized proxy:

    optimizer = PaceMuon(model, lr=1.6e-3, weight_decay=0.05,
                         pullback_c=1e-3, kappa=0.5)

Matrix-shaped tensors use a plain Muon update with a Gram Newton-Schulz polar
step.  Scalar/vector tensors use an AdamW-style fallback.  PACE then applies a
post-step weight-space pullback toward a decaying EMA of iterates.  Evaluation
should normally use the EMA weights:

    with optimizer.use_ema_weights():
        validate(model)

Set ``pullback_c=0`` to get the "Muon @ EMA" control: live training follows
Muon, but evaluation can still swap to EMA weights.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import torch
from torch import nn


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


def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    effective_shape = tuple(int(dim) for dim in x.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        raise ValueError("matrix update requires at least two non-singleton dimensions")
    if len(effective_shape) == 2:
        return x.reshape(effective_shape)
    return x.reshape(effective_shape[0], math.prod(effective_shape[1:]))


def _is_matrix_like(param: torch.Tensor, *, min_matrix_dim: int) -> bool:
    if not param.is_floating_point():
        return False
    effective_shape = tuple(int(dim) for dim in param.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        return False
    return min(effective_shape[0], math.prod(effective_shape[1:])) >= int(min_matrix_dim)


class GramNewtonSchulz:
    """Batched Gram Newton-Schulz polar approximation used by Muon."""

    def __init__(
        self,
        *,
        coefficients: Iterable[Iterable[float]] | None = None,
        epsilon: float = 1e-7,
        reset_iterations: Iterable[int] = (2,),
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.coefficients = tuple(tuple(float(v) for v in row) for row in (coefficients or POLAR_EXPRESS_COEFFICIENTS))
        self.epsilon = float(epsilon)
        self.reset_iterations = set(int(i) for i in reset_iterations)
        self.compute_dtype = compute_dtype

    def _compute_dtype_for(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        return torch.float16 if x.is_cuda else torch.float32

    @torch.no_grad()
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.ndim < 2:
            raise ValueError("GramNewtonSchulz expects a tensor with ndim >= 2")

        original_shape = matrix.shape
        x = matrix
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim > 3:
            x = x.reshape(-1, *x.shape[-2:])

        original_dtype = x.dtype
        x = x.to(torch.float32)
        transposed = x.size(-2) > x.size(-1)
        if transposed:
            x = x.mT

        x = x / (x.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
        x = x.to(self._compute_dtype_for(x))

        if max(x.shape[-2:]) > min(x.shape[-2:]):
            x = self._gram_recurrence(x)
        else:
            x = self._standard_recurrence(x)

        if transposed:
            x = x.mT
        return x.to(original_dtype).reshape(original_shape)

    def _standard_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        for a, b, c in self.coefficients:
            gram = x @ x.mT
            poly = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
            x = torch.baddbmm(x, poly, x, beta=a)
            x = torch.baddbmm(x, poly, x, beta=a)
        return x

    def _gram_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        gram = x @ x.mT
        eye = torch.eye(gram.size(-1), device=x.device, dtype=x.dtype).expand(gram.size(0), -1, -1).contiguous()
        q: torch.Tensor | None = None

        for i, (a, b, c) in enumerate(self.coefficients):
            if i in self.reset_iterations and i != 0:
                if q is None:
                    raise RuntimeError("Gram Newton-Schulz reset reached without an inverse estimate")
                x = q @ x
                gram = x @ x.mT
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
        return q @ x


class PaceMuon(torch.optim.Optimizer):
    """Muon optimizer with PACE post-step pullback and EMA-weight evaluation.

    Args:
        params: ``nn.Module``, iterable of parameters, named parameters, or
            PyTorch-style parameter groups.
        lr: Matrix-path Muon learning rate and fallback default learning rate.
        weight_decay: Decoupled weight decay for matrix tensors and
            matrix-shaped fallback tensors. Scalar/vector fallback tensors
            default to zero decay when ``params`` is a module or flat iterable.
        pullback_c: PACE pullback strength. Use ``1e-3`` as the first trial.
        kappa: EMA/pullback decay exponent. Use ``0.5`` as the first trial.
        pace_precond: ``"adam"`` for per-coordinate PACE gain, ``"row"`` for
            row-wise matrix gain, or ``"scalar"`` for no curvature proxy.
    """

    def __init__(
        self,
        params: nn.Module | Iterable[torch.Tensor] | Iterable[tuple[str, torch.Tensor]] | Iterable[dict[str, Any]],
        *,
        lr: float = 1.6e-3,
        weight_decay: float = 0.05,
        momentum: float = 0.95,
        fallback_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        pullback_c: float = 1e-3,
        kappa: float = 0.5,
        pace_precond: str = "adam",
        pace_beta2: float = 0.999,
        pace_eps: float = 1e-8,
        pace_update_freq: int = 1,
        min_decay: float = 1e-4,
        min_matrix_dim: int = 2,
        ns_epsilon: float = 1e-7,
        ns_compute_dtype: torch.dtype | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if len(fallback_betas) != 2 or not all(0.0 <= beta < 1.0 for beta in fallback_betas):
            raise ValueError("fallback_betas must contain two values in [0, 1)")
        if eps <= 0.0 or pace_eps <= 0.0:
            raise ValueError("epsilon values must be positive")
        if pullback_c < 0.0:
            raise ValueError("pullback_c must be non-negative")
        if not 0.0 < kappa <= 1.0:
            raise ValueError("kappa must be in (0, 1]")
        pace_precond = str(pace_precond).lower()
        if pace_precond not in {"adam", "row", "scalar"}:
            raise ValueError("pace_precond must be 'adam', 'row', or 'scalar'")
        if not 0.0 <= pace_beta2 < 1.0:
            raise ValueError("pace_beta2 must be in [0, 1)")
        if pace_update_freq < 1:
            raise ValueError("pace_update_freq must be >= 1")
        if min_decay < 0.0:
            raise ValueError("min_decay must be non-negative")

        prepared = self._prepare_param_groups(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            fallback_betas=fallback_betas,
            eps=eps,
            min_matrix_dim=min_matrix_dim,
        )
        super().__init__(prepared, defaults={})

        self.pullback_c = float(pullback_c)
        self.kappa = float(kappa)
        self.pace_precond = pace_precond
        self.pace_beta2 = float(pace_beta2)
        self.pace_eps = float(pace_eps)
        self.pace_update_freq = int(pace_update_freq)
        self.min_decay = float(min_decay)
        self.step_index = 0
        self._swapped = False
        self._orthogonalizer = GramNewtonSchulz(epsilon=ns_epsilon, compute_dtype=ns_compute_dtype)

    @classmethod
    def from_model(cls, model: nn.Module, **kwargs: Any) -> "PaceMuon":
        return cls(model, **kwargs)

    @staticmethod
    def _prepare_param_groups(
        params: nn.Module | Iterable[torch.Tensor] | Iterable[tuple[str, torch.Tensor]] | Iterable[dict[str, Any]],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        fallback_betas: tuple[float, float],
        eps: float,
        min_matrix_dim: int,
    ) -> list[dict[str, Any]]:
        if isinstance(params, nn.Module):
            items: list[Any] = list(params.parameters())
        else:
            items = list(params)
        if not items:
            raise ValueError("optimizer got an empty parameter list")

        if isinstance(items[0], dict):
            groups: list[dict[str, Any]] = []
            for group in items:
                group_params = list(group["params"])
                base = {key: value for key, value in group.items() if key != "params"}
                groups.extend(
                    PaceMuon._split_one_param_list(
                        group_params,
                        lr=float(base.pop("lr", lr)),
                        weight_decay=float(base.pop("weight_decay", weight_decay)),
                        momentum=float(base.pop("momentum", momentum)),
                        fallback_betas=base.pop("fallback_betas", fallback_betas),
                        eps=float(base.pop("eps", eps)),
                        min_matrix_dim=min_matrix_dim,
                        respect_scalar_decay=True,
                    )
                )
            return groups

        if isinstance(items[0], tuple) and len(items[0]) == 2 and isinstance(items[0][1], torch.Tensor):
            tensors = [item[1] for item in items]
        else:
            tensors = items
        return PaceMuon._split_one_param_list(
            tensors,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            fallback_betas=fallback_betas,
            eps=eps,
            min_matrix_dim=min_matrix_dim,
            respect_scalar_decay=False,
        )

    @staticmethod
    def _split_one_param_list(
        params: Iterable[torch.Tensor],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        fallback_betas: tuple[float, float],
        eps: float,
        min_matrix_dim: int,
        respect_scalar_decay: bool,
    ) -> list[dict[str, Any]]:
        matrix: list[torch.Tensor] = []
        fallback_decay: list[torch.Tensor] = []
        fallback_nodecay: list[torch.Tensor] = []
        seen: set[int] = set()

        for p in params:
            if not isinstance(p, torch.Tensor):
                raise TypeError("optimizer parameters must be tensors")
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if _is_matrix_like(p, min_matrix_dim=min_matrix_dim):
                matrix.append(p)
            elif respect_scalar_decay or p.ndim >= 2:
                fallback_decay.append(p)
            else:
                fallback_nodecay.append(p)

        groups: list[dict[str, Any]] = []
        if matrix:
            groups.append(
                {
                    "params": matrix,
                    "use_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                }
            )
        if fallback_decay:
            groups.append(
                {
                    "params": fallback_decay,
                    "use_muon": False,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": fallback_betas,
                    "eps": eps,
                }
            )
        if fallback_nodecay:
            groups.append(
                {
                    "params": fallback_nodecay,
                    "use_muon": False,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "betas": fallback_betas,
                    "eps": eps,
                }
            )
        if not groups:
            raise ValueError("optimizer got no trainable parameters")
        return groups

    def _decay_t(self, step: int) -> float:
        return max((1.0 + float(step)) ** (-self.kappa), self.min_decay)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        if self._swapped:
            raise RuntimeError("PaceMuon.step() called while weights are swapped to the EMA")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_index += 1
        decay = self._decay_t(self.step_index)
        self._pace_before_base_step()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)

        self._pace_after_base_step(decay)
        return loss

    def _pace_before_base_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                ema = state.get("pace_ema")
                if ema is None or ema.shape != p.shape:
                    ema = state["pace_ema"] = p.detach().to(torch.float32, copy=True)
                pre = state.get("pace_pre")
                if pre is None or pre.shape != p.shape:
                    pre = state["pace_pre"] = torch.empty_like(p, dtype=torch.float32)
                pre.copy_(p.detach())

                if self.pullback_c <= 0.0:
                    continue
                if self.pace_precond == "adam":
                    v = state.get("pace_v")
                    if v is None or v.shape != p.shape:
                        v = state["pace_v"] = torch.zeros_like(p, dtype=torch.float32)
                    g = p.grad.detach().to(torch.float32)
                    v.mul_(self.pace_beta2).addcmul_(g, g, value=1.0 - self.pace_beta2)
                elif self.pace_precond == "row" and _is_matrix_like(p, min_matrix_dim=2):
                    g_mv = _matrix_view(p.grad.detach()).to(torch.float32)
                    v_row = state.get("pace_v_row")
                    if v_row is None or v_row.shape != (g_mv.shape[0],):
                        v_row = state["pace_v_row"] = torch.zeros(g_mv.shape[0], device=p.device, dtype=torch.float32)
                    v_row.mul_(self.pace_beta2).add_(g_mv.square().mean(dim=1), alpha=1.0 - self.pace_beta2)

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
            matrix = _matrix_view(source)
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
            if "adam_step" not in state:
                state["adam_step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["adam_step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            step = int(state["adam_step"])
            mhat = exp_avg / max(1.0 - beta1**step, 1e-16)
            vhat = exp_avg_sq / max(1.0 - beta2**step, 1e-16)
            p.addcdiv_(mhat.to(p.dtype), vhat.sqrt().add_(eps).to(p.dtype), value=-lr)

    def _pace_after_base_step(self, decay: float) -> None:
        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                ema = state.get("pace_ema")
                pre = state.get("pace_pre")
                if ema is None or pre is None:
                    continue

                if self.pullback_c > 0.0:
                    if self.pace_precond == "adam" and "pace_v" in state:
                        bias2 = max(1.0 - self.pace_beta2**self.step_index, 1e-16)
                        lam = (state["pace_v"] / bias2).sqrt().add_(self.pace_eps).reciprocal_()
                        lam.mul_(lr * self.pullback_c * decay).clamp_(max=1.0)
                        diff = (ema - pre).mul_(lam)
                    elif self.pace_precond == "row" and "pace_v_row" in state:
                        bias2 = max(1.0 - self.pace_beta2**self.step_index, 1e-16)
                        lam_row = (state["pace_v_row"] / bias2).sqrt().add_(self.pace_eps).reciprocal_()
                        lam_row.mul_(lr * self.pullback_c * decay).clamp_(max=1.0)
                        diff = _matrix_view(ema - pre).mul_(lam_row.unsqueeze(1)).reshape_as(p)
                    else:
                        lam_scalar = min(lr * self.pullback_c * decay, 1.0)
                        diff = (ema - pre).mul_(lam_scalar)
                    p.add_(diff.to(p.dtype))

                if self.step_index % self.pace_update_freq == 0:
                    ema.mul_(1.0 - decay).add_(p.detach().to(torch.float32), alpha=decay)

    @torch.no_grad()
    def swap_to_ema(self) -> None:
        """Replace live parameters with PACE EMA parameters for evaluation."""

        if self._swapped:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "pace_ema" not in state:
                    continue
                backup = state.get("pace_live_backup")
                if backup is None or backup.shape != p.shape:
                    backup = state["pace_live_backup"] = torch.empty_like(p)
                backup.copy_(p.detach())
                p.copy_(state["pace_ema"].to(p.dtype))
        self._swapped = True

    @torch.no_grad()
    def swap_to_live(self) -> None:
        """Restore live parameters after ``swap_to_ema()``."""

        if not self._swapped:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "pace_live_backup" not in state:
                    continue
                p.copy_(state["pace_live_backup"])
        self._swapped = False

    @contextmanager
    def use_ema_weights(self) -> Iterator[None]:
        """Context manager that evaluates with EMA weights and always restores."""

        self.swap_to_ema()
        try:
            yield
        finally:
            self.swap_to_live()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["pace_global"] = {
            "step_index": self.step_index,
            "pullback_c": self.pullback_c,
            "kappa": self.kappa,
            "pace_precond": self.pace_precond,
            "pace_beta2": self.pace_beta2,
            "pace_eps": self.pace_eps,
            "pace_update_freq": self.pace_update_freq,
            "min_decay": self.min_decay,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        global_state = state_dict.pop("pace_global", {})
        super().load_state_dict(state_dict)
        self.step_index = int(global_state.get("step_index", 0))
        self.pullback_c = float(global_state.get("pullback_c", self.pullback_c))
        self.kappa = float(global_state.get("kappa", self.kappa))
        self.pace_precond = str(global_state.get("pace_precond", self.pace_precond))
        self.pace_beta2 = float(global_state.get("pace_beta2", self.pace_beta2))
        self.pace_eps = float(global_state.get("pace_eps", self.pace_eps))
        self.pace_update_freq = int(global_state.get("pace_update_freq", self.pace_update_freq))
        self.min_decay = float(global_state.get("min_decay", self.min_decay))
        self._swapped = False


__all__ = ["GramNewtonSchulz", "PaceMuon"]
