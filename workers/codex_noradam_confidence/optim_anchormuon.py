"""AnchorMuon: standalone SODA-AMUSE + PMuonEq optimizer.

AnchorMuon is a compact, dependency-light version of the optimizer stack used in
this repo under the longer name SODA-AMUSE + PMuonEq + Gram Newton-Schulz.

The name is meant to describe the two important pieces:

* ``Anchor``: SODA's initialization anchor and AMUSE's schedule-free fast/average
  iterate bookkeeping.
* ``Muon``: Muon-style matrix updates, orthogonalized by Newton-Schulz.

The optimizer is intentionally ablation-friendly. ``amuse=False`` disables the
schedule-free train/eval interpolation, ``soda="none"`` disables the SODA anchor
pull, ``pmuon_eq=False`` disables row/column PMuonEq scaling, ``mimuon=True``
blends the Muon branch with a normalized momentum-SGD branch, and
``normuon=True`` adds NorMuon-style row/column second-moment normalization after
the Gram Newton-Schulz transform. ``normuon_aspect_scale=True`` optionally adds
the peer-review aspect multiplier after that normalization.
``use_gram=False`` is a diagnostic ablation that replaces the polar transform
with an RMS-matched momentum direction.

For matrix-like parameters, AnchorMuon computes

    m_t = mu * m_{t-1} + (1 - mu) * g_t
    u_t = (1 - mu) * g_t + mu * m_t
    r_t = beta * r_{t-1} + (1 - beta) * mean_cols(g_t^2)
    c_t = beta * c_{t-1} + (1 - beta) * mean_rows(g_t^2)
    d_t = polar(r_t^{-gamma_row} * u_t * c_t^{-gamma_col})

where ``polar`` is approximated with the same quintic Newton-Schulz iteration
commonly used by Muon. The final matrix direction is applied to AMUSE's fast
iterate ``z``; SODA then pulls the averaged/eval iterate toward the initialization
anchor. By default that SODA anchor is applied to matrix/Muon parameters only,
matching the tuned repo stack. Vector and scalar parameters use the same AMUSE
outer loop with a small RMS-style adaptive fallback.

Minimal use:

    from anchormuon import AnchorMuon

    opt = AnchorMuon(model.parameters(), lr=4e-3, row_gamma=0.20, col_gamma=0.0)
    for x, y in loader:
        opt.train()
        loss = model(x).loss(y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    opt.eval()   # swap parameters to averaged/eval weights before validation
    validate(model)
    opt.train()  # swap back before the next training step

Only PyTorch is required. Optimizer states are kept in FP32.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor

__all__ = [
    "AnchorMuon",
    "AnchorMuonEq",
    "gram_newton_schulz",
    "normalized_matrix_direction",
    "normuon_normalize_update",
    "pmuon_eq_precondition",
]


def _as_float(x: Tensor) -> Tensor:
    return x.detach().float()


def _effective_matrix_shape(x: Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in x.shape if int(dim) > 1)


def _matrix_view(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    effective_shape = _effective_matrix_shape(x)
    if len(effective_shape) < 2:
        raise ValueError("matrix view requires at least two non-singleton dimensions")
    shape = tuple(x.shape)
    rows = effective_shape[0]
    cols = math.prod(effective_shape[1:])
    return x.reshape(rows, cols), shape


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
def gram_newton_schulz(update: Tensor, *, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the polar/zero-power direction of a matrix or matrix batch.

    This is the standard Muon quintic Newton-Schulz iteration. Tall matrices are
    transposed internally so the Gram matrix is built on the smaller side. The
    function accepts ``[m, n]`` or ``[batch, m, n]`` tensors and returns FP32.
    """

    if update.ndim not in {2, 3}:
        raise ValueError("gram_newton_schulz expects a matrix or batch of matrices")
    x = update.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    denom = x.norm(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    x = x / denom
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x.float()


@torch.no_grad()
def normalized_matrix_direction(update: Tensor, *, eps: float = 1e-7) -> Tensor:
    """Return a no-polar matrix direction with Muon-compatible RMS scale.

    This is used only for ablations that remove Gram Newton-Schulz. The normal
    Muon path produces an approximately polar matrix whose RMS is about
    ``1 / sqrt(max(rows, cols))`` before AnchorMuon's standard ``0.2 *
    sqrt(max(rows, cols))`` multiplier. Matching that pre-multiplier RMS keeps
    the ablation's layer step scale comparable while removing only the polar
    geometry.
    """

    if update.ndim != 2:
        raise ValueError("normalized_matrix_direction expects a matrix")
    x = update.float()
    rows, cols = x.shape
    target_rms = 1.0 / math.sqrt(max(rows, cols))
    rms = x.square().mean().sqrt().clamp_min(float(eps))
    return x * (target_rms / rms)


def _normalized_inverse_power(values: Tensor, *, gamma: float, eps: float) -> Tensor:
    if float(gamma) == 0.0:
        return torch.ones_like(values, dtype=torch.float32)
    powered = values.float().clamp_min(float(eps)).pow(-float(gamma))
    # Preserve the average update scale. The polar step removes global scale, but
    # this keeps the finite-step Newton-Schulz input numerically comparable.
    return powered * (math.sqrt(powered.numel()) / powered.norm().clamp_min(float(eps)))


@torch.no_grad()
def pmuon_eq_precondition(
    grad_matrix: Tensor,
    momentum_update: Tensor,
    row_ema: Tensor,
    col_ema: Tensor,
    *,
    beta: float = 0.95,
    row_gamma: float = 0.20,
    col_gamma: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """Apply PMuonEq row/column EMA scaling before Muon orthogonalization.

    ``row_ema`` and ``col_ema`` are updated in-place in FP32. ``grad_matrix`` and
    ``momentum_update`` must both be shaped ``[rows, cols]``.
    """

    if grad_matrix.ndim != 2 or momentum_update.ndim != 2:
        raise ValueError("PMuonEq preconditioning expects 2D matrices")
    if grad_matrix.shape != momentum_update.shape:
        raise ValueError("grad_matrix and momentum_update must have the same shape")
    if tuple(row_ema.shape) != (grad_matrix.shape[0],):
        raise ValueError("row_ema shape does not match matrix rows")
    if tuple(col_ema.shape) != (grad_matrix.shape[1],):
        raise ValueError("col_ema shape does not match matrix columns")

    g = grad_matrix.float()
    row_ema.mul_(float(beta)).add_(g.square().mean(dim=1), alpha=1.0 - float(beta))
    col_ema.mul_(float(beta)).add_(g.square().mean(dim=0), alpha=1.0 - float(beta))
    row_scale = _normalized_inverse_power(row_ema, gamma=row_gamma, eps=eps).to(momentum_update.dtype)
    col_scale = _normalized_inverse_power(col_ema, gamma=col_gamma, eps=eps).to(momentum_update.dtype)
    return row_scale[:, None] * momentum_update.float() * col_scale[None, :]


@torch.no_grad()
def normuon_normalize_update(
    update_matrix: Tensor,
    second_moment: Tensor,
    *,
    beta: float = 0.95,
    eps: float = 1e-10,
    aspect_scale: bool = False,
) -> Tensor:
    """Apply NorMuon-style row/column second-moment normalization.

    NorMuon normalizes the already matrix-transformed update, not the raw
    gradient or momentum. For tall matrices this tracks row mean-square update
    energy; for wide matrices it tracks column mean-square update energy. The
    final Frobenius norm is restored so this changes row/column allocation
    without silently changing the whole-layer step size. ``aspect_scale`` then
    applies the optional row-heavy multiplier ``sqrt(max(1, rows / cols))`` as a
    deliberate layer update-scale ablation, matching the cross-worker feedback.
    """

    if update_matrix.ndim != 2:
        raise ValueError("NorMuon normalization expects a 2D matrix")
    if tuple(second_moment.shape) not in {
        (update_matrix.shape[0], 1),
        (1, update_matrix.shape[1]),
    }:
        raise ValueError("second_moment shape does not match NorMuon reduction side")

    update = update_matrix.float()
    old_norm = update.norm(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    if update.shape[0] >= update.shape[1]:
        expected = (update.shape[0], 1)
        reduce_dim = 1
    else:
        expected = (1, update.shape[1])
        reduce_dim = 0
    if tuple(second_moment.shape) != expected:
        raise ValueError(f"second_moment has shape {tuple(second_moment.shape)}, expected {expected}")
    mean_square = update.square().mean(dim=reduce_dim, keepdim=True)
    second_moment.lerp_(mean_square.to(second_moment.dtype), 1.0 - float(beta))
    normalized = update * torch.rsqrt(second_moment.to(update.dtype).clamp_min(float(eps)))
    new_norm = normalized.norm(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    normalized = normalized * (old_norm / new_norm)
    if aspect_scale:
        normalized = normalized * math.sqrt(max(1.0, float(update.shape[0]) / float(max(update.shape[1], 1))))
    return normalized


class AnchorMuon(torch.optim.Optimizer):
    """Standalone SODA-AMUSE + PMuonEq optimizer.

    Parameters can be provided as usual for a PyTorch optimizer. By default,
    tensors with ``ndim >= 2`` and a sufficiently large flattened smaller side are
    treated as Muon matrices; everything else uses the AMUSE adaptive fallback.
    Set ``use_muon=True`` or ``use_muon=False`` on a param group to override this.

    Important knobs:

    * ``lr``: base LR for the fast iterate.
    * ``warmup_steps``: AMUSE LR warmup length. Use ``use_external_lr=True`` if
      another scheduler writes group ``lr`` before each step.
    * ``beta1`` / ``rho``: AMUSE interpolation controls.
    * ``momentum``: Muon/Nesterov momentum.
    * ``pmuon_beta``: row/column gradient-power EMA coefficient.
    * ``row_gamma`` / ``col_gamma``: PMuonEq strength. The TRM-favorable cheap
      default is row-only PMuonEq, ``row_gamma=0.20`` and ``col_gamma=0.0``.
    * ``pmuon_eq``: enables the cheap row/column PMuonEq preconditioner before
      Newton-Schulz. Set false for a plain AMUSE/SODA+Muon ablation.
    * ``use_gram``: enables the Gram Newton-Schulz polar transform. Set false
      only for ablations; the fallback direction is RMS-matched but not polar.
    * ``mimuon``: blends the Muon direction with a normalized momentum direction
      after PMuonEq. This is off by default and exposed for ablations.
    * ``normuon``: applies NorMuon-style row/column second-moment normalization
      after Gram Newton-Schulz and before the existing Muon scale. This is off by
      default and exposed for ablations on top of PMuonEq/SODA.
    * ``normuon_aspect_scale``: after NorMuon Frobenius restoration, multiply
      tall matrices by ``sqrt(rows / cols)``. This is off by default because it
      changes effective per-layer step size and must be tuned.
    * ``soda``: ``"matrix"`` by default, meaning only matrix/Muon parameters get
      SODA's initialization-anchor pull. Use ``True``/``"all"`` for all params or
      ``False``/``"none"`` to disable SODA.
    * ``sync_diagnostics``: off by default. When enabled, ``last_stats`` includes
      RMS diagnostics that require GPU-to-CPU synchronization inside ``step()``.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        beta1: float | None = None,
        beta2: float | None = None,
        rho: float = 0.8,
        warmup_steps: int = 100,
        weight_lr_power: float = 2.0,
        r: float = 0.0,
        use_external_lr: bool = False,
        amuse: bool = True,
        momentum: float = 0.95,
        pmuon_beta: float = 0.95,
        pmuon_eq: bool = True,
        use_gram: bool = True,
        row_gamma: float = 0.20,
        col_gamma: float = 0.0,
        mimuon: bool = False,
        mimuon_mix: float = 0.85,
        normuon: bool = False,
        normuon_beta: float = 0.95,
        normuon_aspect_scale: bool = False,
        normuon_eps: float = 1e-10,
        eps: float = 1e-8,
        pmuon_eps: float = 1e-6,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        soda: bool | str = "matrix",
        soda_disables_weight_decay: bool = True,
        sync_diagnostics: bool = False,
        min_matrix_dim: int = 2,
    ) -> None:
        b1 = float(betas[0] if beta1 is None else beta1)
        b2 = float(betas[1] if beta2 is None else beta2)
        if not (0.0 <= b1 < 1.0):
            raise ValueError("beta1 must be in [0, 1)")
        if not (0.0 <= b2 < 1.0):
            raise ValueError("beta2 must be in [0, 1)")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if not (0.0 <= pmuon_beta < 1.0):
            raise ValueError("pmuon_beta must be in [0, 1)")
        if not (0.0 <= mimuon_mix <= 1.0):
            raise ValueError("mimuon_mix must be in [0, 1]")
        if not (0.0 <= normuon_beta < 1.0):
            raise ValueError("normuon_beta must be in [0, 1)")
        if lr < 0.0:
            raise ValueError("lr must be non-negative")

        defaults = dict(
            lr=float(lr),
            base_lr=float(lr),
            beta1=b1,
            beta2=b2,
            rho=float(rho),
            warmup_steps=int(warmup_steps),
            weight_lr_power=float(weight_lr_power),
            r=float(r),
            use_external_lr=bool(use_external_lr),
            amuse=bool(amuse),
            momentum=float(momentum),
            pmuon_beta=float(pmuon_beta),
            pmuon_eq=bool(pmuon_eq),
            use_gram=bool(use_gram),
            row_gamma=float(row_gamma),
            col_gamma=float(col_gamma),
            mimuon=bool(mimuon),
            mimuon_mix=float(mimuon_mix),
            normuon=bool(normuon),
            normuon_beta=float(normuon_beta),
            normuon_aspect_scale=bool(normuon_aspect_scale),
            normuon_eps=float(normuon_eps),
            eps=float(eps),
            pmuon_eps=float(pmuon_eps),
            ns_steps=int(ns_steps),
            weight_decay=float(weight_decay),
            soda=soda,
            soda_disables_weight_decay=bool(soda_disables_weight_decay),
            sync_diagnostics=bool(sync_diagnostics),
            min_matrix_dim=int(min_matrix_dim),
            anchor_step=0,
            anchor_weight_sum=0.0,
            anchor_beta_current=b1,
        )
        super().__init__(params, defaults)
        self._train_mode = True
        self.last_stats: dict[str, float] = {}

    @torch.no_grad()
    def eval(self) -> "AnchorMuon":
        """Switch parameters from AMUSE training weights ``y`` to averaged weights ``x``."""

        if not self._train_mode:
            return self
        for group in self.param_groups:
            if not bool(group.get("amuse", True)):
                continue
            beta = float(group.get("anchor_beta_current", group["beta1"]))
            beta = min(max(beta, 1e-12), 1.0 - 1e-12)
            for param in group["params"]:
                state = self.state[param]
                z = state.get("z")
                if z is not None:
                    param.lerp_(end=z.to(device=param.device, dtype=param.dtype), weight=1.0 - 1.0 / beta)
        self._train_mode = False
        return self

    @torch.no_grad()
    def train(self) -> "AnchorMuon":
        """Switch parameters from averaged weights ``x`` back to training weights ``y``."""

        if self._train_mode:
            return self
        for group in self.param_groups:
            if not bool(group.get("amuse", True)):
                continue
            beta = float(group.get("anchor_beta_current", group["beta1"]))
            for param in group["params"]:
                state = self.state[param]
                z = state.get("z")
                if z is not None:
                    param.lerp_(end=z.to(device=param.device, dtype=param.dtype), weight=1.0 - beta)
        self._train_mode = True
        return self

    def _group_schedule(self, group: dict[str, Any]) -> tuple[int, float, float, float]:
        k = int(group.get("anchor_step", 0))
        t = k + 1
        warmup = max(1, int(group["warmup_steps"]))
        if bool(group["use_external_lr"]):
            lr = float(group["lr"])
        else:
            base_lr = float(group.get("base_lr", group["lr"]))
            lr = base_lr * min(1.0, t / warmup)

        if not bool(group.get("amuse", True)):
            group["anchor_step"] = t
            group["anchor_weight_sum"] = float(group.get("anchor_weight_sum", 0.0))
            group["anchor_beta_current"] = 1.0
            group["anchor_ckp1"] = 1.0
            group["lr"] = lr
            return t, lr, 1.0, 1.0

        weight = (t ** float(group["r"])) * (lr ** float(group["weight_lr_power"]))
        weight_sum = float(group.get("anchor_weight_sum", 0.0)) + weight
        ckp1 = weight / weight_sum if weight_sum > 0.0 else 1.0

        beta_init = float(group["beta1"])
        if t <= warmup:
            beta = beta_init
            if t == warmup:
                group["anchor_c_warmup"] = ckp1
        else:
            c_warmup = float(group.get("anchor_c_warmup", 1.0 / warmup))
            denom = max(c_warmup * (1.0 - ckp1), 1e-24)
            s_t = (ckp1 * (1.0 - c_warmup)) / denom
            beta = 1.0 - (s_t ** float(group["rho"])) * (1.0 - beta_init)
        beta = min(max(beta, 1e-12), 1.0 - 1e-12)

        group["anchor_step"] = t
        group["anchor_weight_sum"] = weight_sum
        group["anchor_beta_current"] = beta
        group["anchor_ckp1"] = ckp1
        group["lr"] = lr
        return t, lr, ckp1, beta

    def _use_muon_for_param(self, group: dict[str, Any], param: Tensor) -> bool:
        if "use_muon" in group:
            return bool(group["use_muon"]) and len(_effective_matrix_shape(param)) >= 2
        effective_shape = _effective_matrix_shape(param)
        if len(effective_shape) < 2:
            return False
        rows = effective_shape[0]
        cols = math.prod(effective_shape[1:])
        return min(rows, cols) >= int(group["min_matrix_dim"])

    def _soda_for_param(self, group: dict[str, Any], *, use_muon: bool) -> bool:
        mode = group.get("soda", "matrix")
        if isinstance(mode, str):
            normalized = mode.lower()
            if normalized in {"matrix", "matrices", "muon"}:
                return bool(use_muon)
            if normalized in {"all", "true", "yes", "on"}:
                return True
            if normalized in {"none", "false", "no", "off"}:
                return False
            raise ValueError(f"unknown soda mode {mode!r}")
        return bool(mode)

    def _ensure_common_state(self, param: Tensor, state: dict[str, Any], *, soda: bool) -> None:
        if "z" not in state or tuple(state["z"].shape) != tuple(param.shape) or state["z"].device != param.device:
            state["z"] = _as_float(param).clone(memory_format=torch.preserve_format)
        if soda and (
            "soda_init" not in state
            or tuple(state["soda_init"].shape) != tuple(param.shape)
            or state["soda_init"].device != param.device
        ):
            state["soda_init"] = _as_float(param).clone(memory_format=torch.preserve_format)

    def _ensure_matrix_state(self, param: Tensor, state: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        matrix, _ = _matrix_view(param)
        rows, cols = matrix.shape
        momentum = _state_tensor(state, "momentum", shape=(rows, cols), device=param.device, fill=0.0)
        row_ema = _state_tensor(state, "row_ema", shape=(rows,), device=param.device, fill=1.0)
        col_ema = _state_tensor(state, "col_ema", shape=(cols,), device=param.device, fill=1.0)
        return momentum, row_ema, col_ema

    def _ensure_normuon_state(self, param: Tensor, state: dict[str, Any]) -> Tensor:
        matrix, _ = _matrix_view(param)
        rows, cols = matrix.shape
        shape = (rows, 1) if rows >= cols else (1, cols)
        return _state_tensor(state, "normuon_second_moment", shape=shape, device=param.device, fill=0.0)

    def _ensure_fallback_state(self, param: Tensor, state: dict[str, Any]) -> Tensor:
        return _state_tensor(state, "exp_avg_sq", shape=tuple(param.shape), device=param.device, fill=0.0)

    @torch.no_grad()
    def step(self, closure=None):
        if not self._train_mode:
            raise RuntimeError("AnchorMuon.step() called while optimizer is in eval/x mode; call optimizer.train() first")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_params = 0
        matrix_params = 0
        gram_params = 0
        mimuon_params = 0
        normuon_params = 0
        normuon_aspect_params = 0
        update_rms_sum = 0.0
        precond_rms_sum = 0.0
        sync_diagnostics_count = 0
        lr_last = 0.0
        beta_last = 0.0

        for group in self.param_groups:
            t, lr, ckp1, beta = self._group_schedule(group)
            lr_last = lr
            beta_last = beta
            beta2 = float(group["beta2"])
            momentum_beta = float(group["momentum"])
            group_weight_decay = float(group["weight_decay"])
            amuse = bool(group.get("amuse", True))
            sync_diagnostics = bool(group.get("sync_diagnostics", False))

            for param in group["params"]:
                if param.grad is None:
                    continue
                total_params += 1
                grad = _as_float(param.grad)
                state = self.state[param]
                use_muon = self._use_muon_for_param(group, param)
                soda = self._soda_for_param(group, use_muon=use_muon)
                weight_decay = 0.0 if (soda and bool(group["soda_disables_weight_decay"])) else group_weight_decay
                self._ensure_common_state(param, state, soda=soda)
                state["step"] = t
                z = state["z"]

                if amuse:
                    # Convert y_t -> x_t. Gradients were already evaluated at y_t.
                    param.lerp_(end=z.to(device=param.device, dtype=param.dtype), weight=1.0 - 1.0 / beta)
                    old_base = _as_float(param).clone(memory_format=torch.preserve_format) if soda else None
                else:
                    old_base = z.clone(memory_format=torch.preserve_format) if soda else None

                if use_muon:
                    matrix_params += 1
                    grad_matrix, original_shape = _matrix_view(grad)
                    momentum, row_ema, col_ema = self._ensure_matrix_state(param, state)
                    momentum.lerp_(grad_matrix, 1.0 - momentum_beta)
                    nesterov_update = torch.lerp(grad_matrix, momentum, momentum_beta)
                    if bool(group["pmuon_eq"]):
                        preconditioned = pmuon_eq_precondition(
                            grad_matrix,
                            nesterov_update,
                            row_ema,
                            col_ema,
                            beta=float(group["pmuon_beta"]),
                            row_gamma=float(group["row_gamma"]),
                            col_gamma=float(group["col_gamma"]),
                            eps=float(group["pmuon_eps"]),
                        )
                    else:
                        preconditioned = nesterov_update.float()
                    if bool(group.get("use_gram", True)):
                        gram_params += 1
                        muon_matrix = gram_newton_schulz(
                            preconditioned,
                            steps=int(group["ns_steps"]),
                            eps=float(group["pmuon_eps"]),
                        )
                    else:
                        muon_matrix = normalized_matrix_direction(
                            preconditioned,
                            eps=float(group["pmuon_eps"]),
                        )
                    if bool(group["normuon"]):
                        normuon_params += 1
                        if bool(group.get("normuon_aspect_scale", False)):
                            normuon_aspect_params += 1
                        normuon_second = self._ensure_normuon_state(param, state)
                        muon_matrix = normuon_normalize_update(
                            muon_matrix,
                            normuon_second,
                            beta=float(group["normuon_beta"]),
                            eps=float(group["normuon_eps"]),
                            aspect_scale=bool(group.get("normuon_aspect_scale", False)),
                        )
                    muon_matrix = muon_matrix * (0.2 * math.sqrt(max(muon_matrix.shape[-2], muon_matrix.shape[-1])))
                    if bool(group["mimuon"]):
                        mimuon_params += 1
                        raw_matrix = preconditioned.float()
                        raw_rms = raw_matrix.square().mean().sqrt().clamp_min(float(group["pmuon_eps"]))
                        muon_rms = muon_matrix.square().mean().sqrt().clamp_min(float(group["pmuon_eps"]))
                        raw_matrix = raw_matrix * (muon_rms / raw_rms)
                        mix = float(group["mimuon_mix"])
                        update_matrix = mix * muon_matrix + (1.0 - mix) * raw_matrix
                    else:
                        update_matrix = muon_matrix
                    if sync_diagnostics:
                        precond_rms_sum += float(preconditioned.float().square().mean().sqrt().detach().cpu())
                    update = _restore_matrix_view(update_matrix, original_shape)
                else:
                    exp_avg_sq = self._ensure_fallback_state(param, state)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = (exp_avg_sq / max(1.0 - beta2**t, float(group["eps"]))).sqrt().add_(float(group["eps"]))
                    update = grad / denom

                if weight_decay != 0.0:
                    z.mul_(1.0 - lr * weight_decay)
                z.add_(update.to(device=z.device, dtype=torch.float32), alpha=-lr)

                if amuse:
                    # x_{t+1} = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1}
                    param.lerp_(end=z.to(device=param.device, dtype=param.dtype), weight=ckp1)

                    if soda and old_base is not None:
                        init = state["soda_init"].to(device=param.device, dtype=torch.float32)
                        lam = 1.0 / float(t + 1)
                        param.add_((init - old_base).to(param.dtype), alpha=lam)

                    # Return to AMUSE's train-time interpolation point y_{t+1}.
                    param.lerp_(end=z.to(device=param.device, dtype=param.dtype), weight=1.0 - beta)
                else:
                    if soda and old_base is not None:
                        init = state["soda_init"].to(device=param.device, dtype=torch.float32)
                        lam = 1.0 / float(t + 1)
                        z.add_(init - old_base, alpha=lam)
                    param.copy_(z.to(device=param.device, dtype=param.dtype))
                if sync_diagnostics:
                    sync_diagnostics_count += 1
                    update_rms_sum += float(update.float().square().mean().sqrt().detach().cpu())

        self.last_stats = {
            "step": float(max((int(g.get("anchor_step", 0)) for g in self.param_groups), default=0)),
            "lr": float(lr_last),
            "beta": float(beta_last),
            "params": float(total_params),
            "matrix_params": float(matrix_params),
            "gram_params": float(gram_params),
            "mimuon_params": float(mimuon_params),
            "normuon_params": float(normuon_params),
            "normuon_aspect_params": float(normuon_aspect_params),
            "sync_diagnostics": float(sync_diagnostics_count > 0),
        }
        if sync_diagnostics_count > 0:
            self.last_stats.update(
                {
                    "mean_update_rms": update_rms_sum / max(sync_diagnostics_count, 1),
                    "mean_precond_matrix_rms": precond_rms_sum / max(matrix_params, 1),
                }
            )
        return loss


AnchorMuonEq = AnchorMuon
