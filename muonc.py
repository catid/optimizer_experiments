"""Standalone MuonC (MuonC + GS) optimizer.

This file is intentionally self-contained so it can be copied or imported into
another training job.  It is the optimizer used by the canonical ~600M
byte-level H-Net trainer, ported verbatim apart from removing unrelated
ablation optimizers:

    optimizer = MuonC(model, lr=1.2e-3, fallback_lr=6e-4,
                      matrix_weight_decay=0.05, fallback_weight_decay=0.05,
                      row_gamma=0.45, soda_lambda_scale=0.003)

Those are the H-Net trainer's values.  The bare constructor defaults are
inherited from AnchorMuon and were tuned on a CIFAR ViT (``lr=8e-3``, SODA
scale 1.0, no weight decay); do not use them unchanged for language models.

``MuonC`` is the root ``optimizer.py`` AnchorMuon direction (SODA + row-only
PMuonEq + Gram Newton-Schulz + NorMuon, AdamATan2 fallback) with three changes:

1. MuonGS tangent projection (arXiv:2610.37647).  After GramNS/NorMuon have
   shaped the direction ``N_t``, the Frobenius-radial component is removed:

       N'_t = N_t - <W_t, N_t>_F / ||W_t||_F^2 * W_t

   The paper calls this "Gram-Schmidt", but it is a single Frobenius
   tangent-space projection, not an iterative QR.  It runs *after* the
   PMuonEq/NorMuon transforms because a later row normalization would
   reintroduce a radial component.  To first order the update no longer
   changes ``||W||_F``; norm control is left to SODA and the decay term.
   Tag a parameter with ``param._no_gs = True`` to skip the projection.

2. Corrected decoupled weight decay (arXiv:2506.02285).  With scheduled rate
   ``lr_t`` and the group's unscheduled peak rate ``base_lr``:

       W_{t+1} = (1 - weight_decay * lr_t^2 / base_lr) * W_t - lr_t * N'_t

   ``base_lr`` is captured in each param group at construction; trainer-side
   schedulers must only modify ``group["lr"]``.  The fallback group applies
   the same schedule with its own peak ``base_lr``.  Tag a parameter with
   ``param._no_weight_decay = True`` to exempt it.

3. Persistent FP32 masters.  For non-FP32 (e.g. BF16) parameters the SODA
   pull, corrected decay, and update are accumulated in an FP32 master copy
   kept in optimizer state (``master_param``) and the low-precision parameter
   is refreshed from it every step, so small ``lr^2`` decay and AdamATan2
   updates are not lost to parameter quantization.  Because the master is the
   source of truth, code that writes to parameters outside the optimizer
   (EMA swaps, manual re-initialization) must not expect those writes to
   survive the next ``step()`` for non-FP32 parameters.

Full matrix step for weight ``W_t`` with gradient ``G_t``:

       M_t  = beta_m M_{t-1} + (1 - beta_m) G_t
       U_t  = (1 - beta_m) G_t + beta_m M_t
       r_t  = beta_p r_{t-1} + (1 - beta_p) mean_cols(G_t^2)
       A_t  = normalize(r_t^{-gamma_row})[:, None] * U_t
       D_t  = 0.2 * sqrt(max(rows, cols)) * GramNS(A_t)
       N_t  = NorMuon(D_t)                      # Frobenius norm preserved
       N'_t = N_t - <W_t, N_t>_F / ||W_t||_F^2 * W_t
       W_t <- (1 - soda_t) W_t + soda_t W_0     # soda_t = min(1, scale/(t+1)^power)
       W_{t+1} = (1 - wd * lr_t^2 / base_lr) W_t - lr_t N'_t

Differences from root ``optimizer.py`` beyond the three items above (these
come from the H-Net production trainer and also apply to the ``AnchorMuon``
base class defined in *this* file, which is therefore not interchangeable
with the root ``optimizer.AnchorMuon``):

* Tensors with more than two non-singleton dimensions are treated as banks of
  independent ``[rows, cols]`` matrices over the leading dimensions (stacked
  expert weights) rather than flattened to ``[dim0, rest]``.
* When names are available (``MuonC(model)`` / ``named_parameters()``),
  parameters whose name contains ``lm_head``, ``output_head``, ``conv1d`` or
  ``depthwise`` (and >=3-D ``conv`` kernels) are routed to the AdamATan2
  fallback even if matrix-shaped.
* GramNS uses the direct Gram polynomial recurrence with ``bmm`` for both
  square and rectangular matrices; the restarted inverse-estimate recurrence
  diverged for rectangular BF16/FP16 matrices on the production hardware.
* ``matrix_weight_decay`` exists (root AnchorMuon has SODA only), and
  ``use_soda`` / ``use_pmuoneq`` / ``use_normuon`` switches allow matched
  plain-Muon controls.  ``param._no_soda = True`` exempts a tensor from SODA.
* Fallback tensors are updated with batched ``torch._foreach_*`` kernels.

Muon-C (arXiv:2609.09676) is a different, convolution-specific Fourier-domain
optimizer and is not what this file implements.  ``MuonGS`` is exported as an
alias of ``MuonC``.

Learning-rate schedules are trainer-side: update ``group["lr"]`` each step and
leave ``group["base_lr"]`` alone.  The optimizer performs no distributed
communication; use normal DDP gradient all-reduce before ``step()``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import torch


__version__ = "0.5.1"

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
FALLBACK_MODES = {"rms", "atan2", "adamc"}



@torch.no_grad()
def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    effective_shape = tuple(int(dim) for dim in x.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        raise ValueError("matrix update requires at least two non-singleton dimensions")
    if len(effective_shape) == 2:
        return x.reshape(effective_shape)
    # Preserve leading dimensions as independent matrix banks. Transformer
    # Engine GroupedLinear stores expert weights as [experts, out, in];
    # flattening that tensor into [experts, out*in] would couple all experts
    # through one polar update and make the GS projection meaningless per
    # expert. The final two non-singleton dimensions are each matrix, while
    # any preceding dimensions are a batch of independent matrices.
    rows, cols = effective_shape[-2:]
    banks = math.prod(effective_shape[:-2])
    return x.reshape(banks, rows, cols)


@torch.no_grad()
def _diag_inverse_power(values: torch.Tensor, *, gamma: float, eps: float) -> torch.Tensor:
    x = values.to(torch.float32)
    scale = x.mean(dim=-1, keepdim=True).clamp_min(1.0) if x.ndim > 1 else x.mean().clamp_min(1.0)
    lam = (x + eps * scale).clamp_min(eps * scale)
    out = lam.pow(-float(gamma))
    norm = out.norm(dim=-1, keepdim=True).clamp_min(eps) if out.ndim > 1 else out.norm().clamp_min(eps)
    return out * (math.sqrt(out.size(-1)) / norm)


@torch.no_grad()
def _normuon_row_normalize(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    if tuple(second_momentum.shape[-2:]) != (update.shape[-2], 1):
        raise ValueError(
            f"NorMuon row second momentum shape {tuple(second_momentum.shape[-2:])} "
            f"!= {(update.shape[-2], 1)}"
        )
    old_norm = update.norm(dim=(-2, -1), keepdim=True)
    # Keep the running row statistic in the update dtype.  ``dtype`` was an
    # undefined local here, so every GramNS/AnchorMuon step crashed before the
    # first parameter update on the production model.
    row_power = update.square().mean(dim=-1, keepdim=True, dtype=update.dtype)
    second_momentum.lerp_(row_power, 1.0 - beta2)
    # Keep ``eps`` as a Python scalar.  Creating a device tensor here is a
    # host-to-device allocation during CUDA graph capture and makes the first
    # captured optimizer step fail with ``cudaErrorStreamCaptureUnsupported``.
    out = update * torch.rsqrt(second_momentum + eps)
    return out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps))


def _is_matrix_like_parameter(param: torch.Tensor, *, min_matrix_dim: int) -> bool:
    """Return whether ``param`` is large enough for the spectral matrix path.

    Singleton dimensions do not make a tensor matrix-like. For example, ViT
    ``cls_token`` with shape ``[1, 1, width]`` is effectively a vector, while
    ``pos_embed`` with shape ``[1, patches, width]`` is effectively a matrix.
    """

    if not param.is_floating_point():
        return False
    effective_shape = tuple(int(dim) for dim in param.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        return False
    rows, cols = effective_shape[-2:]
    return min(rows, cols) >= int(min_matrix_dim)


def _muon_name_excluded(name: str, param: torch.Tensor) -> bool:
    """Exclude matrices whose semantics are not compatible with Muon.

    Muon's polar update is appropriate for dense projection matrices.  A
    depthwise/convolutional kernel can look matrix-like after singleton
    dimensions are removed, and the vocabulary/output head benefits from the
    AdamATan2 fallback because its rows are directly supervised logits.  Keep
    those tensors on the numerically stable fallback path even when their
    effective shape passes the generic matrix test.
    """

    lowered = str(name).lower()
    if any(token in lowered for token in ("lm_head", "output_head", "conv1d", "depthwise")):
        return True
    if param.ndim >= 3 and any(token in lowered for token in (".conv", "conv.")):
        return True
    return False


class GramNewtonSchulz:
    """Batched Gram Newton-Schulz polar approximation used by AnchorMuon."""

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
        # Keep the recurrence's scalar operation order intact while exposing
        # the two batched matrix products to cuBLAS directly.  ``@`` creates
        # a temporary output for each product; explicit bmm outputs reuse the
        # same workspaces across the five Newton-Schulz iterations.  The
        # elementwise ``poly`` and ``x`` expressions intentionally remain
        # separate: folding them in-place changes BF16 rounding and therefore
        # the optimizer update.
        gram = torch.empty(
            (x.shape[0], x.shape[-2], x.shape[-2]),
            device=x.device,
            dtype=x.dtype,
        )
        gram_square = torch.empty_like(gram)
        product = torch.empty_like(x)
        for a, b, c in self.coefficients:
            torch.bmm(x, x.mT, out=gram)
            torch.bmm(gram, gram, out=gram_square)
            # Avoid cuBLASLt's failing nonzero-beta BF16/FP16 addmm path on
            # B300.  These are algebraically identical to baddbmm calls.
            poly = c * gram_square + b * gram
            torch.bmm(poly, x, out=product)
            x = a * x + product
        return x

    def _gram_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        # The inverse-estimate recurrence used by the upstream research
        # implementation diverges for this B300 workload with rectangular
        # BF16/FP16 matrices.  The direct Gram polynomial is algebraically
        # equivalent to the standard Newton-Schulz update and remains finite;
        # matmul plus scalar operations also avoids cuBLASLt's nonzero-beta
        # addmm failure.  The smaller Gram dimension is used after the caller's
        # transpose, so this remains practical for Transformer matrices.
        gram = torch.empty(
            (x.shape[0], x.shape[-2], x.shape[-2]),
            device=x.device,
            dtype=x.dtype,
        )
        gram_square = torch.empty_like(gram)
        product = torch.empty_like(x)
        for a, b, c in self.coefficients:
            torch.bmm(x, x.mT, out=gram)
            torch.bmm(gram, gram, out=gram_square)
            poly = c * gram_square + b * gram
            torch.bmm(poly, x, out=product)
            x = a * x + product
        return x


class AnchorMuon(torch.optim.Optimizer):
    """PyTorch optimizer implementing the focused AnchorMuon path.

    Preferred use:

        optimizer = AnchorMuon(model)

    Passing a ``torch.nn.Module``, ``model.parameters()``, or
    ``model.named_parameters()`` lets the optimizer route tensors by effective
    shape: matrix-like tensors use PMuonEq/GramNS/NorMuon, while scalar/vector
    tensors use the fallback path. Parameter names are accepted only for
    reporting in ``group_summary()``.

    Call ``optimizer.group_summary()`` before long runs to audit this routing.

    The defaults are intentionally usable. For a new workload, tune ``lr`` and
    ``row_gamma`` first; tune ``fallback_lr`` and ``normuon_beta2`` second;
    leave the remaining arguments fixed unless a targeted diagnostic gives a
    reason to move them. AnchorMuon consumes the current param-group ``lr`` and
    does not own warmup or decay schedules; use normal trainer-side schedulers.
    """

    def __init__(
        self,
        params: ParamsT | Iterable[tuple[str, torch.nn.Parameter]] | torch.nn.Module,
        *,
        lr: float = 8e-3,
        fallback_lr: float | None = None,
        fallback_mode: str = "atan2",
        momentum: float = 0.95,
        fallback_betas: tuple[float, float] = (0.9, 0.95),
        fallback_weight_decay: float = 0.0,
        eps: float = 1e-8,
        soda_lambda_scale: float = 1.0,
        soda_lambda_power: float = 1.0,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.35,
        pmuoneq_eps: float = 1e-6,
        normuon_beta2: float = 0.93,
        normuon_eps: float = 1e-10,
        ns_epsilon: float = 1e-7,
        ns_compute_dtype: torch.dtype | None = None,
        min_matrix_dim: int = 2,
        use_soda: bool = True,
        use_pmuoneq: bool = True,
        use_normuon: bool = True,
        matrix_weight_decay: float = 0.0,
    ) -> None:
        if isinstance(params, torch.nn.Module):
            params = params.named_parameters()
        fallback_lr = lr if fallback_lr is None else fallback_lr
        if lr < 0.0 or fallback_lr < 0.0:
            raise ValueError("learning rates must be non-negative")
        fallback_mode = str(fallback_mode).lower()
        if fallback_mode not in FALLBACK_MODES:
            raise ValueError(f"fallback_mode must be one of {sorted(FALLBACK_MODES)}")
        if fallback_weight_decay < 0.0:
            raise ValueError("fallback_weight_decay must be non-negative")
        if matrix_weight_decay < 0.0:
            raise ValueError("matrix_weight_decay must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= pmuoneq_beta < 1.0:
            raise ValueError("pmuoneq_beta must be in [0, 1)")
        if row_gamma < 0.0:
            raise ValueError("row_gamma must be non-negative")
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError("normuon_beta2 must be in [0, 1)")
        if len(fallback_betas) != 2 or not all(0.0 <= beta < 1.0 for beta in fallback_betas):
            raise ValueError("fallback_betas must contain two values in [0, 1)")
        if eps <= 0.0 or pmuoneq_eps <= 0.0 or normuon_eps <= 0.0 or ns_epsilon <= 0.0:
            raise ValueError("epsilon values must be positive")
        if min_matrix_dim < 1:
            raise ValueError("min_matrix_dim must be >= 1")
        if soda_lambda_scale <= 0.0 and use_soda:
            raise ValueError("soda_lambda_scale must be positive; SODA is always enabled")
        if soda_lambda_power <= 0.0:
            raise ValueError("soda_lambda_power must be positive")
        prepared = self._prepare_param_groups(
            params,
            lr=lr,
            fallback_lr=fallback_lr,
            fallback_mode=fallback_mode,
            momentum=momentum,
            fallback_betas=fallback_betas,
            fallback_weight_decay=fallback_weight_decay,
            eps=eps,
            pmuoneq_beta=pmuoneq_beta,
            row_gamma=row_gamma,
            pmuoneq_eps=pmuoneq_eps,
            normuon_beta2=normuon_beta2,
            normuon_eps=normuon_eps,
            min_matrix_dim=min_matrix_dim,
        )
        super().__init__(prepared, defaults={})

        self.soda_lambda_scale = float(soda_lambda_scale)
        self.soda_lambda_power = float(soda_lambda_power)
        self.use_soda = bool(use_soda)
        self.use_pmuoneq = bool(use_pmuoneq)
        self.use_normuon = bool(use_normuon)
        self.matrix_weight_decay = float(matrix_weight_decay)
        # The canonical MuonC subclass enables persistent FP32 masters after
        # construction.  AnchorMuon keeps its historical in-place parameter
        # arithmetic so the running GramNS control remains reproducible.
        self.use_fp32_masters = False
        self._orthogonalizer = GramNewtonSchulz(epsilon=ns_epsilon, compute_dtype=ns_compute_dtype)
        self.last_stats: dict[str, float] = {}

        for group in self.param_groups:
            group.setdefault("k", 0)

    @classmethod
    def from_model(cls, model: torch.nn.Module, **kwargs: Any) -> "AnchorMuon":
        """Construct the optimizer from ``model.parameters()``.

        This is the least error-prone public API for normal training code.
        """

        # Preserve names so semantic exclusions (depthwise kernels and the
        # output head) are routed to AdamATan2 instead of being flattened into
        # the GramNS matrix path.
        return cls(model.named_parameters(), **kwargs)

    @staticmethod
    def _prepare_param_groups(
        params: ParamsT | Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        fallback_lr: float,
        fallback_mode: str,
        momentum: float,
        fallback_betas: tuple[float, float],
        fallback_weight_decay: float,
        eps: float,
        pmuoneq_beta: float,
        row_gamma: float,
        pmuoneq_eps: float,
        normuon_beta2: float,
        normuon_eps: float,
        min_matrix_dim: int,
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
            return build_param_groups(
                items,  # type: ignore[arg-type]
                lr=lr,
                fallback_lr=fallback_lr,
                fallback_mode=fallback_mode,
                momentum=momentum,
                fallback_betas=fallback_betas,
                fallback_weight_decay=fallback_weight_decay,
                eps=eps,
                pmuoneq_beta=pmuoneq_beta,
                row_gamma=row_gamma,
                pmuoneq_eps=pmuoneq_eps,
                normuon_beta2=normuon_beta2,
                normuon_eps=normuon_eps,
                min_matrix_dim=min_matrix_dim,
            )
        if isinstance(items[0], dict):
            groups: list[dict[str, Any]] = []
            seen_params: set[int] = set()

            def apply_defaults(group: dict[str, Any]) -> None:
                is_matrix = bool(group["use_matrix_update"])
                group.setdefault("lr", lr if is_matrix else fallback_lr)
                group.setdefault("base_lr", float(group["lr"]))
                if "fallback_lr" in group:
                    group.setdefault("base_fallback_lr", float(group["fallback_lr"]))
                group.pop("weight_decay", None)
                if is_matrix:
                    group.setdefault("momentum", momentum)
                    group.setdefault("pmuoneq_beta", pmuoneq_beta)
                    group.setdefault("row_gamma", row_gamma)
                    group.setdefault("pmuoneq_eps", pmuoneq_eps)
                    group.setdefault("normuon_beta2", normuon_beta2)
                    group.setdefault("normuon_eps", normuon_eps)
                    group.setdefault("min_matrix_dim", min_matrix_dim)
                else:
                    group.setdefault("fallback_mode", fallback_mode)
                    group.setdefault("betas", fallback_betas)
                    group.setdefault("fallback_weight_decay", fallback_weight_decay)
                    group.setdefault("eps", eps)
                if is_matrix:
                    group.setdefault("fallback_mode", fallback_mode)
                    group.setdefault("betas", fallback_betas)
                    group.setdefault("fallback_weight_decay", fallback_weight_decay)
                    group.setdefault("eps", eps)

            def normalized_params(group: dict[str, Any]) -> tuple[list[torch.Tensor], list[str], bool]:
                params_value = group.get("params", [])
                params_in = [params_value] if isinstance(params_value, torch.Tensor) else list(params_value)
                names_in = list(group.get("param_names", []))
                keep_names = len(names_in) == len(params_in)
                params_out: list[torch.Tensor] = []
                names_out: list[str] = []
                for idx, p in enumerate(params_in):
                    ident = id(p)
                    if ident in seen_params:
                        continue
                    seen_params.add(ident)
                    params_out.append(p)
                    if keep_names:
                        names_out.append(names_in[idx])
                return params_out, names_out, keep_names

            for original in items:  # type: ignore[assignment]
                group = dict(original)
                params_in, names_in, keep_names = normalized_params(group)
                if not params_in:
                    continue
                group["params"] = params_in
                if keep_names:
                    group["param_names"] = names_in
                else:
                    group.pop("param_names", None)
                route_is_explicit = "use_matrix_update" in group or "use_muon" in group
                if route_is_explicit:
                    if "use_matrix_update" not in group:
                        group["use_matrix_update"] = group["use_muon"]
                    apply_defaults(group)
                    groups.append(group)
                    continue

                matrix_params: list[torch.Tensor] = []
                matrix_names: list[str] = []
                fallback_params: list[torch.Tensor] = []
                fallback_names: list[str] = []
                for idx, p in enumerate(params_in):
                    use_matrix = (
                        isinstance(p, torch.Tensor)
                        and p.requires_grad
                        and _is_matrix_like_parameter(p, min_matrix_dim=min_matrix_dim)
                    )
                    if use_matrix:
                        matrix_params.append(p)
                        if keep_names:
                            matrix_names.append(names_in[idx])
                    else:
                        fallback_params.append(p)
                        if keep_names:
                            fallback_names.append(names_in[idx])

                base_group = {key: value for key, value in group.items() if key not in {"params", "param_names"}}
                if matrix_params:
                    matrix_group = dict(base_group)
                    matrix_group["params"] = matrix_params
                    if keep_names:
                        matrix_group["param_names"] = matrix_names
                    matrix_group["use_matrix_update"] = True
                    apply_defaults(matrix_group)
                    groups.append(matrix_group)
                if fallback_params:
                    fallback_group = dict(base_group)
                    fallback_group["params"] = fallback_params
                    if keep_names:
                        fallback_group["param_names"] = fallback_names
                    fallback_group["use_matrix_update"] = False
                    apply_defaults(fallback_group)
                    groups.append(fallback_group)
            return groups

        matrix_params: list[torch.Tensor] = []
        fallback_params: list[torch.Tensor] = []
        seen_params: set[int] = set()
        for p in items:  # type: ignore[assignment]
            if not isinstance(p, torch.Tensor):
                raise TypeError("params must be tensors or optimizer param-group dictionaries")
            ident = id(p)
            if ident in seen_params:
                continue
            seen_params.add(ident)
            if p.requires_grad and _is_matrix_like_parameter(p, min_matrix_dim=min_matrix_dim):
                matrix_params.append(p)
            else:
                fallback_params.append(p)

        groups: list[dict[str, Any]] = []
        if matrix_params:
            groups.append(
                {
                    "params": matrix_params,
                    "use_matrix_update": True,
                    "lr": lr,
                    "base_lr": lr,
                    "momentum": momentum,
                    "pmuoneq_beta": pmuoneq_beta,
                    "row_gamma": row_gamma,
                    "pmuoneq_eps": pmuoneq_eps,
                    "normuon_beta2": normuon_beta2,
                    "normuon_eps": normuon_eps,
                    "min_matrix_dim": min_matrix_dim,
                }
            )
        if fallback_params:
            groups.append(
                {
                    "params": fallback_params,
                    "use_matrix_update": False,
                    "lr": fallback_lr,
                    "base_lr": fallback_lr,
                    "fallback_mode": fallback_mode,
                    "betas": fallback_betas,
                    "fallback_weight_decay": fallback_weight_decay,
                    "eps": eps,
                }
            )
        return groups

    def train(self) -> "AnchorMuon":
        return self

    def eval(self) -> "AnchorMuon":
        return self

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore custom state without narrowing FP32 accumulators.

        PyTorch's generic loader casts floating state to the parameter dtype.
        MuonC keeps masters, anchors, and optimizer statistics in FP32 for
        BF16 parameters, so restore those tensors by parameter identity after
        the base loader reconstructs its id mapping.  Cloning also prevents a
        caller that reuses ``state_dict()`` from aliasing two optimizers.
        """

        saved_state = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        super().load_state_dict(state_dict)
        fp32_keys = {
            "master_param",
            "momentum_buffer",
            "pmuoneq_row_ema",
            "normuon_second_momentum",
            "exp_avg",
            "exp_avg_sq",
        }
        for saved_group, live_group in zip(saved_groups, self.param_groups, strict=True):
            for saved_id, parameter in zip(
                saved_group.get("params", []), live_group["params"], strict=True
            ):
                source = saved_state.get(saved_id)
                if not isinstance(source, dict):
                    continue
                destination = self.state[parameter]
                for key, value in source.items():
                    if not isinstance(value, torch.Tensor):
                        destination[key] = value
                        continue
                    target_dtype = value.dtype
                    if value.is_floating_point() and (
                        key in fp32_keys or (key == "soda_z0" and self.use_fp32_masters)
                    ):
                        target_dtype = torch.float32
                    destination[key] = value.detach().to(
                        device=parameter.device, dtype=target_dtype
                    ).clone(memory_format=torch.preserve_format)

    def group_summary(self) -> list[dict[str, Any]]:
        """Return a small serializable summary of optimizer parameter routing."""

        summary: list[dict[str, Any]] = []
        for index, group in enumerate(self.param_groups):
            params = list(group["params"])
            names = list(group.get("param_names", []))
            summary.append(
                {
                    "index": index,
                    "use_matrix_update": bool(group.get("use_matrix_update", False)),
                    "lr": float(group.get("lr", 0.0)),
                    "param_count": len(params),
                    "numel": int(sum(p.numel() for p in params)),
                    "named": bool(names),
                    "param_names": names,
                    "fallback_mode": str(group.get("fallback_mode", "atan2")),
                }
            )
        return summary

    def _soda_anchor(self, p: torch.Tensor) -> torch.Tensor:
        # Parameters explicitly marked ``_no_soda`` should not be pulled
        # toward their initialization. Returning the live update tensor (the
        # FP32 master for MuonC) also avoids an unused anchor state entry.
        if getattr(p, "_no_soda", False):
            return self._master_param(p)
        state = self.state[p]
        anchor = state.get("soda_z0")
        master = self._master_param(p)
        if anchor is None:
            anchor = state["soda_z0"] = torch.clone(master, memory_format=torch.preserve_format)
        elif anchor.device != master.device or anchor.dtype != master.dtype:
            anchor = state["soda_z0"] = anchor.to(device=master.device, dtype=master.dtype)
        return anchor

    def _master_param(self, p: torch.Tensor) -> torch.Tensor:
        """Return the update tensor for ``p``.

        MuonC uses a persistent FP32 copy for BF16 parameters so corrected
        decay and small optimizer updates survive parameter quantization.
        AnchorMuon leaves this as the live parameter for control parity.
        """

        if not self.use_fp32_masters or p.dtype == torch.float32:
            return p
        state = self.state[p]
        master = state.get("master_param")
        if not isinstance(master, torch.Tensor) or master.shape != p.shape:
            master = state["master_param"] = p.detach().to(
                device=p.device, dtype=torch.float32
            ).clone(memory_format=torch.preserve_format)
        elif master.device != p.device or master.dtype != torch.float32:
            master = state["master_param"] = master.to(device=p.device, dtype=torch.float32)
        return master

    def _copy_masters_to_parameters(
        self, parameters: list[torch.Tensor], masters: list[torch.Tensor]
    ) -> None:
        """Copy persistent FP32 updates back to low-precision parameters."""

        copy_params: list[torch.Tensor] = []
        copy_masters: list[torch.Tensor] = []
        for parameter, master in zip(parameters, masters, strict=True):
            if master is parameter:
                continue
            copy_params.append(parameter)
            copy_masters.append(master)
        if copy_params:
            # Foreach copy performs the dtype conversion in one multi-tensor
            # operation and is graph-safe because the optimizer runs after
            # graph replay.  Keep a scalar fallback for CPU diagnostics where
            # foreach copy is unavailable on older PyTorch builds.
            try:
                torch._foreach_copy_(copy_params, copy_masters)
            except AttributeError:
                for parameter, master in zip(copy_params, copy_masters, strict=True):
                    parameter.copy_(master.to(dtype=parameter.dtype))

    def _soda_weight(self, t: int, p: torch.Tensor | None = None) -> float:
        if p is not None and getattr(p, "_no_soda", False):
            return 0.0
        if not self.use_soda:
            return 0.0
        return min(1.0, self.soda_lambda_scale / float(t + 1) ** self.soda_lambda_power)

    def _pre_matrix_source(self, p: torch.Tensor, grad: torch.Tensor, momentum_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state[p]
        momentum = state.get("momentum_buffer")
        if isinstance(momentum, torch.Tensor) and momentum.shape == grad.shape:
            if momentum.device != grad.device or momentum.dtype != torch.float32:
                momentum = state["momentum_buffer"] = momentum.to(device=grad.device, dtype=torch.float32)
        else:
            momentum = state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
        g = grad.detach().to(torch.float32)
        momentum.lerp_(g, 1.0 - momentum_beta)
        source = torch.lerp(g, momentum, momentum_beta)
        return _matrix_view(source), _matrix_view(g)

    def _ensure_matrix_state(
        self,
        p: torch.Tensor,
        row_shape: tuple[int, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return PMuonEq/NorMuon state for every independent matrix row.

        A normal weight has ``row_shape=(rows,)``.  Stacked expert weights use
        ``row_shape=(experts, rows)``; keeping that leading bank dimension in
        the state prevents the optimizer from coupling expert matrices by
        flattening them into one large matrix.
        """
        state = self.state[p]
        row_ema = state.get("pmuoneq_row_ema")
        if isinstance(row_ema, torch.Tensor) and tuple(row_ema.shape) == row_shape:
            if row_ema.device != device or row_ema.dtype != torch.float32:
                row_ema = state["pmuoneq_row_ema"] = row_ema.to(device=device, dtype=torch.float32)
        else:
            row_ema = state["pmuoneq_row_ema"] = torch.ones(*row_shape, device=device, dtype=torch.float32)
        second = state.get("normuon_second_momentum")
        second_shape = (*row_shape, 1)
        if isinstance(second, torch.Tensor) and second.shape == second_shape:
            if second.device != device or second.dtype != torch.float32:
                second = state["normuon_second_momentum"] = second.to(device=device, dtype=torch.float32)
        else:
            second = state["normuon_second_momentum"] = torch.zeros(*second_shape, device=device, dtype=torch.float32)
        return row_ema, second

    def _transform_matrix_bucket(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        group: dict[str, Any],
    ) -> torch.Tensor:
        sources = torch.stack([entry[2].to(torch.float32) for entry in entries], dim=0)
        grads = torch.stack([entry[3].to(torch.float32) for entry in entries], dim=0)
        if sources.ndim < 3 or grads.shape != sources.shape:
            raise ValueError("matrix banks must have matching rank >= 3")

        row_buffers: list[torch.Tensor] = []
        second_buffers: list[torch.Tensor] = []
        for p, _anchor, source, _grad in entries:
            # ``source`` is [rows, cols] for ordinary weights and
            # [bank, rows, cols] for stacked expert weights.  State follows
            # every independent matrix in either representation.
            row_shape = tuple(int(v) for v in source.shape[:-1])
            row_ema, second = self._ensure_matrix_state(p, row_shape, grads.device)
            row_buffers.append(row_ema)
            second_buffers.append(second)

        row_stack = torch.stack(row_buffers, dim=0)
        if self.use_pmuoneq:
            beta_p = float(group["pmuoneq_beta"])
            row_stack.mul_(beta_p).add_(grads.square().mean(dim=-1), alpha=1.0 - beta_p)
            row_factor = _diag_inverse_power(
                row_stack, gamma=float(group["row_gamma"]), eps=float(group["pmuoneq_eps"])
            )
        else:
            row_factor = torch.ones_like(row_stack)

        preconditioned = sources * row_factor.unsqueeze(-1)
        update = self._orthogonalizer(preconditioned)
        update = update * (0.2 * math.sqrt(max(update.size(-2), update.size(-1))))

        second_stack = torch.stack(second_buffers, dim=0)
        if self.use_normuon:
            update = _normuon_row_normalize(
                update,
                second_stack,
                beta2=float(group["normuon_beta2"]),
                eps=float(group["normuon_eps"]),
            )

        # The rows in ``row_stack`` and ``second_stack`` are already laid out
        # as one bank per parameter.  Copying each row in a Python loop emits
        # one D2D launch per matrix bucket entry (249 launches in the canonical
        # model).  Foreach copy keeps the state transition identical while
        # issuing one multi-tensor operation per bank; this is also capture
        # safe because all destination tensors are persistent optimizer state.
        row_targets = [
            self.state[p]["pmuoneq_row_ema"]
            for p, _anchor, _source, _grad in entries
        ]
        second_targets = [
            self.state[p]["normuon_second_momentum"]
            for p, _anchor, _source, _grad in entries
        ]
        torch._foreach_copy_(row_targets, list(row_stack.unbind(0)))
        torch._foreach_copy_(second_targets, list(second_stack.unbind(0)))
        return update

    def _postprocess_matrix_updates(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        updates: torch.Tensor,
    ) -> torch.Tensor:
        """Apply an optional direction transform before parameter updates.

        The base AnchorMuon path intentionally preserves its historical update
        exactly.  MuonGS overrides this hook to remove the Frobenius-radial
        component after GramNS/NorMuon has finished shaping the direction.
        """

        del entries
        return updates

    def _matrix_decay_factor(self, group: dict[str, Any], lr: float) -> float:
        """Return the decoupled matrix weight-decay multiplier.

        AnchorMuon historically uses ordinary AdamW-style decay.  MuonC
        overrides this hook with the corrected MuonC schedule; keeping the
        operation behind a hook makes the control path byte-for-byte
        comparable and keeps the schedule free of optimizer-owned state.
        """

        weight_decay = float(self.matrix_weight_decay)
        if weight_decay == 0.0:
            return 1.0
        return 1.0 - lr * weight_decay

    def _fallback_decay_factor(self, group: dict[str, Any], lr: float) -> float:
        """Return the decoupled fallback-parameter decay multiplier."""

        weight_decay = float(group.get("fallback_weight_decay", 0.0))
        if weight_decay == 0.0:
            return 1.0
        return 1.0 - lr * weight_decay

    def _step_matrix_group(self, group: dict[str, Any], *, lr: float, t: int) -> dict[str, float]:
        beta_m = float(group["momentum"])
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        fallback_count = 0
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError(
                    "AnchorMuon does not support sparse gradients. "
                    "Use dense gradients for this parameter or a different optimizer for sparse embeddings."
                )
            if not _is_matrix_like_parameter(grad, min_matrix_dim=int(group.get("min_matrix_dim", 2))):
                fallback_count += self._step_fallback_param(p, group, lr=lr, t=t)
                continue
            anchor = self._soda_anchor(p)
            source, grad_matrix = self._pre_matrix_source(p, grad, beta_m)
            entries.append((p, anchor, source, grad_matrix))

        if not entries:
            return {
                "matrix_count": 0.0,
                "fallback_count": float(fallback_count),
                "soda_weight": float(self._soda_weight(t)),
            }

        buckets: dict[tuple[torch.device, torch.Size], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for entry in entries:
            buckets[(entry[2].device, entry[2].shape)].append(entry)

        soda_weight = self._soda_weight(t)
        matrix_count = 0
        for bucket_entries in buckets.values():
            updates = self._transform_matrix_bucket(bucket_entries, group)
            updates = self._postprocess_matrix_updates(bucket_entries, updates)
            # Most matrix parameters share a shape, so use multi-tensor
            # kernels for the three elementwise optimizer operations instead
            # of launching one lerp/mul/add per projection.  Split by dtype:
            # the model is BF16 while a few residual projections intentionally
            # remain FP32.  ``_no_soda``/``_no_weight_decay`` are kept exact by
            # filtering those semantic exclusions before each foreach call.
            split = defaultdict(list)
            for entry, update in zip(bucket_entries, updates.unbind(0), strict=True):
                p, anchor, _source, _grad = entry
                split[p.dtype].append((p, anchor, update.reshape_as(p)))
            for pairs in split.values():
                ps = [p for p, _a, _u in pairs]
                if self.use_fp32_masters:
                    destinations = [self._master_param(p) for p in ps]
                    anchors = [self._soda_anchor(p) for p in ps]
                else:
                    destinations = ps
                    anchors = [a for _p, a, _u in pairs]
                weights = {self._soda_weight(t, p) for p in ps}
                if len(weights) == 1 and next(iter(weights)):
                    torch._foreach_lerp_(destinations, anchors, next(iter(weights)))
                elif weights:
                    for destination, p, anchor, _u in zip(
                        destinations, ps, anchors, [u for _p, _a, u in pairs], strict=True
                    ):
                        weight = self._soda_weight(t, p)
                        if weight:
                            destination.lerp_(end=anchor, weight=weight)

                if self.matrix_weight_decay:
                    decay_pairs = [
                        destination
                        for destination, p in zip(destinations, ps, strict=True)
                        if not getattr(p, "_no_weight_decay", False)
                    ]
                    if decay_pairs:
                        torch._foreach_mul_(
                            decay_pairs, self._matrix_decay_factor(group, lr)
                        )

                update_values = [
                    u.to(dtype=destinations[0].dtype)
                    for _p, _a, u in pairs
                ]
                torch._foreach_add_(destinations, update_values, alpha=-lr)
                if self.use_fp32_masters:
                    self._copy_masters_to_parameters(ps, destinations)
                matrix_count += len(pairs)

        return {
            "matrix_count": float(matrix_count),
            "fallback_count": float(fallback_count),
            "soda_weight": float(soda_weight),
        }

    def _step_fallback_param(
        self,
        p: torch.Tensor,
        group: dict[str, Any],
        *,
        lr: float,
        t: int,
    ) -> int:
        grad = p.grad
        if grad is None:
            return 0
        if grad.is_sparse:
            raise RuntimeError(
                "AnchorMuon does not support sparse gradients. "
                "Use dense gradients for this parameter or a different optimizer for sparse embeddings."
            )
        state = self.state[p]
        mode = str(group.get("fallback_mode", "atan2")).lower()
        if mode not in FALLBACK_MODES:
            raise ValueError(f"fallback_mode must be one of {sorted(FALLBACK_MODES)}")
        exp_avg_sq = state.get("exp_avg_sq")
        if isinstance(exp_avg_sq, torch.Tensor) and exp_avg_sq.shape == p.shape:
            if exp_avg_sq.device != p.device or exp_avg_sq.dtype != torch.float32:
                exp_avg_sq = state["exp_avg_sq"] = exp_avg_sq.to(device=p.device, dtype=torch.float32)
        else:
            exp_avg_sq = state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] = 0
        exp_avg = state.get("exp_avg")
        if mode in {"atan2", "adamc"} and (
            not isinstance(exp_avg, torch.Tensor) or exp_avg.shape != p.shape
        ):
            exp_avg = state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        elif isinstance(exp_avg, torch.Tensor) and (exp_avg.device != p.device or exp_avg.dtype != torch.float32):
            exp_avg = state["exp_avg"] = exp_avg.to(device=p.device, dtype=torch.float32)
        state["step"] = int(state.get("step", 0)) + 1
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        g = grad.detach().to(torch.float32)
        destination = self._master_param(p)
        destination.lerp_(end=self._soda_anchor(p), weight=self._soda_weight(t, p))
        fallback_weight_decay = 0.0 if getattr(p, "_no_weight_decay", False) else float(group.get("fallback_weight_decay", 0.0))
        if fallback_weight_decay != 0.0:
            destination.mul_(self._fallback_decay_factor(group, lr))

        exp_avg_sq.mul_(float(beta2)).addcmul_(g, g, value=1.0 - float(beta2))
        step = int(state["step"])
        bias2 = max(1.0 - float(beta2) ** step, 1e-16)
        denom = (exp_avg_sq / bias2).sqrt().add_(eps)
        if mode == "rms":
            update = g / denom
        else:
            if exp_avg is None:
                raise RuntimeError("fallback first-moment state was not initialized")
            exp_avg.mul_(float(beta1)).add_(g, alpha=1.0 - float(beta1))
            bias1 = max(1.0 - float(beta1) ** step, 1e-16)
            if mode == "atan2":
                update = torch.atan2(exp_avg / bias1, denom)
            else:
                update = (exp_avg / bias1) / denom
        destination.add_(update.to(destination.dtype), alpha=-lr)
        if self.use_fp32_masters:
            self._copy_masters_to_parameters([p], [destination])
        return 1

    def _step_fallback_group(self, group: dict[str, Any], *, lr: float, t: int) -> dict[str, float]:
        """Apply the scalar/vector fallback in batched foreach launches.

        The production model has roughly one hundred RMS/state/router vectors
        in this group.  Updating each one through ``_step_fallback_param``
        launches a separate CUDA kernel for every multiply/add/reduction.  The
        math is unchanged, but the state updates and elementwise transforms can
        be issued as a handful of multi-tensor kernels.  Keep the old path for
        unusual resumed groups whose per-parameter step counters differ (bias
        correction must remain exact in that case).
        """
        entries = []
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            state = self.state[p]
            exp_avg_sq = state.get("exp_avg_sq")
            exp_avg = state.get("exp_avg")
            mode = str(group.get("fallback_mode", "atan2")).lower()
            if mode not in FALLBACK_MODES:
                raise ValueError(f"fallback_mode must be one of {sorted(FALLBACK_MODES)}")
            if (
                not isinstance(exp_avg_sq, torch.Tensor)
                or exp_avg_sq.shape != p.shape
                or exp_avg_sq.device != p.device
                or exp_avg_sq.dtype != torch.float32
            ):
                exp_avg_sq = state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] = 0
            if mode in {"atan2", "adamc"} and (
                not isinstance(exp_avg, torch.Tensor)
                or exp_avg.shape != p.shape
                or exp_avg.device != p.device
                or exp_avg.dtype != torch.float32
            ):
                exp_avg = state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] = int(state.get("step", 0)) + 1
            entries.append((p, grad.detach().to(torch.float32), exp_avg_sq, exp_avg))

        fallback_count = len(entries)
        if not entries:
            return {"matrix_count": 0.0, "fallback_count": 0.0, "soda_weight": float(self._soda_weight(t))}

        # A resumed optimizer can contain mixed counters when some parameters
        # had no gradient in an earlier step.  Preserve the scalar reference
        # implementation for that rare case; normal DDP training takes the
        # fast common path below.
        steps = {int(self.state[p]["step"]) for p, _g, _sq, _avg in entries}
        if len(steps) != 1:
            for p, _g, _sq, _avg in entries:
                # The batched preparation incremented the counter once; hand
                # control back to the scalar path with the original value so
                # it performs exactly one increment.
                self.state[p]["step"] -= 1
                self._step_fallback_param(p, group, lr=lr, t=t)
            return {"matrix_count": 0.0, "fallback_count": float(fallback_count), "soda_weight": float(self._soda_weight(t))}

        params = [p for p, _g, _sq, _avg in entries]
        grads = [g for _p, g, _sq, _avg in entries]
        exp_avg_sq = [sq for _p, _g, sq, _avg in entries]
        exp_avg = [avg for _p, _g, _sq, avg in entries]
        beta1, beta2 = (float(v) for v in group.get("betas", (0.9, 0.999)))
        eps = float(group.get("eps", 1e-8))
        mode = str(group.get("fallback_mode", "atan2")).lower()

        # SODA/decay are scalar operations on the parameter dtype.  Grouping
        # by the semantic marker keeps no-SODA frequency parameters exact.
        soda_weight = float(self._soda_weight(t))
        soda_params, soda_anchors = [], []
        for p in params:
            weight = self._soda_weight(t, p)
            if weight:
                soda_params.append(p)
                soda_anchors.append(self._soda_anchor(p))
        if soda_params:
            # Foreach kernels require a common destination dtype.  Production
            # groups are predominantly BF16 but contain a small FP32 set of
            # residual projections, so split those without changing math.
            soda_by_dtype = defaultdict(list)
            for p, anchor in zip(soda_params, soda_anchors):
                soda_by_dtype[p.dtype].append((p, anchor))
            for pairs in soda_by_dtype.values():
                ps = [p for p, _a in pairs]
                destinations = [self._master_param(p) for p in ps] if self.use_fp32_masters else ps
                anchors = [self._soda_anchor(p) for p in ps] if self.use_fp32_masters else [a for _p, a in pairs]
                if all(self._soda_weight(t, p) == self._soda_weight(t, ps[0]) for p in ps):
                    torch._foreach_lerp_(destinations, anchors, self._soda_weight(t, ps[0]))
                else:
                    for destination, (p, anchor) in zip(destinations, pairs, strict=True):
                        if self.use_fp32_masters:
                            anchor = self._soda_anchor(p)
                        destination.lerp_(end=anchor, weight=self._soda_weight(t, p))
        destinations = [self._master_param(p) for p in params] if self.use_fp32_masters else params
        for p, destination in zip(params, destinations, strict=True):
            decay = 0.0 if getattr(p, "_no_weight_decay", False) else float(group.get("fallback_weight_decay", 0.0))
            if decay:
                destination.mul_(self._fallback_decay_factor(group, lr))

        torch._foreach_mul_(exp_avg_sq, beta2)
        torch._foreach_addcmul_(exp_avg_sq, grads, grads, value=1.0 - beta2)
        step = next(iter(steps))
        bias2 = max(1.0 - beta2 ** step, 1e-16)
        denoms = torch._foreach_div(exp_avg_sq, bias2)
        denoms = torch._foreach_sqrt(denoms)
        torch._foreach_add_(denoms, eps)
        if mode == "rms":
            updates = torch._foreach_div(grads, denoms)
        else:
            exp_avg = [avg for _p, _g, _sq, avg in entries]
            torch._foreach_mul_(exp_avg, beta1)
            torch._foreach_add_(exp_avg, grads, alpha=1.0 - beta1)
            bias1 = max(1.0 - beta1 ** step, 1e-16)
            numerators = torch._foreach_div(exp_avg, bias1)
            if mode == "atan2":
                # PyTorch has no _foreach_atan2.  This is one launch per
                # vector, but all state updates above are fused; the previous
                # implementation launched several kernels per vector.
                updates = [torch.atan2(num, den) for num, den in zip(numerators, denoms)]
            else:
                updates = torch._foreach_div(numerators, denoms)
        # The optimizer stores BF16 model parameters while the fallback state
        # remains FP32.  Cast once per parameter, then issue grouped adds for
        # each destination dtype (foreach rejects mixed destination dtypes).
        by_dtype = defaultdict(list)
        for p, update in zip(params, updates):
            destination = self._master_param(p) if self.use_fp32_masters else p
            by_dtype[destination.dtype].append((p, destination, update.to(dtype=destination.dtype)))
        for pairs in by_dtype.values():
            torch._foreach_add_([destination for _p, destination, _u in pairs], [u for _p, _destination, u in pairs], alpha=-lr)
            if self.use_fp32_masters:
                self._copy_masters_to_parameters(
                    [p for p, _destination, _u in pairs],
                    [destination for _p, destination, _u in pairs],
                )
        return {
            "matrix_count": 0.0,
            "fallback_count": float(fallback_count),
            "soda_weight": soda_weight,
        }

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stats = {
            "matrix_count": 0.0,
            "fallback_count": 0.0,
            "soda_weight_sum": 0.0,
            "group_count": 0.0,
        }
        for group in self.param_groups:
            k = int(group.get("k", 0))
            t = k + 1
            lr = float(group["lr"])

            if group.get("use_matrix_update", False):
                group_stats = self._step_matrix_group(group, lr=lr, t=t)
            else:
                group_stats = self._step_fallback_group(group, lr=lr, t=t)
            stats["matrix_count"] += group_stats["matrix_count"]
            stats["fallback_count"] += group_stats["fallback_count"]
            stats["soda_weight_sum"] += group_stats["soda_weight"]
            stats["group_count"] += 1.0
            group["k"] = t

        stats["soda_weight"] = stats["soda_weight_sum"] / max(stats["group_count"], 1.0)
        del stats["soda_weight_sum"]
        self.last_stats = stats
        return loss


class MuonC(AnchorMuon):
    """Corrected-decay Muon with the MuonGS Frobenius projection (MuonC+GS).

    Two independent modifications of the AnchorMuon/Muon step are combined:

    * ``gs_projection`` (MuonGS, arXiv:2610.37647): the GramNS/NorMuon
      direction is projected onto the tangent space of the current matrix
      weight before decay and the parameter update.  The projection is
      performed independently for every matrix in a stacked bank, so expert
      matrices never share a radial component.
    * ``corrected_decay`` (MuonC, arXiv:2506.02285): the decay coefficient is
      scaled by the current learning rate divided by the group's peak (base)
      learning rate, so the multiplicative factor is
      ``1 - weight_decay * lr**2 / base_lr`` instead of
      ``1 - lr * weight_decay``.

    Both default to on, which is the H-Net production configuration.  Turning
    one off gives the matched single-modification control; persistent FP32
    masters stay enabled in every configuration so the comparison is not
    confounded by BF16 parameter quantization.

    ``track_gs_stats=True`` records per-step radial diagnostics in
    ``last_stats`` (``gs_radial_cosine_mean``, ``gs_radial_cosine_abs_max``,
    ``gs_tangent_norm_ratio_mean``).  This forces a host synchronization each
    step, so leave it off for throughput runs and CUDA graph capture.
    """

    def __init__(
        self,
        *args: Any,
        gs_eps: float = 1e-12,
        gs_projection: bool = True,
        corrected_decay: bool = True,
        track_gs_stats: bool = False,
        **kwargs: Any,
    ) -> None:
        if gs_eps <= 0.0:
            raise ValueError("gs_eps must be positive")
        super().__init__(*args, **kwargs)
        # Corrected Muon decay and small SODA/AdamATan2 updates must be
        # accumulated in FP32 before the BF16 model copy is refreshed.
        self.use_fp32_masters = True
        self.gs_eps = float(gs_eps)
        self.gs_projection = bool(gs_projection)
        self.corrected_decay = bool(corrected_decay)
        self.track_gs_stats = bool(track_gs_stats)
        self._gs_stat_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _postprocess_matrix_updates(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        updates: torch.Tensor,
    ) -> torch.Tensor:
        if not self.gs_projection and not self.track_gs_stats:
            return updates
        # Keep the projection in FP32 even when the model and GramNS compute
        # path use BF16.  All shapes are static for the canonical trainer, and
        # this contains only device-side tensor operations, so CUDA graph
        # capture/replay never needs a host scalar or synchronization.
        weights = torch.stack(
            [_matrix_view(self._master_param(parameter).detach()) for parameter, _a, _s, _g in entries],
            dim=0,
        ).to(dtype=torch.float32)
        directions = updates.to(dtype=torch.float32)
        if directions.shape != weights.shape:
            raise ValueError(
                "MuonGS matrix/update shape mismatch: "
                f"weights={tuple(weights.shape)} updates={tuple(directions.shape)}"
            )
        dot = (weights * directions).sum(dim=(-2, -1), keepdim=True)
        norm_sq = weights.square().sum(dim=(-2, -1), keepdim=True)
        radial = dot / norm_sq.clamp_min(self.gs_eps)
        projected = directions - radial * weights
        if self.track_gs_stats:
            direction_norm = directions.norm(dim=(-2, -1), keepdim=True).clamp_min(self.gs_eps)
            cosine = dot / (norm_sq.sqrt().clamp_min(self.gs_eps) * direction_norm)
            ratio = projected.norm(dim=(-2, -1), keepdim=True) / direction_norm
            self._gs_stat_buffer.append((cosine.flatten(), ratio.flatten()))
        if not self.gs_projection:
            return updates
        skip = torch.tensor(
            [bool(getattr(parameter, "_no_gs", False)) for parameter, _a, _s, _g in entries],
            device=weights.device,
            dtype=torch.bool,
        ).view(-1, *([1] * (weights.ndim - 1)))
        return torch.where(skip, directions, projected).to(dtype=updates.dtype)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        self._gs_stat_buffer = []
        loss = super().step(closure)
        if self.track_gs_stats and self._gs_stat_buffer:
            cosine = torch.cat([c for c, _r in self._gs_stat_buffer])
            ratio = torch.cat([r for _c, r in self._gs_stat_buffer])
            self.last_stats["gs_radial_cosine_mean"] = float(cosine.mean())
            self.last_stats["gs_radial_cosine_abs_max"] = float(cosine.abs().max())
            self.last_stats["gs_tangent_norm_ratio_mean"] = float(ratio.mean())
        self._gs_stat_buffer = []
        return loss

    @staticmethod
    def _corrected_decay_factor(weight_decay: float, lr: float, base_lr: float) -> float:
        """MuonC's LR-corrected decoupled decay factor.

        ``base_lr`` is the unscheduled peak LR captured when the group is
        constructed.  A zero peak LR means that the group is intentionally
        frozen, so no decay is applied rather than dividing by zero.
        """

        if weight_decay == 0.0 or base_lr <= 0.0 or lr == 0.0:
            return 1.0
        return 1.0 - weight_decay * (lr * lr / base_lr)

    def _matrix_decay_factor(self, group: dict[str, Any], lr: float) -> float:
        if not self.corrected_decay:
            return super()._matrix_decay_factor(group, lr)
        return self._corrected_decay_factor(
            float(self.matrix_weight_decay), lr, float(group.get("base_lr", lr))
        )

    def _fallback_decay_factor(self, group: dict[str, Any], lr: float) -> float:
        if not self.corrected_decay:
            return super()._fallback_decay_factor(group, lr)
        return self._corrected_decay_factor(
            float(group.get("fallback_weight_decay", 0.0)),
            lr,
            float(group.get("base_lr", lr)),
        )


# H-Net exposes this hybrid under both names, so ``MuonGS`` is kept as an alias
# for drop-in compatibility.  Note that the *paper's* MuonGS is the projection
# alone with no matrix weight decay; see the module docstring for that recipe.
MuonGS = MuonC


def build_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float = 8e-3,
    fallback_lr: float | None = None,
    fallback_mode: str = "atan2",
    momentum: float = 0.95,
    fallback_betas: tuple[float, float] = (0.9, 0.95),
    fallback_weight_decay: float = 0.0,
    eps: float = 1e-8,
    pmuoneq_beta: float = 0.90,
    row_gamma: float = 0.35,
    pmuoneq_eps: float = 1e-6,
    normuon_beta2: float = 0.93,
    normuon_eps: float = 1e-10,
    min_matrix_dim: int = 2,
) -> list[dict[str, Any]]:
    """Split named parameters into matrix and fallback groups.

    Routing uses effective tensor shape plus semantic exclusions. Singleton
    dimensions are ignored, so ``[1, 1, width]`` is a vector and
    ``[1, tokens, width]`` is a matrix; depthwise convolution kernels and the
    output head remain on AdamATan2 even when their shape looks matrix-like.
    Parameter names are retained for both routing and diagnostics.

    Tied parameters are deduplicated by object identity. The effective shape of
    the shared tensor decides the route regardless of aliases.
    """

    if min_matrix_dim < 1:
        raise ValueError("min_matrix_dim must be >= 1")
    fallback_lr = lr if fallback_lr is None else fallback_lr
    fallback_mode = str(fallback_mode).lower()
    if fallback_mode not in FALLBACK_MODES:
        raise ValueError(f"fallback_mode must be one of {sorted(FALLBACK_MODES)}")
    if fallback_weight_decay < 0.0:
        raise ValueError("fallback_weight_decay must be non-negative")

    matrix_params: list[torch.nn.Parameter] = []
    matrix_names: list[str] = []
    fallback_params: list[torch.nn.Parameter] = []
    fallback_names: list[str] = []

    unique: dict[int, tuple[torch.nn.Parameter, list[str]]] = {}
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        key = id(p)
        if key not in unique:
            unique[key] = (p, [name])
        else:
            unique[key][1].append(name)

    for _key, (p, names) in unique.items():
        display_name = "|".join(names)
        use_matrix = (
            not _muon_name_excluded(display_name, p)
            and _is_matrix_like_parameter(p, min_matrix_dim=min_matrix_dim)
        )
        if use_matrix:
            matrix_params.append(p)
            matrix_names.append(display_name)
        else:
            fallback_params.append(p)
            fallback_names.append(display_name)

    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "param_names": matrix_names,
                "use_matrix_update": True,
                "lr": lr,
                "base_lr": lr,
                "momentum": momentum,
                "pmuoneq_beta": pmuoneq_beta,
                "row_gamma": row_gamma,
                "pmuoneq_eps": pmuoneq_eps,
                "normuon_beta2": normuon_beta2,
                "normuon_eps": normuon_eps,
                "min_matrix_dim": min_matrix_dim,
            }
        )
    if fallback_params:
        groups.append(
            {
                "params": fallback_params,
                "param_names": fallback_names,
                "use_matrix_update": False,
                "lr": fallback_lr,
                "base_lr": fallback_lr,
                "fallback_mode": fallback_mode,
                "betas": fallback_betas,
                "fallback_weight_decay": fallback_weight_decay,
                "eps": eps,
            }
        )
    return groups


__all__ = [
    "__version__",
    "AnchorMuon",
    "GramNewtonSchulz",
    "MuonC",
    "MuonGS",
    "build_param_groups",
]
