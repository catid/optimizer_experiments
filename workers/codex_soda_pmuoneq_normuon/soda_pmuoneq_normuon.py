"""Standalone SODA-PMuonEq-NorMuon optimizer.

This file packages the best-performing optimizer recipe from the local ViT-5
CIFAR experiments as a focused, copyable PyTorch optimizer. It intentionally
removes the ablation switches from ``sodamuse_eq.py``:

    no AMUSE / schedule-free train-eval sequence
    no MiMuon branch
    no optional PMuonEq disable path
    no optional NorMuon disable path

The matrix path is always:

    SODA anchor + PMuonEq row/column scaling + Gram Newton-Schulz + NorMuon

By default NorMuon is the tuned row-wise variant with the aspect-ratio
multiplier enabled. The ``normuon_mode`` and ``normuon_aspect_scale`` arguments
exist only for compatibility and ablation:

    normuon_mode="row", normuon_aspect_scale=True
        winning/default recipe;

    normuon_mode="row", normuon_aspect_scale=False
        row-wise NorMuon without the extra tall-matrix multiplier;

    normuon_mode="orientation", normuon_aspect_scale=False
        use row statistics for tall matrices and column statistics for wide
        matrices, without the extra aspect multiplier.

    normuon_mode="orientation", normuon_aspect_scale=True
        the missing peer-requested ablation cell. It preserves orientation-aware
        statistics while applying the same tall-matrix aspect multiplier after
        NorMuon.

The fallback path for biases, norms, embeddings, heads, and other non-matrix
parameters is an RMS/AdamW-style second-moment adaptive update with ordinary
weight decay. It intentionally matches the fallback path used in the winning
local recipe.

Math
====

For a matrix parameter W_t and gradient G_t, keep a momentum buffer M_t:

    M_t = beta_m M_{t-1} + (1 - beta_m) G_t
    U_t = (1 - beta_m) G_t + beta_m M_t

PMuonEq uses cheap row/column gradient-power EMAs instead of dense PMuon
covariances:

    r_t = beta_p r_{t-1} + (1 - beta_p) mean_cols(G_t^2)
    c_t = beta_p c_{t-1} + (1 - beta_p) mean_rows(G_t^2)

    a_t = normalize(r_t^{-gamma_row})
    b_t = normalize(c_t^{-gamma_col})

    A_t = a_t[:, None] * U_t * b_t[None, :]

Then Gram Newton-Schulz produces the matrix direction:

    P_t = GramNS(A_t)
    D_t = 0.2 * sqrt(max(rows, cols)) * P_t

NorMuon applies row-wise second-moment normalization after GramNS while
preserving the Frobenius norm of D_t:

    s_t = beta_n s_{t-1} + (1 - beta_n) mean_cols(D_t^2)
    N_t = D_t / sqrt(s_t + eps)
    N_t = N_t * ||D_t||_F / (||N_t||_F + eps)
    N_t = N_t * sqrt(max(1, rows / cols))

The final aspect-ratio multiplier is intentional. It was present in the tuned
local recipe and acts like a fixed layerwise LR scale for tall matrices. Other
NorMuon implementations may preserve only the Frobenius norm; compare those
numbers separately rather than treating them as the same update rule.

SODA applies a scheduled pull toward the initialization anchor W_0 before the
learned update:

    lambda_t = min(1, soda_lambda_scale / (t + 1)^{soda_lambda_power})
    W_t <- (1 - lambda_t) W_t + lambda_t W_0
    W_{t+1} = W_t - lr_t N_t

For matrix parameters, ``soda_disables_matrix_weight_decay=True`` by default.
This keeps the default recipe in the "SODA replaces matrix weight decay" regime.
If matrix weight decay is enabled deliberately as an ablation, set this flag to
``False`` and report the optimizer as SODA plus decoupled matrix weight decay.

Best local tuning result
========================

On ViT-5 tiny / CIFAR-10, the best tuned recipe was:

    matrix_lr       = 8e-3
    adam_lr         = 8e-4
    momentum        = 0.95
    pmuoneq_beta    = 0.90
    row_gamma       = 0.35
    col_gamma       = 0.05
    normuon_beta2   = 0.93
    warmup_steps    = 10
    fallback RMS/AdamW-style weight_decay = 0.05
    matrix weight_decay = 0.0

Confidence runs, 4-GPU DDP:

    CIFAR-10, 50 epochs, 3 seeds:
        SODA-PMuonEq-NorMuon: val loss 0.4786 +/- 0.0196,
                              val acc  86.04% +/- 0.46
        tuned AdamW:          val loss 0.7263 +/- 0.0158,
                              val acc  80.90% +/- 0.55

    CIFAR-100, 30 epochs, 1 seed:
        SODA-PMuonEq-NorMuon: val loss 1.6907, val acc 59.06%
        tuned AdamW:          val loss 2.6706, val acc 48.71%

Speed tradeoff:

    SODA-PMuonEq-NorMuon was about 1.33x AdamW step time on the ViT-5 tiny
    CIFAR-10 confirmation runs. Use it when quality matters more than raw
    iterations/sec.

Latest NorMuon aspect ablation
==============================

The code path used for the latest shared monorepo run is in
``experiments/run_cifar10_normuon_aspect_ablation.py``. After peer feedback,
the final comparison used a 12-epoch tuning pass and 50-epoch confirmation on
seeds 34000, 456, and 789. It launched one single-GPU trial per visible GPU.

ViT-5 tiny / CIFAR-10, 50 epochs, batch size 512 per trial, 3 seeds:

    row + aspect, rg0.35:
        final val loss 0.3975 +/- 0.0150,
        final val acc  87.44% +/- 0.43%,
        34.56 ms/step

    orientation + aspect, rg0.40:
        final val loss 0.4087 +/- 0.0155,
        final val acc  87.04% +/- 0.55%,
        34.86 ms/step

    tuned AdamW:
        final val loss 0.5868 +/- 0.0289,
        final val acc  82.56% +/- 0.89%,
        19.37 ms/step

The missing orientation plus aspect ablation was competitive, but the
three-seed aggregate still supports keeping the row-wise aspect multiplier as
the default. AdamW is about 1.8x faster per step on this small model, so the
optimizer remains a quality-first recipe.

Hyperparameter sensitivity
==========================

Most sensitive:

    1. matrix_lr
       Tuning moved the winner from 7e-3 to 8e-3. Too low under-trains; too
       high can lose the early advantage. Start with {6e-3, 7e-3, 8e-3}.

    2. row_gamma
       Best local value was 0.35. Values around 0.25-0.35 were useful. This is
       the main PMuonEq strength control.

    3. normuon_beta2
       Best local value was 0.93 after a sweep over {0.85, 0.90, 0.93, 0.95}.
       Lower values adapt faster but can add noise; higher values are smoother.

Moderately sensitive:

    4. col_gamma
       A small nonzero value helped. Best was 0.05. Earlier broader sweeps often
       liked 0.1, so tune {0.0, 0.05, 0.1}.

    5. pmuoneq_beta
       0.90 beat 0.95 in the final tight search. Tune {0.90, 0.95}.

Less sensitive in these runs:

    6. momentum
       0.95 remained best for the final recipe. A broader search tested 0.90
       and 0.95.

Recommended tuning order
========================

    1. Tune AdamW baseline fairly first.
    2. Tune matrix_lr with defaults: {6e-3, 7e-3, 8e-3}.
    3. Tune row_gamma and col_gamma jointly:
           row_gamma in {0.25, 0.30, 0.35}
           col_gamma in {0.0, 0.05, 0.10}
    4. Tune normuon_beta2 in {0.85, 0.90, 0.93, 0.95}.
    5. Tune pmuoneq_beta in {0.90, 0.95}.
    6. Only then revisit fallback Adam LR/WD or matrix momentum.

DDP behavior
============

This optimizer does not do any distributed communication. Use ordinary PyTorch
DDP, which all-reduces gradients before ``optimizer.step()``. The optimizer
state is local and deterministic across ranks when DDP gradients are in sync.

Integration notes
=================

Use ``build_soda_pmuoneq_normuon_param_groups(model.named_parameters())`` for
normal model training. Passing raw ``model.parameters()`` is supported, but all
floating tensors with ``ndim >= 2`` are routed through the matrix path, including
embeddings and output heads. The named-parameter helper keeps common
embeddings, heads, norms, and biases in the fallback path.

By default ``step()`` owns a short linear warmup:

    lr_t = base_lr * min(1, t / warmup_steps)

Set ``use_external_lr=True`` on the optimizer or a parameter group if an
external scheduler should write ``group["lr"]`` before each step.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import torch

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


@torch.no_grad()
def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("matrix update requires a tensor with ndim >= 2")
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


@torch.no_grad()
def _diag_inverse_power(diag: torch.Tensor, *, gamma: float, eps: float) -> torch.Tensor:
    x = diag.to(torch.float32)
    scale = x.mean(dim=-1, keepdim=True).clamp_min(1.0) if x.ndim > 1 else x.mean().clamp_min(1.0)
    lam = (x + eps * scale).clamp_min(eps * scale)
    out = lam.pow(-float(gamma))
    norm = out.norm(dim=-1, keepdim=True).clamp_min(eps) if out.ndim > 1 else out.norm().clamp_min(eps)
    return out * (math.sqrt(out.size(-1)) / norm)


@torch.no_grad()
def _normuon_second_shape(rows: int, cols: int, mode: str) -> tuple[int, int]:
    if mode == "row":
        return (rows, 1)
    if mode == "orientation":
        return (rows, 1) if rows >= cols else (1, cols)
    raise ValueError(f"unknown NorMuon mode {mode!r}")


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
    dtype = update.dtype
    eps_t = torch.tensor(eps, dtype=dtype, device=update.device)
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    rows, cols = update.shape[-2], update.shape[-1]
    expected_shape = _normuon_second_shape(rows, cols, mode)
    if tuple(second_momentum.shape[-2:]) != expected_shape:
        raise ValueError(f"NorMuon second momentum shape {tuple(second_momentum.shape[-2:])} != {expected_shape}")
    reduce_dim = -1 if expected_shape[-1] == 1 else -2
    try:
        power = update.square().mean(dim=reduce_dim, keepdim=True, dtype=dtype)
    except TypeError:
        power = update.square().mean(dim=reduce_dim, keepdim=True).to(dtype)
    second_momentum.lerp_(power, 1.0 - beta2)
    out = update * torch.rsqrt(second_momentum + eps_t)
    out = out * (vnorm / (out.norm(dim=(-2, -1), keepdim=True) + eps_t))
    if aspect_scale:
        out = out * math.sqrt(max(1.0, rows / cols))
    return out


def _is_default_fallback_name(name: str) -> bool:
    """Return whether a named parameter should avoid the matrix update path."""

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
        "token_embed",
        "token_embedding",
        "embedding",
        "embeddings",
        "wte",
        "wpe",
        "tok_embeddings",
        "word_embeddings",
        "word_embedding",
        "pos_embed",
        "position_embeddings",
        "cls_token",
        "reg_token",
    }
    if any(part in embed_parts for part in parts):
        return True

    known_head_tokens = ("lm_head", "classifier_head", "unembed")
    if any(token in lower for token in known_head_tokens):
        return True
    if lower == "head.weight" or lower.endswith(".head.weight"):
        return True
    return False


class GramNewtonSchulz:
    """Pure PyTorch Gram Newton-Schulz polar approximation."""

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


class SodaPmuonEqNorMuon(torch.optim.Optimizer):
    """Focused optimizer implementing the winning SODA-PMuonEq-NorMuon path."""

    def __init__(
        self,
        params: ParamsT,
        *,
        matrix_lr: float = 8e-3,
        adam_lr: float = 8e-4,
        momentum: float = 0.95,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-10,
        matrix_weight_decay: float = 0.0,
        adam_weight_decay: float = 0.05,
        warmup_steps: int = 10,
        soda_lambda_scale: float = 1.0,
        soda_lambda_power: float = 1.0,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.35,
        col_gamma: float = 0.05,
        pmuoneq_eps: float = 1e-6,
        normuon_beta2: float = 0.93,
        normuon_eps: float = 1e-10,
        normuon_mode: str = "row",
        normuon_aspect_scale: bool = True,
        soda_disables_matrix_weight_decay: bool = True,
        ns_epsilon: float = 1e-7,
        ns_compute_dtype: torch.dtype | None = None,
        use_external_lr: bool = False,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if soda_lambda_scale < 0.0:
            raise ValueError("soda_lambda_scale must be non-negative")
        if soda_lambda_power <= 0.0:
            raise ValueError("soda_lambda_power must be positive")

        prepared = self._prepare_param_groups(
            params,
            matrix_lr=matrix_lr,
            adam_lr=adam_lr,
            matrix_weight_decay=matrix_weight_decay,
            adam_weight_decay=adam_weight_decay,
            momentum=momentum,
            adam_betas=adam_betas,
            eps=eps,
            pmuoneq_beta=pmuoneq_beta,
            row_gamma=row_gamma,
            col_gamma=col_gamma,
            pmuoneq_eps=pmuoneq_eps,
            normuon_beta2=normuon_beta2,
            normuon_eps=normuon_eps,
            normuon_mode=normuon_mode,
            normuon_aspect_scale=normuon_aspect_scale,
            soda_disables_matrix_weight_decay=soda_disables_matrix_weight_decay,
            use_external_lr=use_external_lr,
        )
        super().__init__(prepared, defaults={})

        self.warmup_steps = int(warmup_steps)
        self.soda_lambda_scale = float(soda_lambda_scale)
        self.soda_lambda_power = float(soda_lambda_power)
        self._orthogonalizer = GramNewtonSchulz(epsilon=ns_epsilon, compute_dtype=ns_compute_dtype)
        self.last_stats: dict[str, float] = {}

        for group in self.param_groups:
            group.setdefault("k", 0)
            group.setdefault("warmup_steps", self.warmup_steps)
            group.setdefault("base_lr", group["lr"])

    @staticmethod
    def _prepare_param_groups(
        params: ParamsT,
        *,
        matrix_lr: float,
        adam_lr: float,
        matrix_weight_decay: float,
        adam_weight_decay: float,
        momentum: float,
        adam_betas: tuple[float, float],
        eps: float,
        pmuoneq_beta: float,
        row_gamma: float,
        col_gamma: float,
        pmuoneq_eps: float,
        normuon_beta2: float,
        normuon_eps: float,
        normuon_mode: str,
        normuon_aspect_scale: bool,
        soda_disables_matrix_weight_decay: bool,
        use_external_lr: bool,
    ) -> list[dict[str, Any]]:
        items = list(params)
        if not items:
            raise ValueError("optimizer got an empty parameter list")
        if isinstance(items[0], dict):
            groups = [dict(group) for group in items]  # type: ignore[arg-type]
            for group in groups:
                group.setdefault("use_matrix_update", group.get("use_muon", False))
                is_matrix = bool(group["use_matrix_update"])
                group.setdefault("lr", matrix_lr if is_matrix else adam_lr)
                group.setdefault("base_lr", group["lr"])
                group.setdefault("use_external_lr", use_external_lr)
                if is_matrix:
                    group.setdefault("weight_decay", matrix_weight_decay)
                    group.setdefault("momentum", momentum)
                    group.setdefault("pmuoneq_beta", pmuoneq_beta)
                    group.setdefault("row_gamma", row_gamma)
                    group.setdefault("col_gamma", col_gamma)
                    group.setdefault("pmuoneq_eps", pmuoneq_eps)
                    group.setdefault("normuon_beta2", normuon_beta2)
                    group.setdefault("normuon_eps", normuon_eps)
                    group.setdefault("normuon_mode", normuon_mode)
                    group.setdefault("normuon_aspect_scale", normuon_aspect_scale)
                    group.setdefault("soda_disables_matrix_weight_decay", soda_disables_matrix_weight_decay)
                else:
                    group.setdefault("weight_decay", adam_weight_decay)
                    group.setdefault("betas", adam_betas)
                    group.setdefault("eps", eps)
            return groups

        matrix_params: list[torch.Tensor] = []
        fallback_params: list[torch.Tensor] = []
        for p in items:  # type: ignore[assignment]
            if not isinstance(p, torch.Tensor):
                raise TypeError("params must be tensors or optimizer param-group dictionaries")
            if p.requires_grad and p.ndim >= 2:
                matrix_params.append(p)
            else:
                fallback_params.append(p)

        groups: list[dict[str, Any]] = []
        if matrix_params:
            groups.append(
                {
                    "params": matrix_params,
                    "use_matrix_update": True,
                    "lr": matrix_lr,
                    "base_lr": matrix_lr,
                    "weight_decay": matrix_weight_decay,
                    "momentum": momentum,
                    "pmuoneq_beta": pmuoneq_beta,
                    "row_gamma": row_gamma,
                    "col_gamma": col_gamma,
                    "pmuoneq_eps": pmuoneq_eps,
                    "normuon_beta2": normuon_beta2,
                    "normuon_eps": normuon_eps,
                    "normuon_mode": normuon_mode,
                    "normuon_aspect_scale": normuon_aspect_scale,
                    "soda_disables_matrix_weight_decay": soda_disables_matrix_weight_decay,
                    "use_external_lr": use_external_lr,
                }
            )
        if fallback_params:
            groups.append(
                {
                    "params": fallback_params,
                    "use_matrix_update": False,
                    "lr": adam_lr,
                    "base_lr": adam_lr,
                    "weight_decay": adam_weight_decay,
                    "betas": adam_betas,
                    "eps": eps,
                    "use_external_lr": use_external_lr,
                }
            )
        return groups

    def train(self) -> "SodaPmuonEqNorMuon":
        return self

    def eval(self) -> "SodaPmuonEqNorMuon":
        return self

    def _soda_anchor(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        anchor = state.get("soda_z0")
        if anchor is None:
            anchor = state["soda_z0"] = torch.clone(p, memory_format=torch.preserve_format)
        return anchor

    def _soda_weight(self, t: int) -> float:
        return min(1.0, self.soda_lambda_scale / float(t + 1) ** self.soda_lambda_power)

    def _pre_matrix_source(self, p: torch.Tensor, grad: torch.Tensor, momentum_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state[p]
        momentum = state.get("momentum_buffer")
        if momentum is None or momentum.shape != grad.shape or momentum.device != grad.device:
            momentum = state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
        g = grad.detach().to(torch.float32)
        momentum.lerp_(g, 1.0 - momentum_beta)
        source = torch.lerp(g, momentum, momentum_beta)
        return _matrix_view(source), _matrix_view(g)

    def _ensure_matrix_state(
        self,
        p: torch.Tensor,
        rows: int,
        cols: int,
        device: torch.device,
        normuon_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state[p]
        row_ema = state.get("pmuoneq_row_ema")
        if row_ema is None or row_ema.shape != (rows,) or row_ema.device != device:
            row_ema = state["pmuoneq_row_ema"] = torch.ones(rows, device=device, dtype=torch.float32)
        col_ema = state.get("pmuoneq_col_ema")
        if col_ema is None or col_ema.shape != (cols,) or col_ema.device != device:
            col_ema = state["pmuoneq_col_ema"] = torch.ones(cols, device=device, dtype=torch.float32)
        row_factor = state.get("pmuoneq_row_factor")
        if row_factor is None or row_factor.shape != (rows,) or row_factor.device != device:
            row_factor = state["pmuoneq_row_factor"] = torch.ones(rows, device=device, dtype=torch.float32)
        col_factor = state.get("pmuoneq_col_factor")
        if col_factor is None or col_factor.shape != (cols,) or col_factor.device != device:
            col_factor = state["pmuoneq_col_factor"] = torch.ones(cols, device=device, dtype=torch.float32)
        second = state.get("normuon_second_momentum")
        second_shape = _normuon_second_shape(rows, cols, normuon_mode)
        if second is None or second.shape != second_shape or second.device != device:
            second = state["normuon_second_momentum"] = torch.zeros(*second_shape, device=device, dtype=torch.float32)
        return row_ema, col_ema, row_factor, col_factor, second

    def _transform_matrix_bucket(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        group: dict[str, Any],
    ) -> torch.Tensor:
        sources = torch.stack([entry[2].to(torch.float32) for entry in entries], dim=0)
        grads = torch.stack([entry[3].to(torch.float32) for entry in entries], dim=0)
        _batch, rows, cols = grads.shape
        row_buffers: list[torch.Tensor] = []
        col_buffers: list[torch.Tensor] = []
        row_factor_buffers: list[torch.Tensor] = []
        col_factor_buffers: list[torch.Tensor] = []
        second_buffers: list[torch.Tensor] = []
        normuon_mode = str(group.get("normuon_mode", "row"))

        for p, _anchor, _source, _grad in entries:
            row_ema, col_ema, row_factor, col_factor, second = self._ensure_matrix_state(p, rows, cols, grads.device, normuon_mode)
            row_buffers.append(row_ema)
            col_buffers.append(col_ema)
            row_factor_buffers.append(row_factor)
            col_factor_buffers.append(col_factor)
            second_buffers.append(second)

        beta_p = float(group["pmuoneq_beta"])
        g2 = grads.square()
        row_stack = torch.stack(row_buffers, dim=0)
        col_stack = torch.stack(col_buffers, dim=0)
        row_stack.mul_(beta_p).add_(g2.mean(dim=2), alpha=1.0 - beta_p)
        col_stack.mul_(beta_p).add_(g2.mean(dim=1), alpha=1.0 - beta_p)
        row_factor = _diag_inverse_power(row_stack, gamma=float(group["row_gamma"]), eps=float(group["pmuoneq_eps"]))
        col_factor = _diag_inverse_power(col_stack, gamma=float(group["col_gamma"]), eps=float(group["pmuoneq_eps"]))

        preconditioned = sources * row_factor[:, :, None] * col_factor[:, None, :]
        update = self._orthogonalizer(preconditioned)
        update = update * (0.2 * math.sqrt(max(update.size(-2), update.size(-1))))

        second_stack = torch.stack(second_buffers, dim=0)
        update = _normuon_normalize(
            update,
            second_stack,
            beta2=float(group["normuon_beta2"]),
            eps=float(group["normuon_eps"]),
            mode=normuon_mode,
            aspect_scale=bool(group.get("normuon_aspect_scale", True)),
        )

        for idx, (p, _anchor, _source, _grad) in enumerate(entries):
            self.state[p]["pmuoneq_row_ema"].copy_(row_stack[idx])
            self.state[p]["pmuoneq_col_ema"].copy_(col_stack[idx])
            self.state[p]["pmuoneq_row_factor"].copy_(row_factor[idx])
            self.state[p]["pmuoneq_col_factor"].copy_(col_factor[idx])
            self.state[p]["normuon_second_momentum"].copy_(second_stack[idx])
        return update

    def _step_matrix_group(self, group: dict[str, Any], *, lr: float, t: int) -> dict[str, float]:
        beta_m = float(group["momentum"])
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        fallback_count = 0
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if grad.ndim < 2:
                fallback_count += self._step_fallback_param(p, group, lr=lr, t=t)
                continue
            anchor = self._soda_anchor(p)
            source, grad_matrix = self._pre_matrix_source(p, grad, beta_m)
            entries.append((p, anchor, source, grad_matrix))

        if not entries:
            return {"matrix_count": 0.0, "fallback_count": float(fallback_count), "soda_weight": float(self._soda_weight(t))}

        buckets: dict[tuple[torch.device, torch.Size], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for entry in entries:
            buckets[(entry[2].device, entry[2].shape)].append(entry)

        soda_weight = self._soda_weight(t)
        weight_decay = float(group.get("weight_decay", 0.0))
        disable_wd = bool(group.get("soda_disables_matrix_weight_decay", True)) and soda_weight > 0.0
        matrix_count = 0
        wd_count = 0
        for bucket_entries in buckets.values():
            updates = self._transform_matrix_bucket(bucket_entries, group)
            for (p, anchor, _source, _grad), update in zip(bucket_entries, updates.unbind(0), strict=True):
                if weight_decay and not disable_wd:
                    p.mul_(1.0 - lr * weight_decay)
                    wd_count += 1
                p.lerp_(end=anchor, weight=soda_weight)
                p.add_(update.reshape_as(p).to(p.dtype), alpha=-lr)
                matrix_count += 1

        return {
            "matrix_count": float(matrix_count),
            "fallback_count": float(fallback_count),
            "soda_weight": float(soda_weight),
            "matrix_weight_decay_count": float(wd_count),
        }

    def _step_fallback_param(self, p: torch.Tensor, group: dict[str, Any], *, lr: float, t: int) -> int:
        grad = p.grad
        if grad is None:
            return 0
        state = self.state[p]
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None or exp_avg_sq.shape != p.shape or exp_avg_sq.device != p.device:
            exp_avg_sq = state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] = 0
        state["step"] = int(state.get("step", 0)) + 1
        _beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-10))
        g = grad.detach().to(torch.float32)
        exp_avg_sq.mul_(float(beta2)).addcmul_(g, g, value=1.0 - float(beta2))
        step = int(state["step"])
        denom = (exp_avg_sq / max(1.0 - float(beta2) ** step, 1e-16)).sqrt().add_(eps)
        update = g / denom
        weight_decay = float(group.get("weight_decay", 0.0))
        if weight_decay:
            update = update + p.detach().to(torch.float32) * weight_decay
        p.lerp_(end=self._soda_anchor(p), weight=self._soda_weight(t))
        p.add_(update.to(p.dtype), alpha=-lr)
        return 1

    def _step_fallback_group(self, group: dict[str, Any], *, lr: float, t: int) -> dict[str, float]:
        fallback_count = 0
        for p in group["params"]:
            fallback_count += self._step_fallback_param(p, group, lr=lr, t=t)
        return {
            "matrix_count": 0.0,
            "fallback_count": float(fallback_count),
            "soda_weight": float(self._soda_weight(t)),
            "matrix_weight_decay_count": 0.0,
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
            "matrix_weight_decay_count": 0.0,
            "soda_weight_sum": 0.0,
            "group_count": 0.0,
        }
        for group in self.param_groups:
            k = int(group.get("k", 0))
            t = k + 1
            warmup_steps = int(group.get("warmup_steps", self.warmup_steps))
            if bool(group.get("use_external_lr", False)):
                lr = float(group["lr"])
            else:
                lr = float(group["base_lr"]) * min(1.0, t / warmup_steps)
                group["lr"] = lr
            if group.get("use_matrix_update", False):
                group_stats = self._step_matrix_group(group, lr=lr, t=t)
            else:
                group_stats = self._step_fallback_group(group, lr=lr, t=t)
            stats["matrix_count"] += group_stats["matrix_count"]
            stats["fallback_count"] += group_stats["fallback_count"]
            stats["matrix_weight_decay_count"] += group_stats["matrix_weight_decay_count"]
            stats["soda_weight_sum"] += group_stats["soda_weight"]
            stats["group_count"] += 1.0
            group["k"] = t
        stats["soda_weight"] = stats["soda_weight_sum"] / max(stats["group_count"], 1.0)
        del stats["soda_weight_sum"]
        self.last_stats = stats
        return loss


def build_soda_pmuoneq_normuon_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    matrix_lr: float = 8e-3,
    adam_lr: float = 8e-4,
    matrix_weight_decay: float = 0.0,
    adam_weight_decay: float = 0.05,
    momentum: float = 0.95,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-10,
    pmuoneq_beta: float = 0.90,
    row_gamma: float = 0.35,
    col_gamma: float = 0.05,
    pmuoneq_eps: float = 1e-6,
    normuon_beta2: float = 0.93,
    normuon_eps: float = 1e-10,
    normuon_mode: str = "row",
    normuon_aspect_scale: bool = True,
    soda_disables_matrix_weight_decay: bool = True,
    matrix_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> list[dict[str, Any]]:
    """Split named parameters into matrix and RMS/AdamW-style fallback groups."""

    matrix_params: list[torch.nn.Parameter] = []
    matrix_names: list[str] = []
    fallback_params: list[torch.nn.Parameter] = []
    fallback_names: list[str] = []

    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        if matrix_filter is not None:
            use_matrix = bool(matrix_filter(name, p))
        else:
            use_matrix = p.ndim >= 2 and not _is_default_fallback_name(name)
        if use_matrix:
            matrix_params.append(p)
            matrix_names.append(name)
        else:
            fallback_params.append(p)
            fallback_names.append(name)

    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "param_names": matrix_names,
                "use_matrix_update": True,
                "lr": matrix_lr,
                "base_lr": matrix_lr,
                "weight_decay": matrix_weight_decay,
                "momentum": momentum,
                "pmuoneq_beta": pmuoneq_beta,
                "row_gamma": row_gamma,
                "col_gamma": col_gamma,
                "pmuoneq_eps": pmuoneq_eps,
                "normuon_beta2": normuon_beta2,
                "normuon_eps": normuon_eps,
                "normuon_mode": normuon_mode,
                "normuon_aspect_scale": normuon_aspect_scale,
                "soda_disables_matrix_weight_decay": soda_disables_matrix_weight_decay,
            }
        )
    if fallback_params:
        groups.append(
            {
                "params": fallback_params,
                "param_names": fallback_names,
                "use_matrix_update": False,
                "lr": adam_lr,
                "base_lr": adam_lr,
                "weight_decay": adam_weight_decay,
                "betas": adam_betas,
                "eps": eps,
            }
        )
    return groups


__all__ = ["SodaPmuonEqNorMuon", "GramNewtonSchulz", "build_soda_pmuoneq_normuon_param_groups"]
