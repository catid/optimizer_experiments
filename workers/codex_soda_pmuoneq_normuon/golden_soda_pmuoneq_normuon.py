"""Golden SODA-PMuonEq-NorMuon optimizer.

This file is the final focused optimizer handoff from
``workers/codex_soda_pmuoneq_normuon``. It intentionally implements only the
repo-wide winning optimizer family for the next language-modeling round:

    SODA + row-only PMuonEq + Gram Newton-Schulz + NorMuon

The implementation deliberately omits the research ablation switches:

    no AMUSE / schedule-free train-eval sequence
    no MiMuon branch
    no full/stale PMuon dense covariance path
    no column PMuonEq scaling
    no post-NorMuon aspect multiplier
    no optional NorMuon disable path

The current best-supported recipe comes from the cleanest cross-worker evidence,
the proper CIFAR-10 train/validation/test split in
``workers/codex_noradam_confidence``:

    trial id: normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93
    model:    vit5_micro, 458,858 trainable parameters
    data:     CIFAR-10 45k train / 5k validation / official 10k test
    training: 50 epochs, 4,350 steps, batch size 512
    seeds:    123, 456, 789

    official test loss: 0.4607 +/- 0.0196
    official test acc:  84.77% +/- 0.65
    step time:          19.95 ms

The aspect-scaled variant was close and sometimes won on other CIFAR proxies,
but no-aspect won the cleanest official-test protocol. This golden file keeps
the no-aspect variant only. If aspect or column gamma is needed for an ablation,
use the older ``soda_pmuoneq_normuon.py`` research file instead.

Matrix update math
==================

For a matrix parameter ``W_t`` with gradient ``G_t``:

1. Momentum / Nesterov-style source:

       M_t = beta_m M_{t-1} + (1 - beta_m) G_t
       U_t = (1 - beta_m) G_t + beta_m M_t

2. Row-only PMuonEq:

       r_t = beta_p r_{t-1} + (1 - beta_p) mean_cols(G_t^2)
       a_t = normalize(r_t^{-gamma_row})
       A_t = a_t[:, None] * U_t

   This is the fast diagonal/row approximation to PMuon. It keeps the useful
   row reliability scaling before orthogonalization without dense covariance
   matrices, QR/eigendecomposition, or inverse-power matrix products.

3. Gram Newton-Schulz:

       P_t = GramNS(A_t)
       D_t = 0.2 * sqrt(max(rows, cols)) * P_t

4. NorMuon after GramNS:

       s_t = beta_n s_{t-1} + (1 - beta_n) mean_cols(D_t^2)
       N_t = D_t / sqrt(s_t + eps)
       N_t = N_t * ||D_t||_F / (||N_t||_F + eps)

   NorMuon changes row allocation after the spectral direction is produced,
   while preserving the whole-matrix Frobenius norm. There is no aspect-ratio
   multiplier in this golden version.

5. SODA anchor pull and update:

       lambda_t = min(1, soda_lambda_scale / (t + 1)^soda_lambda_power)
       W_t <- (1 - lambda_t) W_t + lambda_t W_0
       W_{t+1} = W_t - lr_t N_t

Fallback parameters
===================

Biases, norms, embeddings, heads, scalar/vector tensors, and any parameters
routed to the fallback group use the same SODA anchor plus RMS/AdamW-style
second-moment update used by the winning research path:

       v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2
       update = g_t / sqrt(v_t / (1 - beta_2^t) + eps)

Decoupled fallback weight decay is implemented as an additive term in the
fallback update. Matrix parameters do not use ordinary weight decay; SODA is the
matrix regularization path.

Recommended starting defaults
=============================

Use these defaults before tuning on a new model family:

    matrix_lr       = 8e-3
    fallback_lr     = 8e-4
    momentum        = 0.95
    pmuoneq_beta    = 0.90
    row_gamma       = 0.35
    normuon_beta2   = 0.93
    warmup_steps    = 10
    fallback_weight_decay = 0.05

Language-modeling tuning order:

    1. Verify parameter grouping, especially tied embedding / LM head handling.
    2. Compare against the actual LM baseline optimizer, not only AdamW.
    3. Tune matrix_lr in {0.004, 0.006, 0.008}.
    4. Tune row_gamma in {0.25, 0.35, 0.45}.
    5. Tune normuon_beta2 in {0.90, 0.93, 0.95}.
    6. Tune pmuoneq_beta in {0.90, 0.95}.
    7. Only then revisit fallback LR/weight decay.

DDP and batching
================

The optimizer does not perform distributed communication. Use normal PyTorch
DDP, which all-reduces gradients before ``optimizer.step()``. Optimizer state is
local and deterministic across ranks when gradients are synchronized. For GPU
efficiency, same-shape matrices in a parameter group are bucketed and processed
as a batch through PMuonEq, Gram Newton-Schulz, and NorMuon.
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
def _effective_matrix_shape(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in x.shape if int(dim) > 1)


@torch.no_grad()
def _has_matrix_shape(x: torch.Tensor) -> bool:
    return len(_effective_matrix_shape(x)) >= 2


def _fp32_state_tensor(
    state: dict[str, Any],
    key: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    fill: float,
) -> torch.Tensor:
    value = state.get(key)
    if isinstance(value, torch.Tensor) and tuple(value.shape) == shape:
        if value.device != device or value.dtype != torch.float32:
            value = state[key] = value.to(device=device, dtype=torch.float32)
        return value
    value = torch.full(shape, fill, device=device, dtype=torch.float32)
    state[key] = value
    return value


@torch.no_grad()
def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    effective_shape = _effective_matrix_shape(x)
    if len(effective_shape) < 2:
        raise ValueError("matrix update requires at least two non-singleton dimensions")
    if len(effective_shape) == 2:
        return x.reshape(effective_shape)
    return x.reshape(effective_shape[0], math.prod(effective_shape[1:]))


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
    dtype = update.dtype
    eps_t = torch.tensor(eps, dtype=dtype, device=update.device)
    old_norm = update.norm(dim=(-2, -1), keepdim=True)
    try:
        row_power = update.square().mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        row_power = update.square().mean(dim=-1, keepdim=True).to(dtype)
    second_momentum.lerp_(row_power, 1.0 - beta2)
    out = update * torch.rsqrt(second_momentum + eps_t)
    return out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps_t))


def _is_default_fallback_name(name: str) -> bool:
    lower = name.lower().replace("/", ".")
    parts = [part for part in lower.split(".") if part]
    if not parts:
        return False
    if parts[-1] == "bias" or lower.endswith(".bias"):
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
    return lower == "head.weight" or lower.endswith(".head.weight")


class GramNewtonSchulz:
    """Batched Gram Newton-Schulz polar approximation used by the golden path."""

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


class GoldenSodaPmuonEqNorMuon(torch.optim.Optimizer):
    """PyTorch optimizer implementing only the final golden optimizer path."""

    def __init__(
        self,
        params: ParamsT,
        *,
        matrix_lr: float = 8e-3,
        fallback_lr: float = 8e-4,
        momentum: float = 0.95,
        fallback_betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-10,
        fallback_weight_decay: float = 0.05,
        warmup_steps: int = 10,
        soda_lambda_scale: float = 1.0,
        soda_lambda_power: float = 1.0,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.35,
        pmuoneq_eps: float = 1e-6,
        normuon_beta2: float = 0.93,
        normuon_eps: float = 1e-10,
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
            fallback_lr=fallback_lr,
            fallback_weight_decay=fallback_weight_decay,
            momentum=momentum,
            fallback_betas=fallback_betas,
            eps=eps,
            pmuoneq_beta=pmuoneq_beta,
            row_gamma=row_gamma,
            pmuoneq_eps=pmuoneq_eps,
            normuon_beta2=normuon_beta2,
            normuon_eps=normuon_eps,
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
        fallback_lr: float,
        fallback_weight_decay: float,
        momentum: float,
        fallback_betas: tuple[float, float],
        eps: float,
        pmuoneq_beta: float,
        row_gamma: float,
        pmuoneq_eps: float,
        normuon_beta2: float,
        normuon_eps: float,
        use_external_lr: bool,
    ) -> list[dict[str, Any]]:
        items = list(params)
        if not items:
            raise ValueError("optimizer got an empty parameter list")
        if isinstance(items[0], dict):
            groups = [dict(group) for group in items]  # type: ignore[arg-type]
            seen_params: set[int] = set()
            deduped_groups: list[dict[str, Any]] = []
            for group in groups:
                params_value = group["params"]
                params_in = [params_value] if isinstance(params_value, torch.Tensor) else list(params_value)
                names_in = list(group.get("param_names", []))
                keep_names = len(names_in) == len(params_in)
                params_out: list[torch.Tensor] = []
                names_out: list[str] = []
                for idx, p in enumerate(params_in):
                    if id(p) in seen_params:
                        continue
                    seen_params.add(id(p))
                    params_out.append(p)
                    if keep_names:
                        names_out.append(names_in[idx])
                if not params_out:
                    continue
                group["params"] = params_out
                if keep_names:
                    group["param_names"] = names_out
                else:
                    group.pop("param_names", None)
                group.setdefault("use_matrix_update", group.get("use_muon", False))
                is_matrix = bool(group["use_matrix_update"])
                group.setdefault("lr", matrix_lr if is_matrix else fallback_lr)
                group.setdefault("base_lr", group["lr"])
                group.setdefault("use_external_lr", use_external_lr)
                if is_matrix:
                    group.setdefault("weight_decay", 0.0)
                    group.setdefault("momentum", momentum)
                    group.setdefault("pmuoneq_beta", pmuoneq_beta)
                    group.setdefault("row_gamma", row_gamma)
                    group.setdefault("pmuoneq_eps", pmuoneq_eps)
                    group.setdefault("normuon_beta2", normuon_beta2)
                    group.setdefault("normuon_eps", normuon_eps)
                else:
                    group.setdefault("weight_decay", fallback_weight_decay)
                    group.setdefault("betas", fallback_betas)
                    group.setdefault("eps", eps)
                deduped_groups.append(group)
            return deduped_groups

        matrix_params: list[torch.Tensor] = []
        fallback_params: list[torch.Tensor] = []
        seen_params: set[int] = set()
        for p in items:  # type: ignore[assignment]
            if not isinstance(p, torch.Tensor):
                raise TypeError("params must be tensors or optimizer param-group dictionaries")
            if id(p) in seen_params:
                continue
            seen_params.add(id(p))
            if p.requires_grad and _has_matrix_shape(p):
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
                    "weight_decay": 0.0,
                    "momentum": momentum,
                    "pmuoneq_beta": pmuoneq_beta,
                    "row_gamma": row_gamma,
                    "pmuoneq_eps": pmuoneq_eps,
                    "normuon_beta2": normuon_beta2,
                    "normuon_eps": normuon_eps,
                    "use_external_lr": use_external_lr,
                }
            )
        if fallback_params:
            groups.append(
                {
                    "params": fallback_params,
                    "use_matrix_update": False,
                    "lr": fallback_lr,
                    "base_lr": fallback_lr,
                    "weight_decay": fallback_weight_decay,
                    "betas": fallback_betas,
                    "eps": eps,
                    "use_external_lr": use_external_lr,
                }
            )
        return groups

    def train(self) -> "GoldenSodaPmuonEqNorMuon":
        return self

    def eval(self) -> "GoldenSodaPmuonEqNorMuon":
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
        momentum = _fp32_state_tensor(
            state,
            "momentum_buffer",
            shape=tuple(grad.shape),
            device=grad.device,
            fill=0.0,
        )
        g = grad.detach().to(torch.float32)
        momentum.lerp_(g, 1.0 - momentum_beta)
        source = torch.lerp(g, momentum, momentum_beta)
        return _matrix_view(source), _matrix_view(g)

    def _ensure_matrix_state(
        self,
        p: torch.Tensor,
        rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state[p]
        row_ema = _fp32_state_tensor(state, "pmuoneq_row_ema", shape=(rows,), device=device, fill=1.0)
        row_factor = _fp32_state_tensor(state, "pmuoneq_row_factor", shape=(rows,), device=device, fill=1.0)
        second_shape = (rows, 1)
        second = _fp32_state_tensor(
            state,
            "normuon_second_momentum",
            shape=second_shape,
            device=device,
            fill=0.0,
        )
        return row_ema, row_factor, second

    def _transform_matrix_bucket(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        group: dict[str, Any],
    ) -> torch.Tensor:
        sources = torch.stack([entry[2].to(torch.float32) for entry in entries], dim=0)
        grads = torch.stack([entry[3].to(torch.float32) for entry in entries], dim=0)
        _batch, rows, _cols = grads.shape

        row_buffers: list[torch.Tensor] = []
        row_factor_buffers: list[torch.Tensor] = []
        second_buffers: list[torch.Tensor] = []
        for p, _anchor, _source, _grad in entries:
            row_ema, row_factor, second = self._ensure_matrix_state(p, rows, grads.device)
            row_buffers.append(row_ema)
            row_factor_buffers.append(row_factor)
            second_buffers.append(second)

        beta_p = float(group["pmuoneq_beta"])
        row_stack = torch.stack(row_buffers, dim=0)
        row_stack.mul_(beta_p).add_(grads.square().mean(dim=2), alpha=1.0 - beta_p)
        row_factor = _diag_inverse_power(row_stack, gamma=float(group["row_gamma"]), eps=float(group["pmuoneq_eps"]))

        preconditioned = sources * row_factor[:, :, None]
        update = self._orthogonalizer(preconditioned)
        update = update * (0.2 * math.sqrt(max(update.size(-2), update.size(-1))))

        second_stack = torch.stack(second_buffers, dim=0)
        update = _normuon_row_normalize(
            update,
            second_stack,
            beta2=float(group["normuon_beta2"]),
            eps=float(group["normuon_eps"]),
        )

        for idx, (p, _anchor, _source, _grad) in enumerate(entries):
            self.state[p]["pmuoneq_row_ema"].copy_(row_stack[idx])
            self.state[p]["pmuoneq_row_factor"].copy_(row_factor[idx])
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
            if grad.is_sparse:
                raise RuntimeError("GoldenSodaPmuonEqNorMuon does not support sparse gradients")
            if not _has_matrix_shape(grad):
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
        matrix_count = 0
        for bucket_entries in buckets.values():
            updates = self._transform_matrix_bucket(bucket_entries, group)
            for (p, anchor, _source, _grad), update in zip(bucket_entries, updates.unbind(0), strict=True):
                p.lerp_(end=anchor, weight=soda_weight)
                p.add_(update.reshape_as(p).to(p.dtype), alpha=-lr)
                matrix_count += 1

        return {
            "matrix_count": float(matrix_count),
            "fallback_count": float(fallback_count),
            "soda_weight": float(soda_weight),
        }

    def _step_fallback_param(self, p: torch.Tensor, group: dict[str, Any], *, lr: float, t: int) -> int:
        grad = p.grad
        if grad is None:
            return 0
        if grad.is_sparse:
            raise RuntimeError("GoldenSodaPmuonEqNorMuon does not support sparse gradients")
        state = self.state[p]
        exp_avg_sq = state.get("exp_avg_sq")
        if isinstance(exp_avg_sq, torch.Tensor) and exp_avg_sq.shape == p.shape:
            if exp_avg_sq.device != p.device or exp_avg_sq.dtype != torch.float32:
                exp_avg_sq = state["exp_avg_sq"] = exp_avg_sq.to(device=p.device, dtype=torch.float32)
        else:
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
            stats["soda_weight_sum"] += group_stats["soda_weight"]
            stats["group_count"] += 1.0
            group["k"] = t

        stats["soda_weight"] = stats["soda_weight_sum"] / max(stats["group_count"], 1.0)
        del stats["soda_weight_sum"]
        self.last_stats = stats
        return loss


def build_golden_soda_pmuoneq_normuon_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    matrix_lr: float = 8e-3,
    fallback_lr: float = 8e-4,
    fallback_weight_decay: float = 0.05,
    momentum: float = 0.95,
    fallback_betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-10,
    pmuoneq_beta: float = 0.90,
    row_gamma: float = 0.35,
    pmuoneq_eps: float = 1e-6,
    normuon_beta2: float = 0.93,
    normuon_eps: float = 1e-10,
    matrix_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> list[dict[str, Any]]:
    """Split named parameters into golden matrix and fallback groups.

    The default grouping keeps common embeddings, output heads, normalization
    layers, and biases in the fallback path. For language models, audit tied
    embedding / LM-head parameters before the full run.
    """

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

    for p, names in unique.values():
        if matrix_filter is not None:
            use_matrix = any(bool(matrix_filter(name, p)) for name in names)
        else:
            use_matrix = _has_matrix_shape(p) and not any(_is_default_fallback_name(name) for name in names)
        display_name = "|".join(names)
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
                "lr": matrix_lr,
                "base_lr": matrix_lr,
                "weight_decay": 0.0,
                "momentum": momentum,
                "pmuoneq_beta": pmuoneq_beta,
                "row_gamma": row_gamma,
                "pmuoneq_eps": pmuoneq_eps,
                "normuon_beta2": normuon_beta2,
                "normuon_eps": normuon_eps,
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
                "weight_decay": fallback_weight_decay,
                "betas": fallback_betas,
                "eps": eps,
            }
        )
    return groups


__all__ = [
    "GoldenSodaPmuonEqNorMuon",
    "GramNewtonSchulz",
    "build_golden_soda_pmuoneq_normuon_param_groups",
]
