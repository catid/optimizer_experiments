"""EquiMuse-NorMuon: a standalone PyTorch optimizer recipe.

This file intentionally contains one optimizer recipe rather than an ablation
framework.  The selected recipe is:

    SODA-AMUSE schedule-free outer loop
    + PMuonEq row/column diagonal preconditioning
    + Gram Newton-Schulz matrix orthogonalization
    + NorMuon row-wise second-moment normalization
    + AdamW-style schedule-free fallback for embeddings, heads, biases, norms,
      and vector/scalar parameters.

There are no switches to disable SODA, PMuonEq, GramNS, or NorMuon in this
implementation.  Hyperparameters still exist, but they tune the single recipe
rather than choosing between optimizer families.

Math summary
============

For a matrix parameter W, EquiMuse-NorMuon keeps a fast sequence Z, an averaged
sequence X, and a train/evaluation interpolation Y.  The model parameters hold
Y while training and X while evaluating:

    Y_t = (1 - beta1_t) Z_t + beta1_t X_t
    G_t = grad L(Y_t)

The matrix direction starts from Muon momentum:

    M_t = momentum * M_{t-1} + (1 - momentum) G_t
    U_t = (1 - momentum) G_t + momentum * M_t

PMuonEq is a diagonal row/column approximation to PMuon-style covariance
preconditioning.  It updates gradient-power EMAs

    r_i <- beta_p r_i + (1 - beta_p) mean_j G_ij^2
    c_j <- beta_p c_j + (1 - beta_p) mean_i G_ij^2

and preconditions the Muon momentum before orthogonalization:

    A_ij = r_i^{-gamma_row} U_ij c_j^{-gamma_col}

The spectral branch then applies Gram Newton-Schulz:

    P_t = GramNS(A_t)

NorMuon is applied after GramNS.  It updates a row-wise second-moment EMA of the
orthogonalized update,

    n_i <- beta_n n_i + (1 - beta_n) mean_j P_ij^2
    Q_ij = P_ij / sqrt(n_i + eps)

and restores the Frobenius norm of P so it changes row allocation more than
global step size:

    D_t = ||P_t||_F / ||Q_t||_F * Q_t

An optional controlled ablation applies the peer-tuned tall-matrix aspect
multiplier after Frobenius restoration:

    D_t <- D_t * sqrt(max(1, rows / cols))

This is intentionally a final update-magnitude change, not an input change to
PMuonEq, GramNS, momentum, or the raw gradient path.

The fast sequence is updated with the standard Muon scale:

    Z_{t+1} = SODA_anchor(Z_t) - lr * 0.2 * sqrt(max(m, n)) * D_t

The averaged sequence is updated by the AMUSE/schedule-free weighted average:

    X_{t+1} = (1 - c_{t+1}) X_t + c_{t+1} Z_{t+1}

The fallback path uses a schedule-free AdamW-style normalized gradient update
for parameters where orthogonal matrix updates are inappropriate.  As in the
local SODA-AMUSE implementation this file was extracted from, active SODA
anchoring replaces decoupled weight decay; decoupled weight decay is only
applied on steps where the SODA anchor is inactive.

Hyperparameter sensitivity and recommended tuning order
=======================================================

Recommended starting point for ViT/CIFAR-style runs:

    lr = matrix_lr = 1e-2
    weight_decay = 0.05
    beta1 = 0.6
    rho = 0.5
    warmup_steps = 50
    momentum = 0.95
    pmuoneq_beta = 0.95
    pmuoneq_row_gamma = 0.15
    pmuoneq_col_gamma = 0.15
    normuon_beta2 = 0.9
    ns_steps = 5
    ns_dtype = "float16"

Tune in this order:

1. Matrix learning rate: try {3e-3, 1e-2, 3e-2}.  This dominates quality and
   instability.  If loss spikes or accuracy collapses, lower this first.
2. PMuonEq row/column gamma: try {(0.15, 0.15), (0.30, 0.10), (0.10, 0.30)}.
   Larger values amplify diagonal preconditioning and can help early loss but
   can over-whiten layers and hurt final accuracy.
3. NorMuon beta2: try {0.90, 0.95, 0.98}.  Lower beta2 reacts faster and was
   best in the local ViT-5/CIFAR-10 run; higher beta2 is smoother and safer for
   noisier or smaller batches.
4. Warmup steps: tune only after LR/gammas.  Too little warmup makes the
   schedule-free and diagonal-preconditioned update aggressive early.
5. Weight decay: tune last.  SODA anchoring and schedule-free averaging already
   change long-horizon behavior, so weight decay can interact strongly with LR.

DDP and batching notes
======================

The optimizer is DDP-safe because all state is local to each replicated
parameter and is updated only from the already all-reduced gradient.  No
cross-rank communication is needed inside the optimizer.

For speed, parameters are sorted by shape and same-shape matrix updates are
batched before PMuonEq, GramNS, and NorMuon.  The fallback path uses torch
foreach operations when available.  Per-step GPU-to-CPU diagnostic syncs are
avoided by default; ``last_stats`` contains schedule/count metadata only.

This file has no project-local imports and depends only on PyTorch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor


POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = tuple(
    (
        a / 1.05,
        b / 1.05**3,
        c / 1.05**5,
    )
    for a, b, c in (
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    )
)


def _dtype_from_name(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported Newton-Schulz dtype {name!r}") from exc


@torch.no_grad()
def gram_newton_schulz(
    x: Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    ns_dtype: str | torch.dtype = torch.float16,
    reset_iterations: Sequence[int] = (2,),
) -> Tensor:
    """Orthogonalize one matrix or a batch of same-shaped matrices.

    The method normalizes in fp32, transposes when that makes the Gram matrix
    smaller, runs Polar Express quintic Newton-Schulz coefficients, and restores
    the original dtype and shape.
    """

    if x.ndim < 2:
        raise ValueError("gram_newton_schulz expects at least a 2D tensor")
    original_shape = x.shape
    if x.ndim == 2:
        work = x.unsqueeze(0)
    elif x.ndim > 3:
        work = x.reshape(-1, *x.shape[-2:])
    else:
        work = x

    original_dtype = work.dtype
    work = work.to(torch.float32)
    should_transpose = work.size(-2) > work.size(-1)
    if should_transpose:
        work = work.mT

    work = work / (work.norm(dim=(-2, -1), keepdim=True) + eps)
    work = work.to(_dtype_from_name(ns_dtype))
    coeffs = POLAR_EXPRESS_COEFFICIENTS[: int(steps)]
    if not coeffs:
        raise ValueError("steps must be positive")

    reset_set = {int(i) for i in reset_iterations}
    gram = work @ work.mT
    eye = torch.eye(gram.size(-1), device=work.device, dtype=work.dtype).unsqueeze(0)
    eye = eye.expand(gram.size(0), -1, -1).contiguous()
    q: Tensor | None = None
    for i, (a, b, c) in enumerate(coeffs):
        if i in reset_set and i != 0:
            if q is None:
                raise RuntimeError("Gram Newton-Schulz reset requested before Q was initialized")
            work = q @ work
            gram = work @ work.mT
            q = None

        poly = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
        if i == 0 or i in reset_set:
            q = poly + a * eye
        else:
            if q is None:
                raise RuntimeError("Gram Newton-Schulz Q was not initialized")
            q = torch.baddbmm(q, q, poly, beta=a)

        if i < len(coeffs) - 1 and i + 1 not in reset_set:
            gram_poly = torch.baddbmm(gram, gram, poly, beta=a)
            gram = torch.baddbmm(gram_poly, poly, gram_poly, beta=a)

    if q is None:
        raise RuntimeError("Gram Newton-Schulz failed to build Q")
    work = q @ work
    if should_transpose:
        work = work.mT
    return work.to(original_dtype).reshape(original_shape)


@torch.no_grad()
def normuon_precondition(
    update: Tensor,
    second_momentum: Tensor,
    *,
    beta2: float = 0.9,
    eps: float = 1e-10,
    orientation: str = "row",
    aspect_scale: bool = False,
) -> Tensor:
    """Apply NorMuon second-moment normalization after GramNS.

    ``orientation="row"`` preserves the original EquiMuse-NorMuon recipe.
    ``orientation="auto"`` tracks rows for tall/square matrices and columns for
    wide matrices, matching the peer implementation's orientation-aware variant.
    """

    original_dtype = update.dtype
    work = update.float()
    if orientation == "auto":
        orientation = "row" if work.shape[-2] >= work.shape[-1] else "column"
    if orientation == "row":
        expected = (*work.shape[:-2], work.shape[-2], 1)
        second = work.square().mean(dim=-1, keepdim=True)
    elif orientation == "column":
        expected = (*work.shape[:-2], 1, work.shape[-1])
        second = work.square().mean(dim=-2, keepdim=True)
    else:
        raise ValueError(f"unknown NorMuon orientation: {orientation}")
    if tuple(second_momentum.shape) != tuple(expected):
        raise ValueError(f"second_momentum shape {tuple(second_momentum.shape)} does not match {tuple(expected)}")
    second_momentum.mul_(float(beta2)).add_(second, alpha=1.0 - float(beta2))

    old_norm = work.norm(dim=(-2, -1), keepdim=True)
    scaled = work * torch.rsqrt(second_momentum.clamp_min(float(eps)).to(work.dtype))
    new_norm = scaled.norm(dim=(-2, -1), keepdim=True)
    scaled = scaled * (old_norm / new_norm.clamp_min(float(eps)))
    if aspect_scale:
        rows, cols = work.shape[-2], work.shape[-1]
        scaled = scaled * math.sqrt(max(1.0, float(rows) / float(cols)))
    return scaled.to(original_dtype)


def _bucket_by_tensor(items: list[tuple[Any, ...]], tensor_index: int = 0) -> list[list[tuple[Any, ...]]]:
    buckets: dict[tuple[torch.device, torch.dtype], list[tuple[Any, ...]]] = defaultdict(list)
    for item in items:
        tensor = item[tensor_index]
        buckets[(tensor.device, tensor.dtype)].append(item)
    return list(buckets.values())


def _bucket_muon(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    buckets: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        matrix = item["matrix"]
        buckets[(matrix.device, matrix.dtype, tuple(matrix.shape))].append(item)
    return list(buckets.values())


def _foreach_lerp_(params: list[Tensor], ends: list[Tensor], weight: float, enabled: bool) -> None:
    if not params:
        return
    if enabled and hasattr(torch, "_foreach_lerp_"):
        torch._foreach_lerp_(params, ends, weight)
    else:
        for p, end in zip(params, ends, strict=True):
            p.lerp_(end, weight)


class EquiMuseNorMuon(torch.optim.Optimizer):
    """Standalone SODA-AMUSE + PMuonEq + GramNS + NorMuon optimizer."""

    def __init__(
        self,
        param_groups: Iterable[dict[str, Any]],
        *,
        beta1: float = 0.6,
        rho: float = 0.5,
        warmup_steps: int = 50,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: bool = True,
        batch_muon: bool = True,
        ns_steps: int = 5,
        ns_epsilon: float = 1e-7,
        ns_dtype: str | torch.dtype = torch.float16,
        gram_reset_iterations: Sequence[int] = (2,),
        soda_anchor_scale: float = 1.0,
        soda_warmup_steps: int = 0,
        pmuoneq_beta: float = 0.95,
        pmuoneq_row_gamma: float = 0.15,
        pmuoneq_col_gamma: float = 0.15,
        pmuoneq_eps: float = 1e-6,
        normuon_beta2: float = 0.9,
        normuon_eps: float = 1e-10,
        normuon_orientation: str = "row",
        normuon_aspect_scale: bool = False,
        sort_muon_params: bool = True,
    ) -> None:
        if not 0.0 < float(beta1) < 1.0:
            raise ValueError("beta1 must be in (0, 1)")
        if float(rho) < 0.0:
            raise ValueError("rho must be nonnegative")
        if int(warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        if int(ns_steps) <= 0:
            raise ValueError("ns_steps must be positive")
        if not 0.0 <= float(pmuoneq_beta) < 1.0:
            raise ValueError("pmuoneq_beta must be in [0, 1)")
        if not 0.0 <= float(normuon_beta2) < 1.0:
            raise ValueError("normuon_beta2 must be in [0, 1)")
        if float(soda_anchor_scale) <= 0.0:
            raise ValueError("soda_anchor_scale must be positive for this fixed recipe")

        self.beta1_init = float(beta1)
        self.rho = float(rho)
        self.warmup_steps = int(warmup_steps)
        self.r = float(r)
        self.weight_lr_power = float(weight_lr_power)
        self.foreach = bool(foreach)
        self.batch_muon = bool(batch_muon)
        self.ns_steps = int(ns_steps)
        self.ns_epsilon = float(ns_epsilon)
        self.ns_dtype = _dtype_from_name(ns_dtype)
        self.gram_reset_iterations = tuple(int(i) for i in gram_reset_iterations)
        self.soda_anchor_scale = float(soda_anchor_scale)
        self.soda_warmup_steps = int(soda_warmup_steps)
        self.pmuoneq_beta = float(pmuoneq_beta)
        self.pmuoneq_row_gamma = float(pmuoneq_row_gamma)
        self.pmuoneq_col_gamma = float(pmuoneq_col_gamma)
        self.pmuoneq_eps = float(pmuoneq_eps)
        self.normuon_beta2 = float(normuon_beta2)
        self.normuon_eps = float(normuon_eps)
        if normuon_orientation not in {"row", "auto", "column"}:
            raise ValueError("normuon_orientation must be 'row', 'auto', or 'column'")
        self.normuon_orientation = normuon_orientation
        self.normuon_aspect_scale = bool(normuon_aspect_scale)
        self.train_mode = False
        self._last_stats: dict[str, float] = {}

        defaults = dict(
            use_muon=False,
            weight_decay=0.0,
            beta2=0.999,
            eps=1e-10,
            momentum=0.95,
            warmup_steps=self.warmup_steps,
            k=0,
            weight_sum=0.0,
            beta1=self.beta1_init,
            ckp1=1.0,
            foreach=self.foreach,
            batch_muon=self.batch_muon,
            soda_anchor_scale=self.soda_anchor_scale,
            soda_warmup_steps=self.soda_warmup_steps,
            pmuoneq_beta=self.pmuoneq_beta,
            pmuoneq_row_gamma=self.pmuoneq_row_gamma,
            pmuoneq_col_gamma=self.pmuoneq_col_gamma,
            pmuoneq_eps=self.pmuoneq_eps,
            normuon_beta2=self.normuon_beta2,
            normuon_eps=self.normuon_eps,
            normuon_orientation=self.normuon_orientation,
            normuon_aspect_scale=self.normuon_aspect_scale,
        )

        groups = list(param_groups)
        for group in groups:
            group.setdefault("lr", 1e-3)
            group.setdefault("base_lr", float(group["lr"]))
            group.setdefault("initial_lr", float(group["lr"]))
            if group.get("use_muon", False) and sort_muon_params:
                if "param_names" in group:
                    pairs = sorted(
                        zip(list(group["param_names"]), list(group["params"]), strict=True),
                        key=lambda item: tuple(item[1].size()),
                        reverse=True,
                    )
                    group["param_names"] = [name for name, _ in pairs]
                    group["params"] = [param for _, param in pairs]
                else:
                    group["params"] = sorted(list(group["params"]), key=lambda p: tuple(p.size()), reverse=True)
        super().__init__(groups, defaults)

    @property
    def last_stats(self) -> dict[str, float]:
        return dict(self._last_stats)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        state = super().state_dict()
        state["train_mode"] = self.train_mode
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        state_copy = dict(state_dict)
        self.train_mode = bool(state_copy.pop("train_mode", False))
        super().load_state_dict(state_copy)

    @torch.no_grad()
    def train(self) -> "EquiMuseNorMuon":
        """Swap parameters from averaged/eval weights X to train weights Y."""

        if self.train_mode:
            return self
        for group in self.param_groups:
            beta1 = float(group.get("beta1", self.beta1_init))
            items = [(p.detach(), self.state[p]["z"]) for p in group["params"] if "z" in self.state[p]]
            for bucket in _bucket_by_tensor(items, tensor_index=0):
                _foreach_lerp_([x[0] for x in bucket], [x[1] for x in bucket], 1.0 - beta1, bool(group.get("foreach", self.foreach)))
        self.train_mode = True
        return self

    @torch.no_grad()
    def eval(self) -> "EquiMuseNorMuon":
        """Swap parameters from train weights Y to averaged/eval weights X."""

        if not self.train_mode:
            return self
        for group in self.param_groups:
            beta1 = float(group.get("beta1", self.beta1_init))
            items = [(p.detach(), self.state[p]["z"]) for p in group["params"] if "z" in self.state[p]]
            for bucket in _bucket_by_tensor(items, tensor_index=0):
                _foreach_lerp_([x[0] for x in bucket], [x[1] for x in bucket], 1.0 - 1.0 / beta1, bool(group.get("foreach", self.foreach)))
        self.train_mode = False
        return self

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if not self.train_mode:
            raise RuntimeError("EquiMuseNorMuon.step() requires optimizer.train() before stepping")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_matrices = 0
        fallback_params = 0
        soda_params = 0
        beta1_values: list[float] = []
        ckp1_values: list[float] = []
        lr_values: list[float] = []
        soda_values: list[float] = []
        aspect_enabled = False
        for group in self.param_groups:
            lr, ckp1, beta1, t = self._advance_group_schedule(group)
            soda_lambda = self._soda_lambda(group, t)
            if bool(group.get("use_muon", False)):
                aspect_enabled = aspect_enabled or bool(group.get("normuon_aspect_scale", self.normuon_aspect_scale))
                count = self._step_muon_group(group, lr, ckp1, beta1, soda_lambda)
                muon_matrices += count
            else:
                count = self._step_fallback_group(group, lr, ckp1, beta1, t, soda_lambda)
                fallback_params += count
            if soda_lambda > 0.0:
                soda_params += count
            beta1_values.append(float(beta1))
            ckp1_values.append(float(ckp1))
            lr_values.append(float(lr))
            soda_values.append(float(soda_lambda))

        denom = max(len(beta1_values), 1)
        self._last_stats = {
            "equimuse_normuon_muon_matrix_count": float(muon_matrices),
            "equimuse_normuon_fallback_param_count": float(fallback_params),
            "equimuse_normuon_soda_anchored_param_count": float(soda_params),
            "equimuse_normuon_train_mode": float(self.train_mode),
            "equimuse_normuon_beta1_mean": sum(beta1_values) / denom,
            "equimuse_normuon_ckp1_mean": sum(ckp1_values) / denom,
            "equimuse_normuon_scheduled_lr_mean": sum(lr_values) / denom,
            "equimuse_normuon_soda_lambda_mean": sum(soda_values) / denom,
            "equimuse_normuon_aspect_scale_enabled": float(aspect_enabled),
        }
        return loss

    def _advance_group_schedule(self, group: dict[str, Any]) -> tuple[float, float, float, int]:
        k = int(group["k"])
        t = k + 1
        warmup_steps = int(group.get("warmup_steps", self.warmup_steps))
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")

        base_lr = self._current_base_lr(group)
        lr = base_lr * min(1.0, t / warmup_steps)
        group["lr"] = lr
        group["scheduled_lr"] = lr

        weight = (t**self.r) * (lr**self.weight_lr_power)
        future_weight_sum = float(group.get("weight_sum", 0.0)) + weight
        ckp1 = weight / future_weight_sum if future_weight_sum > 0.0 else 1.0
        group["weight_sum"] = future_weight_sum
        group["ckp1"] = ckp1

        beta1 = self._compute_beta1(group, t, ckp1, warmup_steps)
        group["beta1"] = beta1
        group["k"] = t
        return lr, ckp1, beta1, t

    @staticmethod
    def _current_base_lr(group: dict[str, Any]) -> float:
        current = float(group.get("lr", group.get("base_lr", 0.0)))
        base = float(group.get("base_lr", current))
        scheduled = group.get("scheduled_lr")
        if scheduled is None and not math.isclose(current, base, rel_tol=0.0, abs_tol=1e-18):
            group["base_lr"] = current
            return current
        if scheduled is not None and not math.isclose(current, float(scheduled), rel_tol=0.0, abs_tol=1e-18):
            group["base_lr"] = current
            return current
        return base

    def _compute_beta1(self, group: dict[str, Any], t: int, ckp1: float, warmup_steps: int) -> float:
        if t <= warmup_steps:
            if t == warmup_steps:
                group["c_warmup"] = ckp1
            return self.beta1_init
        c_warmup = float(group.get("c_warmup", 1.0 / warmup_steps))
        s_t = (ckp1 * (1.0 - c_warmup)) / max(c_warmup * (1.0 - ckp1), 1e-30)
        return 1.0 - (s_t**self.rho) * (1.0 - self.beta1_init)

    def _get_z(self, p: Tensor) -> Tensor:
        state = self.state[p]
        if "z" not in state:
            state["z"] = torch.clone(p.detach(), memory_format=torch.preserve_format)
        return state["z"]

    def _soda_lambda(self, group: dict[str, Any], t: int) -> float:
        warmup = int(group.get("soda_warmup_steps", self.soda_warmup_steps))
        if t <= warmup:
            return 0.0
        scale = float(group.get("soda_anchor_scale", self.soda_anchor_scale))
        return scale / float(t - warmup + 1)

    def _get_soda_anchor(self, p: Tensor, z: Tensor) -> Tensor:
        state = self.state[p]
        if "soda_z0" not in state:
            state["soda_z0"] = torch.clone(z.detach(), memory_format=torch.preserve_format)
        return state["soda_z0"]

    @staticmethod
    def _apply_soda_anchor_one(z: Tensor, anchor: Tensor, soda_lambda: float) -> None:
        if soda_lambda > 0.0:
            z.lerp_(anchor, soda_lambda)

    def _apply_soda_anchor_bucket(self, zs: list[Tensor], anchors: list[Tensor], soda_lambda: float) -> None:
        if soda_lambda <= 0.0:
            return
        if self.foreach and hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(zs, anchors, soda_lambda)
        else:
            for z, anchor in zip(zs, anchors, strict=True):
                z.lerp_(anchor, soda_lambda)

    def _step_fallback_group(
        self,
        group: dict[str, Any],
        lr: float,
        ckp1: float,
        beta1: float,
        t: int,
        soda_lambda: float,
    ) -> int:
        items: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            z = self._get_z(p)
            anchor = self._get_soda_anchor(p, z)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            items.append((p, p.grad, z, state["exp_avg_sq"], anchor))

        if bool(group.get("foreach", self.foreach)) and hasattr(torch, "_foreach_mul_"):
            for bucket in _bucket_by_tensor(items, tensor_index=0):
                self._step_fallback_bucket_foreach(bucket, group, lr, ckp1, beta1, t, soda_lambda)
        else:
            for item in items:
                self._step_fallback_one(item, group, lr, ckp1, beta1, t, soda_lambda)
        return len(items)

    def _step_fallback_bucket_foreach(
        self,
        bucket: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
        group: dict[str, Any],
        lr: float,
        ckp1: float,
        beta1: float,
        t: int,
        soda_lambda: float,
    ) -> None:
        params = [x[0] for x in bucket]
        grads = [x[1] for x in bucket]
        zs = [x[2] for x in bucket]
        exp_avg_sqs = [x[3] for x in bucket]
        anchors = [x[4] for x in bucket]
        beta2 = float(group.get("beta2", 0.999))
        eps = float(group.get("eps", 1e-10))
        wd = float(group.get("weight_decay", 0.0))
        bias_correction2 = 1.0 - beta2**t

        _foreach_lerp_(params, zs, 1.0 - 1.0 / beta1, True)
        self._apply_soda_anchor_bucket(zs, anchors, soda_lambda)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        denoms = torch._foreach_div(exp_avg_sqs, bias_correction2)
        torch._foreach_sqrt_(denoms)
        torch._foreach_add_(denoms, eps)
        updates = torch._foreach_div(grads, denoms)
        if wd and soda_lambda <= 0.0:
            updates = torch._foreach_add(updates, zs, alpha=wd)
        torch._foreach_add_(zs, updates, alpha=-lr)
        _foreach_lerp_(params, zs, ckp1, True)
        _foreach_lerp_(params, zs, 1.0 - beta1, True)

    def _step_fallback_one(
        self,
        item: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        group: dict[str, Any],
        lr: float,
        ckp1: float,
        beta1: float,
        t: int,
        soda_lambda: float,
    ) -> None:
        p, grad, z, exp_avg_sq, anchor = item
        beta2 = float(group.get("beta2", 0.999))
        eps = float(group.get("eps", 1e-10))
        wd = float(group.get("weight_decay", 0.0))
        p.lerp_(z, 1.0 - 1.0 / beta1)
        self._apply_soda_anchor_one(z, anchor, soda_lambda)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        update = grad.div(exp_avg_sq.div(1.0 - beta2**t).sqrt_().add_(eps))
        if wd and soda_lambda <= 0.0:
            update = update.add(z, alpha=wd)
        z.add_(update, alpha=-lr)
        p.lerp_(z, ckp1)
        p.lerp_(z, 1.0 - beta1)

    def _step_muon_group(self, group: dict[str, Any], lr: float, ckp1: float, beta1: float, soda_lambda: float) -> int:
        items: list[dict[str, Any]] = []
        momentum_beta = float(group.get("momentum", 0.95))
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.ndim < 2:
                raise ValueError("use_muon=True parameters must have ndim >= 2")
            state = self.state[p]
            z = self._get_z(p)
            anchor = self._get_soda_anchor(p, z)
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            p.lerp_(z, 1.0 - 1.0 / beta1)
            state["momentum_buffer"].lerp_(p.grad, 1.0 - momentum_beta)
            update = p.grad.lerp(state["momentum_buffer"], momentum_beta)
            raw_grad = p.grad.detach()
            if update.ndim > 2:
                matrix = update.reshape(update.shape[0], -1)
                raw_matrix = raw_grad.reshape(raw_grad.shape[0], -1)
            else:
                matrix = update.reshape(update.shape[-2], update.shape[-1])
                raw_matrix = raw_grad.reshape(raw_grad.shape[-2], raw_grad.shape[-1])
            items.append(
                {
                    "param": p,
                    "state": state,
                    "z": z,
                    "anchor": anchor,
                    "matrix": matrix,
                    "raw_matrix": raw_matrix,
                    "param_shape": p.shape,
                    "matrix_shape": matrix.shape,
                }
            )

        if bool(group.get("batch_muon", self.batch_muon)):
            for bucket in _bucket_muon(items):
                updates = self._orthogonalize(self._pmuoneq_bucket(bucket, group))
                updates = self._normuon_bucket(bucket, updates, group)
                for item, update in zip(bucket, updates, strict=True):
                    self._apply_muon_update(item, update, group, lr, ckp1, beta1, soda_lambda)
        else:
            for item in items:
                update = self._orthogonalize(self._pmuoneq_matrix(item, group))
                update = self._normuon_matrix(item, update, group)
                self._apply_muon_update(item, update, group, lr, ckp1, beta1, soda_lambda)
        return len(items)

    @staticmethod
    def _get_diag_state(state: dict[str, Any], key: str, size: int, device: torch.device) -> Tensor:
        value = state.get(key)
        if value is None or value.shape != (size,) or value.device != device:
            value = torch.zeros(size, device=device, dtype=torch.float32)
            state[key] = value
        return value

    @torch.no_grad()
    def _pmuoneq_matrix(self, item: dict[str, Any], group: dict[str, Any]) -> Tensor:
        row_gamma = float(group.get("pmuoneq_row_gamma", self.pmuoneq_row_gamma))
        col_gamma = float(group.get("pmuoneq_col_gamma", self.pmuoneq_col_gamma))
        beta = float(group.get("pmuoneq_beta", self.pmuoneq_beta))
        eps = float(group.get("pmuoneq_eps", self.pmuoneq_eps))
        raw = item["raw_matrix"].detach().float()
        update = item["matrix"].float()
        rows, cols = raw.shape
        row_ema = self._get_diag_state(item["state"], "pmuoneq_row_ema", rows, raw.device)
        col_ema = self._get_diag_state(item["state"], "pmuoneq_col_ema", cols, raw.device)
        row_ema.mul_(beta).add_(raw.square().mean(dim=1), alpha=1.0 - beta)
        col_ema.mul_(beta).add_(raw.square().mean(dim=0), alpha=1.0 - beta)
        update = update * row_ema.clamp_min(eps).pow(-row_gamma).to(update.dtype).unsqueeze(1)
        update = update * col_ema.clamp_min(eps).pow(-col_gamma).to(update.dtype).unsqueeze(0)
        return update.to(item["matrix"].dtype)

    @torch.no_grad()
    def _pmuoneq_bucket(self, bucket: list[dict[str, Any]], group: dict[str, Any]) -> Tensor:
        row_gamma = float(group.get("pmuoneq_row_gamma", self.pmuoneq_row_gamma))
        col_gamma = float(group.get("pmuoneq_col_gamma", self.pmuoneq_col_gamma))
        beta = float(group.get("pmuoneq_beta", self.pmuoneq_beta))
        eps = float(group.get("pmuoneq_eps", self.pmuoneq_eps))
        updates = torch.stack([item["matrix"].float() for item in bucket], dim=0)
        raw = torch.stack([item["raw_matrix"].detach().float() for item in bucket], dim=0)
        _, rows, cols = raw.shape
        row_emas = torch.stack(
            [self._get_diag_state(item["state"], "pmuoneq_row_ema", rows, raw.device) for item in bucket],
            dim=0,
        )
        col_emas = torch.stack(
            [self._get_diag_state(item["state"], "pmuoneq_col_ema", cols, raw.device) for item in bucket],
            dim=0,
        )
        row_emas.mul_(beta).add_(raw.square().mean(dim=2), alpha=1.0 - beta)
        col_emas.mul_(beta).add_(raw.square().mean(dim=1), alpha=1.0 - beta)
        updates = updates * row_emas.clamp_min(eps).pow(-row_gamma).to(updates.dtype).unsqueeze(-1)
        updates = updates * col_emas.clamp_min(eps).pow(-col_gamma).to(updates.dtype).unsqueeze(-2)
        for i, item in enumerate(bucket):
            item["state"]["pmuoneq_row_ema"].copy_(row_emas[i])
            item["state"]["pmuoneq_col_ema"].copy_(col_emas[i])
        return updates.to(bucket[0]["matrix"].dtype)

    def _orthogonalize(self, update: Tensor) -> Tensor:
        return gram_newton_schulz(
            update,
            steps=self.ns_steps,
            eps=self.ns_epsilon,
            ns_dtype=self.ns_dtype,
            reset_iterations=self.gram_reset_iterations,
        )

    @staticmethod
    def _normuon_effective_orientation(rows: int, cols: int, orientation: str) -> str:
        if orientation == "auto":
            return "row" if rows >= cols else "column"
        if orientation not in {"row", "column"}:
            raise ValueError("normuon_orientation must be 'row', 'auto', or 'column'")
        return orientation

    @staticmethod
    def _get_normuon_second_state(
        state: dict[str, Any],
        rows: int,
        cols: int,
        device: torch.device,
        orientation: str,
    ) -> Tensor:
        effective = EquiMuseNorMuon._normuon_effective_orientation(rows, cols, orientation)
        shape = (rows, 1) if effective == "row" else (1, cols)
        key = f"normuon_{effective}_second_moment"
        value = state.get(key)
        if value is None or tuple(value.shape) != shape or value.device != device:
            value = torch.zeros(shape, device=device, dtype=torch.float32)
            state[key] = value
        return value

    def _normuon_matrix(self, item: dict[str, Any], update: Tensor, group: dict[str, Any]) -> Tensor:
        rows, cols = update.shape[-2], update.shape[-1]
        orientation = str(group.get("normuon_orientation", self.normuon_orientation))
        moment = self._get_normuon_second_state(item["state"], rows, cols, update.device, orientation)
        return normuon_precondition(
            update,
            moment,
            beta2=float(group.get("normuon_beta2", self.normuon_beta2)),
            eps=float(group.get("normuon_eps", self.normuon_eps)),
            orientation=orientation,
            aspect_scale=bool(group.get("normuon_aspect_scale", self.normuon_aspect_scale)),
        )

    def _normuon_bucket(self, bucket: list[dict[str, Any]], updates: Tensor, group: dict[str, Any]) -> Tensor:
        rows, cols = updates.shape[-2], updates.shape[-1]
        orientation = str(group.get("normuon_orientation", self.normuon_orientation))
        effective = self._normuon_effective_orientation(rows, cols, orientation)
        moments = torch.stack(
            [self._get_normuon_second_state(item["state"], rows, cols, updates.device, orientation) for item in bucket],
            dim=0,
        )
        out = normuon_precondition(
            updates,
            moments,
            beta2=float(group.get("normuon_beta2", self.normuon_beta2)),
            eps=float(group.get("normuon_eps", self.normuon_eps)),
            orientation=orientation,
            aspect_scale=bool(group.get("normuon_aspect_scale", self.normuon_aspect_scale)),
        )
        for i, item in enumerate(bucket):
            item["state"][f"normuon_{effective}_second_moment"].copy_(moments[i])
        return out

    def _apply_muon_update(
        self,
        item: dict[str, Any],
        update_matrix: Tensor,
        group: dict[str, Any],
        lr: float,
        ckp1: float,
        beta1: float,
        soda_lambda: float,
    ) -> None:
        rows, cols = item["matrix_shape"]
        update = (update_matrix * (0.2 * math.sqrt(max(rows, cols)))).reshape(item["param_shape"])
        z = item["z"]
        self._apply_soda_anchor_one(z, item["anchor"], soda_lambda)
        wd = float(group.get("weight_decay", 0.0))
        if wd and soda_lambda <= 0.0:
            z.mul_(1.0 - lr * wd)
        z.add_(update.to(z.dtype), alpha=-lr)
        item["param"].lerp_(z, ckp1)
        item["param"].lerp_(z, 1.0 - beta1)


def build_equimuse_normuon_param_groups(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float = 1e-3,
    matrix_lr: float | None = None,
    weight_decay: float = 0.0,
    beta2: float = 0.999,
    eps: float = 1e-10,
    momentum: float = 0.95,
    fallback_weight_decay: bool = True,
    soda_anchor_scale: float = 1.0,
    soda_warmup_steps: int = 0,
    pmuoneq_beta: float = 0.95,
    pmuoneq_row_gamma: float = 0.15,
    pmuoneq_col_gamma: float = 0.15,
    pmuoneq_eps: float = 1e-6,
    normuon_beta2: float = 0.9,
    normuon_eps: float = 1e-10,
    normuon_orientation: str = "row",
    normuon_aspect_scale: bool = False,
) -> list[dict[str, Any]]:
    """Split model parameters into fallback and EquiMuse-NorMuon matrix groups."""

    fallback: list[torch.nn.Parameter] = []
    fallback_names: list[str] = []
    muon: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    seen: set[int] = set()
    for name, param in named_params:
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        lower = name.lower().removeprefix("module.")
        is_embedding = any(token in lower for token in ("embed", "embedding", "wte", "wpe", "token_emb", "tok_embeddings"))
        is_lm_head = "lm_head" in lower or "unembed" in lower or lower.endswith("head.weight")
        is_scalar_or_bias = param.ndim < 2 or lower.endswith("bias")
        if is_embedding or is_lm_head or is_scalar_or_bias:
            fallback.append(param)
            fallback_names.append(name)
        else:
            muon.append(param)
            muon_names.append(name)

    groups: list[dict[str, Any]] = []
    if fallback:
        groups.append(
            {
                "params": fallback,
                "param_names": fallback_names,
                "lr": float(lr),
                "beta2": float(beta2),
                "eps": float(eps),
                "use_muon": False,
                "weight_decay": float(weight_decay if fallback_weight_decay else 0.0),
                "soda_anchor_scale": float(soda_anchor_scale),
                "soda_warmup_steps": int(soda_warmup_steps),
            }
        )
    if muon:
        groups.append(
            {
                "params": muon,
                "param_names": muon_names,
                "lr": float(lr if matrix_lr is None else matrix_lr),
                "momentum": float(momentum),
                "use_muon": True,
                "weight_decay": float(weight_decay),
                "soda_anchor_scale": float(soda_anchor_scale),
                "soda_warmup_steps": int(soda_warmup_steps),
                "pmuoneq_beta": float(pmuoneq_beta),
                "pmuoneq_row_gamma": float(pmuoneq_row_gamma),
                "pmuoneq_col_gamma": float(pmuoneq_col_gamma),
                "pmuoneq_eps": float(pmuoneq_eps),
                "normuon_beta2": float(normuon_beta2),
                "normuon_eps": float(normuon_eps),
                "normuon_orientation": normuon_orientation,
                "normuon_aspect_scale": bool(normuon_aspect_scale),
            }
        )
    return groups


__all__ = [
    "EquiMuseNorMuon",
    "build_equimuse_normuon_param_groups",
    "gram_newton_schulz",
    "normuon_precondition",
    "POLAR_EXPRESS_COEFFICIENTS",
]
