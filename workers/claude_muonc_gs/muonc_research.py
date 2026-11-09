"""Research variants of MuonC used by the MuonC x MuonGS combination study.

``MuonCResearch`` extends the root ``muonc.MuonC`` with switches that are only
needed to compare ways of combining corrected decay (C) with the Gram-Schmidt
tangent projection (GS).  With default arguments it is step-for-step the root
``MuonC`` (covered by a test).

Extra switches
--------------
``gs_strength``           fraction of the radial component removed (1 = GS,
                          0 = no projection, in between = partial projection).
                          This replaces the root ``gs_projection`` flag.
``gs_mode``               ``"both"`` removes the radial component whatever its
                          sign; ``"outward"`` removes it only when the step
                          would *grow* the Frobenius norm.
``matrix_decay_law``      ``"corrected"`` (``1 - wd * lr^2 / base_lr``) or
                          ``"standard"`` (``1 - wd * lr``) on the matrix path.
                          These two replace the root ``corrected_decay`` flag.
``fallback_decay_law``    same choice for fallback (Adam-family) tensors.
``unprojected_decay_law`` decay law for matrices tagged ``_no_gs``; ``None``
                          follows ``matrix_decay_law``.
``norm_mode``             ``"free"`` or ``"fixed"``.  ``"fixed"`` rescales each
                          matrix back to its initial Frobenius norm after the
                          step (a MuonH-style hard constraint), which is the
                          limit of GS + decay held exactly at equilibrium.
``norm_target``           ``None`` or ``kappa > 0``.  Replaces the global decay
                          coefficient by a per-matrix corrected decay
                          ``d_t = lr_t^2 |N_t|^2 / (2 kappa^2 |W_0|^2)`` whose
                          fixed point is ``|W| = kappa |W_0|`` for *every*
                          projected matrix, whatever its initial scale.  A
                          global ``wd`` instead drives all matrices to one
                          absolute RMS (``0.2 sqrt(lr_max / (2 wd))``), erasing
                          e.g. residual-depth-scaled initialisation.  Requires
                          the full projection; matrices tagged ``_no_gs`` or
                          ``_no_weight_decay`` and zero-initialised matrices
                          get no decay in this mode.

``radial_lr``             ``c >= 0``.  With the projection on, the loss can no
                          longer move a matrix's norm at all.  This gives the
                          radial degree of freedom back as its own scalar
                          optimizer per matrix: AdamATan2 on the log-scale
                          gradient ``h = <G, W>_F = dL/dlog|W|`` produces a
                          bounded ``u`` and the weight is multiplied by
                          ``1 - c * lr_t * u`` (folded into the direction as
                          ``+ c * u * W``).  ``c = 0`` is plain MuonGS; the
                          tangential step is unchanged for every ``c``.

Diagnostics
-----------
Set ``optimizer.diag_next = True`` before a step to record, for every matrix
parameter: the radial cosine ``<W, N> / (|W| |N|)`` of the direction *before*
projection, the norm of the projected direction that is applied (excluding
any ``norm_target`` shrink term), its tangential part, and ``|W|``.
``pop_diag()`` returns them keyed by parameter name.  Recording costs a host
sync, so it is opt-in per step rather than always on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from muonc import MuonC, _matrix_view  # noqa: E402

DECAY_LAWS = {"standard", "corrected"}
_REPLACED = {
    "gs_projection": "gs_strength",
    "corrected_decay": "matrix_decay_law / fallback_decay_law",
    "track_gs_stats": "diag_next / pop_diag()",
}


class MuonCResearch(MuonC):
    def __init__(
        self,
        *args: Any,
        gs_strength: float = 1.0,
        gs_mode: str = "both",
        matrix_decay_law: str = "corrected",
        fallback_decay_law: str = "corrected",
        unprojected_decay_law: str | None = None,
        norm_mode: str = "free",
        norm_target: float | None = None,
        radial_lr: float = 0.0,
        radial_betas: tuple[float, float] = (0.9, 0.95),
        **kwargs: Any,
    ) -> None:
        for name, replacement in _REPLACED.items():
            if name in kwargs:
                raise TypeError(f"MuonCResearch does not take {name!r}; use {replacement}")
        if not 0.0 <= gs_strength <= 1.0:
            raise ValueError("gs_strength must be in [0, 1]")
        if gs_mode not in {"both", "outward"}:
            raise ValueError("gs_mode must be 'both' or 'outward'")
        for law in (matrix_decay_law, fallback_decay_law, unprojected_decay_law or "standard"):
            if law not in DECAY_LAWS:
                raise ValueError(f"decay law must be one of {sorted(DECAY_LAWS)}")
        if norm_mode not in {"free", "fixed"}:
            raise ValueError("norm_mode must be 'free' or 'fixed'")
        if norm_target is not None:
            if norm_target <= 0.0:
                raise ValueError("norm_target must be positive")
            if float(kwargs.get("matrix_weight_decay", 0.0)) != 0.0:
                raise ValueError("norm_target replaces matrix_weight_decay; pass only one")
            if gs_strength != 1.0 or gs_mode != "both" or norm_mode != "free":
                raise ValueError("norm_target needs the full projection and norm_mode='free'")
        if radial_lr < 0.0:
            raise ValueError("radial_lr must be non-negative")
        if radial_lr > 0.0 and (gs_strength != 1.0 or gs_mode != "both" or norm_mode != "free"):
            raise ValueError("radial_lr needs the full projection and norm_mode='free'")
        super().__init__(*args, gs_projection=gs_strength > 0.0, **kwargs)
        self.radial_lr = float(radial_lr)
        self.radial_betas = (float(radial_betas[0]), float(radial_betas[1]))
        self.gs_strength = float(gs_strength)
        self.gs_mode = gs_mode
        self.matrix_decay_law = matrix_decay_law
        self.fallback_decay_law = fallback_decay_law
        self.unprojected_decay_law = unprojected_decay_law
        self.norm_mode = norm_mode
        self.norm_target = None if norm_target is None else float(norm_target)
        self.diag_next = False
        self._lr_now = 0.0
        self._diag: dict[int, tuple[torch.Tensor, ...]] = {}
        self._radial_diag: dict[int, torch.Tensor] = {}
        self._names: dict[int, str] = {}
        for group in self.param_groups:
            names = list(group.get("param_names", []))
            if len(names) == len(group["params"]):
                for name, parameter in zip(names, group["params"]):
                    self._names[id(parameter)] = name

    # ------------------------------------------------------------------ GS
    def _flags(self, entries, weights: torch.Tensor, attribute: str) -> torch.Tensor:
        return torch.tensor(
            [bool(getattr(parameter, attribute, False)) for parameter, _a, _s, _g in entries],
            device=weights.device,
            dtype=torch.bool,
        ).view(-1, *([1] * (weights.ndim - 1)))

    def _postprocess_matrix_updates(self, entries, updates):  # type: ignore[override]
        if self.gs_strength == 0.0 and not self.diag_next:
            return updates
        weights = torch.stack(
            [_matrix_view(self._master_param(parameter).detach()) for parameter, _a, _s, _g in entries],
            dim=0,
        ).to(dtype=torch.float32)
        directions = updates.to(dtype=torch.float32)
        if directions.shape != weights.shape:
            raise ValueError("MuonGS matrix/update shape mismatch")
        dims = (-2, -1)
        dot = (weights * directions).sum(dim=dims, keepdim=True)
        norm_sq = weights.square().sum(dim=dims, keepdim=True)
        radial = dot / norm_sq.clamp_min(self.gs_eps)
        skip = self._flags(entries, weights, "_no_gs")
        if self.gs_strength > 0.0:
            removed = radial.clamp(max=0.0) if self.gs_mode == "outward" else radial
            projected = directions - self.gs_strength * removed * weights
            applied = torch.where(skip, directions, projected)
        else:
            applied = directions
        if self.diag_next:
            direction_norm = directions.norm(dim=dims, keepdim=True).clamp_min(self.gs_eps)
            cosine = dot / (norm_sq.sqrt().clamp_min(self.gs_eps) * direction_norm)
            applied_sq = applied.square().sum(dim=dims, keepdim=True)
            applied_radial_sq = (weights * applied).sum(dim=dims, keepdim=True).square() / norm_sq.clamp_min(self.gs_eps)
            tangent_sq = (applied_sq - applied_radial_sq).clamp_min(0.0)
            for index, (parameter, _a, _s, _g) in enumerate(entries):
                self._diag[id(parameter)] = (
                    cosine[index].mean(),
                    applied_sq[index].sum().sqrt(),
                    tangent_sq[index].sum().sqrt(),
                    norm_sq[index].sum().sqrt(),
                )
        if self.radial_lr > 0.0:
            # Scalar AdamATan2 on the log-scale gradient h = <G, W> of each matrix.
            grads = torch.stack([entry[3] for entry in entries], dim=0).to(dtype=torch.float32)
            h = (grads * weights).sum(dim=dims, keepdim=True)
            beta1, beta2 = self.radial_betas
            controls = []
            for index, (parameter, _a, _s, _g) in enumerate(entries):
                state = self.state[parameter]
                if "radial_m" not in state:
                    state["radial_m"] = torch.zeros_like(h[index])
                    state["radial_v"] = torch.zeros_like(h[index])
                    state["radial_t"] = 0
                state["radial_t"] += 1
                count = state["radial_t"]
                state["radial_m"].lerp_(h[index], 1.0 - beta1)
                state["radial_v"].lerp_(h[index].square(), 1.0 - beta2)
                m_hat = state["radial_m"] / (1.0 - beta1**count)
                v_hat = state["radial_v"] / (1.0 - beta2**count)
                controls.append(torch.atan2(m_hat, v_hat.sqrt()))
            control = torch.stack(controls, dim=0)
            if self.diag_next:
                for index, (parameter, _a, _s, _g) in enumerate(entries):
                    self._radial_diag[id(parameter)] = control[index].mean()
            applied = torch.where(skip, applied, applied + self.radial_lr * control * weights)
        if self.norm_target is not None:
            # Per-matrix corrected decay folded into the direction:
            #   W <- W - lr (N + c W),  c = lr |N|^2 / (2 kappa^2 |W_0|^2)
            # so the multiplicative factor is 1 - lr^2 |N|^2 / (2 kappa^2 |W_0|^2).
            r0_sq = torch.stack(
                [
                    self.state[parameter]["gs_r0"].to(weights.device).square().reshape(weights.shape[1:-2] + (1, 1))
                    for parameter, _a, _s, _g in entries
                ],
                dim=0,
            )
            tau_sq = applied.square().sum(dim=dims, keepdim=True)
            coefficient = self._lr_now * tau_sq / (2.0 * self.norm_target**2 * r0_sq.clamp_min(self.gs_eps))
            exempt = skip | (r0_sq <= 0) | self._flags(entries, weights, "_no_weight_decay")
            applied = torch.where(exempt, applied, applied + coefficient * weights)
        elif self.gs_strength == 0.0:
            return updates
        return applied.to(dtype=updates.dtype)

    def pop_diag(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for ident, (cosine, applied_norm, tangent_norm, weight_norm) in self._diag.items():
            out[self._names.get(ident, str(ident))] = {
                "radial_cos": float(cosine),
                "update_norm": float(applied_norm),
                "tangent_norm": float(tangent_norm),
                "weight_norm": float(weight_norm),
            }
            if ident in self._radial_diag:
                out[self._names.get(ident, str(ident))]["radial_control"] = float(self._radial_diag[ident])
        self._diag = {}
        self._radial_diag = {}
        return out

    # --------------------------------------------------------------- decay
    def _decay(self, law: str, weight_decay: float, lr: float, base_lr: float) -> float:
        if weight_decay == 0.0:
            return 1.0
        if law == "corrected":
            return self._corrected_decay_factor(weight_decay, lr, base_lr)
        return 1.0 - lr * weight_decay

    def _matrix_decay_factor(self, group, lr):  # type: ignore[override]
        return self._decay(
            self.matrix_decay_law, float(self.matrix_weight_decay), lr, float(group.get("base_lr", lr))
        )

    def _fallback_decay_factor(self, group, lr):  # type: ignore[override]
        return self._decay(
            self.fallback_decay_law,
            float(group.get("fallback_weight_decay", 0.0)),
            lr,
            float(group.get("base_lr", lr)),
        )

    # ---------------------------------------------------------------- step
    def _step_matrix_group(self, group, *, lr, t):  # type: ignore[override]
        self._lr_now = float(lr)
        return super()._step_matrix_group(group, lr=lr, t=t)

    def _matrix_parameters(self):
        for group in self.param_groups:
            if group.get("use_matrix_update", False):
                yield from ((group, parameter) for parameter in group["params"])

    def _refresh(self, parameter: torch.Tensor, master: torch.Tensor) -> None:
        if master is not parameter:
            self._copy_masters_to_parameters([parameter], [master])

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        fixed = self.norm_mode == "fixed"
        need_r0 = fixed or self.norm_target is not None
        relaw = (
            self.unprojected_decay_law is not None
            and self.unprojected_decay_law != self.matrix_decay_law
            and self.matrix_weight_decay != 0.0
        )
        corrections: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        if need_r0 or relaw:
            for group, parameter in self._matrix_parameters():
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                master = self._master_param(parameter)
                if need_r0 and "gs_r0" not in state:
                    state["gs_r0"] = _matrix_view(master.detach()).float().norm(dim=(-2, -1), keepdim=True)
                if relaw and getattr(parameter, "_no_gs", False) and not getattr(parameter, "_no_weight_decay", False):
                    lr = float(group["lr"])
                    base_lr = float(group.get("base_lr", lr))
                    wd = float(self.matrix_weight_decay)
                    applied = self._decay(self.matrix_decay_law, wd, lr, base_lr)
                    wanted = self._decay(self.unprojected_decay_law, wd, lr, base_lr)
                    # The base step computes applied * L - lr * N with L the
                    # SODA-pulled weight.  Adding (wanted - applied) * L after
                    # the step turns that into wanted * L - lr * N exactly.
                    weight = self._soda_weight(int(group.get("k", 0)) + 1, parameter)
                    pulled = master.detach().clone()
                    if weight:
                        pulled.lerp_(self._soda_anchor(parameter), weight)
                    corrections.append((parameter, pulled, wanted - applied))
        super().step(None)
        self.diag_next = False
        for parameter, pulled, delta in corrections:
            master = self._master_param(parameter)
            master.add_(pulled.to(master.dtype), alpha=delta)
            self._refresh(parameter, master)
        if fixed:
            for _group, parameter in self._matrix_parameters():
                if parameter.grad is None or getattr(parameter, "_no_gs", False):
                    continue
                state = self.state[parameter]
                master = self._master_param(parameter)
                view = _matrix_view(master)
                if view.data_ptr() != master.data_ptr():
                    raise RuntimeError("fixed-norm mode needs a contiguous parameter")
                norm = view.float().norm(dim=(-2, -1), keepdim=True)
                r0 = state["gs_r0"]
                scale = torch.where(r0 > 0, r0 / norm.clamp_min(self.gs_eps), torch.ones_like(norm))
                view.mul_(scale.to(view.dtype))
                self._refresh(parameter, master)
        return loss


__all__ = ["MuonCResearch"]
