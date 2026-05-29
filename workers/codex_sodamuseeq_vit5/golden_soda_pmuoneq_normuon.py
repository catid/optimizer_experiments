"""Golden fixed optimizer for the best Codex SodaMuseEq result.

This module exposes the exact optimizer recipe that produced the best result in
``workers/codex_sodamuseeq_vit5/results/focused_peer_aspect``:

    SODA + PMuonEq + Gram/Newton-Schulz + NorMuon row+aspect
    AMUSE off
    MiMuon off

It intentionally does not expose the broad ablation switches from
``vit5/optim_sodamuseeq.py``. Internally it delegates to the same implementation
used for the reported run so deterministic update parity is possible.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vit5.optim_sodamuseeq import (
    BEST_KNOWN_CONFIG as _VIT5_BEST_KNOWN_CONFIG,
    GramNewtonSchulzProjector,
    SodaMuseEq,
    make_sodamuseeq_param_groups,
)


GOLDEN_CONFIG: dict[str, Any] = {
    **_VIT5_BEST_KNOWN_CONFIG,
    "algorithm": "SODA+PMuonEq+Gram+NorMuon row+aspect",
    "source": "workers/codex_sodamuseeq_vit5/results/focused_peer_aspect",
    "use_mimuon": False,
}


class GoldenSodaPmuonEqNorMuon(SodaMuseEq):
    """Fixed wrapper around the best row+aspect SodaMuseEq recipe.

    Public knobs are limited to LR, warmup, precision/batching, and a few state
    decay constants needed for reproducing or retuning the winning recipe.
    Algorithm switches such as AMUSE, MiMuon, PMuonEq off, Gram off, NorMuon
    off, orientation mode, and aspect off are deliberately not exposed.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.012,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        beta1: float = 0.6,
        beta2: float = 0.999,
        rho: float = 0.8,
        warmup_steps: int = 500,
        soda_warmup_steps: int = 500,
        pmuon_beta: float = 0.90,
        pmuon_row_gamma: float = 0.15,
        normuon_beta2: float = 0.90,
        batch_project: bool = True,
        stats_interval: int = 100,
        projection_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            rho=rho,
            warmup_steps=warmup_steps,
            eps=1e-10,
            use_soda=True,
            soda_warmup_steps=soda_warmup_steps,
            soda_anchor="warmup",
            soda_replaces_weight_decay=True,
            use_amuse=False,
            use_pmuoneq=True,
            pmuon_beta=pmuon_beta,
            pmuon_gamma=0.0,
            pmuon_row_gamma=pmuon_row_gamma,
            pmuon_col_gamma=0.0,
            pmuon_sides="both",
            pmuon_eps=1e-6,
            use_gram=True,
            use_mimuon=False,
            mimuon_tau=0.005,
            use_normuon=True,
            normuon_beta2=normuon_beta2,
            normuon_eps=1e-10,
            normuon_mode="row",
            normuon_aspect_scale=True,
            batch_project=batch_project,
            stats_interval=stats_interval,
            projection_dtype=projection_dtype,
        )


def build_golden_param_groups(
    model_or_named_parameters: nn.Module | Iterable[tuple[str, nn.Parameter]],
    *,
    lr: float = 0.012,
    weight_decay: float = 0.0,
    aux_lr: float | None = None,
    aux_weight_decay: float | None = None,
) -> list[dict[str, Any]]:
    """Build the same groups used by the best focused ViT-5 run.

    Hidden 2D/4D matrices go through the golden spectral path. Biases, norm
    scales, positional/register/class tokens, token embeddings, and heads stay
    in the fallback path by default.
    """

    if isinstance(model_or_named_parameters, nn.Module):
        model = model_or_named_parameters
    else:
        model = _NamedParameterModule(model_or_named_parameters)
    return make_sodamuseeq_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        aux_lr=aux_lr,
        aux_weight_decay=aux_weight_decay,
    )


class _NamedParameterModule(nn.Module):
    """Adapter so the proven grouping helper can consume named parameters."""

    def __init__(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> None:
        super().__init__()
        self._named_params = list(named_parameters)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):  # type: ignore[override]
        seen: set[int] = set()
        for name, param in self._named_params:
            if remove_duplicate and id(param) in seen:
                continue
            seen.add(id(param))
            yield name, param


__all__ = [
    "GOLDEN_CONFIG",
    "GoldenSodaPmuonEqNorMuon",
    "GramNewtonSchulzProjector",
    "build_golden_param_groups",
]
