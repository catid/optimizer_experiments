"""Optimizer factory extensions for ViT-5 experiments."""

from __future__ import annotations

from typing import Any

from timm.optim import create_optimizer as create_timm_optimizer

from optim_anchormuon import AnchorMuon


def _no_weight_decay_names(model) -> set[str]:
    if hasattr(model, "no_weight_decay"):
        return set(model.no_weight_decay())
    return set()


def _anchor_param_groups(model, weight_decay: float) -> list[dict[str, Any]]:
    no_weight_decay = _no_weight_decay_names(model)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def create_optimizer(args, model):
    """Create a timm optimizer or AnchorMuon depending on ``args.opt``."""

    opt_name = args.opt.lower()
    if opt_name not in {"anchormuon", "anchor_muon", "anchor-muon"}:
        return create_timm_optimizer(args, model)

    betas = tuple(args.opt_betas) if args.opt_betas is not None else (0.9, 0.95)
    return AnchorMuon(
        _anchor_param_groups(model, float(args.weight_decay)),
        lr=float(args.lr),
        betas=betas,  # type: ignore[arg-type]
        beta1=float(args.anchor_beta1),
        beta2=float(args.anchor_beta2),
        rho=float(args.anchor_rho),
        warmup_steps=int(args.anchor_warmup_steps),
        use_external_lr=bool(args.anchor_use_external_lr),
        amuse=bool(args.anchor_amuse),
        momentum=float(args.anchor_momentum),
        pmuon_beta=float(args.anchor_pmuon_beta),
        pmuon_eq=bool(args.anchor_pmuon_eq),
        row_gamma=float(args.anchor_row_gamma),
        col_gamma=float(args.anchor_col_gamma),
        mimuon=bool(args.anchor_mimuon),
        mimuon_mix=float(args.anchor_mimuon_mix),
        normuon=bool(args.anchor_normuon),
        normuon_beta=float(args.anchor_normuon_beta),
        normuon_aspect_scale=bool(getattr(args, "anchor_normuon_aspect_scale", False)),
        normuon_eps=float(args.anchor_normuon_eps),
        eps=float(args.opt_eps),
        pmuon_eps=float(args.anchor_pmuon_eps),
        ns_steps=int(args.anchor_ns_steps),
        soda=args.anchor_soda,
        min_matrix_dim=int(args.anchor_min_matrix_dim),
    )
