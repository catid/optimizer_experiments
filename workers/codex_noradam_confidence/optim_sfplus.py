"""Experimental ScheduleFree+ wrapper for the AnchorMuon matrix direction.

This module is intentionally kept out of the root optimizer.  It exists to test
whether ScheduleFree+ mechanisms can improve the current direct AnchorMuon WSD
recipe.  The SF+ outer loop owns x/y/z iterate bookkeeping, Polyak LR, c_t
averaging, beta annealing, and AdamC-style decay.  The matrix path only supplies
the same PMuonEq + GramNS + NorMuon direction used by the current winner.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor

from optim_anchormuon import (
    _matrix_view,
    _restore_matrix_view,
    _state_tensor,
    gram_newton_schulz,
    normuon_normalize_update,
    pmuon_eq_precondition,
)


def _sfplus_beta(beta1: float, beta1_max: float, anneal_steps: int, step: int, enabled: bool) -> float:
    if not enabled or anneal_steps <= 0:
        return float(beta1)
    progress = min(float(step) / float(max(1, anneal_steps)), 1.0)
    return 1.0 - math.exp(math.log(1.0 - beta1) * (1.0 - progress) + math.log(1.0 - beta1_max) * progress)


class SFPlusAnchorMuon(torch.optim.Optimizer):
    """ScheduleFree+ outer loop with AnchorMuon's matrix direction.

    Toggle flags correspond to the five SF+ ideas under test:

    * ``sfplus_polyak``: use Polyak-style online LR multiplier.
    * ``sfplus_c_warmup_enabled``: force c_t=1 for ``sfplus_c_warmup`` steps.
    * ``sfplus_beta_anneal``: anneal beta_sf toward ``sfplus_beta1_max``.
    * ``sfplus_adamc_decay``: apply AdamC-style ``lr^2 * weight_decay`` decay.
    * ``sfplus_inner_momentum``: use inner momentum for matrix and fallback paths.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.95),
        sfplus_beta1: float = 0.9,
        sfplus_beta1_max: float = 0.965,
        sfplus_beta1_anneal_steps: int = 0,
        sfplus_polyak_beta: float = 0.0,
        sfplus_c_warmup: int = 0,
        sfplus_r: float = 0.0,
        sfplus_weight_lr_power: float = 2.0,
        sfplus_polyak: bool = True,
        sfplus_c_warmup_enabled: bool = True,
        sfplus_beta_anneal: bool = True,
        sfplus_adamc_decay: bool = True,
        sfplus_inner_momentum: bool = True,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        pmuon_beta: float = 0.90,
        row_gamma: float = 0.35,
        col_gamma: float = 0.0,
        normuon_beta: float = 0.93,
        eps: float = 1e-8,
        pmuon_eps: float = 1e-6,
        normuon_eps: float = 1e-10,
        ns_steps: int = 5,
        min_matrix_dim: int = 2,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            sfplus_beta1=float(sfplus_beta1),
            sfplus_beta1_max=float(sfplus_beta1_max),
            sfplus_beta1_anneal_steps=int(sfplus_beta1_anneal_steps),
            sfplus_polyak_beta=float(sfplus_polyak_beta),
            sfplus_c_warmup=int(sfplus_c_warmup),
            sfplus_r=float(sfplus_r),
            sfplus_weight_lr_power=float(sfplus_weight_lr_power),
            sfplus_polyak=bool(sfplus_polyak),
            sfplus_c_warmup_enabled=bool(sfplus_c_warmup_enabled),
            sfplus_beta_anneal=bool(sfplus_beta_anneal),
            sfplus_adamc_decay=bool(sfplus_adamc_decay),
            sfplus_inner_momentum=bool(sfplus_inner_momentum),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            pmuon_beta=float(pmuon_beta),
            row_gamma=float(row_gamma),
            col_gamma=float(col_gamma),
            normuon_beta=float(normuon_beta),
            eps=float(eps),
            pmuon_eps=float(pmuon_eps),
            normuon_eps=float(normuon_eps),
            ns_steps=int(ns_steps),
            min_matrix_dim=int(min_matrix_dim),
            sfplus_k=0,
            sfplus_weight_sum=0.0,
            sfplus_lr_max=float(eps),
            sfplus_grad_l1_ema=0.0,
            sfplus_scheduled_lr=0.0,
            sfplus_ckp1=1.0,
            sfplus_beta_current=float(sfplus_beta1),
            sfplus_polyak_lr=1.0,
        )
        super().__init__(params, defaults)
        self._train_mode = True
        self.last_stats: dict[str, float] = {}

    def _use_matrix_for_param(self, group: dict[str, Any], param: Tensor) -> bool:
        if "use_matrix_update" in group:
            return bool(group["use_matrix_update"])
        if "use_muon" in group:
            return bool(group["use_muon"])
        if param.ndim < 2:
            return False
        rows = int(param.shape[0])
        cols = int(param.numel() // max(rows, 1))
        return min(rows, cols) >= int(group["min_matrix_dim"])

    def _ensure_sf_state(self, param: Tensor, state: dict[str, Any]) -> None:
        if "sfplus_z" not in state or tuple(state["sfplus_z"].shape) != tuple(param.shape) or state["sfplus_z"].device != param.device:
            base = param.detach().float()
            state["sfplus_z"] = base.clone(memory_format=torch.preserve_format)
            state["sfplus_x"] = base.clone(memory_format=torch.preserve_format)
            state["sfplus_y"] = base.clone(memory_format=torch.preserve_format)

    def _ensure_matrix_state(self, param: Tensor, state: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        matrix, _ = _matrix_view(param)
        rows, cols = matrix.shape
        momentum = _state_tensor(state, "sfplus_matrix_momentum", shape=(rows, cols), device=param.device, fill=0.0)
        row_ema = _state_tensor(state, "sfplus_row_ema", shape=(rows,), device=param.device, fill=1.0)
        col_ema = _state_tensor(state, "sfplus_col_ema", shape=(cols,), device=param.device, fill=1.0)
        norm_shape = (rows, 1) if rows >= cols else (1, cols)
        norm = _state_tensor(state, "sfplus_normuon_second", shape=norm_shape, device=param.device, fill=1.0)
        return momentum, row_ema, col_ema, norm

    def _ensure_adam_state(self, param: Tensor, state: dict[str, Any]) -> tuple[Tensor, Tensor]:
        exp_avg = _state_tensor(state, "sfplus_exp_avg", shape=tuple(param.shape), device=param.device, fill=0.0)
        exp_avg_sq = _state_tensor(state, "sfplus_exp_avg_sq", shape=tuple(param.shape), device=param.device, fill=0.0)
        return exp_avg, exp_avg_sq

    @torch.no_grad()
    def eval(self) -> "SFPlusAnchorMuon":
        if not self._train_mode:
            return self
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                x = state.get("sfplus_x")
                if x is not None:
                    param.copy_(x.to(device=param.device, dtype=param.dtype))
        self._train_mode = False
        return self

    @torch.no_grad()
    def train(self) -> "SFPlusAnchorMuon":
        if self._train_mode:
            return self
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                y = state.get("sfplus_y")
                if y is not None:
                    param.copy_(y.to(device=param.device, dtype=param.dtype))
        self._train_mode = True
        return self

    @torch.no_grad()
    def _prepare_context(self, function_value: float | None) -> dict[str, float]:
        first = self.param_groups[0]
        k = int(first.get("sfplus_k", 0))
        sf_beta = _sfplus_beta(
            float(first["sfplus_beta1"]),
            float(first["sfplus_beta1_max"]),
            int(first["sfplus_beta1_anneal_steps"]),
            k,
            bool(first["sfplus_beta_anneal"]),
        )
        grad_l1 = 0.0
        ip_term = 0.0
        param_count = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("SFPlusAnchorMuon does not support sparse gradients")
                state = self.state[param]
                self._ensure_sf_state(param, state)
                grad = param.grad.detach().float()
                grad_l1 += float(grad.abs().sum().detach().cpu())
                z = state["sfplus_z"]
                x = state["sfplus_x"]
                ip_term += sf_beta * float((grad * (z - x)).sum().detach().cpu())
                param_count += 1

        polyak_enabled = bool(first["sfplus_polyak"])
        if polyak_enabled:
            if function_value is None:
                raise RuntimeError("SFPlusAnchorMuon requires function_value when Polyak LR is enabled")
            polyak_beta = float(first["sfplus_polyak_beta"])
            ema = polyak_beta * float(first.get("sfplus_grad_l1_ema", 0.0)) + (1.0 - polyak_beta) * grad_l1 * math.sqrt(math.pi / 2.0)
            denom = ema / max(1.0 - polyak_beta ** (k + 1), float(first["eps"]))
            polyak_lr = max(0.0, float(function_value) + ip_term) / max(denom, float(first["eps"]))
        else:
            ema = grad_l1
            denom = max(grad_l1, float(first["eps"]))
            polyak_lr = 1.0

        for group in self.param_groups:
            group["sfplus_grad_l1_ema"] = float(ema)
            group["sfplus_grad_l1_ema_corr"] = float(denom)
            group["sfplus_ip_term"] = float(ip_term)
            group["sfplus_function_value"] = 0.0 if function_value is None else float(function_value)
            group["sfplus_polyak_lr"] = float(polyak_lr)
            group["sfplus_beta_current"] = float(sf_beta)

        return {
            "sf_beta": float(sf_beta),
            "polyak_lr": float(polyak_lr),
            "grad_l1": float(grad_l1),
            "ip_term": float(ip_term),
            "param_count": float(param_count),
        }

    def _group_schedule(self, group: dict[str, Any], polyak_lr: float) -> tuple[int, float, float]:
        k = int(group.get("sfplus_k", 0))
        t = k + 1
        group_lr = max(float(group["lr"]), float(group["eps"])) * float(polyak_lr)
        group["sfplus_scheduled_lr"] = group_lr
        lr_max = max(group_lr, float(group.get("sfplus_lr_max", group["eps"])))
        group["sfplus_lr_max"] = lr_max

        c_warmup = int(group["sfplus_c_warmup"]) if bool(group["sfplus_c_warmup_enabled"]) else 0
        if k < c_warmup:
            ckp1 = 1.0
        else:
            weight = ((k + 1) ** float(group["sfplus_r"])) * (lr_max ** float(group["sfplus_weight_lr_power"]))
            weight_sum = float(group.get("sfplus_weight_sum", 0.0)) + weight
            group["sfplus_weight_sum"] = weight_sum
            ckp1 = weight / weight_sum if weight_sum > 0 else 1.0
        group["sfplus_ckp1"] = float(ckp1)
        group["sfplus_k"] = t
        return t, group_lr, float(ckp1)

    def _matrix_direction(self, param: Tensor, grad: Tensor, group: dict[str, Any], state: dict[str, Any]) -> Tensor:
        grad_matrix, original_shape = _matrix_view(grad)
        momentum, row_ema, col_ema, normuon_second = self._ensure_matrix_state(param, state)
        if bool(group["sfplus_inner_momentum"]):
            beta_m = float(group["momentum"])
            momentum.lerp_(grad_matrix, 1.0 - beta_m)
            source = torch.lerp(grad_matrix, momentum, beta_m)
        else:
            source = grad_matrix.float()
        precond = pmuon_eq_precondition(
            grad_matrix,
            source,
            row_ema,
            col_ema,
            beta=float(group["pmuon_beta"]),
            row_gamma=float(group["row_gamma"]),
            col_gamma=float(group["col_gamma"]),
            eps=float(group["pmuon_eps"]),
        )
        update = gram_newton_schulz(precond, steps=int(group["ns_steps"]), eps=float(group["pmuon_eps"]))
        update = normuon_normalize_update(
            update,
            normuon_second,
            beta=float(group["normuon_beta"]),
            eps=float(group["normuon_eps"]),
            aspect_scale=False,
        )
        update = update * (0.2 * math.sqrt(max(update.shape[-2], update.shape[-1])))
        return _restore_matrix_view(update, original_shape)

    def _adamc_direction(self, param: Tensor, grad: Tensor, group: dict[str, Any], state: dict[str, Any], step: int) -> Tensor:
        beta1, beta2 = group["betas"]
        exp_avg, exp_avg_sq = self._ensure_adam_state(param, state)
        if bool(group["sfplus_inner_momentum"]):
            exp_avg.mul_(float(beta1)).add_(grad, alpha=1.0 - float(beta1))
            bias1 = max(1.0 - float(beta1) ** step, float(group["eps"]))
            numer = exp_avg / bias1
        else:
            numer = grad
        exp_avg_sq.mul_(float(beta2)).addcmul_(grad, grad, value=1.0 - float(beta2))
        bias2 = max(1.0 - float(beta2) ** step, float(group["eps"]))
        denom = (exp_avg_sq / bias2).sqrt().add_(float(group["eps"]))
        return numer / denom

    @torch.no_grad()
    def step(self, closure=None, *, function_value: float | None = None):
        if not self._train_mode:
            raise RuntimeError("SFPlusAnchorMuon.step() called in eval/x mode; call optimizer.train() first")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            if function_value is None and loss is not None:
                function_value = float(loss.detach().cpu())

        context = self._prepare_context(function_value)
        matrix_params = 0
        fallback_params = 0
        lr_last = 0.0
        ckp1_last = 0.0

        for group in self.param_groups:
            step, group_lr, ckp1 = self._group_schedule(group, context["polyak_lr"])
            lr_last = group_lr
            ckp1_last = ckp1
            sf_beta = context["sf_beta"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                if grad.is_sparse:
                    raise RuntimeError("SFPlusAnchorMuon does not support sparse gradients")
                state = self.state[param]
                self._ensure_sf_state(param, state)
                z = state["sfplus_z"]
                x = state["sfplus_x"]
                y = state["sfplus_y"]

                if bool(group["sfplus_adamc_decay"]) and float(group["weight_decay"]) != 0.0:
                    z.sub_(y, alpha=group_lr * group_lr * float(group["weight_decay"]))

                if self._use_matrix_for_param(group, param):
                    matrix_params += 1
                    direction = self._matrix_direction(param, grad, group, state)
                else:
                    fallback_params += 1
                    direction = self._adamc_direction(param, grad, group, state, step)

                z.add_(direction.to(device=z.device, dtype=torch.float32), alpha=-group_lr)
                x.mul_(1.0 - ckp1).add_(z, alpha=ckp1)
                y.copy_(x.mul(sf_beta).add_(z, alpha=1.0 - sf_beta))
                param.copy_(y.to(device=param.device, dtype=param.dtype))

        self.last_stats = {
            "sfplus": 1.0,
            "sfplus_polyak": float(bool(self.param_groups[0]["sfplus_polyak"])),
            "sfplus_c_warmup_enabled": float(bool(self.param_groups[0]["sfplus_c_warmup_enabled"])),
            "sfplus_beta_anneal": float(bool(self.param_groups[0]["sfplus_beta_anneal"])),
            "sfplus_adamc_decay": float(bool(self.param_groups[0]["sfplus_adamc_decay"])),
            "sfplus_inner_momentum": float(bool(self.param_groups[0]["sfplus_inner_momentum"])),
            "sfplus_lr": float(lr_last),
            "sfplus_polyak_lr": float(context["polyak_lr"]),
            "sfplus_beta": float(context["sf_beta"]),
            "sfplus_ckp1": float(ckp1_last),
            "sfplus_grad_l1": float(context["grad_l1"]),
            "sfplus_ip_term": float(context["ip_term"]),
            "matrix_params": float(matrix_params),
            "fallback_params": float(fallback_params),
        }
        return loss

