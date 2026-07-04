"""The PACE optimizer: AdamW plus a per-coordinate pullback toward an EMA of the
iterates. See the README for the update rule and the parameter table.
"""

import math
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim

try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class _StepStats:
    """Accumulates per-step telemetry across params; used only when ``log_stats``.

    Each ``add_*`` call does a ``.item()`` (a GPU sync), so it is gated behind the
    ``log_stats`` flag in ``step``. ``store`` writes the aggregates onto the optimizer.
    """

    def __init__(self):
        self.raw_sum = self.effective_sum = self.lam_sum = 0.0
        self.raw_max = self.effective_max = self.lam_max = -float("inf")
        self.raw_min = self.effective_min = self.lam_min = float("inf")
        self.clipped = self.count = 0
        self.adam_sq = self.pull_sq = self.upd_sq = 0.0

    def add_raw_gain(self, lam):           # raw gain, before c and lr
        self.raw_sum += lam.sum().item()
        self.raw_max = max(self.raw_max, lam.max().item())
        self.raw_min = min(self.raw_min, lam.min().item())
        self.count += lam.numel()

    def add_clipped(self, lam):            # coords hitting the <=1 clamp
        self.clipped += (lam >= 1.0).sum().item()

    def add_effective_gain(self, lam):      # gain after c and lr, before clamp
        self.effective_sum += lam.sum().item()
        self.effective_max = max(self.effective_max, lam.max().item())
        self.effective_min = min(self.effective_min, lam.min().item())

    def add_final_gain(self, lam):         # gain after c, lr and clamp
        self.lam_sum += lam.sum().item()
        self.lam_max = max(self.lam_max, lam.max().item())
        self.lam_min = min(self.lam_min, lam.min().item())

    def add_sq_norm(self, attr, delta):    # running squared norm of a displacement
        setattr(self, attr, getattr(self, attr) + delta.norm().square().item())

    def store(self, opt):                  # write aggregates onto the optimizer
        if self.count > 0:
            opt._last_raw_lambda_mean = self.raw_sum / self.count
            opt._last_raw_lambda_max = self.raw_max
            opt._last_raw_lambda_min = self.raw_min
            opt._last_lambda_mean = self.lam_sum / self.count
            opt._last_lambda_max = self.lam_max
            opt._last_lambda_min = self.lam_min
            opt._last_lambda_clipped_frac = self.clipped / self.count
            # Effective gain = lr * c * raw (before the clamp)
            opt._last_effective_lambda_mean = self.effective_sum / self.count
            opt._last_effective_lambda_max = self.effective_max
            opt._last_effective_lambda_min = self.effective_min
        opt._last_adam_step_norm = math.sqrt(self.adam_sq)
        opt._last_pullback_norm = math.sqrt(self.pull_sq)
        opt._last_update_norm = math.sqrt(self.upd_sq)


class PACE(torch.optim.Optimizer):
    """AdamW with an additive, lr-scaled, per-coordinate pullback toward an EMA.

    Args:
        params: iterable of parameters or parameter groups.
        lr: learning rate.
        betas: Adam coefficients ``(beta1, beta2)``.
        eps: term added to the denominator for numerical stability.
        weight_decay: decoupled (AdamW) weight decay.
        lambda_pullback: pullback strength ``c``. ``0`` disables the pullback
            entirely (the optimizer then reduces to AdamW, or to the EMA-eval
            baseline if ``use_ema_eval`` is set).
        clamp_pullback: clamp each per-coordinate gain to ``<= 1`` (interpolate
            toward the EMA rather than extrapolate past it).
        beta_ema: fixed EMA decay, used only when ``ema_kappa is None``.
        use_ema_eval: if True, ``eval()`` swaps the parameters to the EMA weights
            and ``train()`` swaps them back.
        ema_kappa: exponent ``kappa`` of the decaying EMA schedule. ``None`` uses
            the fixed ``beta_ema`` instead.
        ema_rho: offset ``rho`` before the schedule begins to decay. Default ``0``
            gives the paper's ``(1 + gamma*t)^(-kappa)`` schedule.
        ema_gamma: schedule rate ``gamma``.
        ema_update_freq: update the EMA every this many steps.
        log_stats: if True, compute per-step telemetry (lambda/clip-fraction and
            update norms) exposed via ``get_step_stats``. Off by default because each
            stat calls ``.item()`` (a GPU sync) every step; the optimizer math is
            identical either way. Enable it for analysis/figure reproduction.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        lambda_pullback: float = 0.0,
        clamp_pullback: bool = True,
        beta_ema: float = 0.01,
        use_ema_eval: bool = False,
        ema_kappa: Optional[float] = None,
        ema_rho: float = 0.0,
        ema_gamma: float = 1.0,
        ema_update_freq: int = 1,
        log_stats: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if lambda_pullback < 0.0:
            raise ValueError(f"Invalid lambda_pullback: {lambda_pullback}")
        if ema_kappa is None and not 0.0 < beta_ema <= 1.0:
            raise ValueError(f"Invalid beta_ema: {beta_ema}")
        if ema_update_freq < 1:
            raise ValueError(f"Invalid ema_update_freq: {ema_update_freq}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            lambda_pullback=lambda_pullback,
            clamp_pullback=clamp_pullback,
            beta_ema=beta_ema,
            use_ema_eval=use_ema_eval,
            ema_kappa=ema_kappa,
            ema_rho=ema_rho,
            ema_gamma=ema_gamma,
            ema_update_freq=ema_update_freq,
            log_stats=log_stats,
            train_mode=True,
        )
        super().__init__(params, defaults)
        # Per-step stats for logging
        self._last_raw_lambda_mean = None   # decay / (sqrt(v_hat) + eps), before c and lr
        self._last_raw_lambda_max = None
        self._last_raw_lambda_min = None
        self._last_lambda_mean = None       # lr * c * raw_lambda, after clamp
        self._last_lambda_max = None
        self._last_lambda_min = None
        self._last_lambda_clipped_frac = None
        self._last_pullback_norm = None
        self._last_adam_step_norm = None
        self._last_update_norm = None
        # Effective pullback strength = lr * c * raw_lambda_mean (lr-comparable)
        self._last_effective_lambda_mean = None
        self._last_effective_lambda_max = None
        self._last_effective_lambda_min = None

    @torch.no_grad()
    def eval(self):
        """Swap parameters to EMA weights for evaluation."""
        for group in self.param_groups:
            if not group['use_ema_eval']:
                continue
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'ema' in state:
                        state['theta'].copy_(p.data)
                        p.data.copy_(state['ema'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """Swap parameters back to training weights."""
        for group in self.param_groups:
            if not group['use_ema_eval']:
                continue
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'theta' in state:
                        p.data.copy_(state['theta'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        for group in self.param_groups:
            if group['use_ema_eval'] and not group['train_mode']:
                raise RuntimeError("PACE.step() called while parameters are swapped to EMA weights; call train() first")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stats = _StepStats() if any(group['log_stats'] for group in self.param_groups) else None

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            lambda_pullback = group['lambda_pullback']
            clamp_pullback = group['clamp_pullback']
            beta_ema = group['beta_ema']
            use_ema_eval = group['use_ema_eval']
            ema_kappa = group['ema_kappa']
            ema_rho = group['ema_rho']
            ema_gamma = group['ema_gamma']
            log_stats = group['log_stats']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Init state on first step; keep it fp32 (like torch AdamW)
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['v'] = torch.zeros_like(p.data, dtype=torch.float32)
                    if use_ema_eval or lambda_pullback > 0:
                        state['ema'] = p.data.clone().float()
                        state['theta'] = p.data.clone().float()
                        state['last_ema_step'] = 0

                state['step'] += 1
                t = state['step']

                m = state['m']
                v = state['v']

                # Phase A: standard Adam moment updates (in fp32)
                grad_fp32 = grad.float()
                m.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                m_hat = m / bias_correction1
                v_hat_sqrt = (v / bias_correction2).sqrt_().add_(eps)

                # Snapshot the pre-Adam weights for the pullback reference; clone
                # because in fp32 .float() would alias and the in-place steps below
                # would otherwise overwrite it
                p_before = p.data.clone().float()

                # Decoupled weight decay, then the Adam step
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.addcdiv_(m_hat, v_hat_sqrt, value=-lr)

                if log_stats:
                    stats.add_sq_norm("adam_sq", p.data.float() - p_before)

                # Phase B: pull toward the EMA by lam = clamp(lr*c*decay/(sqrt(v_hat)+eps), 1),
                # measured from the pre-Adam theta so the Adam step is preserved
                if lambda_pullback > 0 and 'ema' in state:
                    if ema_kappa is not None:
                        t_eff = max(0, t - ema_rho)
                        decay = max((1.0 + ema_gamma * t_eff) ** (-ema_kappa), 1e-4)
                    else:
                        decay = 1.0  # fixed beta_ema -> no decay

                    # Reuse v_hat_sqrt (done with it) to build the gain
                    lam = v_hat_sqrt.reciprocal_().mul_(decay)

                    if log_stats:
                        stats.add_raw_gain(lam)

                    lam.mul_(lr * lambda_pullback)
                    if log_stats:
                        stats.add_effective_gain(lam)
                    if clamp_pullback:
                        if log_stats:
                            stats.add_clipped(lam)
                        lam.clamp_(max=1.0)

                    if log_stats:
                        stats.add_final_gain(lam)

                    diff = state['ema'] - p_before
                    diff.mul_(lam)
                    p.data.add_(diff)
                    if log_stats:
                        stats.add_sq_norm("pull_sq", diff)

                if log_stats:
                    stats.add_sq_norm("upd_sq", p.data.float() - p_before)

                # Phase C: EMA update (gated by ema_update_freq)
                if 'ema' in state:
                    ema_update_freq = group['ema_update_freq']
                    if (t - state['last_ema_step']) % ema_update_freq == 0:
                        if ema_kappa is not None:
                            t_eff = max(0, t - ema_rho)
                            beta_ema_t = max((1.0 + ema_gamma * t_eff) ** (-ema_kappa), 1e-4)
                        else:
                            beta_ema_t = beta_ema
                        state['ema'].mul_(1.0 - beta_ema_t).add_(p.data.float(), alpha=beta_ema_t)
                        state['last_ema_step'] = t
                    # Keep theta in sync every step for train/eval swaps
                    state['theta'].copy_(p.data)

        if stats is not None:
            stats.store(self)

        return loss

    def get_step_stats(self) -> dict:
        """Return optimizer step stats for logging."""
        stats = {}
        if self._last_raw_lambda_mean is not None:
            stats['raw_lambda_mean'] = self._last_raw_lambda_mean
            stats['raw_lambda_max'] = self._last_raw_lambda_max
            stats['raw_lambda_min'] = self._last_raw_lambda_min
        if self._last_lambda_mean is not None:
            stats['lambda_mean'] = self._last_lambda_mean
            stats['lambda_max'] = self._last_lambda_max
            stats['lambda_min'] = self._last_lambda_min
            stats['lambda_clipped_frac'] = self._last_lambda_clipped_frac
        if self._last_effective_lambda_mean is not None:
            stats['effective_lambda_mean'] = self._last_effective_lambda_mean
            stats['effective_lambda_max'] = self._last_effective_lambda_max
            stats['effective_lambda_min'] = self._last_effective_lambda_min
        if self._last_adam_step_norm is not None:
            stats['adam_step_norm'] = self._last_adam_step_norm
        if self._last_pullback_norm is not None:
            stats['pullback_norm'] = self._last_pullback_norm
        if self._last_update_norm is not None:
            stats['update_norm'] = self._last_update_norm
        return stats

    def get_ema_beta_t(self) -> Optional[float]:
        """Return the current EMA interpolation rate for logging."""
        for group in self.param_groups:
            if not group['use_ema_eval'] and group['lambda_pullback'] == 0:
                continue
            ema_kappa = group['ema_kappa']
            if ema_kappa is not None:
                for p in group['params']:
                    state = self.state[p]
                    if 'step' in state:
                        t = state['step']
                        t_eff = max(0, t - group['ema_rho'])
                        return max((1.0 + group['ema_gamma'] * t_eff) ** (-ema_kappa), 1e-4)
            else:
                return group['beta_ema']
        return None


__all__ = ["PACE"]
