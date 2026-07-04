"""Learning-rate schedules used in the paper.

All schedules optionally start with a linear warmup over ``warmup_steps`` and
return a multiplicative factor in ``[0, 1]`` applied to the base learning rate
(suitable for ``torch.optim.lr_scheduler.LambdaLR``).

  - ``const``            : warmup, then a constant learning rate.
  - ``cos``              : warmup, then a cosine decay **to zero** (the "cosine"
                           schedule in the paper; all schedules end at LR 0).
  - ``wsd``              : warmup-stable-decay — constant for the first
                           ``1 - decay_frac`` of training, then a linear decay to
                           zero over the final ``decay_frac`` (default 0.2).

PACE is designed to run at a *constant* learning rate; the decay schedules are
provided for the baseline comparisons.
"""

import math
from typing import Callable

from torch.optim.lr_scheduler import LambdaLR


def _warmup_factor(step: int, warmup_steps: int) -> float | None:
    """Return the warmup multiplier if still warming up, else None."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    return None


def make_lr_lambda(name: str, total_steps: int, warmup_steps: int = 0,
                   decay_frac: float = 0.2) -> Callable[[int], float]:
    """Build the LambdaLR multiplier function for a named schedule."""
    name = (name or "const").lower()

    def const(step: int) -> float:
        w = _warmup_factor(step, warmup_steps)
        return w if w is not None else 1.0

    def cos(step: int) -> float:
        w = _warmup_factor(step, warmup_steps)
        if w is not None:
            return w
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / denom)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def wsd(step: int) -> float:
        w = _warmup_factor(step, warmup_steps)
        if w is not None:
            return w
        decay_steps = max(1, int(round(decay_frac * total_steps)))
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / decay_steps)
        return max(0.0, 1.0 - progress)

    table = {"const": const, "cos": cos, "wsd": wsd}
    if name not in table:
        raise ValueError(f"Unknown schedule '{name}'. Options: {sorted(table)}")
    return table[name]


def build_scheduler(optimizer, name: str, total_steps: int, warmup_steps: int = 0,
                    decay_frac: float = 0.2) -> LambdaLR:
    """Return a LambdaLR implementing the named schedule."""
    return LambdaLR(optimizer, make_lr_lambda(name, total_steps, warmup_steps, decay_frac))
