from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
sys.modules.pop("optim_sfplus", None)

from optim_sfplus import SFPlusAnchorMuon


class TinySFNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(6, 10, bias=False)
        self.norm = torch.nn.LayerNorm(10)
        self.head = torch.nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.proj(x)).relu())


def _loss(model: TinySFNet) -> torch.Tensor:
    x = torch.randn(8, 6)
    y = torch.randint(0, 3, (8,))
    return torch.nn.functional.cross_entropy(model(x), y)


def _flags(mask: int) -> dict[str, bool]:
    return {
        "sfplus_polyak": bool(mask & 1),
        "sfplus_c_warmup_enabled": bool(mask & 2),
        "sfplus_beta_anneal": bool(mask & 4),
        "sfplus_adamc_decay": bool(mask & 8),
        "sfplus_inner_momentum": bool(mask & 16),
    }


def test_all_sfplus_toggle_combinations_run_one_step_without_mutating_gradients() -> None:
    for mask in range(32):
        torch.manual_seed(100 + mask)
        flags = _flags(mask)
        model = TinySFNet()
        opt = SFPlusAnchorMuon(
            model.parameters(),
            lr=1.0 if flags["sfplus_polyak"] else 8e-3,
            weight_decay=0.2,
            row_gamma=0.2,
            col_gamma=0.0,
            pmuon_beta=0.90,
            momentum=0.95,
            normuon_beta=0.93,
            sfplus_c_warmup=2,
            sfplus_beta1_anneal_steps=8,
            **flags,
        )

        loss = _loss(model)
        loss.backward()
        grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
        kwargs = {"function_value": float(loss.detach())} if flags["sfplus_polyak"] else {}
        opt.step(**kwargs)

        for grad_before, param in zip(grads, model.parameters()):
            if grad_before is not None:
                assert torch.equal(grad_before, param.grad)
            assert torch.isfinite(param).all()

        assert opt.last_stats["sfplus"] == 1.0
        assert opt.last_stats["matrix_params"] > 0
        assert opt.last_stats["fallback_params"] > 0
        assert opt.last_stats["sfplus_polyak"] == float(flags["sfplus_polyak"])
        assert opt.last_stats["sfplus_c_warmup_enabled"] == float(flags["sfplus_c_warmup_enabled"])
        assert opt.last_stats["sfplus_beta_anneal"] == float(flags["sfplus_beta_anneal"])
        assert opt.last_stats["sfplus_adamc_decay"] == float(flags["sfplus_adamc_decay"])
        assert opt.last_stats["sfplus_inner_momentum"] == float(flags["sfplus_inner_momentum"])

        # Schedule-free train/eval sequence swapping should be available for
        # checkpoint/evaluation code, even though this experiment did not win.
        opt.eval()
        assert opt._train_mode is False
        opt.train()
        assert opt._train_mode is True


def test_polyak_mode_requires_current_function_value() -> None:
    torch.manual_seed(200)
    model = TinySFNet()
    opt = SFPlusAnchorMuon(model.parameters(), lr=1.0, sfplus_polyak=True)
    loss = _loss(model)
    loss.backward()
    with pytest.raises(RuntimeError, match="function_value"):
        opt.step()
