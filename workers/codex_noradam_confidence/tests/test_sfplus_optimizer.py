from __future__ import annotations

import copy
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


def test_sfplus_constructor_deduplicates_shared_params_in_raw_inputs() -> None:
    for params in (
        lambda p: [p, p],
        lambda p: [{"params": [p, p], "use_muon": False}],
        lambda p: [{"params": p, "use_muon": False}],
    ):
        param = torch.nn.Parameter(torch.ones(1))
        opt = SFPlusAnchorMuon(params(param), lr=0.1, sfplus_polyak=False)

        assert sum(candidate is param for group in opt.param_groups for candidate in group["params"]) == 1

        param.grad = torch.ones_like(param)
        opt.step()

        assert opt.last_stats["fallback_params"] == 1.0


def test_sfplus_uses_effective_matrix_shape_for_leading_singleton_params() -> None:
    torch.manual_seed(300)
    param = torch.nn.Parameter(torch.randn(1, 6, 8) * 0.02)
    opt = SFPlusAnchorMuon([param], lr=1e-3, sfplus_polyak=False, min_matrix_dim=2)

    param.grad = torch.randn_like(param)
    opt.step()

    assert opt.last_stats["matrix_params"] == 1.0
    assert opt.last_stats["fallback_params"] == 0.0
    state = opt.state[param]
    assert state["sfplus_row_ema"].shape == (6,)
    assert state["sfplus_col_ema"].shape == (8,)
    assert "sfplus_exp_avg" not in state


def test_sfplus_fallback_bias_correction_uses_per_parameter_step() -> None:
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    opt = SFPlusAnchorMuon(
        [{"params": [p1, p2], "use_muon": False}],
        lr=0.1,
        betas=(0.0, 0.9),
        sfplus_polyak=False,
        sfplus_c_warmup_enabled=True,
        sfplus_c_warmup=10,
        sfplus_beta_anneal=False,
        sfplus_adamc_decay=False,
        sfplus_inner_momentum=False,
        weight_decay=0.0,
    )

    p1.grad = torch.tensor([1.0])
    p2.grad = None
    opt.step()
    opt.zero_grad(set_to_none=True)
    p1.grad = None
    p2.grad = torch.tensor([1.0])
    opt.step()

    assert torch.allclose(p2.detach(), torch.tensor([0.9]), atol=1e-6, rtol=1e-6)
    assert opt.state[p2]["sfplus_fallback_step"] == 1


def test_sfplus_eval_mode_survives_state_dict_round_trip() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    opt = SFPlusAnchorMuon([p], lr=0.1, sfplus_polyak=False)
    for grad in (
        torch.tensor([0.3, -0.2, 0.1]),
        torch.tensor([0.1, 0.2, -0.3]),
    ):
        p.grad = grad
        opt.step()
        opt.zero_grad(set_to_none=True)
    opt.eval()
    eval_value = p.detach().clone()

    restored_param = torch.nn.Parameter(eval_value.clone())
    restored = SFPlusAnchorMuon([restored_param], lr=0.1, sfplus_polyak=False)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))

    assert restored._train_mode is False
    restored.train()
    assert not torch.allclose(restored_param.detach(), eval_value)


def test_sfplus_bfloat16_state_dict_load_restores_fp32_state() -> None:
    param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    opt = SFPlusAnchorMuon(
        [{"params": [param], "use_matrix_update": True}],
        lr=1e-3,
        sfplus_polyak=False,
    )
    param.grad = torch.randn_like(param)
    opt.step()

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = SFPlusAnchorMuon(
        [{"params": [restored_param], "use_matrix_update": True}],
        lr=1e-3,
        sfplus_polyak=False,
    )
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))

    restored_param.grad = torch.randn_like(restored_param)
    restored.step()

    state = restored.state[restored_param]
    assert state["sfplus_z"].dtype == torch.float32
    assert state["sfplus_x"].dtype == torch.float32
    assert state["sfplus_y"].dtype == torch.float32
    assert state["sfplus_matrix_momentum"].dtype == torch.float32
    assert state["sfplus_row_ema"].dtype == torch.float32
    assert state["sfplus_col_ema"].dtype == torch.float32
    assert state["sfplus_normuon_second"].dtype == torch.float32
