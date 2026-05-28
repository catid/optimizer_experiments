from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optim_anchormuon import AnchorMuon, gram_newton_schulz, normuon_normalize_update, pmuon_eq_precondition
from optim_factory import _anchor_param_groups


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Parameter(torch.randn(1, 3, 8) * 0.02)
        self.linear1 = torch.nn.Linear(8, 16, bias=False)
        self.norm = torch.nn.LayerNorm(16)
        self.linear2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.embed.mean(dim=1)
        x = self.linear1(x)
        x = self.norm(x).relu()
        return self.linear2(x)


def _loss(model: TinyNet) -> torch.Tensor:
    x = torch.randn(12, 8)
    y = torch.randint(0, 4, (12,))
    return torch.nn.functional.cross_entropy(model(x), y)


def _step(mode: dict) -> tuple[TinyNet, AnchorMuon]:
    torch.manual_seed(11)
    model = TinyNet()
    opt = AnchorMuon(
        model.parameters(),
        lr=1e-3,
        warmup_steps=2,
        soda=mode.get("soda", "matrix"),
        pmuon_eq=mode.get("pmuon_eq", True),
        row_gamma=mode.get("row_gamma", 0.20),
        col_gamma=mode.get("col_gamma", 0.0),
        mimuon=mode.get("mimuon", False),
        mimuon_mix=mode.get("mimuon_mix", 0.85),
        normuon=mode.get("normuon", False),
        normuon_beta=mode.get("normuon_beta", 0.95),
        amuse=mode.get("amuse", True),
        sync_diagnostics=mode.get("sync_diagnostics", False),
    )
    loss = _loss(model)
    loss.backward()
    grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    opt.step()
    for grad_before, param in zip(grads, model.parameters()):
        if grad_before is not None:
            assert torch.equal(grad_before, param.grad)
    assert opt.last_stats["params"] > 0
    assert opt.last_stats["matrix_params"] > 0
    for param in model.parameters():
        assert torch.isfinite(param).all()
    return model, opt


def test_gram_newton_schulz_shape_and_finiteness() -> None:
    torch.manual_seed(1)
    for shape in [(7, 3), (3, 7), (2, 5, 3)]:
        update = gram_newton_schulz(torch.randn(*shape))
        assert update.shape == shape
        assert torch.isfinite(update).all()


def test_pmuon_eq_updates_state_and_preserves_shape() -> None:
    torch.manual_seed(2)
    grad = torch.randn(5, 7)
    momentum = torch.randn(5, 7)
    row = torch.ones(5)
    col = torch.ones(7)
    out = pmuon_eq_precondition(grad, momentum, row, col, row_gamma=0.2, col_gamma=0.1)
    assert out.shape == momentum.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(row, torch.ones_like(row))
    assert not torch.equal(col, torch.ones_like(col))


def test_normuon_normalizes_rows_or_columns_and_preserves_norm() -> None:
    torch.manual_seed(3)
    update = torch.randn(7, 3)
    second = torch.ones(7, 1)
    old_norm = update.norm()
    out = normuon_normalize_update(update, second, beta=0.90)
    assert out.shape == update.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(second, torch.ones_like(second))
    assert torch.allclose(out.norm(), old_norm, atol=1e-5, rtol=1e-5)

    wide = torch.randn(3, 7)
    second_wide = torch.ones(1, 7)
    wide_out = normuon_normalize_update(wide, second_wide, beta=0.90)
    assert wide_out.shape == wide.shape
    assert torch.isfinite(wide_out).all()
    assert torch.allclose(wide_out.norm(), wide.norm(), atol=1e-5, rtol=1e-5)


def test_modes_run_without_nan_and_keep_gradients_unchanged() -> None:
    modes = [
        {"soda": "matrix", "pmuon_eq": True, "mimuon": False},
        {"soda": "none", "pmuon_eq": True, "mimuon": False},
        {"soda": "all", "pmuon_eq": True, "mimuon": False},
        {"soda": "matrix", "pmuon_eq": False, "mimuon": False},
        {"soda": "matrix", "pmuon_eq": True, "mimuon": True, "mimuon_mix": 0.75},
        {"soda": "all", "pmuon_eq": True, "normuon": True, "normuon_beta": 0.90},
        {"soda": "all", "pmuon_eq": True, "mimuon": True, "mimuon_mix": 0.85, "normuon": True},
        {"soda": "all", "pmuon_eq": True, "normuon": True, "normuon_beta": 0.90, "amuse": False},
    ]
    for mode in modes:
        model, opt = _step(mode)
        if mode.get("mimuon"):
            assert opt.last_stats["mimuon_params"] > 0
        if mode.get("normuon"):
            assert opt.last_stats["normuon_params"] > 0
        assert isinstance(model, TinyNet)


def test_sync_diagnostics_are_opt_in() -> None:
    _model, opt = _step({"soda": "all", "amuse": False, "normuon": True})
    assert opt.last_stats["sync_diagnostics"] == 0.0
    assert "mean_update_rms" not in opt.last_stats
    assert "mean_precond_matrix_rms" not in opt.last_stats

    _model, sync_opt = _step({"soda": "all", "amuse": False, "normuon": True, "sync_diagnostics": True})
    assert sync_opt.last_stats["sync_diagnostics"] == 1.0
    assert sync_opt.last_stats["mean_update_rms"] > 0.0
    assert sync_opt.last_stats["mean_precond_matrix_rms"] > 0.0


def test_reported_normuon_base_recipe_flags_are_explicit() -> None:
    torch.manual_seed(13)
    model = TinyNet()
    opt = AnchorMuon(
        _anchor_param_groups(model, weight_decay=0.05),
        lr=8e-3,
        warmup_steps=80,
        soda="all",
        amuse=False,
        pmuon_eq=True,
        pmuon_beta=0.90,
        row_gamma=0.30,
        col_gamma=0.0,
        momentum=0.95,
        normuon=True,
        normuon_beta=0.95,
        mimuon=False,
    )
    assert all(group["soda"] == "all" for group in opt.param_groups)
    assert all(group["amuse"] is False for group in opt.param_groups)
    assert all(group["pmuon_eq"] is True for group in opt.param_groups)
    assert all(group["normuon"] is True for group in opt.param_groups)
    assert all(group["mimuon"] is False for group in opt.param_groups)
    assert {group["sync_diagnostics"] for group in opt.param_groups} == {False}

    loss = _loss(model)
    loss.backward()
    opt.step()
    assert opt.last_stats["matrix_params"] > 0
    assert opt.last_stats["normuon_params"] == opt.last_stats["matrix_params"]


def test_factory_currently_routes_2d_head_through_muon_path() -> None:
    torch.manual_seed(17)
    model = TinyNet()
    groups = _anchor_param_groups(model, weight_decay=0.05)
    opt = AnchorMuon(groups, lr=1e-3, warmup_steps=2, soda="all", amuse=False, normuon=True)
    loss = _loss(model)
    loss.backward()
    opt.step()
    # This documents the exact grouping used by the committed confidence runs:
    # _anchor_param_groups is name-free, so a 2D classifier/head matrix is still
    # eligible for Muon/NorMuon unless the model's no_weight_decay() excludes it.
    assert opt._use_muon_for_param(opt.param_groups[0], model.linear2.weight)


def test_reported_recipe_state_dict_resume_matches_uninterrupted_step() -> None:
    def make_opt(model: TinyNet) -> AnchorMuon:
        return AnchorMuon(
            _anchor_param_groups(model, weight_decay=0.05),
            lr=8e-3,
            warmup_steps=80,
            soda="all",
            amuse=False,
            pmuon_eq=True,
            pmuon_beta=0.90,
            row_gamma=0.30,
            col_gamma=0.0,
            momentum=0.95,
            normuon=True,
            normuon_beta=0.95,
            mimuon=False,
        )

    def seeded_step(model: TinyNet, opt: AnchorMuon, seed: int) -> None:
        torch.manual_seed(seed)
        loss = _loss(model)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.manual_seed(19)
    uninterrupted = TinyNet()
    opt_uninterrupted = make_opt(uninterrupted)
    seeded_step(uninterrupted, opt_uninterrupted, seed=20)

    resumed = copy.deepcopy(uninterrupted)
    opt_resumed = make_opt(resumed)
    opt_resumed.load_state_dict(copy.deepcopy(opt_uninterrupted.state_dict()))

    seeded_step(uninterrupted, opt_uninterrupted, seed=21)
    seeded_step(resumed, opt_resumed, seed=21)

    for p_expected, p_actual in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.allclose(p_expected, p_actual, atol=1e-6, rtol=1e-6)
    for group_expected, group_actual in zip(opt_uninterrupted.param_groups, opt_resumed.param_groups):
        assert group_expected["anchor_step"] == group_actual["anchor_step"]


def test_eval_train_swap_roundtrip() -> None:
    model, opt = _step({"soda": "matrix"})
    opt.zero_grad(set_to_none=True)
    loss = _loss(model)
    loss.backward()
    opt.step()
    train_params = [p.detach().clone() for p in model.parameters()]
    opt.eval()
    eval_params = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(train_params, eval_params))
    opt.train()
    restored = [p.detach().clone() for p in model.parameters()]
    for before, after in zip(train_params, restored):
        assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)


def test_eval_train_is_noop_when_amuse_is_disabled() -> None:
    model, opt = _step({"soda": "all", "amuse": False, "normuon": True})
    params = [p.detach().clone() for p in model.parameters()]
    opt.eval()
    after_eval = [p.detach().clone() for p in model.parameters()]
    opt.train()
    after_train = [p.detach().clone() for p in model.parameters()]
    for before, eval_param, train_param in zip(params, after_eval, after_train):
        assert torch.equal(before, eval_param)
        assert torch.equal(before, train_param)


def test_disabled_pmuoneq_changes_direction_relative_to_default() -> None:
    torch.manual_seed(9)
    model_a = TinyNet()
    model_b = copy.deepcopy(model_a)
    opt_a = AnchorMuon(model_a.parameters(), lr=1e-3, warmup_steps=2, pmuon_eq=True)
    opt_b = AnchorMuon(model_b.parameters(), lr=1e-3, warmup_steps=2, pmuon_eq=False)
    torch.manual_seed(10)
    loss_a = _loss(model_a)
    torch.manual_seed(10)
    loss_b = _loss(model_b)
    loss_a.backward()
    loss_b.backward()
    opt_a.step()
    opt_b.step()
    diffs = [(pa - pb).abs().max().item() for pa, pb in zip(model_a.parameters(), model_b.parameters())]
    assert max(diffs) > 0.0
