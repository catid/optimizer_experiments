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
for module_name in ("golden_soda_pmuoneq_normuon", "optim_anchormuon"):
    sys.modules.pop(module_name, None)

from golden_soda_pmuoneq_normuon import (  # noqa: E402
    GoldenSodaPmuonEqNorMuon,
    build_golden_param_groups,
    golden_gram_newton_schulz,
)
from optim_anchormuon import AnchorMuon  # noqa: E402


class TinyGoldenNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(17, 8)
        self.linear1 = torch.nn.Linear(8, 16, bias=False)
        self.norm = torch.nn.LayerNorm(16)
        self.linear2 = torch.nn.Linear(16, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens).mean(dim=1)
        x = self.linear1(x)
        x = self.norm(x).relu()
        return self.linear2(x)


class TinyTiedLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(11, 8)
        self.proj = torch.nn.Linear(8, 8, bias=False)
        self.lm_head = torch.nn.Linear(8, 11, bias=False)
        self.lm_head.weight = self.wte.weight


def test_sparse_gradients_fail_with_clear_error() -> None:
    embedding = torch.nn.Embedding(16, 8, sparse=True)
    opt = GoldenSodaPmuonEqNorMuon(
        [{"params": [embedding.weight], "use_matrix": False}],
        lr=0.1,
        warmup_steps=1,
    )

    loss = embedding(torch.tensor([1, 2, 3])).sum()
    loss.backward()

    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        opt.step()


def test_golden_constructor_deduplicates_shared_params_in_raw_inputs() -> None:
    for params in (
        lambda p: [p, p],
        lambda p: [{"params": [p, p], "use_matrix": False}],
        lambda p: [{"params": p, "use_matrix": False}],
    ):
        param = torch.nn.Parameter(torch.ones(1))
        opt = GoldenSodaPmuonEqNorMuon(params(param), lr=0.1, warmup_steps=1)

        assert sum(candidate is param for group in opt.param_groups for candidate in group["params"]) == 1

        param.grad = torch.ones_like(param)
        opt.step()

        assert opt.last_stats["fallback_params"] == 1.0


def _loss(model: TinyGoldenNet, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    tokens = torch.randint(0, 17, (10, 5))
    y = torch.randint(0, 4, (10,))
    return torch.nn.functional.cross_entropy(model(tokens), y)


def _raw_groups(model: torch.nn.Module) -> list[dict]:
    return [{"params": list(model.parameters())}]


def _make_anchor(model: torch.nn.Module) -> AnchorMuon:
    return AnchorMuon(
        _raw_groups(model),
        lr=8e-3,
        warmup_steps=7,
        soda="all",
        amuse=False,
        pmuon_eq=True,
        pmuon_beta=0.90,
        row_gamma=0.35,
        col_gamma=0.0,
        momentum=0.95,
        normuon=True,
        normuon_beta=0.93,
        normuon_aspect_scale=False,
        mimuon=False,
    )


def _make_golden(model: torch.nn.Module) -> GoldenSodaPmuonEqNorMuon:
    return GoldenSodaPmuonEqNorMuon(_raw_groups(model), lr=8e-3, warmup_steps=7)


def test_golden_gram_newton_schulz_shape_and_finiteness() -> None:
    torch.manual_seed(1)
    for shape in [(7, 3), (3, 7), (2, 5, 3)]:
        update = golden_gram_newton_schulz(torch.randn(*shape))
        assert update.shape == shape
        assert torch.isfinite(update).all()


def test_golden_matches_anchormuon_winning_path_for_raw_groups() -> None:
    torch.manual_seed(2)
    anchor_model = TinyGoldenNet()
    golden_model = copy.deepcopy(anchor_model)
    anchor = _make_anchor(anchor_model)
    golden = _make_golden(golden_model)

    for seed in [3, 4, 5]:
        anchor_loss = _loss(anchor_model, seed)
        golden_loss = _loss(golden_model, seed)
        assert torch.allclose(anchor_loss, golden_loss, atol=1e-7, rtol=1e-7)
        anchor_loss.backward()
        golden_loss.backward()
        anchor.step()
        golden.step()
        anchor.zero_grad(set_to_none=True)
        golden.zero_grad(set_to_none=True)

    for expected, actual in zip(anchor_model.parameters(), golden_model.parameters()):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    assert golden.last_stats["matrix_params"] == anchor.last_stats["matrix_params"]


def test_golden_param_groups_are_lm_safe_and_deduplicate_tied_weights() -> None:
    model = TinyTiedLM()
    duplicate_named = [
        ("wte.weight", model.wte.weight),
        ("proj.weight", model.proj.weight),
        ("lm_head.weight", model.lm_head.weight),
    ]
    groups = build_golden_param_groups(duplicate_named)
    matrix_group = next(group for group in groups if group["use_matrix"] is True)
    fallback_group = next(group for group in groups if group["use_matrix"] is False)

    assert model.proj.weight in matrix_group["params"]
    assert model.wte.weight in fallback_group["params"]
    assert sum(id(param) == id(model.wte.weight) for group in groups for param in group["params"]) == 1


def test_golden_uses_effective_matrix_shape_for_leading_singleton_params() -> None:
    torch.manual_seed(15)
    param = torch.nn.Parameter(torch.randn(1, 6, 8) * 0.02)
    opt = GoldenSodaPmuonEqNorMuon([param], lr=1e-3, warmup_steps=1)

    param.grad = torch.randn_like(param)
    opt.step()

    assert opt.last_stats["matrix_params"] == 1.0
    state = opt.state[param]
    assert state["row_ema"].shape == (6,)
    assert state["normuon_second_moment"].shape == (1, 8)


def test_golden_fallback_bias_correction_uses_per_parameter_step() -> None:
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    opt = GoldenSodaPmuonEqNorMuon(
        [{"params": [p1, p2], "use_matrix": False}],
        lr=0.1,
        warmup_steps=1,
        beta2=0.9,
        eps=1e-10,
    )

    p1.grad = torch.tensor([1.0])
    p2.grad = None
    opt.step()
    opt.zero_grad(set_to_none=True)
    p1.grad = None
    p2.grad = torch.tensor([1.0])
    opt.step()

    assert torch.allclose(p2.detach(), torch.tensor([0.9]), atol=1e-6, rtol=1e-6)
    assert opt.state[p2]["fallback_step"] == 1


def test_golden_step_keeps_gradients_unchanged_and_state_fp32() -> None:
    torch.manual_seed(6)
    model = TinyGoldenNet()
    opt = GoldenSodaPmuonEqNorMuon(build_golden_param_groups(model.named_parameters()), lr=8e-3, warmup_steps=4)
    loss = _loss(model, seed=7)
    loss.backward()
    grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    opt.step()

    for grad_before, param in zip(grads, model.parameters()):
        if grad_before is not None:
            assert torch.equal(grad_before, param.grad)
        assert torch.isfinite(param).all()
        for value in opt.state[param].values():
            if torch.is_tensor(value):
                assert value.dtype == torch.float32
    assert opt.last_stats["matrix_params"] > 0
    assert opt.last_stats["fallback_params"] > 0


def test_golden_state_dict_resume_matches_uninterrupted_step() -> None:
    def seeded_step(model: TinyGoldenNet, opt: GoldenSodaPmuonEqNorMuon, seed: int) -> None:
        loss = _loss(model, seed=seed)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.manual_seed(8)
    uninterrupted = TinyGoldenNet()
    opt_uninterrupted = GoldenSodaPmuonEqNorMuon(_raw_groups(uninterrupted), lr=8e-3, warmup_steps=4)
    seeded_step(uninterrupted, opt_uninterrupted, seed=9)

    resumed = copy.deepcopy(uninterrupted)
    opt_resumed = GoldenSodaPmuonEqNorMuon(_raw_groups(resumed), lr=8e-3, warmup_steps=4)
    opt_resumed.load_state_dict(copy.deepcopy(opt_uninterrupted.state_dict()))

    seeded_step(uninterrupted, opt_uninterrupted, seed=10)
    seeded_step(resumed, opt_resumed, seed=10)

    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_golden_bfloat16_state_dict_load_restores_fp32_state() -> None:
    param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    opt = GoldenSodaPmuonEqNorMuon(
        [{"params": [param], "use_matrix": True}],
        lr=1e-3,
        warmup_steps=1,
    )
    param.grad = torch.randn_like(param)
    opt.step()

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = GoldenSodaPmuonEqNorMuon(
        [{"params": [restored_param], "use_matrix": True}],
        lr=1e-3,
        warmup_steps=1,
    )
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))

    restored_param.grad = torch.randn_like(restored_param)
    restored.step()

    state = restored.state[restored_param]
    assert state["z"].dtype == torch.float32
    assert state["soda_init"].dtype == torch.float32
    assert state["momentum"].dtype == torch.float32
    assert state["row_ema"].dtype == torch.float32
    assert state["normuon_second_moment"].dtype == torch.float32


def test_golden_train_eval_are_noop_compatibility_methods() -> None:
    torch.manual_seed(11)
    model = TinyGoldenNet()
    opt = GoldenSodaPmuonEqNorMuon(_raw_groups(model), lr=8e-3)
    loss = _loss(model, seed=12)
    loss.backward()
    opt.step()
    before = [param.detach().clone() for param in model.parameters()]
    assert opt.eval() is opt
    assert opt.train() is opt
    after = [param.detach().clone() for param in model.parameters()]
    for expected, actual in zip(before, after):
        assert torch.equal(expected, actual)
