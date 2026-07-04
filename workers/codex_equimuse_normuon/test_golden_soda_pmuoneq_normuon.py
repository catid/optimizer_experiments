from __future__ import annotations

import copy
import sys
from io import BytesIO
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
for module_name in ("direct_soda_pmuoneq_normuon", "golden_soda_pmuoneq_normuon"):
    sys.modules.pop(module_name, None)

from direct_soda_pmuoneq_normuon import (
    SodaPmuonEqNorMuon,
    build_soda_pmuoneq_normuon_param_groups,
)
from golden_soda_pmuoneq_normuon import (
    GoldenSodaPmuonEqNorMuon,
    build_golden_soda_pmuoneq_normuon_param_groups,
)


def _make_params() -> list[torch.nn.Parameter]:
    torch.manual_seed(20260529)
    return [
        torch.nn.Parameter(torch.randn(4, 8) * 0.1),
        torch.nn.Parameter(torch.randn(4, 8) * 0.1),
        torch.nn.Parameter(torch.randn(5) * 0.1),
    ]


def _clone_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(p.detach().clone()) for p in params]


def _assign_grads(params_a: list[torch.nn.Parameter], params_b: list[torch.nn.Parameter], step: int) -> None:
    for i, (pa, pb) in enumerate(zip(params_a, params_b, strict=True)):
        grad = torch.arange(pa.numel(), dtype=pa.dtype).reshape_as(pa)
        grad = (grad + 1.0 + 0.13 * i) * (0.017 + 0.003 * step)
        pa.grad = grad.clone()
        pb.grad = grad.clone()


@pytest.mark.parametrize("optimizer_cls", [GoldenSodaPmuonEqNorMuon, SodaPmuonEqNorMuon])
def test_sparse_gradients_fail_with_clear_error(optimizer_cls: type[torch.optim.Optimizer]) -> None:
    embedding = torch.nn.Embedding(16, 8, sparse=True)
    opt = optimizer_cls(
        [{"params": [embedding.weight], "use_matrix_update": False}],
        warmup_steps=1,
    )

    loss = embedding(torch.tensor([1, 2, 3])).sum()
    loss.backward()

    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        opt.step()


@pytest.mark.parametrize("optimizer_cls", [GoldenSodaPmuonEqNorMuon, SodaPmuonEqNorMuon])
def test_bfloat16_state_dict_load_restores_fp32_matrix_state(optimizer_cls: type[torch.optim.Optimizer]) -> None:
    param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    fallback = torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
    opt = optimizer_cls(
        [
            {"params": [param], "use_matrix_update": True, "lr": 1e-3},
            {"params": [fallback], "use_matrix_update": False, "lr": 1e-3},
        ],
        warmup_steps=1,
        use_external_lr=True,
        ns_compute_dtype=torch.float32,
    )
    param.grad = torch.randn_like(param)
    fallback.grad = torch.randn_like(fallback)
    opt.step()

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored_fallback = torch.nn.Parameter(fallback.detach().clone())
    restored = optimizer_cls(
        [
            {"params": [restored_param], "use_matrix_update": True, "lr": 1e-3},
            {"params": [restored_fallback], "use_matrix_update": False, "lr": 1e-3},
        ],
        warmup_steps=1,
        use_external_lr=True,
        ns_compute_dtype=torch.float32,
    )
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))

    restored_param.grad = torch.randn_like(restored_param)
    restored_fallback.grad = torch.randn_like(restored_fallback)
    restored.step()

    state = restored.state[restored_param]
    fallback_state = restored.state[restored_fallback]
    assert state["momentum_buffer"].dtype == torch.float32
    assert state["pmuoneq_row_ema"].dtype == torch.float32
    assert state["normuon_second_momentum"].dtype == torch.float32
    assert fallback_state["exp_avg_sq"].dtype == torch.float32


def test_golden_param_group_helper_excludes_lm_sensitive_params() -> None:
    embed = torch.nn.Parameter(torch.zeros(8, 16))
    lm_head = torch.nn.Parameter(torch.zeros(16, 8))
    norm = torch.nn.Parameter(torch.zeros(16))
    hidden = torch.nn.Parameter(torch.zeros(16, 16))
    q_proj = torch.nn.Parameter(torch.zeros(16, 16))
    named = [
        ("transformer.wte.weight", embed),
        ("lm_head.weight", lm_head),
        ("blocks.0.norm.weight", norm),
        ("blocks.0.mlp.fc1.weight", hidden),
        ("blocks.0.attn.q_proj.weight", q_proj),
    ]

    groups = build_golden_soda_pmuoneq_normuon_param_groups(named)

    assert [group["use_matrix_update"] for group in groups] == [True, False]
    assert {id(p) for p in groups[0]["params"]} == {id(hidden), id(q_proj)}
    assert {id(p) for p in groups[1]["params"]} == {id(embed), id(lm_head), id(norm)}
    assert groups[0]["col_gamma"] == 0.05
    assert "normuon_aspect_scale" not in groups[0]


def test_golden_param_group_helper_deduplicates_tied_aliases() -> None:
    tied = torch.nn.Parameter(torch.zeros(8, 8))
    groups = build_golden_soda_pmuoneq_normuon_param_groups(
        [
            ("token_embed.weight", tied),
            ("lm_head.weight", tied),
        ]
    )

    params = [p for group in groups for p in group["params"]]
    assert params == [tied]
    assert groups[0]["use_matrix_update"] is False
    assert groups[0]["param_names"] == ["token_embed.weight|lm_head.weight"]


def test_matrix_group_vector_fallback_uses_configured_defaults() -> None:
    vector = torch.nn.Parameter(torch.ones(5))
    opt = GoldenSodaPmuonEqNorMuon(
        [{"params": [vector], "use_matrix_update": True}],
        eps=1e-4,
        fallback_beta2=0.8,
        fallback_weight_decay=0.2,
        warmup_steps=1,
    )

    group = opt.param_groups[0]
    assert group["fallback_beta2"] == 0.8
    assert group["eps"] == 1e-4
    assert group["weight_decay"] == 0.2

    vector.grad = torch.full_like(vector, 0.1)
    opt.step()

    assert torch.isfinite(vector).all()
    assert opt.state[vector]["exp_avg_sq"].dtype == torch.float32


def test_direct_and_golden_use_effective_shape_for_leading_singleton_params() -> None:
    torch.manual_seed(304)
    direct_param = torch.nn.Parameter(torch.randn(1, 6, 8))
    golden_param = torch.nn.Parameter(direct_param.detach().clone())
    direct = SodaPmuonEqNorMuon([direct_param], warmup_steps=1, ns_compute_dtype=torch.float32)
    golden = GoldenSodaPmuonEqNorMuon([golden_param], warmup_steps=1, ns_compute_dtype=torch.float32)

    grad = torch.randn_like(direct_param)
    direct_param.grad = grad.clone()
    golden_param.grad = grad.clone()
    direct.step()
    golden.step()

    assert direct.state[direct_param]["pmuoneq_row_ema"].shape == (6,)
    assert direct.state[direct_param]["pmuoneq_col_ema"].shape == (8,)
    assert golden.state[golden_param]["pmuoneq_row_ema"].shape == (6,)
    assert golden.state[golden_param]["pmuoneq_col_ema"].shape == (8,)


def test_direct_optimizer_deduplicates_shared_parameters() -> None:
    shared = torch.nn.Parameter(torch.ones(4, 4))
    vector = torch.nn.Parameter(torch.ones(4))

    opt = SodaPmuonEqNorMuon([shared, shared, vector, vector], warmup_steps=1)
    params = [p for group in opt.param_groups for p in group["params"]]

    assert sum(p is shared for p in params) == 1
    assert sum(p is vector for p in params) == 1


def test_direct_optimizer_accepts_single_tensor_dict_group() -> None:
    vector = torch.nn.Parameter(torch.ones(4))
    opt = SodaPmuonEqNorMuon(
        [{"params": vector, "use_matrix_update": False}],
        warmup_steps=1,
    )

    assert opt.param_groups[0]["params"] == [vector]

    vector.grad = torch.ones_like(vector)
    opt.step()

    assert opt.last_stats["fallback_count"] == 1.0


def test_golden_optimizer_deduplicates_shared_parameters() -> None:
    for params in (
        lambda p: [p, p],
        lambda p: [{"params": [p, p], "use_matrix_update": False}],
        lambda p: [{"params": p, "use_matrix_update": False}],
    ):
        vector = torch.nn.Parameter(torch.ones(4))
        opt = GoldenSodaPmuonEqNorMuon(params(vector), warmup_steps=1)

        assert sum(candidate is vector for group in opt.param_groups for candidate in group["params"]) == 1

        vector.grad = torch.ones_like(vector)
        opt.step()

        assert opt.last_stats["fallback_count"] == 1.0


def test_direct_param_group_builder_deduplicates_tied_weights() -> None:
    shared = torch.nn.Parameter(torch.ones(4, 4))
    vector = torch.nn.Parameter(torch.ones(4))
    groups = build_soda_pmuoneq_normuon_param_groups([
        ("embed.weight", shared),
        ("lm_head.weight", shared),
        ("norm.weight", vector),
        ("norm_alias.weight", vector),
    ])
    params = [p for group in groups for p in group["params"]]

    assert sum(p is shared for p in params) == 1
    assert sum(p is vector for p in params) == 1


def test_direct_param_group_builder_uses_all_aliases_for_tied_routing() -> None:
    shared = torch.nn.Parameter(torch.ones(4, 4))
    groups = build_soda_pmuoneq_normuon_param_groups([
        ("blocks.0.weight", shared),
        ("lm_head.weight", shared),
    ])

    assert len(groups) == 1
    assert groups[0]["use_matrix_update"] is False
    assert groups[0]["params"] == [shared]
    assert groups[0]["param_names"] == ["blocks.0.weight|lm_head.weight"]


def test_golden_matches_direct_best_result_configuration() -> None:
    golden_params = _make_params()
    direct_params = _clone_params(golden_params)

    golden = GoldenSodaPmuonEqNorMuon(
        [
            {
                "params": golden_params[:2],
                "use_matrix_update": True,
                "lr": 0.02,
                "base_lr": 0.02,
            },
            {
                "params": golden_params[2:],
                "use_matrix_update": False,
                "lr": 0.002,
                "base_lr": 0.002,
            },
        ],
        warmup_steps=2,
        ns_compute_dtype=torch.float32,
    )
    direct = SodaPmuonEqNorMuon(
        [
            {
                "params": direct_params[:2],
                "use_matrix_update": True,
                "lr": 0.02,
                "base_lr": 0.02,
                "weight_decay": 0.0,
                "col_gamma": 0.05,
                "normuon_aspect_scale": True,
            },
            {
                "params": direct_params[2:],
                "use_matrix_update": False,
                "lr": 0.002,
                "base_lr": 0.002,
                "weight_decay": 0.05,
            },
        ],
        momentum=0.9,
        eps=1e-8,
        warmup_steps=2,
        ns_compute_dtype=torch.float32,
    )

    for step in range(6):
        _assign_grads(golden_params, direct_params, step)
        golden.step()
        direct.step()

    for golden_param, direct_param in zip(golden_params, direct_params, strict=True):
        assert torch.allclose(golden_param, direct_param, atol=1e-6, rtol=1e-6)


def test_golden_has_column_and_aspect_state_for_best_result_path() -> None:
    params = _make_params()
    opt = GoldenSodaPmuonEqNorMuon(params, warmup_steps=2, ns_compute_dtype=torch.float32)
    for i, p in enumerate(params):
        p.grad = torch.ones_like(p) * (0.01 + 0.01 * i)
    opt.step()

    assert opt.last_stats["column_preconditioning_enabled"] == 1.0
    assert opt.last_stats["aspect_scale_enabled"] == 1.0
    for p in params[:2]:
        state = opt.state[p]
        assert "pmuoneq_row_ema" in state
        assert "pmuoneq_row_factor" in state
        assert "pmuoneq_col_ema" in state
        assert "pmuoneq_col_factor" in state


def test_golden_state_dict_resume_parity() -> None:
    params = _make_params()
    opt = GoldenSodaPmuonEqNorMuon(params, warmup_steps=2, ns_compute_dtype=torch.float32)
    for step in range(3):
        for i, p in enumerate(params):
            p.grad = torch.ones_like(p) * (0.02 + i * 0.01 + step * 0.005)
        opt.step()

    buffer = BytesIO()
    torch.save({"params": [p.detach().clone() for p in params], "optimizer": opt.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=False)

    resumed_params = [torch.nn.Parameter(p.clone()) for p in checkpoint["params"]]
    resumed = GoldenSodaPmuonEqNorMuon(resumed_params, warmup_steps=2, ns_compute_dtype=torch.float32)
    resumed.load_state_dict(checkpoint["optimizer"])

    for step in range(3, 7):
        _assign_grads(params, resumed_params, step)
        opt.step()
        resumed.step()

    for a, b in zip(params, resumed_params, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_golden_builder_can_match_named_direct_builder_for_best_result_subset() -> None:
    params = _make_params()
    named = [
        ("blocks.0.mlp.weight", params[0]),
        ("blocks.1.attn.q_proj.weight", params[1]),
        ("head.bias", params[2]),
    ]
    golden_groups = build_golden_soda_pmuoneq_normuon_param_groups(named)
    direct_groups = build_soda_pmuoneq_normuon_param_groups(
        named,
        col_gamma=0.05,
        normuon_aspect_scale=True,
        matrix_weight_decay=0.0,
        adam_weight_decay=0.05,
    )

    assert [g["use_matrix_update"] for g in golden_groups] == [g["use_matrix_update"] for g in direct_groups]
    assert {id(p) for p in golden_groups[0]["params"]} == {id(p) for p in direct_groups[0]["params"]}
    assert {id(p) for p in golden_groups[1]["params"]} == {id(p) for p in direct_groups[1]["params"]}
