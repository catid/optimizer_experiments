import copy
import math
import sys
from io import BytesIO
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimizer as optimizer_module
from optimizer import AnchorMuon, __version__, build_param_groups


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(16)
        self.classifier_head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(self.norm(F.gelu(self.fc1(x))))


class TwoMatrixNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(8, 8, bias=False)
        self.right = nn.Linear(8, 8, bias=False)
        self.classifier_head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(F.gelu(self.right(F.gelu(self.left(x)))))


class SparseEmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8, sparse=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x).sum()


def _run_step(model: nn.Module, opt: torch.optim.Optimizer, step: int) -> float:
    torch.manual_seed(1000 + step)
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_anchor_param_groups_use_shape_only_for_matrix_routing() -> None:
    params = [
        ("blocks.0.mlp.fc1.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.attn.head_projection.weight", nn.Parameter(torch.zeros(8, 8))),
        ("lm_head.weight", nn.Parameter(torch.zeros(8, 8))),
        ("head.weight", nn.Parameter(torch.zeros(8, 8))),
        ("token_embed.weight", nn.Parameter(torch.zeros(8, 8))),
        ("cls_token", nn.Parameter(torch.zeros(1, 1, 8))),
        ("pos_embed", nn.Parameter(torch.zeros(1, 16, 8))),
        ("reg_token", nn.Parameter(torch.zeros(1, 4, 8))),
        ("blocks.0.layernorm.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.mlp.fc1.bias", nn.Parameter(torch.zeros(8))),
    ]
    groups = build_param_groups(params)
    assert len(groups) == 2
    matrix_names = set(groups[0]["param_names"])
    fallback_names = set(groups[1]["param_names"])
    assert "blocks.0.mlp.fc1.weight" in matrix_names
    assert "blocks.0.attn.head_projection.weight" in matrix_names
    assert "lm_head.weight" in matrix_names
    assert "head.weight" in matrix_names
    assert "token_embed.weight" in matrix_names
    assert "pos_embed" in matrix_names
    assert "reg_token" in matrix_names
    assert "blocks.0.layernorm.weight" in matrix_names
    assert "cls_token" in fallback_names
    assert "blocks.0.mlp.fc1.bias" in fallback_names


def test_public_api_is_only_anchormuon_and_build_param_groups() -> None:
    tied = nn.Parameter(torch.zeros(8, 8))
    groups = build_param_groups(
        [
            ("blocks.0.mlp.fc1.weight", nn.Parameter(torch.zeros(8, 8))),
            ("token_embed.weight", tied),
            ("lm_head.weight", tied),
        ]
    )
    assert set(optimizer_module.__all__) == {"__version__", "AnchorMuon", "build_param_groups"}
    assert [name for name in optimizer_module.__all__ if name not in {"__version__", "AnchorMuon", "build_param_groups"}] == []
    assert isinstance(__version__, str)
    assert len(groups) == 1
    assert sum(len(group["params"]) for group in groups) == 2
    assert "token_embed.weight|lm_head.weight" in set(groups[0]["param_names"])
    assert "param_aliases" not in groups[0]


def test_named_parameters_constructor_keeps_names_for_reporting_only() -> None:
    model = TinyClassifier()
    opt = AnchorMuon(model.named_parameters(), lr=1e-3, fallback_lr=1e-4)
    assert len(opt.param_groups) == 2
    matrix_group = next(group for group in opt.param_groups if group["use_matrix_update"])
    fallback_group = next(group for group in opt.param_groups if not group["use_matrix_update"])
    matrix_names = set(matrix_group["param_names"])
    fallback_names = set(fallback_group["param_names"])
    assert "classifier_head.weight" in matrix_names
    assert "norm.weight" in fallback_names
    assert "fc1.bias" in fallback_names
    summary = opt.group_summary()
    assert any(row["named"] for row in summary)


def test_optimizer_uses_trainer_owned_lr_values() -> None:
    model = TinyClassifier()
    opt = AnchorMuon(model, lr=1e-3, fallback_lr=1e-4)
    for group in opt.param_groups:
        group["lr"] = 7e-4

    _run_step(model, opt, 0)

    assert {group["lr"] for group in opt.param_groups} == {7e-4}
    assert all("base_lr" not in group for group in opt.param_groups)
    assert all("use_external_lr" not in group for group in opt.param_groups)


def test_soda_cannot_be_disabled_with_zero_scale() -> None:
    model = TinyClassifier()
    with pytest.raises(ValueError, match="SODA is always enabled"):
        AnchorMuon(model.named_parameters(), soda_lambda_scale=0.0)


def test_min_matrix_dim_keeps_tiny_matrices_in_fallback() -> None:
    params = [
        ("blocks.0.mlp.fc1.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.router.weight", nn.Parameter(torch.zeros(1, 8))),
    ]
    groups = build_param_groups(params, min_matrix_dim=2)
    assert len(groups) == 2
    assert groups[0]["param_names"] == ["blocks.0.mlp.fc1.weight"]
    assert groups[1]["param_names"] == ["blocks.0.router.weight"]


def test_unnamed_parameter_constructor_deduplicates_shared_tensors() -> None:
    shared = nn.Parameter(torch.zeros(8, 8))
    opt = AnchorMuon([shared, shared])
    assert len(opt.param_groups) == 1
    assert len(opt.param_groups[0]["params"]) == 1


def test_sparse_gradients_fail_with_clear_error() -> None:
    model = SparseEmbeddingNet()
    opt = AnchorMuon(model)
    loss = model(torch.tensor([1, 2, 3]))
    loss.backward()
    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        opt.step()


def test_default_recipe_matches_root_documented_winner() -> None:
    model = TinyClassifier()
    opt = AnchorMuon(model)
    assert {group["lr"] for group in opt.param_groups} == {8e-3}
    assert all("base_lr" not in group for group in opt.param_groups)
    matrix_group = next(group for group in opt.param_groups if group["use_matrix_update"])
    fallback_group = next(group for group in opt.param_groups if not group["use_matrix_update"])
    assert matrix_group["momentum"] == 0.95
    assert matrix_group["pmuoneq_beta"] == 0.90
    assert matrix_group["row_gamma"] == 0.35
    assert matrix_group["normuon_beta2"] == 0.93
    assert fallback_group["fallback_mode"] == "atan2"
    assert fallback_group["betas"] == (0.9, 0.95)
    assert fallback_group["fallback_weight_decay"] == 0.0
    assert fallback_group["eps"] == 1e-8


def test_fallback_modes_are_validated() -> None:
    model = TinyClassifier()
    with pytest.raises(ValueError, match="fallback_mode"):
        AnchorMuon(model.named_parameters(), fallback_mode="not-a-mode")
    with pytest.raises(ValueError, match="fallback_weight_decay"):
        AnchorMuon(model.named_parameters(), fallback_weight_decay=-0.1)


def test_atan2_fallback_scales_only_fallback_update() -> None:
    p = nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = AnchorMuon(
        [{"params": [p], "use_matrix_update": False}],
        lr=0.1,
        fallback_mode="atan2",
        fallback_betas=(0.5, 0.5),
    )
    p.grad = torch.tensor([0.5, -0.25])
    opt.step()

    expected = torch.tensor([1.0, -2.0]) - 0.1 * torch.tensor([math.pi / 4.0, -math.pi / 4.0])
    assert torch.allclose(p.detach(), expected, atol=1e-6, rtol=1e-6)
    assert "exp_avg" in opt.state[p]
    assert "exp_avg_sq" in opt.state[p]


def test_adamc_fallback_applies_optional_lr_squared_decay() -> None:
    p = nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = AnchorMuon(
        [{"params": [p], "use_matrix_update": False}],
        lr=0.1,
        fallback_mode="adamc",
        fallback_betas=(0.5, 0.5),
        fallback_weight_decay=1.0,
    )
    p.grad = torch.tensor([0.5, -0.25])
    opt.step()

    expected = torch.tensor([1.0, -2.0]) * 0.99 - 0.1 * torch.tensor([1.0, -1.0])
    assert torch.allclose(p.detach(), expected, atol=1e-6, rtol=1e-6)
    assert "exp_avg" in opt.state[p]
    assert "exp_avg_sq" in opt.state[p]


def test_from_model_constructor_matches_module_constructor() -> None:
    torch.manual_seed(5)
    direct_model = TinyClassifier()
    factory_model = copy.deepcopy(direct_model)
    direct = AnchorMuon(direct_model, lr=1e-3, fallback_lr=1e-4)
    factory = AnchorMuon.from_model(factory_model, lr=1e-3, fallback_lr=1e-4)

    for step in range(3):
        assert _run_step(direct_model, direct, step) == _run_step(factory_model, factory, step)

    for a, b in zip(direct_model.parameters(), factory_model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_anchor_groups_do_not_expose_removed_ablation_flags() -> None:
    model = TinyClassifier()
    groups = build_param_groups(model.named_parameters())
    forbidden = {
        "amuse",
        "mimuon",
        "mimuon_mix",
        "col_gamma",
        "normuon_mode",
        "normuon_aspect_scale",
        "matrix_filter",
        "base_lr",
        "use_external_lr",
    }
    for group in groups:
        assert forbidden.isdisjoint(group.keys())


def test_anchor_step_is_finite_and_creates_only_row_pmuoneq_state() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    opt = AnchorMuon(build_param_groups(model.named_parameters(), lr=1e-3, fallback_lr=1e-4))
    before = model.fc1.weight.detach().clone()

    for step in range(4):
        assert torch.isfinite(torch.tensor(_run_step(model, opt, step)))

    assert not torch.allclose(before, model.fc1.weight.detach())
    state = opt.state[model.fc1.weight]
    assert "momentum_buffer" in state
    assert "pmuoneq_row_ema" in state
    assert "pmuoneq_row_factor" in state
    assert "normuon_second_momentum" in state
    assert "pmuoneq_col_ema" not in state
    assert "pmuoneq_col_factor" not in state
    assert opt.last_stats["matrix_count"] >= 1
    assert opt.last_stats["fallback_count"] >= 1


def test_anchor_same_shape_bucket_matches_split_matrix_groups() -> None:
    torch.manual_seed(9)
    bucketed = TwoMatrixNet()
    split = copy.deepcopy(bucketed)

    opt_bucketed = AnchorMuon(build_param_groups(bucketed.named_parameters(), lr=1e-3, fallback_lr=1e-4))
    split_groups = [
        {"params": [split.left.weight], "use_matrix_update": True, "lr": 1e-3},
        {"params": [split.right.weight], "use_matrix_update": True, "lr": 1e-3},
        {"params": [split.classifier_head.weight], "use_matrix_update": True, "lr": 1e-3},
        {"params": [split.classifier_head.bias], "use_matrix_update": False, "lr": 1e-4},
    ]
    opt_split = AnchorMuon(split_groups)

    for step in range(5):
        _run_step(bucketed, opt_bucketed, step)
        _run_step(split, opt_split, step)

    for a, b in zip(bucketed.parameters(), split.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_anchor_state_dict_resume_matches_uninterrupted_training() -> None:
    torch.manual_seed(11)
    uninterrupted = TinyClassifier()
    resume_source = copy.deepcopy(uninterrupted)
    opt_uninterrupted = AnchorMuon(
        build_param_groups(uninterrupted.named_parameters(), lr=1e-3, fallback_lr=1e-4)
    )
    opt_resume_source = AnchorMuon(
        build_param_groups(resume_source.named_parameters(), lr=1e-3, fallback_lr=1e-4)
    )

    for step in range(3):
        _run_step(uninterrupted, opt_uninterrupted, step)
        _run_step(resume_source, opt_resume_source, step)

    model_blob = BytesIO()
    opt_blob = BytesIO()
    torch.save(resume_source.state_dict(), model_blob)
    torch.save(opt_resume_source.state_dict(), opt_blob)
    model_blob.seek(0)
    opt_blob.seek(0)

    resumed = TinyClassifier()
    resumed.load_state_dict(torch.load(model_blob, weights_only=True))
    opt_resumed = AnchorMuon(build_param_groups(resumed.named_parameters(), lr=1e-3, fallback_lr=1e-4))
    opt_resumed.load_state_dict(torch.load(opt_blob, weights_only=False))

    for step in range(3, 6):
        _run_step(uninterrupted, opt_uninterrupted, step)
        _run_step(resumed, opt_resumed, step)

    for a, b in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_anchor_short_training_sanity_loss_decreases() -> None:
    torch.manual_seed(42)
    model = TinyClassifier()
    opt = AnchorMuon(build_param_groups(model.named_parameters(), lr=2e-3, fallback_lr=2e-4))
    x = torch.randn(32, 8)
    y = torch.randint(0, 4, (32,))
    losses = []
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert all(torch.isfinite(torch.tensor(losses)))
    assert min(losses[-4:]) < losses[0]
