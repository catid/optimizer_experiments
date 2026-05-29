import copy
import sys
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REF_ROOT = ROOT / "workers" / "codex_soda_pmuoneq_normuon"
if str(REF_ROOT) not in sys.path:
    sys.path.insert(0, str(REF_ROOT))

from optimizer import (
    GoldenSodaPmuonEqNorMuon,
    GoldenMuon,
    SodaPmuonEqNorMuon,
    __version__,
    build_param_groups,
    build_golden_soda_pmuoneq_normuon_param_groups,
    build_soda_pmuoneq_normuon_param_groups,
)
from soda_pmuoneq_normuon import (
    SodaPmuonEqNorMuon as ReferenceSodaPmuonEqNorMuon,
    build_soda_pmuoneq_normuon_param_groups as build_reference_param_groups,
)


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


def test_golden_param_groups_keep_heads_norms_and_embeddings_in_fallback() -> None:
    params = [
        ("blocks.0.mlp.fc1.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.attn.head_projection.weight", nn.Parameter(torch.zeros(8, 8))),
        ("lm_head.weight", nn.Parameter(torch.zeros(8, 8))),
        ("head.weight", nn.Parameter(torch.zeros(8, 8))),
        ("token_embed.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.layernorm.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.mlp.fc1.bias", nn.Parameter(torch.zeros(8))),
    ]
    groups = build_golden_soda_pmuoneq_normuon_param_groups(params)
    assert len(groups) == 2
    matrix_names = set(groups[0]["param_names"])
    fallback_names = set(groups[1]["param_names"])
    assert "blocks.0.mlp.fc1.weight" in matrix_names
    assert "blocks.0.attn.head_projection.weight" in matrix_names
    assert "lm_head.weight" in fallback_names
    assert "head.weight" in fallback_names
    assert "token_embed.weight" in fallback_names
    assert "blocks.0.layernorm.weight" in fallback_names
    assert "blocks.0.mlp.fc1.bias" in fallback_names


def test_public_names_and_tied_parameter_grouping_are_safe() -> None:
    tied = nn.Parameter(torch.zeros(8, 8))
    groups = build_soda_pmuoneq_normuon_param_groups(
        [
            ("blocks.0.mlp.fc1.weight", nn.Parameter(torch.zeros(8, 8))),
            ("token_embed.weight", tied),
            ("lm_head.weight", tied),
        ]
    )
    assert GoldenSodaPmuonEqNorMuon is SodaPmuonEqNorMuon
    assert GoldenMuon is SodaPmuonEqNorMuon
    assert build_param_groups is build_soda_pmuoneq_normuon_param_groups
    assert isinstance(__version__, str)
    assert len(groups) == 2
    assert sum(len(group["params"]) for group in groups) == 2
    fallback_names = set(groups[1]["param_names"])
    assert "token_embed.weight|lm_head.weight" in fallback_names
    assert groups[1]["param_aliases"] == [["token_embed.weight", "lm_head.weight"]]


def test_named_parameters_constructor_hides_grouping_from_training_code() -> None:
    model = TinyClassifier()
    opt = SodaPmuonEqNorMuon(model.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4, warmup_steps=2)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["use_matrix_update"] is True
    assert opt.param_groups[1]["use_matrix_update"] is False
    fallback_names = set(opt.param_groups[1]["param_names"])
    assert "classifier_head.weight" in fallback_names
    assert "norm.weight" in fallback_names
    summary = opt.group_summary()
    assert summary[0]["use_matrix_update"] is True
    assert summary[0]["param_count"] == 1
    assert summary[1]["use_matrix_update"] is False
    assert summary[1]["named"] is True


def test_named_constructor_preserves_external_lr_flag() -> None:
    model = TinyClassifier()
    opt = SodaPmuonEqNorMuon(
        model,
        matrix_lr=1e-3,
        fallback_lr=1e-4,
        warmup_steps=100,
        use_external_lr=True,
    )
    for group in opt.param_groups:
        group["lr"] = 7e-4

    _run_step(model, opt, 0)

    assert {group["lr"] for group in opt.param_groups} == {7e-4}
    assert all(group["use_external_lr"] is True for group in opt.param_groups)


def test_min_matrix_dim_keeps_tiny_matrices_in_fallback() -> None:
    params = [
        ("blocks.0.mlp.fc1.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.router.weight", nn.Parameter(torch.zeros(1, 8))),
    ]
    groups = build_soda_pmuoneq_normuon_param_groups(params, min_matrix_dim=2)
    assert len(groups) == 2
    assert groups[0]["param_names"] == ["blocks.0.mlp.fc1.weight"]
    assert groups[1]["param_names"] == ["blocks.0.router.weight"]


def test_unnamed_parameter_constructor_deduplicates_shared_tensors() -> None:
    shared = nn.Parameter(torch.zeros(8, 8))
    with pytest.warns(UserWarning, match="unnamed parameters"):
        opt = SodaPmuonEqNorMuon([shared, shared], warmup_steps=2)
    assert len(opt.param_groups) == 1
    assert len(opt.param_groups[0]["params"]) == 1


def test_sparse_gradients_fail_with_clear_error() -> None:
    model = SparseEmbeddingNet()
    opt = SodaPmuonEqNorMuon(model, warmup_steps=2)
    loss = model(torch.tensor([1, 2, 3]))
    loss.backward()
    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        opt.step()


def test_default_recipe_matches_root_documented_winner() -> None:
    model = TinyClassifier()
    opt = SodaPmuonEqNorMuon(model)
    assert opt.warmup_steps == 80
    assert {group["lr"] for group in opt.param_groups} == {8e-3}
    assert {group["base_lr"] for group in opt.param_groups} == {8e-3}
    matrix_group = next(group for group in opt.param_groups if group["use_matrix_update"])
    fallback_group = next(group for group in opt.param_groups if not group["use_matrix_update"])
    assert matrix_group["momentum"] == 0.95
    assert matrix_group["pmuoneq_beta"] == 0.90
    assert matrix_group["row_gamma"] == 0.35
    assert matrix_group["normuon_beta2"] == 0.93
    assert fallback_group["betas"] == (0.9, 0.95)
    assert fallback_group["eps"] == 1e-8
    assert fallback_group["weight_decay"] == 0.05


def test_from_model_constructor_matches_named_parameters_constructor() -> None:
    torch.manual_seed(5)
    direct_model = TinyClassifier()
    factory_model = copy.deepcopy(direct_model)
    direct = SodaPmuonEqNorMuon(direct_model.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4, warmup_steps=2)
    module_direct = copy.deepcopy(direct_model)
    factory = SodaPmuonEqNorMuon.from_model(factory_model, matrix_lr=1e-3, fallback_lr=1e-4, warmup_steps=2)
    module_opt = SodaPmuonEqNorMuon(module_direct, matrix_lr=1e-3, fallback_lr=1e-4, warmup_steps=2)

    for step in range(3):
        assert _run_step(direct_model, direct, step) == _run_step(factory_model, factory, step)
        _run_step(module_direct, module_opt, step)

    for a, b in zip(direct_model.parameters(), factory_model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)
    for a, b in zip(direct_model.parameters(), module_direct.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_golden_groups_do_not_expose_removed_ablation_flags() -> None:
    model = TinyClassifier()
    groups = build_golden_soda_pmuoneq_normuon_param_groups(model.named_parameters())
    forbidden = {
        "amuse",
        "mimuon",
        "mimuon_mix",
        "col_gamma",
        "normuon_mode",
        "normuon_aspect_scale",
        "soda_disables_matrix_weight_decay",
    }
    for group in groups:
        assert forbidden.isdisjoint(group.keys())


def test_golden_step_is_finite_and_creates_only_row_pmuoneq_state() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    opt = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(model.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4),
        warmup_steps=2,
    )
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


def test_golden_matches_legacy_configured_as_winning_no_aspect_path() -> None:
    torch.manual_seed(7)
    golden_model = TwoMatrixNet()
    legacy_model = copy.deepcopy(golden_model)

    golden = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(
            golden_model.named_parameters(),
            matrix_lr=1e-3,
            fallback_lr=1e-4,
            fallback_betas=(0.9, 0.999),
            eps=1e-10,
        ),
        warmup_steps=2,
    )
    legacy = ReferenceSodaPmuonEqNorMuon(
        build_reference_param_groups(
            legacy_model.named_parameters(),
            matrix_lr=1e-3,
            adam_lr=1e-4,
            pmuoneq_beta=0.90,
            row_gamma=0.35,
            col_gamma=0.0,
            normuon_beta2=0.93,
            normuon_mode="row",
            normuon_aspect_scale=False,
        ),
        warmup_steps=2,
    )

    for step in range(5):
        golden_loss = _run_step(golden_model, golden, step)
        legacy_loss = _run_step(legacy_model, legacy, step)
        assert golden_loss == legacy_loss

    for golden_param, legacy_param in zip(golden_model.parameters(), legacy_model.parameters(), strict=True):
        assert torch.allclose(golden_param, legacy_param, atol=1e-6, rtol=1e-6)


def test_golden_same_shape_bucket_matches_split_matrix_groups() -> None:
    torch.manual_seed(9)
    bucketed = TwoMatrixNet()
    split = copy.deepcopy(bucketed)

    opt_bucketed = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(bucketed.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4),
        warmup_steps=2,
    )
    split_groups = [
        {
            "params": [split.left.weight],
            "param_names": ["left.weight"],
            "use_matrix_update": True,
            "lr": 1e-3,
            "base_lr": 1e-3,
        },
        {
            "params": [split.right.weight],
            "param_names": ["right.weight"],
            "use_matrix_update": True,
            "lr": 1e-3,
            "base_lr": 1e-3,
        },
        {
            "params": list(split.classifier_head.parameters()),
            "param_names": ["classifier_head.weight", "classifier_head.bias"],
            "use_matrix_update": False,
            "lr": 1e-4,
            "base_lr": 1e-4,
        },
    ]
    opt_split = GoldenSodaPmuonEqNorMuon(split_groups, warmup_steps=2)

    for step in range(5):
        _run_step(bucketed, opt_bucketed, step)
        _run_step(split, opt_split, step)

    for a, b in zip(bucketed.parameters(), split.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_golden_state_dict_resume_matches_uninterrupted_training() -> None:
    torch.manual_seed(11)
    uninterrupted = TinyClassifier()
    resume_source = copy.deepcopy(uninterrupted)
    opt_uninterrupted = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(uninterrupted.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4),
        warmup_steps=2,
    )
    opt_resume_source = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(resume_source.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4),
        warmup_steps=2,
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
    opt_resumed = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(resumed.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4),
        warmup_steps=2,
    )
    opt_resumed.load_state_dict(torch.load(opt_blob, weights_only=False))

    for step in range(3, 6):
        _run_step(uninterrupted, opt_uninterrupted, step)
        _run_step(resumed, opt_resumed, step)

    for a, b in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_golden_short_training_sanity_loss_decreases() -> None:
    torch.manual_seed(42)
    model = TinyClassifier()
    opt = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(model.named_parameters(), matrix_lr=2e-3, fallback_lr=2e-4),
        warmup_steps=2,
    )
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
