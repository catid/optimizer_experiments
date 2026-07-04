import copy
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

from soda_pmuoneq_normuon import SodaPmuonEqNorMuon, build_soda_pmuoneq_normuon_param_groups


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


class WideMatrixNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wide = nn.Linear(16, 8, bias=False)
        self.classifier_head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(F.gelu(self.wide(x)))


def test_sparse_gradients_fail_with_clear_error() -> None:
    embedding = nn.Embedding(16, 8, sparse=True)
    opt = SodaPmuonEqNorMuon(
        [{"params": [embedding.weight], "use_matrix_update": False}],
        warmup_steps=1,
    )

    loss = embedding(torch.tensor([1, 2, 3])).sum()
    loss.backward()

    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        opt.step()


def test_param_group_builder_keeps_head_and_norm_in_fallback() -> None:
    model = TinyClassifier()
    groups = build_soda_pmuoneq_normuon_param_groups(model.named_parameters())
    assert len(groups) == 2
    matrix_names = set(groups[0]["param_names"])
    fallback_names = set(groups[1]["param_names"])
    assert groups[0]["use_matrix_update"] is True
    assert groups[1]["use_matrix_update"] is False
    assert "fc1.weight" in matrix_names
    assert "norm.weight" in fallback_names
    assert "classifier_head.weight" in fallback_names


def test_param_group_builder_does_not_overmatch_head_substrings() -> None:
    params = [
        ("blocks.0.attn.head_projection.weight", nn.Parameter(torch.zeros(8, 8))),
        ("lm_head.weight", nn.Parameter(torch.zeros(8, 8))),
        ("head.weight", nn.Parameter(torch.zeros(8, 8))),
        ("token_embed.weight", nn.Parameter(torch.zeros(8, 8))),
        ("blocks.0.layernorm.weight", nn.Parameter(torch.zeros(8, 8))),
    ]
    groups = build_soda_pmuoneq_normuon_param_groups(params)
    matrix_names = set(groups[0]["param_names"])
    fallback_names = set(groups[1]["param_names"])
    assert "blocks.0.attn.head_projection.weight" in matrix_names
    assert "lm_head.weight" in fallback_names
    assert "head.weight" in fallback_names
    assert "token_embed.weight" in fallback_names
    assert "blocks.0.layernorm.weight" in fallback_names


def test_param_group_builder_deduplicates_tied_parameter_aliases() -> None:
    tied = nn.Parameter(torch.zeros(8, 8))
    groups = build_soda_pmuoneq_normuon_param_groups(
        [
            ("token_embed.weight", tied),
            ("lm_head.weight", tied),
        ]
    )

    assert len(groups) == 1
    assert groups[0]["use_matrix_update"] is False
    assert groups[0]["params"] == [tied]
    assert groups[0]["param_names"] == ["token_embed.weight|lm_head.weight"]


def test_constructor_deduplicates_shared_params_in_raw_inputs() -> None:
    for params in (
        lambda p: [p, p],
        lambda p: [{"params": [p, p], "use_matrix_update": False}],
        lambda p: [{"params": p, "use_matrix_update": False}],
    ):
        param = nn.Parameter(torch.ones(1))
        opt = SodaPmuonEqNorMuon(params(param), warmup_steps=1)

        assert sum(candidate is param for group in opt.param_groups for candidate in group["params"]) == 1

        param.grad = torch.ones_like(param)
        opt.step()

        assert opt.last_stats["fallback_count"] == 1.0


def test_optimizer_step_is_finite_and_creates_matrix_state() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    opt = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(model.named_parameters(), matrix_lr=1e-3, adam_lr=1e-4),
        warmup_steps=2,
    )
    x = torch.randn(12, 8)
    y = torch.randint(0, 4, (12,))
    before = model.fc1.weight.detach().clone()

    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()

    assert not torch.allclose(before, model.fc1.weight.detach())
    state = opt.state[model.fc1.weight]
    assert "momentum_buffer" in state
    assert "pmuoneq_row_ema" in state
    assert "pmuoneq_col_ema" in state
    assert "normuon_second_momentum" in state
    assert opt.last_stats["matrix_count"] >= 1
    assert opt.last_stats["fallback_count"] >= 1


def test_leading_singleton_param_uses_effective_matrix_shape() -> None:
    torch.manual_seed(101)
    param = nn.Parameter(torch.randn(1, 6, 8))
    groups = build_soda_pmuoneq_normuon_param_groups([("blocks.0.rel.weight", param)])
    opt = SodaPmuonEqNorMuon(groups, warmup_steps=1, ns_compute_dtype=torch.float32)

    param.grad = torch.randn_like(param)
    opt.step()

    state = opt.state[param]
    assert state["pmuoneq_row_ema"].shape == (6,)
    assert state["pmuoneq_col_ema"].shape == (8,)
    assert opt.last_stats["matrix_count"] == 1.0


def test_effective_vector_shape_routes_to_fallback() -> None:
    param = nn.Parameter(torch.randn(1, 1, 8))
    groups = build_soda_pmuoneq_normuon_param_groups([("blocks.0.scale.weight", param)])

    assert len(groups) == 1
    assert groups[0]["use_matrix_update"] is False
    assert groups[0]["params"] == [param]


def test_normuon_aspect_and_orientation_ablation_controls() -> None:
    torch.manual_seed(13)
    row_aspect = TinyClassifier()
    row_noaspect = copy.deepcopy(row_aspect)
    opt_aspect = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(
            row_aspect.named_parameters(),
            matrix_lr=1e-3,
            adam_lr=1e-4,
            normuon_mode="row",
            normuon_aspect_scale=True,
        ),
        warmup_steps=1,
    )
    opt_noaspect = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(
            row_noaspect.named_parameters(),
            matrix_lr=1e-3,
            adam_lr=1e-4,
            normuon_mode="row",
            normuon_aspect_scale=False,
        ),
        warmup_steps=1,
    )
    _run_step(row_aspect, opt_aspect, 0)
    _run_step(row_noaspect, opt_noaspect, 0)
    assert not torch.allclose(row_aspect.fc1.weight, row_noaspect.fc1.weight)

    wide = WideMatrixNet()
    opt_orient = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(
            wide.named_parameters(),
            matrix_lr=1e-3,
            adam_lr=1e-4,
            normuon_mode="orientation",
            normuon_aspect_scale=False,
        ),
        warmup_steps=1,
    )
    torch.manual_seed(10_001)
    x = torch.randn(16, 16)
    y = torch.randint(0, 4, (16,))
    opt_orient.zero_grad(set_to_none=True)
    F.cross_entropy(wide(x), y).backward()
    opt_orient.step()
    second = opt_orient.state[wide.wide.weight]["normuon_second_momentum"]
    assert tuple(second.shape) == (1, wide.wide.weight.shape[1])


def test_train_eval_are_noops_for_standalone_optimizer() -> None:
    model = TinyClassifier()
    opt = SodaPmuonEqNorMuon(build_soda_pmuoneq_normuon_param_groups(model.named_parameters()))
    before = [p.detach().clone() for p in model.parameters()]
    assert opt.train() is opt
    assert opt.eval() is opt
    after = [p.detach().clone() for p in model.parameters()]
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b)


def test_short_training_sanity_loss_decreases() -> None:
    torch.manual_seed(42)
    model = TinyClassifier()
    opt = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(model.named_parameters(), matrix_lr=2e-3, adam_lr=2e-4),
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


def _run_step(model: nn.Module, opt: torch.optim.Optimizer, step: int) -> float:
    torch.manual_seed(10_000 + step)
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_state_dict_resume_matches_uninterrupted_training() -> None:
    torch.manual_seed(7)
    uninterrupted = TinyClassifier()
    resume_source = copy.deepcopy(uninterrupted)
    opt_uninterrupted = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(uninterrupted.named_parameters(), matrix_lr=1e-3, adam_lr=1e-4),
        warmup_steps=2,
    )
    opt_resume_source = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(resume_source.named_parameters(), matrix_lr=1e-3, adam_lr=1e-4),
        warmup_steps=2,
    )

    for step in range(3):
        assert _run_step(uninterrupted, opt_uninterrupted, step) == _run_step(resume_source, opt_resume_source, step)

    model_blob = BytesIO()
    opt_blob = BytesIO()
    torch.save(resume_source.state_dict(), model_blob)
    torch.save(opt_resume_source.state_dict(), opt_blob)
    model_blob.seek(0)
    opt_blob.seek(0)

    resumed = TinyClassifier()
    resumed.load_state_dict(torch.load(model_blob, weights_only=True))
    opt_resumed = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(resumed.named_parameters(), matrix_lr=1e-3, adam_lr=1e-4),
        warmup_steps=2,
    )
    opt_resumed.load_state_dict(torch.load(opt_blob, weights_only=False))

    for step in range(3, 6):
        _run_step(uninterrupted, opt_uninterrupted, step)
        _run_step(resumed, opt_resumed, step)

    for a, b in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_same_shape_bucket_matches_split_matrix_groups() -> None:
    torch.manual_seed(9)
    bucketed = TwoMatrixNet()
    split = copy.deepcopy(bucketed)

    opt_bucketed = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(bucketed.named_parameters(), matrix_lr=1e-3, adam_lr=1e-4),
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
    opt_split = SodaPmuonEqNorMuon(split_groups, warmup_steps=2)

    for step in range(5):
        _run_step(bucketed, opt_bucketed, step)
        _run_step(split, opt_split, step)

    for a, b in zip(bucketed.parameters(), split.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_external_lr_is_not_overwritten_by_internal_warmup() -> None:
    torch.manual_seed(12)
    model = TinyClassifier()
    opt = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(model.named_parameters(), matrix_lr=1e-3, adam_lr=1e-4),
        warmup_steps=100,
        use_external_lr=True,
    )
    for group in opt.param_groups:
        group["lr"] = 3e-4
    _run_step(model, opt, 0)
    assert {group["lr"] for group in opt.param_groups} == {3e-4}


def test_soda_disables_matrix_weight_decay_by_default() -> None:
    def make_model_and_grad() -> tuple[nn.Linear, torch.Tensor]:
        torch.manual_seed(31)
        model = nn.Linear(8, 8, bias=False)
        x = torch.randn(16, 8)
        y = torch.randn(16, 8)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        return model, model.weight.grad.detach().clone()

    disabled, grad = make_model_and_grad()
    enabled = copy.deepcopy(disabled)
    enabled.weight.grad = grad.clone()

    opt_disabled = SodaPmuonEqNorMuon(
        [{"params": [disabled.weight], "use_matrix_update": True, "weight_decay": 0.5}],
        matrix_lr=1e-3,
        warmup_steps=1,
        soda_disables_matrix_weight_decay=True,
    )
    opt_enabled = SodaPmuonEqNorMuon(
        [{"params": [enabled.weight], "use_matrix_update": True, "weight_decay": 0.5}],
        matrix_lr=1e-3,
        warmup_steps=1,
        soda_disables_matrix_weight_decay=False,
    )

    opt_disabled.step()
    opt_enabled.step()

    assert opt_disabled.last_stats["matrix_weight_decay_count"] == 0.0
    assert opt_enabled.last_stats["matrix_weight_decay_count"] == 1.0
    assert not torch.allclose(disabled.weight, enabled.weight)
