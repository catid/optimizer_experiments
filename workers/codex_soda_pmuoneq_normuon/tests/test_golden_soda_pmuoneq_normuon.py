import copy
import sys
from io import BytesIO
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
for module_name in ("golden_soda_pmuoneq_normuon", "soda_pmuoneq_normuon"):
    sys.modules.pop(module_name, None)

from golden_soda_pmuoneq_normuon import (
    GoldenSodaPmuonEqNorMuon,
    build_golden_soda_pmuoneq_normuon_param_groups,
)
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


@pytest.mark.parametrize("optimizer_cls", [GoldenSodaPmuonEqNorMuon, SodaPmuonEqNorMuon])
def test_sparse_gradients_fail_with_clear_error(optimizer_cls: type[torch.optim.Optimizer]) -> None:
    embedding = nn.Embedding(16, 8, sparse=True)
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
    param = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    fallback = nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
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

    restored_param = nn.Parameter(param.detach().clone())
    restored_fallback = nn.Parameter(fallback.detach().clone())
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


def test_golden_param_groups_deduplicate_tied_parameter_aliases() -> None:
    tied = nn.Parameter(torch.zeros(8, 8))
    groups = build_golden_soda_pmuoneq_normuon_param_groups(
        [
            ("token_embed.weight", tied),
            ("lm_head.weight", tied),
        ]
    )

    assert len(groups) == 1
    assert groups[0]["use_matrix_update"] is False
    assert groups[0]["params"] == [tied]
    assert groups[0]["param_names"] == ["token_embed.weight|lm_head.weight"]


def test_golden_constructor_deduplicates_shared_params_in_raw_inputs() -> None:
    for params in (
        lambda p: [p, p],
        lambda p: [{"params": [p, p], "use_matrix_update": False}],
        lambda p: [{"params": p, "use_matrix_update": False}],
    ):
        param = nn.Parameter(torch.ones(1))
        opt = GoldenSodaPmuonEqNorMuon(params(param), warmup_steps=1)

        assert sum(candidate is param for group in opt.param_groups for candidate in group["params"]) == 1

        param.grad = torch.ones_like(param)
        opt.step()

        assert opt.last_stats["fallback_count"] == 1.0


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


def test_golden_leading_singleton_param_uses_effective_matrix_shape() -> None:
    torch.manual_seed(101)
    param = nn.Parameter(torch.randn(1, 6, 8))
    groups = build_golden_soda_pmuoneq_normuon_param_groups([("blocks.0.rel.weight", param)])
    opt = GoldenSodaPmuonEqNorMuon(groups, warmup_steps=1, ns_compute_dtype=torch.float32)

    param.grad = torch.randn_like(param)
    opt.step()

    state = opt.state[param]
    assert state["pmuoneq_row_ema"].shape == (6,)
    assert opt.last_stats["matrix_count"] == 1.0


def test_golden_matches_legacy_configured_as_winning_no_aspect_path() -> None:
    torch.manual_seed(7)
    golden_model = TwoMatrixNet()
    legacy_model = copy.deepcopy(golden_model)

    golden = GoldenSodaPmuonEqNorMuon(
        build_golden_soda_pmuoneq_normuon_param_groups(golden_model.named_parameters(), matrix_lr=1e-3, fallback_lr=1e-4),
        warmup_steps=2,
    )
    legacy = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(
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
