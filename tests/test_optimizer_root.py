import copy

import torch
from torch import nn

from optimizer import SodaPmuonEqNorMuon, build_param_groups


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.fc1 = nn.Linear(8, 32)
        self.norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 4)
        self.lm_head = nn.Linear(4, 16, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embed(x).mean(dim=1)
        y = self.fc1(y).relu()
        y = self.norm(y)
        return self.lm_head(self.fc2(y))


def _one_step(model: nn.Module, opt: torch.optim.Optimizer) -> float:
    x = torch.randint(0, 16, (8, 5))
    y = torch.randint(0, 16, (8,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach())


def test_named_grouping_keeps_embeddings_heads_and_norms_in_fallback() -> None:
    model = TinyModel()
    groups = build_param_groups(model.named_parameters())

    matrix_names = [name for group in groups if group["use_matrix_update"] for name in group["param_names"]]
    fallback_names = [name for group in groups if not group["use_matrix_update"] for name in group["param_names"]]

    assert "fc1.weight" in matrix_names
    assert "fc2.weight" in matrix_names
    assert "embed.weight" in fallback_names
    assert "norm.weight" in fallback_names
    assert "lm_head.weight" in fallback_names


def test_tied_aliases_route_to_fallback_if_any_alias_is_head_or_embedding() -> None:
    shared = nn.Parameter(torch.randn(16, 8))
    groups = build_param_groups(
        [
            ("blocks.0.mlp.weight", shared),
            ("lm_head.weight", shared),
        ]
    )

    matrix_names = [name for group in groups if group["use_matrix_update"] for name in group["param_names"]]
    fallback_names = [name for group in groups if not group["use_matrix_update"] for name in group["param_names"]]

    assert not matrix_names
    assert fallback_names == ["blocks.0.mlp.weight|lm_head.weight"]


def test_module_constructor_uses_safe_named_grouping() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    opt = SodaPmuonEqNorMuon(model)

    loss = _one_step(model, opt)

    assert torch.isfinite(torch.tensor(loss))
    assert opt.last_stats["matrix_count"] == 2.0
    assert opt.last_stats["fallback_count"] == 6.0


def test_named_parameter_constructor_uses_safe_grouping() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    opt = SodaPmuonEqNorMuon(model.named_parameters())

    loss = _one_step(model, opt)

    assert torch.isfinite(torch.tensor(loss))
    assert opt.last_stats["matrix_count"] == 2.0
    assert opt.last_stats["fallback_count"] == 6.0


def test_default_optimizer_step_is_finite() -> None:
    torch.manual_seed(1)
    model = TinyModel()
    opt = SodaPmuonEqNorMuon(build_param_groups(model.named_parameters()))

    for _ in range(3):
        loss = _one_step(model, opt)
        assert torch.isfinite(torch.tensor(loss))

    assert opt.last_stats["matrix_count"] == 2.0
    assert opt.last_stats["fallback_count"] == 6.0
    assert all(torch.isfinite(param).all() for param in model.parameters())


def test_default_fallback_weight_decay_is_active_when_soda_does_not_replace_it() -> None:
    p = nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = SodaPmuonEqNorMuon(
        [p],
        matrix_lr=0.1,
        fallback_lr=0.1,
        fallback_weight_decay=0.1,
        warmup_steps=1,
        soda_lambda_scale=0.0,
    )

    p.grad = torch.zeros_like(p)
    opt.step()

    assert torch.allclose(p, torch.tensor([0.99, -1.98]))


def test_disagreement_options_step_is_finite() -> None:
    torch.manual_seed(2)
    model = TinyModel()
    opt = SodaPmuonEqNorMuon(
        build_param_groups(model.named_parameters()),
        col_gamma=0.05,
        normuon_aspect_scale=True,
        normuon_mode="orientation",
        ns_variant="classic_muon",
    )

    loss = _one_step(model, opt)
    assert torch.isfinite(torch.tensor(loss))
    assert opt.last_stats["matrix_count"] == 2.0


def test_state_dict_resume_matches_continuous_run() -> None:
    torch.manual_seed(3)
    base = TinyModel()
    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)
    opt_a = SodaPmuonEqNorMuon(build_param_groups(model_a.named_parameters()))
    opt_b = SodaPmuonEqNorMuon(build_param_groups(model_b.named_parameters()))

    torch.manual_seed(4)
    _one_step(model_a, opt_a)
    torch.manual_seed(4)
    _one_step(model_b, opt_b)

    model_c = copy.deepcopy(model_b)
    opt_c = SodaPmuonEqNorMuon(build_param_groups(model_c.named_parameters()))
    opt_c.load_state_dict(opt_b.state_dict())

    torch.manual_seed(5)
    _one_step(model_a, opt_a)
    torch.manual_seed(5)
    _one_step(model_c, opt_c)

    for p_a, p_c in zip(model_a.parameters(), model_c.parameters(), strict=True):
        assert torch.allclose(p_a, p_c, atol=0.0, rtol=0.0)
