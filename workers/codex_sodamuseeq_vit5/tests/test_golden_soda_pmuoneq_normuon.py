import copy
import inspect
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden_soda_pmuoneq_normuon import (  # noqa: E402
    GOLDEN_CONFIG,
    GoldenSodaPmuonEqNorMuon,
    build_golden_param_groups,
)
from vit5.optim_sodamuseeq import SodaMuseEq, make_sodamuseeq_param_groups  # noqa: E402


class TinyLMish(nn.Module):
    def __init__(self, tied: bool = False):
        super().__init__()
        self.wte = nn.Embedding(32, 12)
        self.patch_embed = nn.Linear(12, 12, bias=False)
        self.block = nn.Sequential(nn.Linear(12, 24), nn.GELU(), nn.Linear(24, 12))
        self.norm = nn.LayerNorm(12)
        self.lm_head = nn.Linear(12, 32, bias=False)
        if tied:
            self.lm_head.weight = self.wte.weight

    def forward(self, tokens):
        x = self.wte(tokens).mean(dim=1)
        x = self.patch_embed(x)
        x = self.norm(x + self.block(x))
        return self.lm_head(x)


def _batch():
    torch.manual_seed(2026)
    return torch.randint(0, 32, (10, 5)), torch.randint(0, 32, (10,))


def _make_golden(model):
    return GoldenSodaPmuonEqNorMuon(
        build_golden_param_groups(model, lr=0.012, weight_decay=0.0),
        warmup_steps=4,
        soda_warmup_steps=2,
        projection_dtype=torch.float32,
        stats_interval=1,
    )


def _make_flexible_best(model):
    return SodaMuseEq(
        make_sodamuseeq_param_groups(model, lr=0.012, weight_decay=0.0),
        lr=0.012,
        weight_decay=0.0,
        momentum=0.95,
        beta1=0.6,
        beta2=0.999,
        rho=0.8,
        warmup_steps=4,
        use_soda=True,
        soda_warmup_steps=2,
        soda_replaces_weight_decay=True,
        use_amuse=False,
        use_pmuoneq=True,
        pmuon_beta=0.90,
        pmuon_gamma=0.0,
        pmuon_row_gamma=0.15,
        pmuon_col_gamma=0.0,
        use_gram=True,
        use_mimuon=False,
        use_normuon=True,
        normuon_beta2=0.90,
        normuon_mode="row",
        normuon_aspect_scale=True,
        batch_project=True,
        stats_interval=1,
        projection_dtype=torch.float32,
    )


def _step(model, opt, x, y):
    opt.train()
    opt.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    return float(loss.detach())


def _clone_model_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_models_close(a, b):
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-6)


def test_golden_config_and_constructor_have_no_ablation_switches():
    assert GOLDEN_CONFIG["algorithm"] == "SODA+PMuonEq+Gram+NorMuon row+aspect"
    assert GOLDEN_CONFIG["use_amuse"] is False
    assert GOLDEN_CONFIG["use_mimuon"] is False
    assert GOLDEN_CONFIG["use_normuon"] is True
    assert GOLDEN_CONFIG["normuon_aspect_scale"] is True
    params = inspect.signature(GoldenSodaPmuonEqNorMuon).parameters
    for name in ("use_amuse", "use_mimuon", "use_normuon", "use_gram", "normuon_aspect_scale", "pmuon_col_gamma"):
        assert name not in params


def test_named_grouping_keeps_lm_special_tensors_out_of_matrix_path():
    model = TinyLMish()
    groups = build_golden_param_groups(model)
    assert len(groups) == 2
    matrix_names = {getattr(p, "_sodamuseeq_param_name") for p in groups[0]["params"]}
    fallback_names = {getattr(p, "_sodamuseeq_param_name") for p in groups[1]["params"]}
    assert {"patch_embed.weight", "block.0.weight", "block.2.weight"} == matrix_names
    assert "wte.weight" in fallback_names
    assert "lm_head.weight" in fallback_names
    assert "norm.weight" in fallback_names
    assert "block.0.bias" in fallback_names


def test_tied_embedding_and_lm_head_are_not_grouped_twice():
    model = TinyLMish(tied=True)
    groups = build_golden_param_groups(model.named_parameters())
    params = [param for group in groups for param in group["params"]]
    assert len(params) == len({id(param) for param in params})
    names = [getattr(param, "_sodamuseeq_param_name") for group in groups for param in group["params"]]
    assert "wte.weight" in names
    assert "lm_head.weight" not in names


def test_golden_reproduces_flexible_best_update_path():
    x, y = _batch()
    torch.manual_seed(1)
    golden_model = TinyLMish()
    flexible_model = copy.deepcopy(golden_model)
    golden = _make_golden(golden_model)
    flexible = _make_flexible_best(flexible_model)

    for _ in range(5):
        loss_g = _step(golden_model, golden, x, y)
        loss_f = _step(flexible_model, flexible, x, y)
        assert abs(loss_g - loss_f) < 1e-7
        _assert_models_close(golden_model, flexible_model)

    stats = golden.get_last_stats()
    assert stats["matrix_count"] == 3.0
    assert stats["fallback_count"] == 6.0
    assert stats["pmuoneq_right_count"] == 0.0
    state = golden.state[golden_model.block[0].weight]
    assert "pmuoneq_left_ema" in state
    assert "pmuoneq_right_ema" not in state
    assert "normuon_second_moment" in state


def test_golden_state_dict_resume_matches_uninterrupted_training():
    x, y = _batch()
    torch.manual_seed(2)
    ref = TinyLMish()
    ckpt = copy.deepcopy(ref)
    opt_ref = _make_golden(ref)
    opt_ckpt = _make_golden(ckpt)
    for _ in range(4):
        _step(ref, opt_ref, x, y)
        _step(ckpt, opt_ckpt, x, y)

    model_state = _clone_model_state(ckpt)
    opt_state = copy.deepcopy(opt_ckpt.state_dict())
    _step(ref, opt_ref, x, y)

    restored = TinyLMish()
    restored.load_state_dict(model_state)
    opt_restored = _make_golden(restored)
    opt_restored.load_state_dict(opt_state)
    _step(restored, opt_restored, x, y)
    _assert_models_close(ref, restored)
