import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sodamuseeq import SodaMuseEq, make_sodamuseeq_param_groups


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.block = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 4)

    def forward(self, tokens):
        x = self.embed(tokens).mean(dim=1)
        x = self.norm(x + self.block(x))
        return self.head(x)


class WideModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.Linear(16, 8, bias=False)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        return self.head(torch.tanh(self.wide(x)))


def _batch():
    torch.manual_seed(123)
    return torch.randint(0, 16, (12, 5)), torch.randint(0, 4, (12,))


def _make_opt(model):
    return SodaMuseEq(
        make_sodamuseeq_param_groups(model, lr=1e-2, weight_decay=0.01),
        warmup_steps=2,
        soda_warmup_steps=1,
        projection_dtype=torch.float32,
    )


def _step(model, opt, x, y):
    opt.train()
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    return float(loss.detach())


def _clone_model_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_models_close(model_a, model_b):
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-6)


def test_named_group_helper_keeps_embed_norm_and_head_in_fallback():
    model = TinyModel()
    groups = make_sodamuseeq_param_groups(model, lr=1e-2, weight_decay=0.01)
    assert len(groups) == 2
    matrix_names = {getattr(p, "_sodamuseeq_param_name") for p in groups[0]["params"]}
    aux_names = {getattr(p, "_sodamuseeq_param_name") for p in groups[1]["params"]}
    assert {"block.0.weight", "block.2.weight"} == matrix_names
    assert {"embed.weight", "block.0.bias", "block.2.bias", "norm.weight", "norm.bias", "head.weight", "head.bias"} == aux_names


def test_standalone_normuon_aspect_and_orientation_controls():
    torch.manual_seed(6)
    base = TinyModel()
    row_aspect = copy.deepcopy(base)
    row_noaspect = copy.deepcopy(base)
    x, y = _batch()
    opt_aspect = SodaMuseEq(
        make_sodamuseeq_param_groups(row_aspect, lr=1e-2, weight_decay=0.01),
        warmup_steps=1,
        soda_warmup_steps=99,
        use_normuon=True,
        normuon_aspect_scale=True,
        projection_dtype=torch.float32,
    )
    opt_noaspect = SodaMuseEq(
        make_sodamuseeq_param_groups(row_noaspect, lr=1e-2, weight_decay=0.01),
        warmup_steps=1,
        soda_warmup_steps=99,
        use_normuon=True,
        normuon_aspect_scale=False,
        projection_dtype=torch.float32,
    )
    _step(row_aspect, opt_aspect, x, y)
    _step(row_noaspect, opt_noaspect, x, y)
    assert not torch.allclose(row_aspect.block[0].weight, row_noaspect.block[0].weight)

    wide = WideModel()
    opt_wide = SodaMuseEq(
        make_sodamuseeq_param_groups(wide, lr=1e-2, weight_decay=0.01),
        warmup_steps=1,
        soda_warmup_steps=99,
        use_normuon=True,
        normuon_mode="orientation",
        projection_dtype=torch.float32,
    )
    torch.manual_seed(66)
    xw = torch.randn(12, 16)
    yw = torch.randint(0, 4, (12,))
    _step(wide, opt_wide, xw, yw)
    second = opt_wide.state[wide.wide.weight]["normuon_second_moment"]
    assert tuple(second.shape) == (1, wide.wide.weight.shape[1])


def test_standalone_step_train_eval_and_stats_are_finite():
    x, y = _batch()
    torch.manual_seed(7)
    model = TinyModel()
    opt = _make_opt(model)
    losses = [_step(model, opt, x, y) for _ in range(4)]
    assert all(torch.isfinite(torch.tensor(losses)))
    train_weights = [p.detach().clone() for p in model.parameters()]
    assert opt.eval() is opt
    assert opt.train() is opt
    for expected, actual in zip(train_weights, model.parameters(), strict=True):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    stats = opt.get_last_stats()
    assert stats["matrix_count"] == 2.0
    assert stats["fallback_count"] == 7.0
    assert stats["soda_applied_count"] > 0.0


def test_standalone_state_dict_resume_matches_uninterrupted_checkpoint():
    x, y = _batch()
    torch.manual_seed(8)
    model_ref = TinyModel()
    model_ckpt = TinyModel()
    model_ckpt.load_state_dict(model_ref.state_dict())
    opt_ref = _make_opt(model_ref)
    opt_ckpt = _make_opt(model_ckpt)
    for _ in range(4):
        _step(model_ref, opt_ref, x, y)
        _step(model_ckpt, opt_ckpt, x, y)

    model_state = _clone_model_state(model_ckpt)
    opt_state = copy.deepcopy(opt_ckpt.state_dict())
    assert opt_state["sodamuseeq_extra"]["soda_step_idx"] == opt_ckpt.soda_step_idx
    _step(model_ref, opt_ref, x, y)

    model_new = TinyModel()
    model_new.load_state_dict(model_state)
    opt_new = _make_opt(model_new)
    opt_new.load_state_dict(opt_state)
    _step(model_new, opt_new, x, y)
    _assert_models_close(model_ref, model_new)


def test_standalone_eval_checkpoint_restores_train_path():
    x, y = _batch()
    torch.manual_seed(9)
    model_ref = TinyModel()
    model_ckpt = TinyModel()
    model_ckpt.load_state_dict(model_ref.state_dict())
    opt_ref = _make_opt(model_ref)
    opt_ckpt = _make_opt(model_ckpt)
    for _ in range(4):
        _step(model_ref, opt_ref, x, y)
        _step(model_ckpt, opt_ckpt, x, y)

    opt_ref.eval()
    opt_ckpt.eval()
    model_state = _clone_model_state(model_ckpt)
    opt_state = copy.deepcopy(opt_ckpt.state_dict())
    assert opt_state["sodamuseeq_extra"]["train_mode"] is False
    _step(model_ref, opt_ref, x, y)

    model_new = TinyModel()
    model_new.load_state_dict(model_state)
    opt_new = _make_opt(model_new)
    opt_new.load_state_dict(opt_state)
    assert not opt_new.train_mode
    _step(model_new, opt_new, x, y)
    _assert_models_close(model_ref, model_new)
