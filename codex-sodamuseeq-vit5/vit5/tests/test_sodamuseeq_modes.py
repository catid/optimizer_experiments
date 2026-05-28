import itertools
import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim_sodamuseeq import SodaMuseEq, make_sodamuseeq_param_groups


class TinyViTLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(12, 16)
        self.norm = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.head = nn.Linear(16, 5)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.norm(x + self.mlp(self.norm(x)))
        return self.head(x)


class SameShapeBucketModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(8, 8, bias=False)
        self.proj_b = nn.Linear(8, 8, bias=False)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        x = torch.tanh(self.proj_a(x) + self.proj_b(x))
        return self.head(self.norm(x))


def _batch():
    torch.manual_seed(123)
    return torch.randn(24, 12), torch.randint(0, 5, (24,))


def _same_shape_batch():
    torch.manual_seed(124)
    return torch.randn(24, 8), torch.randint(0, 5, (24,))


def _step(model, opt, x, y):
    opt.train()
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    return float(loss.detach())


def _make_opt(model, **kwargs):
    groups = make_sodamuseeq_param_groups(model, lr=1e-2, weight_decay=0.01)
    return SodaMuseEq(
        groups,
        warmup_steps=2,
        soda_warmup_steps=1,
        stats_interval=1,
        projection_dtype=torch.float32,
        **kwargs,
    )


def _clone_model_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_models_close(model_a, model_b, *, atol=1e-6, rtol=1e-6):
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=atol, rtol=rtol)


def test_all_ablation_modes_run_without_nan():
    x, y = _batch()
    flags = ["use_soda", "use_amuse", "use_pmuoneq", "use_gram", "use_mimuon", "use_normuon"]
    for values in itertools.product([False, True], repeat=len(flags)):
        torch.manual_seed(7)
        model = TinyViTLike()
        kwargs = dict(zip(flags, values, strict=True))
        opt = _make_opt(model, **kwargs)
        losses = [_step(model, opt, x, y) for _ in range(3)]
        assert all(torch.isfinite(torch.tensor(losses)))
        stats = opt.get_last_stats()
        assert stats["matrix_count"] == 3.0
        assert stats["fallback_count"] == 7.0
        if kwargs["use_pmuoneq"]:
            assert stats["pmuoneq_factor_count"] > 0.0
        else:
            assert stats["pmuoneq_factor_count"] == 0.0
        if kwargs["use_normuon"]:
            assert stats["normuon_applied_count"] > 0.0
        else:
            assert stats["normuon_applied_count"] == 0.0


def test_batch_projection_matches_individual_projection():
    x, y = _batch()
    torch.manual_seed(8)
    model_a = TinyViTLike()
    model_b = TinyViTLike()
    model_b.load_state_dict(model_a.state_dict())
    opt_a = _make_opt(model_a, use_soda=False, batch_project=True)
    opt_b = _make_opt(model_b, use_soda=False, batch_project=False)
    for _ in range(3):
        _step(model_a, opt_a, x, y)
        _step(model_b, opt_b, x, y)
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-6)


def test_same_shape_bucket_projection_matches_individual_projection():
    x, y = _same_shape_batch()
    torch.manual_seed(80)
    model_a = SameShapeBucketModel()
    model_b = SameShapeBucketModel()
    model_b.load_state_dict(model_a.state_dict())
    opt_a = SodaMuseEq(
        make_sodamuseeq_param_groups(model_a, lr=1e-2, weight_decay=0.01),
        warmup_steps=2,
        soda_warmup_steps=1,
        stats_interval=1,
        projection_dtype=torch.float32,
        batch_project=True,
    )
    opt_b = SodaMuseEq(
        make_sodamuseeq_param_groups(model_b, lr=1e-2, weight_decay=0.01),
        warmup_steps=2,
        soda_warmup_steps=1,
        stats_interval=1,
        projection_dtype=torch.float32,
        batch_project=False,
    )
    for _ in range(4):
        _step(model_a, opt_a, x, y)
        _step(model_b, opt_b, x, y)
    _assert_models_close(model_a, model_b)


def test_train_eval_roundtrip_restores_train_weights():
    x, y = _batch()
    torch.manual_seed(9)
    model = TinyViTLike()
    opt = _make_opt(model)
    for _ in range(3):
        _step(model, opt, x, y)
    train_weights = [p.detach().clone() for p in model.parameters()]
    opt.eval()
    eval_weights = [p.detach().clone() for p in model.parameters()]
    opt.train()
    restored = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(train_weights, eval_weights, strict=True))
    for a, b in zip(train_weights, restored, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_repeated_train_eval_roundtrips_are_idempotent():
    x, y = _batch()
    torch.manual_seed(91)
    model = TinyViTLike()
    opt = _make_opt(model)
    for _ in range(4):
        _step(model, opt, x, y)
    train_weights = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        assert opt.eval() is opt
        assert opt.train() is opt
        restored = [p.detach().clone() for p in model.parameters()]
        for expected, actual in zip(train_weights, restored, strict=True):
            assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_state_dict_resume_matches_uninterrupted_train_checkpoint():
    x, y = _batch()
    torch.manual_seed(92)
    model_ref = TinyViTLike()
    model_ckpt = TinyViTLike()
    model_ckpt.load_state_dict(model_ref.state_dict())
    opt_ref = _make_opt(model_ref)
    opt_ckpt = _make_opt(model_ckpt)
    for _ in range(4):
        _step(model_ref, opt_ref, x, y)
        _step(model_ckpt, opt_ckpt, x, y)

    model_state = _clone_model_state(model_ckpt)
    opt_state = copy.deepcopy(opt_ckpt.state_dict())
    _step(model_ref, opt_ref, x, y)

    model_new = TinyViTLike()
    model_new.load_state_dict(model_state)
    opt_new = _make_opt(model_new)
    opt_new.load_state_dict(opt_state)
    assert opt_new.global_step == opt_ckpt.global_step
    _step(model_new, opt_new, x, y)
    _assert_models_close(model_ref, model_new)


def test_state_dict_resume_matches_uninterrupted_eval_checkpoint():
    x, y = _batch()
    torch.manual_seed(93)
    model_ref = TinyViTLike()
    model_ckpt = TinyViTLike()
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

    model_new = TinyViTLike()
    model_new.load_state_dict(model_state)
    opt_new = _make_opt(model_new)
    opt_new.load_state_dict(opt_state)
    assert not opt_new.train_mode
    _step(model_new, opt_new, x, y)
    _assert_models_close(model_ref, model_new)


def test_stats_interval_avoids_expensive_sync_stats_between_intervals():
    x, y = _batch()
    torch.manual_seed(10)
    model = TinyViTLike()
    opt = SodaMuseEq(
        make_sodamuseeq_param_groups(model, lr=1e-2, weight_decay=0.01),
        warmup_steps=2,
        soda_warmup_steps=1,
        stats_interval=3,
        projection_dtype=torch.float32,
    )
    _step(model, opt, x, y)
    assert torch.isfinite(torch.tensor(opt.get_last_stats()["stiefel_defect"]))
    _step(model, opt, x, y)
    assert torch.isnan(torch.tensor(opt.get_last_stats()["stiefel_defect"]))
    _step(model, opt, x, y)
    assert torch.isfinite(torch.tensor(opt.get_last_stats()["stiefel_defect"]))


def test_soda_replaces_weight_decay_flag_changes_update():
    x, y = _batch()
    torch.manual_seed(11)
    model_pure = TinyViTLike()
    model_hybrid = TinyViTLike()
    model_hybrid.load_state_dict(model_pure.state_dict())
    groups_pure = make_sodamuseeq_param_groups(model_pure, lr=1e-2, weight_decay=0.2)
    groups_hybrid = make_sodamuseeq_param_groups(model_hybrid, lr=1e-2, weight_decay=0.2)
    opt_pure = SodaMuseEq(groups_pure, warmup_steps=2, soda_warmup_steps=99, stats_interval=1, projection_dtype=torch.float32)
    opt_hybrid = SodaMuseEq(
        groups_hybrid,
        warmup_steps=2,
        soda_warmup_steps=99,
        stats_interval=1,
        projection_dtype=torch.float32,
        soda_replaces_weight_decay=False,
    )
    _step(model_pure, opt_pure, x, y)
    _step(model_hybrid, opt_hybrid, x, y)
    assert opt_pure.get_last_stats()["soda_replaces_weight_decay"] == 1.0
    assert opt_hybrid.get_last_stats()["soda_replaces_weight_decay"] == 0.0
    assert any(
        not torch.allclose(pure, hybrid)
        for pure, hybrid in zip(model_pure.parameters(), model_hybrid.parameters(), strict=True)
    )
