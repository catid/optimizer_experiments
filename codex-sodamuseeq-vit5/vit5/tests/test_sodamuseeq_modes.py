import itertools
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


def _batch():
    torch.manual_seed(123)
    return torch.randn(24, 12), torch.randint(0, 5, (24,))


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
