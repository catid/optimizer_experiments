import sys
from pathlib import Path

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

