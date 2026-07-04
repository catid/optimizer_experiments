import pytest
import torch

from pace_muon import PaceMuon


def _train_step(model: torch.nn.Module, optimizer: PaceMuon) -> None:
    optimizer.zero_grad()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()


def test_pace_muon_steps_and_tracks_ema_for_matrix_and_fallback_params() -> None:
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU(), torch.nn.Linear(3, 2))
    optimizer = PaceMuon(model, lr=1e-2, weight_decay=0.01, pullback_c=1e-3, kappa=0.5)

    _train_step(model, optimizer)
    _train_step(model, optimizer)

    assert optimizer.step_index == 2
    for parameter in model.parameters():
        assert "pace_ema" in optimizer.state[parameter]
        assert optimizer.state[parameter]["pace_ema"].shape == parameter.shape


def test_pace_muon_ema_swap_round_trips_and_guards_step() -> None:
    torch.manual_seed(456)
    model = torch.nn.Linear(4, 2)
    optimizer = PaceMuon(model, lr=1e-2, weight_decay=0.0, pullback_c=1e-3, kappa=0.5)
    _train_step(model, optimizer)
    _train_step(model, optimizer)

    live = [p.detach().clone() for p in model.parameters()]
    with optimizer.use_ema_weights():
        ema = [p.detach().clone() for p in model.parameters()]
        assert any(not torch.allclose(a, b) for a, b in zip(live, ema))
        with pytest.raises(RuntimeError, match="swapped"):
            optimizer.step()

    for parameter, expected in zip(model.parameters(), live):
        assert torch.allclose(parameter, expected)


def test_pace_muon_state_dict_restores_global_step_and_ema() -> None:
    torch.manual_seed(789)
    model = torch.nn.Linear(4, 2)
    optimizer = PaceMuon(model, lr=1e-2, weight_decay=0.0, pullback_c=1e-3, kappa=0.5)
    _train_step(model, optimizer)
    _train_step(model, optimizer)

    restored_model = torch.nn.Linear(4, 2)
    restored_model.load_state_dict(model.state_dict())
    restored = PaceMuon(restored_model, lr=1e-2, weight_decay=0.0, pullback_c=0.0, kappa=0.7)
    restored.load_state_dict(optimizer.state_dict())

    assert restored.step_index == optimizer.step_index
    assert restored.pullback_c == pytest.approx(optimizer.pullback_c)
    assert restored.kappa == pytest.approx(optimizer.kappa)
    for original_param, restored_param in zip(model.parameters(), restored_model.parameters()):
        assert torch.allclose(
            optimizer.state[original_param]["pace_ema"],
            restored.state[restored_param]["pace_ema"],
        )
