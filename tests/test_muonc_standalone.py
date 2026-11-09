import math

import pytest
import torch

from muonc import AnchorMuon, MuonC, MuonGS, _matrix_view


def _model(dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 8), torch.nn.GELU(), torch.nn.Linear(8, 4)
    )
    return model.to(dtype)


def _backward(model: torch.nn.Module, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    dtype = next(model.parameters()).dtype
    x = torch.randn(16, 6, generator=generator).to(dtype)
    y = torch.randn(16, 4, generator=generator).to(dtype)
    for parameter in model.parameters():
        parameter.grad = None
    torch.nn.functional.mse_loss(model(x).float(), y.float()).backward()


def _plain(cls, model, **kwargs):
    """Plain-Muon direction so the radial algebra is easy to reason about."""

    return cls(model, use_soda=False, use_pmuoneq=False, use_normuon=False, **kwargs)


def test_muongs_is_an_alias_and_routes_by_shape() -> None:
    assert MuonGS is MuonC
    optimizer = MuonC(_model(), lr=1e-2)
    summary = optimizer.group_summary()
    assert [group["use_matrix_update"] for group in summary] == [True, False]
    assert summary[0]["param_names"] == ["0.weight", "2.weight"]
    assert summary[1]["param_names"] == ["0.bias", "2.bias"]


def test_output_head_names_use_the_fallback_path() -> None:
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    summary = MuonC(Net(), lr=1e-2).group_summary()
    assert summary[0]["param_names"] == ["body.weight"]
    assert "lm_head.weight" in summary[1]["param_names"]


def test_projected_step_is_frobenius_tangent_to_the_weight() -> None:
    model = _model()
    optimizer = _plain(MuonC, model, lr=1e-2)
    weights = [model[0].weight, model[2].weight]
    before = [w.detach().clone() for w in weights]
    _backward(model)
    optimizer.step()
    for w0, w1 in zip(before, weights):
        delta = w1.detach() - w0
        cosine = (delta * w0).sum() / (delta.norm() * w0.norm())
        assert delta.norm() > 0
        assert abs(float(cosine)) < 1e-5
        # Pythagoras: the norm can only grow at second order.
        assert float(w1.detach().norm() ** 2) == pytest.approx(
            float(w0.norm() ** 2 + delta.norm() ** 2), rel=1e-5
        )


def test_gs_projection_can_be_disabled_and_per_parameter_skipped() -> None:
    reference = _model()
    _backward(reference)
    base = _plain(AnchorMuon, reference, lr=1e-2)
    base.step()

    disabled_model = _model()
    _backward(disabled_model)
    _plain(MuonC, disabled_model, lr=1e-2, gs_projection=False).step()

    tagged_model = _model()
    tagged_model[0].weight._no_gs = True
    _backward(tagged_model)
    _plain(MuonC, tagged_model, lr=1e-2).step()

    torch.testing.assert_close(disabled_model[0].weight, reference[0].weight)
    torch.testing.assert_close(disabled_model[2].weight, reference[2].weight)
    torch.testing.assert_close(tagged_model[0].weight, reference[0].weight)
    assert not torch.allclose(tagged_model[2].weight, reference[2].weight)


@pytest.mark.parametrize("corrected", [True, False])
def test_decay_schedule(corrected: bool) -> None:
    base_lr, lr, weight_decay = 1e-2, 2.5e-3, 0.5

    def run(decay: float) -> torch.nn.Module:
        model = _model()
        optimizer = _plain(
            MuonC,
            model,
            lr=base_lr,
            matrix_weight_decay=decay,
            fallback_weight_decay=decay,
            corrected_decay=corrected,
        )
        for group in optimizer.param_groups:
            assert group["base_lr"] == base_lr
            group["lr"] = lr
        _backward(model)
        optimizer.step()
        return model

    init, no_decay, decayed = _model(), run(0.0), run(weight_decay)
    factor = 1.0 - weight_decay * (lr * lr / base_lr if corrected else lr)
    for name, w0 in init.named_parameters():
        step = dict(no_decay.named_parameters())[name] - w0
        expected = factor * w0 + step
        torch.testing.assert_close(
            dict(decayed.named_parameters())[name], expected, rtol=1e-5, atol=1e-7
        )


def test_corrected_decay_equals_standard_decay_at_peak_lr() -> None:
    results = []
    for corrected in (True, False):
        model = _model()
        optimizer = _plain(
            MuonC, model, lr=1e-2, matrix_weight_decay=0.1, corrected_decay=corrected
        )
        _backward(model)
        optimizer.step()
        results.append(model[0].weight.detach().clone())
    torch.testing.assert_close(results[0], results[1])


def test_no_weight_decay_tag_is_respected() -> None:
    model = _model()
    reference = _model()
    model[0].weight._no_weight_decay = True
    _backward(model)
    _backward(reference)
    _plain(MuonC, model, lr=1e-2, matrix_weight_decay=0.5).step()
    _plain(MuonC, reference, lr=1e-2, matrix_weight_decay=0.0).step()
    torch.testing.assert_close(model[0].weight, reference[0].weight)
    assert not torch.allclose(model[2].weight, reference[2].weight)


def test_bf16_parameters_keep_fp32_masters() -> None:
    model = _model(torch.bfloat16)
    optimizer = MuonC(model, lr=1e-4, matrix_weight_decay=0.01, fallback_weight_decay=0.01)
    start = model[0].weight.detach().float().clone()
    for step in range(20):
        _backward(model, seed=step)
        optimizer.step()
    for parameter in model.parameters():
        master = optimizer.state[parameter]["master_param"]
        assert master.dtype == torch.float32
        assert parameter.dtype == torch.bfloat16
        torch.testing.assert_close(parameter.detach().float(), master.to(torch.bfloat16).float())
    master = optimizer.state[model[0].weight]["master_param"]
    # Sub-ulp progress is retained in the master even where BF16 cannot move.
    assert not torch.equal(master, master.to(torch.bfloat16).float())
    assert not torch.equal(master, start)


def test_fp32_parameters_do_not_allocate_masters() -> None:
    model = _model()
    optimizer = MuonC(model, lr=1e-3)
    _backward(model)
    optimizer.step()
    assert all("master_param" not in optimizer.state[p] for p in model.parameters())


def test_state_dict_round_trip_preserves_fp32_state_and_trajectory() -> None:
    kwargs = dict(lr=1e-3, matrix_weight_decay=0.01, fallback_weight_decay=0.01)
    model = _model(torch.bfloat16)
    optimizer = MuonC(model, **kwargs)
    for step in range(3):
        _backward(model, seed=step)
        optimizer.step()

    restored_model = _model(torch.bfloat16)
    restored_model.load_state_dict(model.state_dict())
    restored = MuonC(restored_model, **kwargs)
    restored.load_state_dict(optimizer.state_dict())
    for parameter in restored_model.parameters():
        for key in ("master_param", "soda_z0"):
            assert restored.state[parameter][key].dtype == torch.float32
    assert restored.param_groups[0]["k"] == 3

    _backward(model, seed=7)
    _backward(restored_model, seed=7)
    optimizer.step()
    restored.step()
    for a, b in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(
            optimizer.state[a]["master_param"], restored.state[b]["master_param"]
        )


def test_stacked_expert_bank_is_projected_per_matrix() -> None:
    torch.manual_seed(3)
    bank = torch.nn.Parameter(torch.randn(3, 5, 4))
    assert _matrix_view(bank).shape == (3, 5, 4)
    before = bank.detach().clone()
    bank.grad = torch.randn(3, 5, 4)
    optimizer = _plain(MuonC, [bank], lr=1e-2)
    optimizer.step()
    delta = bank.detach() - before
    per_matrix = (delta * before).sum(dim=(-2, -1))
    assert float(per_matrix.abs().max()) < 1e-5
    assert tuple(optimizer.state[bank]["pmuoneq_row_ema"].shape) == (3, 5)


def test_gs_stats_report_the_removed_radial_component() -> None:
    model = _model()
    optimizer = _plain(MuonC, model, lr=1e-2, track_gs_stats=True)
    _backward(model)
    optimizer.step()
    stats = optimizer.last_stats
    cosine_max = stats["gs_radial_cosine_abs_max"]
    assert 0.0 < cosine_max <= 1.0
    assert abs(stats["gs_radial_cosine_mean"]) <= cosine_max
    ratio = stats["gs_tangent_norm_ratio_mean"]
    assert math.sqrt(1.0 - cosine_max**2) - 1e-5 <= ratio <= 1.0 + 1e-6

    quiet = _plain(MuonC, _model(), lr=1e-2)
    assert "gs_radial_cosine_mean" not in quiet.last_stats


def test_invalid_gs_eps_is_rejected() -> None:
    with pytest.raises(ValueError, match="gs_eps"):
        MuonC(_model(), gs_eps=0.0)
