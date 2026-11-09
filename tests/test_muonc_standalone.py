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
def test_decay_schedule_uses_each_groups_own_peak_lr(corrected: bool) -> None:
    # Different peak LRs and different schedule fractions per group, so a
    # fallback group that borrowed the matrix group's base_lr would be caught.
    peaks = {True: 1e-2, False: 4e-3}
    fractions = {True: 0.25, False: 0.5}
    weight_decay = 0.5

    def run(decay: float) -> torch.nn.Module:
        model = _model()
        optimizer = _plain(
            MuonC,
            model,
            lr=peaks[True],
            fallback_lr=peaks[False],
            matrix_weight_decay=decay,
            fallback_weight_decay=decay,
            corrected_decay=corrected,
        )
        for group in optimizer.param_groups:
            is_matrix = bool(group["use_matrix_update"])
            assert group["base_lr"] == peaks[is_matrix]
            group["lr"] = peaks[is_matrix] * fractions[is_matrix]
        _backward(model)
        optimizer.step()
        return model

    init, no_decay, decayed = _model(), run(0.0), run(weight_decay)
    for name, w0 in init.named_parameters():
        is_matrix = name.endswith("weight")
        lr, base_lr = peaks[is_matrix] * fractions[is_matrix], peaks[is_matrix]
        factor = 1.0 - weight_decay * (lr * lr / base_lr if corrected else lr)
        step = dict(no_decay.named_parameters())[name] - w0
        torch.testing.assert_close(
            dict(decayed.named_parameters())[name], factor * w0 + step, rtol=1e-5, atol=1e-7
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


def test_fp32_master_accumulates_updates_too_small_for_bf16() -> None:
    """A constant sub-ulp update moves a BF16 weight only when a master integrates it."""

    def run(cls) -> tuple[torch.Tensor, int]:
        weight = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        optimizer = _plain(cls, [weight], lr=2e-4)
        gradient = torch.eye(4, dtype=torch.bfloat16)
        first_change = -1
        for step in range(1, 81):
            weight.grad = gradient.clone()
            optimizer.step()
            if first_change < 0 and not torch.equal(weight.detach(), torch.ones_like(weight)):
                first_change = step
        return weight.detach().float(), first_change

    # The BF16 spacing just below 1.0 is 2**-8 = 0.0039; each step moves the
    # diagonal by lr * 0.2 * sqrt(4) = 8e-5, far below half of that.
    control, control_change = run(AnchorMuon)
    assert control_change == -1  # every in-place BF16 update rounds straight back
    assert torch.equal(control, torch.ones(4, 4))
    mastered, master_change = run(MuonC)
    assert master_change > 5  # nothing visible until enough sub-ulp steps accumulate
    assert float(mastered.diagonal().max()) < 1.0
    assert torch.equal(mastered - torch.diag(mastered.diagonal()), torch.ones(4, 4) - torch.eye(4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_precision_parameters_do_not_allocate_masters(dtype: torch.dtype) -> None:
    model = _model(dtype)
    optimizer = MuonC(model, lr=1e-3)
    _backward(model)
    optimizer.step()
    assert all("master_param" not in optimizer.state[p] for p in model.parameters())
    assert all(p.dtype == dtype for p in model.parameters())


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


@pytest.mark.parametrize("strength", [0.25, 0.5])
def test_partial_projection_removes_that_fraction_of_the_radial_step(strength: float) -> None:
    reference, model = _model(), _model()
    _backward(reference)
    _backward(model)
    before = [w.detach().clone() for w in (model[0].weight, model[2].weight)]
    _plain(MuonC, reference, lr=1e-2, gs_projection=False).step()
    _plain(MuonC, model, lr=1e-2, gs_projection=strength).step()
    for w0, w_ref, w in zip(before, (reference[0].weight, reference[2].weight), (model[0].weight, model[2].weight)):
        radial_ref = float(((w_ref.detach() - w0) * w0).sum())
        radial = float(((w.detach() - w0) * w0).sum())
        assert radial == pytest.approx((1.0 - strength) * radial_ref, rel=1e-4, abs=1e-7)


def test_projection_strength_one_is_the_boolean_projection() -> None:
    a, b = _model(), _model()
    _backward(a)
    _backward(b)
    MuonC(a, lr=1e-2, gs_projection=True).step()
    MuonC(b, lr=1e-2, gs_projection=1.0).step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    with pytest.raises(ValueError, match="gs_projection"):
        MuonC(_model(), gs_projection=1.5)


def test_equilibrium_rms_is_the_fixed_point_of_the_projected_norm() -> None:
    """Random tangent steps of known size + corrected decay settle at the predicted RMS."""

    lr, wd = 0.05, 0.5
    rms = MuonC.equilibrium_rms(lr, wd)
    assert MuonC.weight_decay_for_rms(lr, rms) == pytest.approx(wd)
    assert MuonC.equilibrium_rms(1.2e-3, 0.05) == pytest.approx(0.0219, abs=1e-4)
    torch.manual_seed(0)
    weight = torch.nn.Parameter(0.5 * torch.randn(24, 16))  # starts well above the fixed point
    optimizer = _plain(MuonC, [weight], lr=lr, matrix_weight_decay=wd)
    generator = torch.Generator().manual_seed(1)
    realised: list[float] = []
    for step in range(600):
        current = lr * (1.0 if step < 300 else 0.5)  # halving the LR must not move the fixed point
        for group in optimizer.param_groups:
            group["lr"] = current
        weight.grad = torch.randn(24, 16, generator=generator)
        before = weight.detach().clone()
        optimizer.step()
        applied = (before * (1.0 - wd * current**2 / lr) - weight.detach()) / current
        realised.append(float(applied.square().mean().sqrt()))
        if step in (299, 599):
            # The orthogonalization is approximate, so use the realised update RMS.
            update_rms = sum(realised[-100:]) / 100
            assert 0.10 < update_rms < 0.21  # ideal is 0.2; small random matrices orthogonalize less tightly
            measured = float(weight.detach().square().mean().sqrt())
            assert measured == pytest.approx(MuonC.equilibrium_rms(lr, wd, update_rms=update_rms), rel=0.03)


def test_invalid_gs_eps_is_rejected() -> None:
    with pytest.raises(ValueError, match="gs_eps"):
        MuonC(_model(), gs_eps=0.0)
