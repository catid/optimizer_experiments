import math
import sys
from pathlib import Path

import pytest
import torch

WORKER = Path(__file__).resolve().parents[1]
ROOT = WORKER.parents[1]
for path in (ROOT, WORKER, WORKER / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from muonc import MuonC  # noqa: E402
from muonc_research import MuonCResearch  # noqa: E402


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))


def _backward(model: torch.nn.Module, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 6, generator=generator)
    y = torch.randn(16, 4, generator=generator)
    for parameter in model.parameters():
        parameter.grad = None
    torch.nn.functional.mse_loss(model(x), y).backward()


PLAIN = dict(use_soda=False, use_pmuoneq=False, use_normuon=False)


def _weights(model):
    return [model[0].weight, model[2].weight]


def test_defaults_match_root_muonc_exactly() -> None:
    kwargs = dict(lr=3e-3, matrix_weight_decay=0.1, fallback_weight_decay=0.1)
    a, b = _model(), _model()
    opt_a, opt_b = MuonC(a, **kwargs), MuonCResearch(b, **kwargs)
    for step in range(4):
        for model, optimizer in ((a, opt_a), (b, opt_b)):
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * (0.5 if step > 1 else 1.0)
            _backward(model, seed=step)
            optimizer.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("strength", [0.0, 0.25, 1.0])
def test_partial_projection_removes_that_fraction_of_the_radial_step(strength: float) -> None:
    reference, model = _model(), _model()
    _backward(reference)
    _backward(model)
    before = [w.detach().clone() for w in _weights(model)]
    MuonCResearch(reference, lr=1e-2, gs_strength=0.0, **PLAIN).step()
    MuonCResearch(model, lr=1e-2, gs_strength=strength, **PLAIN).step()
    for w0, w_ref, w in zip(before, _weights(reference), _weights(model)):
        radial_ref = ((w_ref.detach() - w0) * w0).sum()
        radial = ((w.detach() - w0) * w0).sum()
        assert float(radial) == pytest.approx((1.0 - strength) * float(radial_ref), rel=1e-4, abs=1e-7)


def test_outward_mode_only_removes_norm_growing_components() -> None:
    torch.manual_seed(1)
    grow = torch.nn.Parameter(torch.randn(5, 4))
    shrink = torch.nn.Parameter(grow.detach().clone())
    grow.grad = -grow.detach().clone()  # descent step moves along +W: norm grows
    shrink.grad = shrink.detach().clone()  # descent step moves along -W: norm shrinks
    start = float(grow.detach().norm())
    MuonCResearch([grow, shrink], lr=1e-2, gs_mode="outward", **PLAIN).step()
    grown = float(grow.detach().norm()) / start - 1.0
    shrunk = 1.0 - float(shrink.detach().norm()) / start
    assert shrunk > 1e-3  # the inward radial step is kept
    assert 0.0 <= grown < 0.1 * shrunk  # the outward one is removed (second-order growth only)


def test_fixed_norm_mode_holds_the_initial_frobenius_norm() -> None:
    model = _model()
    start = [float(w.detach().norm()) for w in _weights(model)]
    optimizer = MuonCResearch(model, lr=2e-2, norm_mode="fixed", **PLAIN)
    for step in range(10):
        _backward(model, seed=step)
        optimizer.step()
    for w, r0 in zip(_weights(model), start):
        assert float(w.detach().norm()) == pytest.approx(r0, rel=1e-5)


def test_gs_norm_follows_the_closed_form_recursion() -> None:
    """|W_{t+1}|^2 = f_t^2 |W_t|^2 + lr_t^2 |N_t|^2 with the applied direction N_t."""

    model = _model()
    base_lr, wd = 2e-2, 0.5
    optimizer = MuonCResearch(model, lr=base_lr, matrix_weight_decay=wd, **PLAIN)
    names = {id(p): n for n, p in model.named_parameters()}
    for step in range(8):
        lr = base_lr * (1.0 - 0.1 * step)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * (lr / base_lr)
        before = {names[id(w)]: float(w.detach().norm()) for w in _weights(model)}
        _backward(model, seed=step)
        optimizer.diag_next = True
        optimizer.step()
        diag = optimizer.pop_diag()
        factor = 1.0 - wd * lr * lr / base_lr
        for w in _weights(model):
            name = names[id(w)]
            assert diag[name]["weight_norm"] == pytest.approx(before[name], rel=1e-5)
            predicted = math.sqrt((factor * before[name]) ** 2 + (lr * diag[name]["update_norm"]) ** 2)
            assert float(w.detach().norm()) == pytest.approx(predicted, rel=1e-4)


@pytest.mark.parametrize("law", ["standard", "corrected"])
def test_matrix_and_fallback_decay_laws_are_independent(law: str) -> None:
    base_lr, lr, wd = 1e-2, 2.5e-3, 0.5
    other = "corrected" if law == "standard" else "standard"

    def run(matrix_wd: float, fallback_wd: float) -> torch.nn.Module:
        model = _model()
        optimizer = MuonCResearch(
            model, lr=base_lr, matrix_weight_decay=matrix_wd, fallback_weight_decay=fallback_wd,
            matrix_decay_law=law, fallback_decay_law=other, gs_strength=0.0, **PLAIN,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        _backward(model)
        optimizer.step()
        return model

    init, plain, decayed = _model(), run(0.0, 0.0), run(wd, wd)
    factors = {"standard": 1.0 - wd * lr, "corrected": 1.0 - wd * lr * lr / base_lr}
    for name, w0 in init.named_parameters():
        factor = factors[law] if name.endswith("weight") else factors[other]
        expected = factor * w0 + (dict(plain.named_parameters())[name] - w0)
        torch.testing.assert_close(dict(decayed.named_parameters())[name], expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("soda", [False, True])
def test_unprojected_matrices_can_use_a_different_decay_law(soda: bool) -> None:
    """Order is SODA pull -> decay -> update, so compare against wanted * L - lr * N."""

    base_lr, lr, wd, steps = 1e-2, 2.5e-3, 0.5, 3
    kwargs = dict(use_pmuoneq=False, use_normuon=False, use_soda=soda, soda_lambda_scale=0.6)

    def run(law: str | None, matrix_law: str) -> torch.nn.Module:
        model = _model()
        model[0].weight._no_gs = True
        optimizer = MuonCResearch(
            model, lr=base_lr, matrix_weight_decay=wd, matrix_decay_law=matrix_law,
            unprojected_decay_law=law, **kwargs,
        )
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr
            _backward(model, seed=step)
            optimizer.step()
        return model

    mixed = run("standard", "corrected")
    all_standard = run(None, "standard")
    all_corrected = run(None, "corrected")
    # The unprojected matrix follows the standard law, the projected one the corrected law.
    # (Gradients differ slightly between runs after step 1, hence the loose tolerance on
    # the projected matrix; the unprojected matrix is compared after a single step below.)
    assert not torch.allclose(all_standard[0].weight, all_corrected[0].weight, rtol=1e-4, atol=0)

    def one_step(law: str | None, matrix_law: str) -> torch.nn.Module:
        model = _model()
        model[0].weight._no_gs = True
        optimizer = MuonCResearch(
            model, lr=base_lr, matrix_weight_decay=wd, matrix_decay_law=matrix_law,
            unprojected_decay_law=law, **kwargs,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        _backward(model)
        optimizer.step()
        return model

    mixed, all_standard, all_corrected = (
        one_step("standard", "corrected"), one_step(None, "standard"), one_step(None, "corrected")
    )
    torch.testing.assert_close(mixed[0].weight, all_standard[0].weight, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(mixed[2].weight, all_corrected[2].weight, rtol=1e-6, atol=1e-8)
    assert not torch.allclose(mixed[0].weight, all_corrected[0].weight, rtol=1e-5, atol=0)


@pytest.mark.parametrize("name", ["gs_projection", "corrected_decay", "track_gs_stats"])
def test_replaced_root_switches_are_rejected(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        MuonCResearch(_model(), **{name: True})


def test_norm_target_requires_the_full_projection() -> None:
    with pytest.raises(ValueError, match="full projection"):
        MuonCResearch(_model(), norm_target=1.0, gs_strength=0.5)


def test_plain_configuration_is_the_same_optimizer_as_the_repo_plain_muon() -> None:
    """gs=0, standard decay, Adam fallback at 1x LR == PlainMuon up to GramNS rounding.

    On CPU both orthogonalizers run in FP32; the repo uses the restarted Gram
    recurrence and muonc the direct polynomial, which agree in exact arithmetic.
    """

    codex = ROOT / "workers" / "codex_noradam_confidence" / "experiments"
    if str(codex) not in sys.path:
        sys.path.insert(0, str(codex))
    from run_muonc_gs_lm import Trial, make_optimizer
    from run_synthetic_llm50m import TinyGPT

    def run(opt: str) -> torch.nn.Module:
        torch.manual_seed(7)
        model = TinyGPT(vocab_size=97, block_size=12, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
        trial = Trial(name="t", opt=opt, lr=2e-3, wd=0.1, gs=0.0, decay="standard")
        optimizer = make_optimizer(model, trial)
        generator = torch.Generator().manual_seed(11)
        for _step in range(4):
            tokens = torch.randint(0, 97, (4, 13), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens[:, :-1])
            torch.nn.functional.cross_entropy(logits.reshape(-1, 97), tokens[:, 1:].reshape(-1)).backward()
            optimizer.step()
        return model

    reference, ported = run("repo_muon"), run("muonc")
    for (name, a), (_n, b) in zip(reference.named_parameters(), ported.named_parameters()):
        torch.testing.assert_close(b, a, rtol=2e-3, atol=2e-5, msg=lambda m, name=name: f"{name}: {m}")


def test_launcher_refuses_to_reuse_a_trial_run_with_other_settings(tmp_path) -> None:
    import json
    from dataclasses import asdict

    from run_muonc_gs_lm import Trial, launch

    old = Trial(name="same-name", lr=1e-3)
    trial_dir = tmp_path / "trials" / old.name
    trial_dir.mkdir(parents=True)
    (trial_dir / "trial.json").write_text(json.dumps(asdict(old)))
    (trial_dir / "summary.json").write_text("{}")
    launch([Trial(name="same-name", lr=1e-3, stage="other")], tmp_path, tmp_path, [0], 1)  # identical: reused
    with pytest.raises(ValueError, match="different settings"):
        launch([Trial(name="same-name", lr=2e-3)], tmp_path, tmp_path, [0], 1)


@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
def test_norm_target_converges_to_kappa_times_each_initial_norm(kappa: float) -> None:
    """Two matrices with very different initial scales each settle at kappa * |W_0|."""

    torch.manual_seed(0)
    small = torch.nn.Parameter(0.01 * torch.randn(6, 5))
    large = torch.nn.Parameter(1.0 * torch.randn(6, 5))
    starts = [float(p.detach().norm()) for p in (small, large)]
    optimizer = MuonCResearch([small, large], lr=1.0, norm_target=kappa, **PLAIN)
    generator = torch.Generator().manual_seed(1)
    for step in range(600):
        # A fresh random gradient each step keeps the direction norm stationary.
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * (0.02 if step < 500 else 0.01)
        for parameter in (small, large):
            parameter.grad = torch.randn(6, 5, generator=generator)
        optimizer.step()
    # |N| is 0.2 * sqrt(6) * sqrt(5) ~ 1.1, so the relaxation time at lr=0.02 is
    # ~ kappa^2 |W_0|^2 / (lr |N|)^2 steps: tiny for `small`, long for `large`.
    assert float(small.detach().norm()) == pytest.approx(kappa * starts[0], rel=0.05)
    ratio_large = float(large.detach().norm()) / starts[1]
    assert (ratio_large - 1.0) * (kappa - 1.0) >= 0.0  # moves toward kappa, never away
    assert abs(ratio_large - 1.0) <= abs(kappa - 1.0) + 1e-6


def test_norm_target_rejects_a_second_decay_source() -> None:
    with pytest.raises(ValueError, match="norm_target"):
        MuonCResearch(_model(), norm_target=1.0, matrix_weight_decay=0.1)


def test_radial_step_moves_only_the_norm_and_follows_the_scale_gradient() -> None:
    """With a loss that depends on |W| alone, c > 0 changes the norm and c = 0 cannot."""

    def run(c: float, sign: float) -> tuple[float, float]:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(6, 5))
        start = weight.detach().clone()
        optimizer = MuonCResearch([weight], lr=1e-2, radial_lr=c, **PLAIN)
        generator = torch.Generator().manual_seed(1)
        for _step in range(50):
            # dL/dW = sign * W / |W|  (pure scale gradient)  +  tangential noise.
            noise = torch.randn(6, 5, generator=generator)
            unit = weight.detach() / weight.detach().norm()
            noise = noise - (noise * unit).sum() * unit
            weight.grad = sign * unit + 0.5 * noise
            optimizer.step()
        ratio = float(weight.detach().norm() / start.norm())
        cosine = float((weight.detach() * start).sum() / (weight.detach().norm() * start.norm()))
        return ratio, cosine

    pinned, _ = run(0.0, +1.0)
    assert 1.0 <= pinned < 1.02  # projection alone: second-order growth only
    shrunk, _ = run(1.0, +1.0)  # loss decreases with smaller norm
    grown, _ = run(1.0, -1.0)  # loss decreases with larger norm
    assert shrunk < 0.75 and grown > 1.3
    # The tangential motion is the same optimizer either way.
    assert abs(run(1.0, +1.0)[1] - run(0.0, +1.0)[1]) < 0.05


def test_radial_step_needs_the_full_projection() -> None:
    with pytest.raises(ValueError, match="radial_lr"):
        MuonCResearch(_model(), radial_lr=1.0, gs_strength=0.5)


def test_runner_schedules_and_grid_names() -> None:
    import grids
    from run_muonc_gs_lm import Trial, lr_scale

    wsd = Trial(name="w", schedule="wsd", steps=1000, warmup_steps=100, final_lr_scale=0.1, decay_frac=0.2)
    assert lr_scale(wsd, 50) == pytest.approx(0.5)
    assert lr_scale(wsd, 800) == 1.0
    assert lr_scale(wsd, 1000) == pytest.approx(0.1)
    cosine = Trial(name="c", schedule="cosine", steps=1000, warmup_steps=100, final_lr_scale=0.1)
    assert lr_scale(cosine, 100) == 1.0
    assert lr_scale(cosine, 550) == pytest.approx(0.55)
    assert lr_scale(cosine, 1000) == pytest.approx(0.1)
    linear = Trial(name="l", schedule="linear", steps=1000, warmup_steps=0, final_lr_scale=0.0, decay_frac=0.85)
    assert lr_scale(linear, 150) == 1.0
    assert lr_scale(linear, 1000) == pytest.approx(0.0)
    seen: dict[str, dict] = {}
    for stage in grids.STAGES:
        trials = grids.stage_trials(stage)
        names = [t.name for t in trials]
        assert len(names) == len(set(names)), stage
        for trial in trials:
            config = {k: v for k, v in vars(trial).items() if k != "stage"}
            # A name may recur across stages only for the identical configuration.
            assert seen.setdefault(trial.name, config) == config, trial.name
