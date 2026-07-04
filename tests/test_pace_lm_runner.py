import importlib.util
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "workers/codex_noradam_confidence/experiments/run_wikitext_llm50m.py"
    spec = importlib.util.spec_from_file_location("test_pace_lm_runner_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_pace_lm_runner_mod"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sgd_param(value: float = 1.0, lr: float = 0.1) -> tuple[torch.nn.Parameter, torch.optim.SGD]:
    p = torch.nn.Parameter(torch.tensor([value]))
    return p, torch.optim.SGD([p], lr=lr)


def _load_reference_pace():
    pace_src = Path(__file__).resolve().parents[1] / "papers/code/pace/src"
    if str(pace_src) not in sys.path:
        sys.path.insert(0, str(pace_src))
    from pace.optimizer import PACE

    return PACE


def test_pace_wrapper_matches_reference_pace_adamw_update():
    runner = _load_runner()
    reference_pace = _load_reference_pace()
    torch.manual_seed(123)
    init = torch.randn(3, 4)
    grads = [torch.randn(3, 4) for _ in range(5)]

    p_ref = torch.nn.Parameter(init.clone())
    ref = reference_pace(
        [p_ref],
        lr=0.03,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.02,
        lambda_pullback=0.01,
        use_ema_eval=True,
        ema_kappa=0.5,
        ema_update_freq=2,
    )
    p_wrap = torch.nn.Parameter(init.clone())
    base = torch.optim.AdamW(
        [p_wrap],
        lr=0.03,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.02,
    )
    wrapped = runner.PaceOptimizer(
        base,
        pullback_c=0.01,
        kappa=0.5,
        precond="adam",
        beta2=0.999,
        update_freq=2,
    )

    for grad in grads:
        p_ref.grad = grad.clone()
        ref.step()
        p_wrap.grad = grad.clone()
        wrapped.step()

        assert torch.allclose(p_wrap.detach(), p_ref.detach(), atol=3e-7, rtol=0.0)
        assert torch.allclose(wrapped.state[p_wrap]["ema"], ref.state[p_ref]["ema"], atol=3e-7, rtol=0.0)


def test_pace_c0_matches_base_optimizer_trajectory():
    runner = _load_runner()
    torch.manual_seed(0)
    init = torch.randn(4, 3)
    grads = [torch.randn(4, 3) for _ in range(5)]

    p_ref = torch.nn.Parameter(init.clone())
    ref = torch.optim.SGD([p_ref], lr=0.05)
    p_pace = torch.nn.Parameter(init.clone())
    pace = runner.PaceOptimizer(torch.optim.SGD([p_pace], lr=0.05), pullback_c=0.0, kappa=0.5)

    for g in grads:
        p_ref.grad = g.clone()
        ref.step()
        p_pace.grad = g.clone()
        pace.step()

    assert torch.equal(p_ref.detach(), p_pace.detach())


def test_pace_pullback_matches_hand_computation():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=1.0, kappa=0.5, beta2=0.999)

    # Step 1: EMA starts at theta0, so the pullback is exactly zero.
    p.grad = torch.tensor([1.0])
    pace.step()
    assert float(p.detach()) == pytest.approx(0.9, abs=1e-6)
    decay1 = 2.0 ** -0.5
    ema1 = (1.0 - decay1) * 1.0 + decay1 * 0.9
    assert float(pace.state[p]["ema"]) == pytest.approx(ema1, abs=1e-6)

    # Step 2: v_hat is exactly g^2 = 1 by bias correction, so
    # lam = lr * c * (1+2)^-0.5 / (1 + eps) and the pullback is measured from
    # the pre-step weights (0.9), not the post-step weights (0.8).
    p.grad = torch.tensor([1.0])
    pace.step()
    decay2 = 3.0 ** -0.5
    lam = 0.1 * 1.0 * decay2 / (1.0 + 1e-8)
    expected = 0.8 + lam * (ema1 - 0.9)
    assert float(p.detach()) == pytest.approx(expected, abs=1e-6)


def test_pace_clamped_gain_transports_to_ema_plus_step():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=1e9, kappa=0.5)

    p.grad = torch.tensor([1.0])
    pace.step()
    ema1 = float(pace.state[p]["ema"])

    p.grad = torch.tensor([1.0])
    pace.step()
    # lam clamps to 1: theta2 = theta' + (ema1 - pre) = 0.8 + (ema1 - 0.9).
    assert float(p.detach()) == pytest.approx(0.8 + (ema1 - 0.9), abs=1e-6)


def test_pace_scalar_precond_uses_unpreconditioned_gain():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=1.0, kappa=0.5, precond="scalar")

    p.grad = torch.tensor([4.0])
    pace.step()
    ema1 = float(pace.state[p]["ema"])
    pre2 = float(p.detach())

    p.grad = torch.tensor([4.0])
    pace.step()
    lam = min(0.1 * 1.0 * 3.0 ** -0.5, 1.0)
    expected = (pre2 - 0.4) + lam * (ema1 - pre2)
    assert float(p.detach()) == pytest.approx(expected, abs=1e-6)
    assert "v" not in pace.state[p]


def test_pace_swap_round_trip_and_step_guard():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=0.01, kappa=0.5)
    for _ in range(3):
        p.grad = torch.tensor([1.0])
        pace.step()

    live = p.detach().clone()
    ema = pace.state[p]["ema"].clone()
    pace.swap_to_ema()
    assert float(p.detach()) == pytest.approx(float(ema), abs=1e-6)
    with pytest.raises(RuntimeError, match="swapped"):
        pace.step()
    pace.swap_to_live()
    assert torch.equal(p.detach(), live)


def test_pace_update_freq_gates_ema_but_not_pullback():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=1.0, kappa=0.5, update_freq=2)

    p.grad = torch.tensor([1.0])
    pace.step()
    # t=1 is not a multiple of uf=2, so the EMA still equals theta0.
    assert float(pace.state[p]["ema"]) == pytest.approx(1.0, abs=1e-6)

    p.grad = torch.tensor([1.0])
    pace.step()
    # Pullback still fired on step 2, measured against the frozen EMA.
    decay2 = 3.0 ** -0.5
    lam = 0.1 * decay2 / (1.0 + 1e-8)
    expected = 0.8 + lam * (1.0 - 0.9)
    assert float(p.detach()) == pytest.approx(expected, abs=1e-6)
    # And the EMA updated at t=2 from the post-pullback weights.
    ema2 = (1.0 - decay2) * 1.0 + decay2 * expected
    assert float(pace.state[p]["ema"]) == pytest.approx(ema2, abs=1e-6)


def test_pace_state_dict_remaps_state_to_new_parameters():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(
        base,
        pullback_c=0.5,
        kappa=0.3,
        precond="adam",
        beta2=0.95,
        update_freq=2,
    )
    for grad in (1.0, 0.5, -0.25):
        p.grad = torch.tensor([grad])
        pace.step()

    p2, base2 = _sgd_param(float(p.detach()), lr=0.1)
    pace2 = runner.PaceOptimizer(base2, pullback_c=0.0, kappa=0.5)
    pace2.load_state_dict(pace.state_dict())

    assert p2 in pace2.state
    assert p not in pace2.state
    assert pace2.step_index == pace.step_index
    assert pace2.pullback_c == pytest.approx(0.5)
    assert pace2.kappa == pytest.approx(0.3)
    assert pace2.beta2 == pytest.approx(0.95)
    assert pace2.update_freq == 2
    assert torch.equal(pace2.state[p2]["ema"], pace.state[p]["ema"])
    assert pace2.state[p2]["ema"].data_ptr() != pace.state[p]["ema"].data_ptr()

    p.grad = torch.tensor([0.75])
    p2.grad = torch.tensor([0.75])
    pace.step()
    pace2.step()
    assert float(p2.detach()) == pytest.approx(float(p.detach()), abs=1e-6)


def test_pace_state_dict_refreshes_public_param_groups_after_load():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=0.5, kappa=0.5)
    p.grad = torch.tensor([1.0])
    pace.step()
    state = pace.state_dict()

    p2, base2 = _sgd_param(float(p.detach()), lr=0.2)
    pace2 = runner.PaceOptimizer(base2, pullback_c=0.0, kappa=0.5)
    pace2.load_state_dict(state)

    assert pace2.param_groups is base2.param_groups
    pace2.param_groups[0]["lr"] = 0.03
    assert base2.param_groups[0]["lr"] == pytest.approx(0.03)


def test_pace_step_closure_initializes_wrapper_state():
    runner = _load_runner()
    p, base = _sgd_param(1.0, lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=1.0, kappa=0.5)

    def closure():
        base.zero_grad()
        loss = (p * p).sum()
        loss.backward()
        return loss

    loss = pace.step(closure=closure)

    assert float(loss.detach()) == pytest.approx(1.0)
    assert p in pace.state
    assert "ema" in pace.state[p]


def test_pace_inject_muon_state_dict_preserves_ema_state():
    runner = _load_runner()
    model = _tiny_matrix_model()
    matrix, fallback = runner.split_muon_groups(model, 0.0)
    opt = runner.PaceInjectMuon(matrix, fallback, lr=1e-2, weight_decay=0.0, pullback_c=0.1, kappa=0.5)
    _train_steps(model, opt, steps=3)

    state = copy.deepcopy(opt.state_dict())
    model2 = _tiny_matrix_model()
    model2.load_state_dict(copy.deepcopy(model.state_dict()))
    matrix2, fallback2 = runner.split_muon_groups(model2, 0.0)
    opt2 = runner.PaceInjectMuon(matrix2, fallback2, lr=1e-2, weight_decay=0.0, pullback_c=0.1, kappa=0.5)
    opt2.load_state_dict(state)

    assert opt2._pace_state
    opt2.swap_to_ema()
    assert opt2._swapped
    assert any(
        not torch.equal(p.detach(), live.detach())
        for p, live in zip(model2.parameters(), model.parameters(), strict=True)
    )


def _tiny_matrix_model(seed: int = 7) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(6, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 4),
    )


def _train_steps(model: torch.nn.Module, opt, steps: int = 3, seed: int = 11) -> None:
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(8, 6, generator=gen)
        y = torch.randn(8, 4, generator=gen)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_stackmuon_all_toggles_off_matches_plain_muon():
    runner = _load_runner()
    model_a = _tiny_matrix_model()
    model_b = _tiny_matrix_model()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.detach(), pb.detach())

    matrix_a, fallback_a = runner.split_muon_groups(model_a, 0.05)
    plain = runner.PlainMuon(matrix_a, fallback_a, lr=1e-2, weight_decay=0.05)
    matrix_b, fallback_b = runner.split_muon_groups(model_b, 0.05)
    stack = runner.StackMuon(matrix_b, fallback_b, lr=1e-2, weight_decay=0.05)

    _train_steps(model_a, plain)
    _train_steps(model_b, stack)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.detach(), pb.detach())


def test_stackmuon_soda_zeroes_decay_and_pulls_toward_anchor():
    runner = _load_runner()
    model = _tiny_matrix_model()
    matrix, fallback = runner.split_muon_groups(model, 0.05)
    stack = runner.StackMuon(
        matrix,
        fallback,
        lr=1e-2,
        weight_decay=0.05,
        use_soda=True,
        soda_lambda_scale=0.5,
    )
    assert all(float(group.get("weight_decay", 0.0)) == 0.0 for group in stack.param_groups)

    # Zero gradients throughout: momentum and the polar update stay zero, so
    # only the SODA pull can move parameters.
    for q in model.parameters():
        q.grad = torch.zeros_like(q)
    stack.step()
    p = next(model.parameters())
    anchor = stack._soda_anchors[id(p)].clone()
    assert torch.equal(p.detach(), anchor)

    with torch.no_grad():
        p.add_(1.0)
    before = p.detach().clone()
    stack.step()
    weight = min(1.0, 0.5 / 3.0)  # t=2 -> scale / (t+1)
    expected = before + (anchor - before) * weight
    assert torch.allclose(p.detach(), expected, atol=1e-6)


def test_stackmuon_pmuoneq_and_normuon_alter_updates_and_stay_finite():
    runner = _load_runner()
    variants = {}
    for key, kwargs in {
        "plain": {},
        "pmuoneq": {"row_gamma": 0.35},
        "normuon": {"use_normuon": True},
    }.items():
        model = _tiny_matrix_model()
        matrix, fallback = runner.split_muon_groups(model, 0.05)
        stack = runner.StackMuon(matrix, fallback, lr=1e-2, weight_decay=0.05, **kwargs)
        _train_steps(model, stack)
        variants[key] = (model, stack)
        for p in model.parameters():
            assert torch.isfinite(p).all()

    plain_first = next(variants["plain"][0].parameters()).detach()
    pm_model, pm_stack = variants["pmuoneq"]
    nm_model, nm_stack = variants["normuon"]
    assert not torch.equal(plain_first, next(pm_model.parameters()).detach())
    assert not torch.equal(plain_first, next(nm_model.parameters()).detach())
    pm_state = pm_stack.state[next(pm_model.parameters())]
    assert pm_state["pmuoneq_row_ema"].shape == (10,)
    nm_state = nm_stack.state[next(nm_model.parameters())]
    assert nm_state["normuon_second_momentum"].shape == (10, 1)


def test_make_optimizer_constructs_every_pace_family():
    runner = _load_runner()
    families = [
        "pace_adamw",
        "emaeval_adamw",
        "pace_muon",
        "emaeval_muon",
        "muon_soda",
        "pace_muon_soda",
        "muon_pmuoneq",
        "pace_muon_pmuoneq",
        "muon_normuon",
        "pace_muon_normuon",
        "pace_anchormuon",
    ]
    for family in families:
        model = _tiny_matrix_model()
        trial = runner.TrialConfig(
            f"unit_{family}",
            family,
            1e-3,
            row_gamma=0.25,
            soda_lambda_scale=0.003,
            pace_c=0.01,
            pace_kappa=0.5,
        )
        opt = runner.make_optimizer(model, trial)
        _train_steps(model, opt, steps=2)
        for p in model.parameters():
            assert torch.isfinite(p).all(), family
        if family.startswith(("pace_", "emaeval_")):
            opt.swap_to_ema()
            opt.swap_to_live()


def test_fineweb_pace_hpo_preset_covers_pace_ladder():
    runner = _load_runner()
    args = SimpleNamespace(
        preset="fineweb_pace_hpo",
        hpo_steps=600,
        eval_bins=8,
        warmup_steps=100,
        max_hpo_trials=0,
    )
    trials = runner.hpo_trials(args)
    families = {trial.family for trial in trials}

    assert {
        "adamw",
        "muon",
        "ema_muon",
        "anchormuon",
        "emaeval_adamw",
        "emaeval_muon",
        "pace_adamw",
        "pace_muon",
        "muon_soda",
        "pace_muon_soda",
        "muon_pmuoneq",
        "pace_muon_pmuoneq",
        "muon_normuon",
        "pace_muon_normuon",
        "pace_anchormuon",
    } <= families
    pace_muon_trials = [t for t in trials if t.family == "pace_muon"]
    assert {t.pace_precond for t in pace_muon_trials} == {"adam", "scalar"}
    const_trials = [t for t in trials if t.final_lr_scale == 1.0]
    assert const_trials and all(t.wsd_decay_frac == 0.0 for t in const_trials)
    emaeval_trials = [t for t in trials if t.family.startswith("emaeval_")]
    assert emaeval_trials and all(t.pace_c == 0.0 for t in emaeval_trials)
    assert all(trial.eval_every == 75 for trial in trials)
    assert all(trial.warmup_steps == 100 for trial in trials)
    assert len(trials) >= 200


def test_final_trials_from_hpo_carries_pace_and_schedule_fields():
    runner = _load_runner()
    args = SimpleNamespace(final_steps=100, eval_bins=4, warmup_steps=20, seed=7, final_top_per_family=1)
    hpo_rows = [
        {
            "name": "pace_best",
            "family": "pace_muon",
            "lr": 0.0014,
            "best_val_loss": 1.0,
            "pace_c": 0.01,
            "pace_kappa": 0.3,
            "pace_precond": "scalar",
            "pace_update_freq": "1",
            "final_lr_scale": 1.0,
            "wsd_decay_frac": 0.0,
        },
        {
            "name": "soda_best",
            "family": "pace_muon_soda",
            "lr": 0.0016,
            "best_val_loss": 1.1,
            "pace_c": 0.003,
            "soda_lambda_scale": 0.001,
        },
    ]

    trials = runner.final_trials_from_hpo(args, hpo_rows)
    by_family = {t.family: t for t in trials}

    pace = by_family["pace_muon"]
    assert pace.pace_c == 0.01
    assert pace.pace_kappa == 0.3
    assert pace.pace_precond == "scalar"
    assert pace.final_lr_scale == 1.0
    assert pace.wsd_decay_frac == 0.0
    soda = by_family["pace_muon_soda"]
    assert soda.pace_c == 0.003
    assert soda.soda_lambda_scale == 0.001
    assert soda.final_lr_scale == 0.1


def test_pace_rejects_bad_arguments():
    runner = _load_runner()
    p, base = _sgd_param()
    with pytest.raises(ValueError):
        runner.PaceOptimizer(base, pullback_c=-1.0, kappa=0.5)
    with pytest.raises(ValueError):
        runner.PaceOptimizer(base, pullback_c=0.1, kappa=0.0)
    with pytest.raises(ValueError):
        runner.PaceOptimizer(base, pullback_c=0.1, kappa=0.5, precond="rowwise")
    with pytest.raises(ValueError):
        runner.PaceOptimizer(base, pullback_c=0.1, kappa=0.5, update_freq=0)


def test_pace_row_precond_broadcasts_row_gain():
    runner = _load_runner()
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(3, 4))
    base = torch.optim.SGD([p], lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=1.0, kappa=0.5, precond="row", beta2=0.999)

    g = torch.zeros(3, 4)
    g[0].fill_(4.0)
    g[1].fill_(1.0)
    g[2].fill_(0.25)
    p.grad = g.clone()
    pace.step()
    ema1 = pace.state[p]["ema"].clone()
    pre2 = p.detach().clone()

    p.grad = g.clone()
    pace.step()
    # v_row bias-corrects to exactly mean(g_row^2); each row's gain is
    # lr * c * (1+2)^-0.5 / (sqrt(v_row) + eps), clamped at 1.
    decay2 = 3.0 ** -0.5
    for i, row_rms in enumerate((4.0, 1.0, 0.25)):
        lam = min(0.1 * decay2 / (row_rms + 1e-8), 1.0)
        expected = (pre2[i] - 0.1 * g[i]) + lam * (ema1[i] - pre2[i])
        assert torch.allclose(p.detach()[i], expected, atol=1e-5), f"row {i}"
    # Rows with weaker gradients get strictly stronger pullback.
    state = pace.state[p]
    v_hat = (state["v_row"] / (1.0 - 0.999**2)).sqrt()
    assert v_hat[2] < v_hat[1] < v_hat[0]


def test_pace_row_precond_uses_scalar_gain_for_vectors():
    runner = _load_runner()
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base = torch.optim.SGD([p], lr=0.1)
    pace = runner.PaceOptimizer(base, pullback_c=1.0, kappa=0.5, precond="row")

    for _ in range(2):
        p.grad = torch.tensor([1.0, 1.0])
        pace.step()
    assert "v_row" not in pace.state[p]
    assert torch.isfinite(p.detach()).all()


def test_pace_inject_muon_c0_matches_plain_muon():
    runner = _load_runner()
    model_a = _tiny_matrix_model()
    model_b = _tiny_matrix_model()

    matrix_a, fallback_a = runner.split_muon_groups(model_a, 0.05)
    plain = runner.PlainMuon(matrix_a, fallback_a, lr=1e-2, weight_decay=0.05)
    matrix_b, fallback_b = runner.split_muon_groups(model_b, 0.05)
    inject = runner.PaceInjectMuon(
        matrix_b, fallback_b, lr=1e-2, weight_decay=0.05, pullback_c=0.0, kappa=0.5
    )

    _train_steps(model_a, plain)
    _train_steps(model_b, inject)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.detach(), pb.detach())
    # EMA state exists and differs from the live weights (it lags them).
    first = next(model_b.parameters())
    ema = inject._pace_state[id(first)]["ema"]
    assert not torch.equal(ema, first.detach().to(torch.float32))


def test_pace_inject_muon_initializes_ema_from_pre_step_weights():
    runner = _load_runner()
    model = _tiny_matrix_model()
    initial = [p.detach().clone() for p in model.parameters()]
    matrix, fallback = runner.split_muon_groups(model, 0.05)
    inject = runner.PaceInjectMuon(
        matrix,
        fallback,
        lr=1e-2,
        weight_decay=0.05,
        pullback_c=0.0,
        kappa=0.5,
        update_freq=1,
    )

    _train_steps(model, inject, steps=1)
    live = [p.detach().clone() for p in model.parameters()]
    decay = 2.0 ** -0.5

    inject.swap_to_ema()

    assert any(not torch.equal(p.detach(), live_p) for p, live_p in zip(model.parameters(), live))
    for p, init_p, live_p in zip(model.parameters(), initial, live):
        expected = init_p.to(torch.float32).mul(1.0 - decay).add(live_p.to(torch.float32), alpha=decay)
        assert torch.allclose(p.detach().to(torch.float32), expected, atol=1e-6)


def test_pace_inject_muon_blends_pullback_into_polar_source():
    runner = _load_runner()
    model_a = _tiny_matrix_model()
    model_b = _tiny_matrix_model()

    matrix_a, fallback_a = runner.split_muon_groups(model_a, 0.05)
    zero = runner.PaceInjectMuon(matrix_a, fallback_a, lr=1e-2, weight_decay=0.05, pullback_c=0.0, kappa=0.5)
    matrix_b, fallback_b = runner.split_muon_groups(model_b, 0.05)
    inject = runner.PaceInjectMuon(matrix_b, fallback_b, lr=1e-2, weight_decay=0.05, pullback_c=1.0, kappa=0.5)

    # Step 1: EMA == theta0 for both, so injection has zero displacement and
    # the trajectories match exactly.
    _train_steps(model_a, zero, steps=1)
    _train_steps(model_b, inject, steps=1)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.detach(), pb.detach())

    # Step 2: displacement is now nonzero, so matrix params must diverge.
    _train_steps(model_a, zero, steps=1, seed=12)
    _train_steps(model_b, inject, steps=1, seed=12)
    diverged = any(
        not torch.equal(pa.detach(), pb.detach())
        for pa, pb in zip(model_a.parameters(), model_b.parameters())
        if pa.ndim >= 2
    )
    assert diverged
    for pb in model_b.parameters():
        assert torch.isfinite(pb).all()


def test_pace_inject_muon_swap_round_trip_and_guard():
    runner = _load_runner()
    model = _tiny_matrix_model()
    matrix, fallback = runner.split_muon_groups(model, 0.05)
    inject = runner.PaceInjectMuon(matrix, fallback, lr=1e-2, weight_decay=0.05, pullback_c=0.1, kappa=0.5)
    _train_steps(model, inject, steps=3)

    first = next(model.parameters())
    live = first.detach().clone()
    ema = inject._pace_state[id(first)]["ema"].clone()
    inject.swap_to_ema()
    assert torch.allclose(first.detach(), ema.to(first.dtype))
    with pytest.raises(RuntimeError, match="swapped"):
        inject.step()
    inject.swap_to_live()
    assert torch.equal(first.detach(), live)


def test_make_optimizer_constructs_inject_and_row_families():
    runner = _load_runner()
    for family, extra in (
        ("pace_inject_muon", {}),
        ("pace_muon", {"pace_precond": "row"}),
    ):
        model = _tiny_matrix_model()
        trial = runner.TrialConfig(
            f"unit_{family}_row", family, 1e-3, pace_c=0.01, pace_kappa=0.5, **extra
        )
        opt = runner.make_optimizer(model, trial)
        _train_steps(model, opt, steps=2)
        for p in model.parameters():
            assert torch.isfinite(p).all()
        opt.swap_to_ema()
        opt.swap_to_live()
