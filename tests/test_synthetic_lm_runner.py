import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "workers/codex_noradam_confidence/experiments/run_synthetic_llm50m.py"
    spec = importlib.util.spec_from_file_location("test_synthetic_lm_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_synthetic_lm_runner"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_trials_from_hpo_skips_families_missing_after_truncation():
    runner = _load_runner()
    args = SimpleNamespace(final_steps=10, eval_bins=2, warmup_steps=4, seed=123)
    hpo_rows = [
        {
            "name": "adamw_lr0.001",
            "family": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.05,
            "best_val_loss": 2.0,
        }
    ]

    trials = runner.final_trials_from_hpo(args, hpo_rows)

    assert len(trials) == 1
    assert trials[0].family == "adamw"
    assert trials[0].name == "final_adamw_lr0.001"


def test_final_trials_from_hpo_ignores_nan_best_loss_when_finite_exists():
    runner = _load_runner()
    args = SimpleNamespace(final_steps=10, eval_bins=2, warmup_steps=4, seed=123)
    hpo_rows = [
        {"name": "bad_nan", "family": "muon", "lr": 0.1, "best_val_loss": "nan"},
        {"name": "good_finite", "family": "muon", "lr": 0.2, "best_val_loss": 1.0},
    ]

    trials = runner.final_trials_from_hpo(args, hpo_rows)

    assert [trial.name for trial in trials] == ["final_good_finite"]


def test_final_trials_from_hpo_skips_family_with_only_nonfinite_loss():
    runner = _load_runner()
    args = SimpleNamespace(final_steps=10, eval_bins=2, warmup_steps=4, seed=123)
    hpo_rows = [
        {"name": "bad_nan", "family": "muon", "lr": 0.1, "best_val_loss": "nan"},
    ]

    trials = runner.final_trials_from_hpo(args, hpo_rows)

    assert trials == []


def test_write_summary_supports_minimal_final_only_hpo_rows(tmp_path):
    runner = _load_runner()
    args = SimpleNamespace(
        output_dir=tmp_path,
        n_layer=1,
        n_embd=8,
        n_head=1,
        block_size=8,
        vocab_size=32,
        batch_size=2,
        hpo_steps=4,
        final_steps=8,
    )
    hpo_rows = [
        {"name": "selected_muon", "family": "muon", "lr": "0.001", "best_val_loss": "1.0"},
    ]
    final_rows = [
        {
            "name": "final_selected_muon",
            "family": "muon",
            "lr": 0.001,
            "weight_decay": 0.05,
            "param_count": 123,
            "final_val_loss": 0.9,
            "best_val_loss": 0.8,
            "final_val_acc": 0.25,
            "mean_step_time_ms": 1.2,
            "tokens_per_sec": 4096,
        }
    ]

    runner.write_summary(args, hpo_rows, final_rows)

    report = (tmp_path / "summary.md").read_text()
    assert "`selected_muon`" in report
    assert "n/a" in report


def test_smoke_trials_finish_warmup():
    runner = _load_runner()
    args = SimpleNamespace(preset="smoke")

    trials = runner.hpo_trials(args)

    assert trials
    assert all(trial.steps > trial.warmup_steps for trial in trials)
    assert all(
        runner.lr_scale(trial.warmup_steps + 1, trial.steps, trial.warmup_steps, trial.final_lr_scale, trial.wsd_decay_frac)
        == 1.0
        for trial in trials
    )
