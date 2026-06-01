import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


def _load_runner():
    repo = Path(__file__).resolve().parents[1]
    worker_root = repo / "workers/codex_noradam_confidence"
    for stale_root in (
        repo / "workers/codex_soda_pmuoneq_normuon",
        repo / "workers/codex_equimuse_normuon",
        worker_root,
    ):
        if str(stale_root) in sys.path:
            sys.path.remove(str(stale_root))
    sys.path.insert(0, str(worker_root))
    for module_name in ("golden_soda_pmuoneq_normuon", "models_vit5"):
        sys.modules.pop(module_name, None)
    path = worker_root / "experiments/run_cifar10_ablation.py"
    spec = importlib.util.spec_from_file_location("test_cifar_runner_args_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_cifar_runner_args_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cifar_runner_defaults_to_train_split_validation(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", ["runner"])

    args = runner.parse_args()

    assert args.val_source == "train_split"
    runner.validate_experiment_args(args)


def test_cifar_runner_blocks_supervisor_hpo_on_test_split(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", ["runner", "--val-source", "test"])

    args = runner.parse_args()

    with pytest.raises(ValueError, match="not allowed"):
        runner.validate_experiment_args(args)


def test_cifar_runner_blocks_overlapping_validation_and_test(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", ["runner", "--worker", "--val-source", "test", "--eval-test"])

    args = runner.parse_args()

    with pytest.raises(ValueError, match="cannot be combined"):
        runner.validate_experiment_args(args)


def test_non_constant_anchor_schedules_are_trainer_owned():
    runner = _load_runner()

    wsd_anchor = runner.TrialConfig("anchor", "anchormuon", 1e-3, lr_schedule="wsd")
    constant_anchor = runner.TrialConfig("anchor", "anchormuon", 1e-3, lr_schedule="constant")
    sfplus_wsd = runner.TrialConfig("sfplus", "sfplus", 1e-3, lr_schedule="wsd")

    assert runner.trainer_owns_lr_schedule(wsd_anchor)
    assert not runner.trainer_owns_lr_schedule(constant_anchor)
    assert runner.trainer_owns_lr_schedule(sfplus_wsd)


def test_best_by_family_selects_mean_over_seed_luck(tmp_path):
    runner = _load_runner()

    def row(trial: str, seed: int, acc: float, loss: float) -> dict:
        return {
            "trial": trial,
            "optimizer": "anchormuon",
            "lr": 1e-3,
            "seed": seed,
            "lr_schedule": "wsd",
            "weight_decay": 0.05,
            "soda": "matrix",
            "pmuon_eq": True,
            "use_gram": True,
            "row_gamma": 0.25,
            "col_gamma": 0.0,
            "pmuon_beta": 0.95,
            "momentum": 0.95,
            "amuse": True,
            "mimuon": False,
            "mimuon_mix": 0.85,
            "normuon": False,
            "normuon_beta": 0.95,
            "normuon_aspect_scale": False,
            "root_grouping": "anchor",
            "root_normuon_mode": "row",
            "fallback_mode": "rms",
            "fallback_lr_mult": 1.0,
            "fallback_beta1": 0.9,
            "fallback_beta2": 0.95,
            "fallback_weight_decay": 0.0,
            "sfplus_polyak": False,
            "sfplus_c_warmup_enabled": False,
            "sfplus_beta_anneal": False,
            "sfplus_adamc_decay": False,
            "sfplus_inner_momentum": False,
            "sfplus_beta1": 0.9,
            "sfplus_beta1_max": 0.965,
            "sfplus_beta1_anneal_steps": 0,
            "sfplus_polyak_beta": 0.0,
            "sfplus_c_warmup": 0,
            "sfplus_r": 0.0,
            "sfplus_weight_lr_power": 2.0,
            "external_lr": False,
            "ema_beta": 0.0,
            "ema_gamma": 0.99,
            "ema_warmup_frac": 0.3,
            "ema_rest_frac": 0.2,
            "muown_mag_lr_mult": 1.0,
            "best_val_acc": acc,
            "best_val_loss": loss,
        }

    rows = [
        row("anchormuon_lucky_seed1", 1, 90.0, 0.20),
        row("anchormuon_lucky_seed2", 2, 70.0, 0.80),
        row("anchormuon_stable_seed1", 1, 82.0, 0.40),
        row("anchormuon_stable_seed2", 2, 82.0, 0.40),
    ]
    for item in rows:
        trial_dir = tmp_path / str(item["trial"])
        trial_dir.mkdir()
        (trial_dir / "summary.json").write_text(json.dumps(item) + "\n")

    runner.summarize(tmp_path, make_plots=False)

    best_by_family = json.loads((tmp_path / "best_by_family.json").read_text())
    selected = best_by_family["anchormuon_full"]
    assert selected["trial"] == "anchormuon_stable"
    assert selected["seed"] == ""
    assert selected["hpo_selected_by"] == "mean_over_seeds"
    assert selected["hpo_seed_count"] == 2
    assert selected["hpo_best_val_acc_mean"] == 82.0


def test_test_val_source_train_subset_uses_split_seed_not_trial_seed(monkeypatch, tmp_path):
    runner = _load_runner()

    class FakeCIFAR10(torch.utils.data.Dataset):
        def __init__(self, root, train=True, transform=None, download=False):
            self.train = train
            self.transform = transform
            self.size = 50 if train else 20

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            image = torch.zeros(3, 32, 32)
            label = int(index % 10)
            return image, label

    monkeypatch.setattr(runner.datasets, "CIFAR10", FakeCIFAR10)

    def make_args(seed: int) -> Namespace:
        return Namespace(
            data_path=tmp_path,
            seed=seed,
            split_seed=77,
            val_source="test",
            train_val_size=10,
            train_subset=12,
            val_subset=8,
            test_subset=0,
            batch_size=4,
            num_workers=0,
            eval_test=False,
        )

    train_a, _, _, info_a = runner.cifar10_loaders(make_args(seed=1))
    train_b, _, _, info_b = runner.cifar10_loaders(make_args(seed=999))

    assert info_a["split_seed"] == 77
    assert info_b["split_seed"] == 77
    assert train_a.dataset.indices == train_b.dataset.indices


def test_final_epoch_row_guard_reports_empty_training():
    runner = _load_runner()

    with pytest.raises(RuntimeError, match="No training epoch completed"):
        runner.final_epoch_row_or_raise([])

    assert runner.final_epoch_row_or_raise([{"epoch": 0, "val_acc": 12.5}])["val_acc"] == 12.5
