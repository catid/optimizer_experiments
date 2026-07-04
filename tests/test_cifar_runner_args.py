import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


def _load_aspect_runner():
    repo = Path(__file__).resolve().parents[1]
    worker_root = repo / "workers/codex_soda_pmuoneq_normuon"
    noradam_root = repo / "workers/codex_noradam_confidence"
    for stale_root in (
        repo / "workers/codex_equimuse_normuon",
        worker_root,
        noradam_root,
    ):
        if str(stale_root) in sys.path:
            sys.path.remove(str(stale_root))
    sys.path.insert(0, str(noradam_root))
    sys.path.insert(0, str(worker_root))
    for module_name in ("soda_pmuoneq_normuon", "models_vit5"):
        sys.modules.pop(module_name, None)
    path = worker_root / "experiments/run_cifar10_normuon_aspect_ablation.py"
    spec = importlib.util.spec_from_file_location("test_cifar_aspect_runner_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_cifar_aspect_runner_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cifar_runner_defaults_to_train_split_validation(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", ["runner"])

    args = runner.parse_args()

    assert args.val_source == "train_split"
    runner.validate_experiment_args(args)


def test_aspect_runner_validation_uses_train_holdout(monkeypatch, tmp_path):
    runner = _load_aspect_runner()

    class FakeCIFAR10(torch.utils.data.Dataset):
        def __init__(self, _root, *, train: bool, transform=None, download: bool = False):
            self.train = train
            self.transform = transform

        def __len__(self) -> int:
            return 20

        def __getitem__(self, index: int):
            return torch.zeros(3, 32, 32), index % 10

    monkeypatch.setattr(runner.datasets, "CIFAR10", FakeCIFAR10)
    args = Namespace(
        data_path=tmp_path,
        train_subset=0,
        val_subset=0,
        batch_size=4,
        num_workers=0,
        seed=123,
    )

    train_loader, val_loader = runner.make_loaders(args)

    assert train_loader.dataset.dataset.train is True
    assert val_loader.dataset.dataset.train is True
    assert set(train_loader.dataset.indices).isdisjoint(set(val_loader.dataset.indices))


def test_aspect_runner_rejects_zero_training_steps():
    runner = _load_aspect_runner()

    class EmptyLoader:
        def __len__(self) -> int:
            return 0

    args = Namespace(epochs=1, max_steps=0)

    with pytest.raises(RuntimeError, match="no training steps"):
        runner.total_training_steps(args, EmptyLoader())


def test_aspect_summarize_ranks_finite_metrics_before_nan(tmp_path):
    runner = _load_aspect_runner()

    rows = [
        {
            "name": "aaa_nan",
            "best_val_loss": float("nan"),
            "best_val_accuracy": float("nan"),
            "final_val_loss": float("nan"),
            "final_val_accuracy": float("nan"),
            "overall_examples_per_s": 1.0,
            "mean_step_s": 0.01,
        },
        {
            "name": "zzz_finite",
            "best_val_loss": 0.5,
            "best_val_accuracy": 0.8,
            "final_val_loss": 0.6,
            "final_val_accuracy": 0.7,
            "overall_examples_per_s": 1.0,
            "mean_step_s": 0.01,
        },
    ]
    for row in rows:
        trial_dir = tmp_path / str(row["name"])
        trial_dir.mkdir()
        (trial_dir / "summary.json").write_text(json.dumps(row) + "\n")

    runner.summarize(tmp_path)

    data_lines = [
        line
        for line in (tmp_path / "report.md").read_text().splitlines()
        if line.startswith("| aaa_") or line.startswith("| zzz_")
    ]
    assert data_lines[0].startswith("| zzz_finite ")


def test_cifar_ema_nesterov_state_dict_remaps_state_to_new_parameters():
    runner = _load_runner()
    p = torch.nn.Parameter(torch.tensor([1.0]))
    base = torch.optim.SGD([p], lr=0.1)
    opt = runner.EMANesterovOptimizer(
        base,
        total_steps=10,
        beta=1.0,
        gamma=0.0,
        warmup_frac=0.0,
        rest_frac=0.0,
    )
    opt.zero_grad()
    opt.state[p]["ema_delta"].fill_(0.5)
    state = opt.state_dict()

    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    base2 = torch.optim.SGD([p2], lr=0.1)
    opt2 = runner.EMANesterovOptimizer(
        base2,
        total_steps=10,
        beta=1.0,
        gamma=0.0,
        warmup_frac=0.0,
        rest_frac=0.0,
    )
    opt2.load_state_dict(state)

    assert p2 in opt2.state
    assert p not in opt2.state
    assert torch.equal(opt2.state[p2]["ema_delta"], opt.state[p]["ema_delta"])
    assert opt2.state[p2]["ema_delta"].data_ptr() != opt.state[p]["ema_delta"].data_ptr()
    opt2.zero_grad()
    assert float(p2.detach()) == pytest.approx(1.5)


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


def test_summarize_ignores_nan_best_metrics_when_finite_run_exists(tmp_path):
    runner = _load_runner()

    def row(trial: str, acc, loss) -> dict:
        return {
            "trial": trial,
            "optimizer": "muon",
            "lr": 1e-3,
            "seed": 1,
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

    for item in (row("aaa_nan", float("nan"), float("nan")), row("zzz_finite", 80.0, 0.5)):
        trial_dir = tmp_path / str(item["trial"])
        trial_dir.mkdir()
        (trial_dir / "summary.json").write_text(json.dumps(item) + "\n")

    runner.summarize(tmp_path, make_plots=False)

    best = json.loads((tmp_path / "best_run.json").read_text())
    by_family = json.loads((tmp_path / "best_by_family.json").read_text())
    assert best["trial"] == "zzz_finite"
    assert [row["trial"] for row in by_family.values()] == ["zzz_finite"]


def test_summarize_plots_skip_nan_speed_metrics(tmp_path, capsys):
    runner = _load_runner()

    def row(trial: str, avg_step_ms, best_acc: float) -> dict:
        return {
            "trial": trial,
            "optimizer": "muon",
            "lr": 1e-3,
            "seed": 1,
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
            "best_val_acc": best_acc,
            "best_val_loss": 0.5,
            "overall_examples_per_sec": 100.0,
            "avg_step_ms": avg_step_ms,
        }

    for item in (
        row("aaa_nan_speed", float("nan"), 80.0),
        row("zzz_finite_speed", 1.0, 81.0),
    ):
        trial_dir = tmp_path / str(item["trial"])
        trial_dir.mkdir()
        (trial_dir / "summary.json").write_text(json.dumps(item) + "\n")

    runner.summarize(tmp_path, make_plots=True)

    captured = capsys.readouterr()
    assert "plotting skipped" not in captured.out
    assert (tmp_path / "step_time_ms_bar.png").exists()


def test_launch_trials_reuses_freed_gpu(monkeypatch, tmp_path):
    runner = _load_runner()
    launches = []

    class FakeProc:
        def __init__(self, statuses):
            self.statuses = list(statuses)

        def poll(self):
            if len(self.statuses) > 1:
                return self.statuses.pop(0)
            return self.statuses[0]

    fake_processes = [
        FakeProc([None, 0]),
        FakeProc([0]),
        FakeProc([0]),
    ]

    def fake_popen(cmd, env=None, cwd=None):
        launches.append(env["CUDA_VISIBLE_DEVICES"])
        return fake_processes[len(launches) - 1]

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(runner, "ensure_cifar10_downloaded", lambda _path: None)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        data_path=tmp_path / "data",
        skip_completed=False,
        model="vit5_micro",
        epochs=1,
        max_steps=1,
        eval_bins=1,
        train_subset=8,
        val_subset=4,
        val_source="train_split",
        train_val_size=4,
        split_seed=1,
        test_subset=0,
        batch_size=2,
        num_workers=0,
        seed=1,
        warmup_steps=1,
        lr_final_scale=0.1,
        wsd_decay_frac=0.2,
        log_every=1,
        eval_test=False,
        sync_step_timing=False,
    )
    trials = [runner.TrialConfig(f"trial{i}", "adamw", 1e-3) for i in range(3)]

    runner.launch_trials(args, trials)

    assert launches == ["0", "1", "1"]


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
