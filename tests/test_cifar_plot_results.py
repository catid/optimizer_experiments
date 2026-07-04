import importlib.util
import sys
from pathlib import Path


def _load_plot_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "workers/codex_soda_pmuoneq_normuon/experiments/plot_cifar10_normuon_results.py"
    spec = importlib.util.spec_from_file_location("test_cifar_plot_results_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_cifar_plot_results_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cifar_plot_report_ranks_finite_loss_before_nan(tmp_path):
    plotter = _load_plot_module()
    (tmp_path / "figures").mkdir()
    summaries = [
        {
            "name": "aaa_nan",
            "best_val_loss": "nan",
            "best_val_accuracy": "nan",
            "final_val_loss": "nan",
            "final_val_accuracy": "nan",
            "overall_examples_per_s": "100",
            "mean_step_s": "0.01",
        },
        {
            "name": "zzz_finite",
            "best_val_loss": "0.4",
            "best_val_accuracy": "0.8",
            "final_val_loss": "0.5",
            "final_val_accuracy": "0.7",
            "overall_examples_per_s": "100",
            "mean_step_s": "0.01",
        },
    ]
    curves = [
        {"event": "bin", "trial": "aaa_nan", "step": "1", "val_accuracy": "0.1"},
        {"event": "bin", "trial": "zzz_finite", "step": "1", "val_accuracy": "0.7"},
    ]

    best_name = plotter.plot_accuracy_best_vs_adamw(tmp_path, curves, summaries)
    plotter.write_report(tmp_path, summaries, best_name)

    assert best_name == "zzz_finite"
    data_lines = [line for line in (tmp_path / "report.md").read_text().splitlines() if line.startswith("| 1 |")]
    assert data_lines[0].startswith("| 1 | zzz_finite |")
