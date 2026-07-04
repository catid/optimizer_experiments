import importlib.util
import sys
from pathlib import Path


def _load_aggregate_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "workers/codex_soda_pmuoneq_normuon/experiments/aggregate_cifar10_peer_feedback.py"
    spec = importlib.util.spec_from_file_location("test_peer_feedback_aggregate_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_peer_feedback_aggregate_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_summary(result_dir: Path, name: str, final_loss: str, best_loss: str) -> None:
    result_dir.mkdir()
    (result_dir / "summary.csv").write_text(
        "seed,name,optimizer,best_val_loss,final_val_loss,best_val_accuracy,"
        "final_val_accuracy,overall_examples_per_s,mean_step_s\n"
        f"1,{name},opt,{best_loss},{final_loss},0.8,0.75,1000,0.01\n",
        encoding="utf-8",
    )


def test_peer_feedback_aggregate_ranks_finite_losses_before_nan(tmp_path):
    aggregate = _load_aggregate_module()
    _write_summary(tmp_path / "bad", "bad", "nan", "nan")
    _write_summary(tmp_path / "good", "good", "0.5", "0.4")

    per_seed, rows = aggregate.aggregate_summaries([tmp_path / "bad", tmp_path / "good"])
    out = tmp_path / "out"
    (out / "figures").mkdir(parents=True)
    aggregate.write_report(out, rows, per_seed)

    assert [row["name"] for row in rows] == ["good", "bad"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Best mean final validation loss: **good**" in report
    assert "| 1 | good | 0.5000 | good | 0.4000 |" in report
