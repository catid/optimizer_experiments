"""End-to-end test of the training pipeline.

Imports and runs each stage on the small model: prepare data (``pace.helpers.data``),
expand a figure config and select one run (``pace_figures.reproduce``), build its config
from ``conf/``, and train it (``pace.helpers.train.run_training``).

The default mode uses a small data slice and a few steps to check that the pipeline runs
and reaches a reasonable loss. With ``PACE_TEST_FULL=1`` it trains on the full dataset and
asserts that the final validation loss matches the committed reference
(``tests/fixtures/results/``) within ``PACE_REPRO_TOL``.
"""
import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pace.helpers.data import prepare_sft
from pace.helpers.train import run_training
from pace_figures.reproduce import expand_figure

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "src" / "pace" / "helpers" / "conf"
FIG = REPO / "src" / "pace_figures" / "conf" / "fig" / "fig3_135m_grid.yaml"
REF = json.load(open(REPO / "tests" / "fixtures" / "results" / "smollm2_135m_pace.json"))

# the SmolLM2-135M fig3 cell whose number we reproduce (a real cell of the grid)
TARGET = {"lr": 0.005, "lambda_pullback": 0.03, "ema_kappa": 0.3}
FULL = os.environ.get("PACE_TEST_FULL") == "1"
TOL = float(os.environ.get("PACE_REPRO_TOL", "0.05"))
CACHE = Path(os.environ.get("PACE_TEST_DATA_CACHE", Path.home() / ".cache" / "pace_test"))


def _matches(curve):
    o = curve["optimizer"]
    return (o["name"] == "pace"
            and abs(o["lr"] - TARGET["lr"]) < 1e-9
            and abs(o["lambda_pullback"] - TARGET["lambda_pullback"]) < 1e-9
            and abs(o.get("ema_kappa", -1) - TARGET["ema_kappa"]) < 1e-9)


def _compose(curve, data_path, full):
    from hydra import compose, initialize_config_dir
    o = curve["optimizer"]
    overrides = [
        f"model={curve['model']}", f"optimizer={o['name']}",
        f"schedule={curve['schedule']}", f"seed={curve['seed']}",
        f"optimizer.lr={o['lr']}", f"optimizer.lambda_pullback={o['lambda_pullback']}",
        f"optimizer.ema_kappa={o['ema_kappa']}", f"dataset.path={data_path}",
    ]
    if full:  # match the reference run: full epoch, eval the full val set
        overrides += ["training.batch_size=32", "training.gradient_accumulation_steps=4",
                      "training.epochs=1", "training.eval_interval=1000"]
    else:     # quick: a few real steps on the small slice
        overrides += ["training.gradient_accumulation_steps=1", "training.max_steps=5",
                      "training.eval_interval=2", "training.eval_samples=64"]
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_pipeline_reproduces_fig_number(tmp_path):
    # The run we reproduce is a real cell of the figure yaml
    curve = next((c for c in expand_figure(OmegaConf.load(FIG)) if _matches(c)), None)
    assert curve is not None, "target cell not found in fig3_135m_grid.yaml"

    data = CACHE / ("full_smoltalk" if FULL else "pipeline_135m")
    if not (data / "train").exists():
        try:
            if FULL:
                prepare_sft(str(data), tokenizer_name="HuggingFaceTB/SmolLM2-135M")
            else:
                prepare_sft(str(data), tokenizer_name="HuggingFaceTB/SmolLM2-135M",
                            max_seq_len=256, max_train=1024, max_val=128)
        except Exception as e:
            pytest.skip(f"could not fetch model/data from HuggingFace: {e}")

    os.environ["PACE_RUN_DIR"] = str(tmp_path / "run")
    fv = run_training(_compose(curve, data, FULL))

    if FULL:
        expected = REF["summary"]["final_val_loss"]
        print(f"  reproduced final_val_loss={fv:.4f} vs reference {expected:.4f} (tol {TOL})")
        assert abs(fv - expected) <= TOL, f"{fv:.4f} != {expected:.4f} (> {TOL})"
    else:
        print(f"  pipeline ran, final_val_loss={fv:.4f}")
        assert fv == fv and fv < 100, f"bad final_val_loss {fv}"
