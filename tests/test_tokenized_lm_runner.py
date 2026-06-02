import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "workers/codex_noradam_confidence/experiments/run_wikitext_llm50m.py"
    spec = importlib.util.spec_from_file_location("test_lm_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_lm_runner"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    vocab_size = 128
    eos_token_id = 127
    is_fast = True

    def __len__(self):
        return self.vocab_size

    def __call__(self, batch, *, add_special_tokens=False, return_attention_mask=False):
        assert add_special_tokens is False
        assert return_attention_mask is False
        return {"input_ids": [[(ord(ch) % 64) + 1 for ch in text] for text in batch]}


def test_consume_text_tokens_adds_eos_and_respects_limit():
    runner = _load_runner()
    rows = [{"text": "abc"}, {"text": "de"}, {"text": "fg"}]

    out = runner._consume_text_tokens(
        iter(rows),
        text_field="text",
        tokenizer=FakeTokenizer(),
        max_tokens=7,
        add_eos=True,
        batch_rows=2,
    )

    assert out.dtype == np.uint16
    assert out.shape == (7,)
    assert int(FakeTokenizer.eos_token_id) in out.tolist()


def test_effective_tokenizer_vocab_size_includes_added_tokens():
    runner = _load_runner()

    class AddedTokenTokenizer:
        vocab_size = 10
        eos_token_id = 11

        def __len__(self):
            return 12

    assert runner._effective_tokenizer_vocab_size(AddedTokenTokenizer()) == 12


def test_atomic_save_npy_writes_complete_file_and_cleans_temp(tmp_path):
    runner = _load_runner()
    path = tmp_path / "cache.npy"
    array = np.arange(16, dtype=np.uint16)

    runner._atomic_save_npy(path, array)

    assert np.array_equal(np.load(path), array)
    assert not list(tmp_path.glob(".*.tmp"))


def test_worker_hf_token_cache_requires_parent_seeded_cache(tmp_path):
    runner = _load_runner()
    args = SimpleNamespace(
        dataset_source="hf_text",
        tokenizer_mode="hf",
        cache_dir=tmp_path,
        hf_dataset="demo/dataset",
        hf_config="",
        hf_split="train",
        hf_text_field="text",
        tokenizer_name="gpt2",
        tokenizer_add_eos=True,
        hf_shuffle_buffer=17,
        seed=777,
        max_train_tokens=128,
        max_val_tokens=64,
    )

    with pytest.raises(FileNotFoundError, match="seed777"):
        runner.prepare_text_cache(args, require_existing=True)


def test_worker_command_forwards_supervisor_seed(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--seed",
            "777",
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    args = runner.parse_args()

    cmd = runner.build_worker_command(args, tmp_path / "trial.json")

    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "777"


def test_token_stream_batches_integer_token_arrays(tmp_path):
    runner = _load_runner()
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(64, dtype=np.uint32))

    stream = runner.TokenStream(path, torch.device("cpu"))
    gen = torch.Generator(device="cpu").manual_seed(123)
    x, y = stream.batch(batch_size=4, block_size=8, generator=gen)

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_token_stream_accepts_exactly_one_window(tmp_path):
    runner = _load_runner()
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(9, dtype=np.uint16))

    stream = runner.TokenStream(path, torch.device("cpu"), vocab_size=16)
    gen = torch.Generator(device="cpu").manual_seed(123)
    x, y = stream.batch(batch_size=3, block_size=8, generator=gen)

    assert torch.equal(x, torch.arange(8).repeat(3, 1))
    assert torch.equal(y, torch.arange(1, 9).repeat(3, 1))


def test_token_stream_rejects_ids_outside_model_vocab(tmp_path):
    runner = _load_runner()
    path = tmp_path / "tokens.npy"
    np.save(path, np.array([0, 1, 2, 9], dtype=np.uint16))

    with pytest.raises(ValueError, match="model vocab size"):
        runner.TokenStream(path, torch.device("cpu"), vocab_size=9)


def test_token_stream_sequential_batch_is_deterministic(tmp_path):
    runner = _load_runner()
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(32, dtype=np.uint16))

    stream = runner.TokenStream(path, torch.device("cpu"), vocab_size=64)
    x0, y0 = stream.sequential_batch(batch_index=0, batch_size=2, block_size=4)
    x1, y1 = stream.sequential_batch(batch_index=1, batch_size=2, block_size=4)

    assert torch.equal(x0, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert torch.equal(y0, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
    assert torch.equal(x1, torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]]))
    assert torch.equal(y1, torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]]))


def test_ema_nesterov_restores_base_weights_before_step():
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
    opt.state[p]["ema_delta"].fill_(-0.3)
    opt.zero_grad()
    assert float(p.detach()) == pytest.approx(0.7)

    p.grad = torch.tensor([2.0])
    opt.step()

    assert float(p.detach()) == pytest.approx(0.8)


def test_anchor_muown_optimizer_step_keeps_parameters_finite():
    runner = _load_runner()
    torch.manual_seed(123)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(8, 3, bias=False),
    )
    opt = runner.AnchorMuown(
        model,
        lr=1e-3,
        fallback_lr=1e-3,
        row_gamma=0.25,
        soda_lambda_scale=0.003,
        muown_mag_lr_mult=0.5,
    )

    x = torch.randn(5, 4)
    target = torch.randn(5, 3)
    loss = torch.nn.functional.mse_loss(model(x), target)
    opt.zero_grad()
    loss.backward()
    opt.step()

    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_fineweb_ema_anchor_hpo_preset_has_focused_ema_anchor_grid():
    runner = _load_runner()
    args = SimpleNamespace(
        preset="fineweb_ema_anchor_hpo",
        hpo_steps=123,
        eval_bins=3,
        warmup_steps=99,
        max_hpo_trials=0,
    )

    trials = runner.hpo_trials(args)
    families = {trial.family for trial in trials}
    ema_anchor_trials = [trial for trial in trials if trial.family == "ema_anchormuon"]

    assert {"anchormuon", "ema_anchormuon", "muon", "ema_muon", "adamw", "adamatan2"} <= families
    assert len(ema_anchor_trials) > 500
    assert {trial.ema_warmup_frac for trial in ema_anchor_trials} >= {0.0, 0.2, 0.3}
    assert {trial.ema_rest_frac for trial in ema_anchor_trials} >= {0.0, 0.1, 0.2}
    assert all(trial.eval_every == 41 for trial in trials)
    assert all(trial.warmup_steps == 30 for trial in trials)


def test_final_trials_from_hpo_can_replay_multiple_top_configs_per_family():
    runner = _load_runner()
    args = SimpleNamespace(final_steps=100, eval_bins=4, warmup_steps=20, seed=7, final_top_per_family=2)
    hpo_rows = [
        {"name": "anchor_bad", "family": "anchormuon", "lr": 0.1, "best_val_loss": 2.0},
        {"name": "anchor_best", "family": "anchormuon", "lr": 0.2, "best_val_loss": 1.0},
        {"name": "anchor_second", "family": "anchormuon", "lr": 0.3, "best_val_loss": 1.5},
        {"name": "ema_best", "family": "ema_anchormuon", "lr": 0.4, "best_val_loss": 0.9, "ema_beta": 0.3},
        {"name": "ema_second", "family": "ema_anchormuon", "lr": 0.5, "best_val_loss": 1.1, "ema_beta": 0.5},
        {"name": "ema_bad", "family": "ema_anchormuon", "lr": 0.6, "best_val_loss": 1.2, "ema_beta": 0.7},
    ]

    trials = runner.final_trials_from_hpo(args, hpo_rows)
    names = [trial.name for trial in trials]

    assert names == [
        "final_anchor_best",
        "final_anchor_second",
        "final_ema_best",
        "final_ema_second",
    ]
    assert [trial.eval_every for trial in trials] == [25, 25, 25, 25]
    assert [trial.warmup_steps for trial in trials] == [20, 20, 20, 20]


def test_read_csv_rows_supports_selected_final_replay(tmp_path):
    runner = _load_runner()
    path = tmp_path / "hpo_summary.csv"
    path.write_text("name,family,lr,best_val_loss\nmuon_a,muon,0.1,2.0\nmuon_b,muon,0.2,1.0\n")

    rows = runner.read_csv_rows(path)

    assert rows == [
        {"name": "muon_a", "family": "muon", "lr": "0.1", "best_val_loss": "2.0"},
        {"name": "muon_b", "family": "muon", "lr": "0.2", "best_val_loss": "1.0"},
    ]


def test_launch_trials_skips_existing_summaries(monkeypatch, tmp_path):
    runner = _load_runner()
    trial = runner.TrialConfig("already_done", "adamw", 1e-3)
    summary_dir = tmp_path / "hpo" / trial.name
    summary_dir.mkdir(parents=True)
    summary = {"name": trial.name, "family": trial.family, "best_val_loss": 1.23}
    (summary_dir / "summary.json").write_text(__import__("json").dumps(summary))
    args = SimpleNamespace(output_dir=tmp_path, resume_existing=True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)

    rows = runner.launch_trials(args, [trial])

    assert rows == [summary]
