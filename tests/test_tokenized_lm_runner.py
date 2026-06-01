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
