import importlib.util
import sys
from pathlib import Path

import numpy as np
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
