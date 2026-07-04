import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


PACE_SRC = Path(__file__).resolve().parents[1] / "papers/code/pace/src"
if str(PACE_SRC) not in sys.path:
    sys.path.insert(0, str(PACE_SRC))

from pace.helpers.data import tokenize_conversation
from pace.optimizer import PACE


def _load_train_module(monkeypatch):
    hydra = types.ModuleType("hydra")
    hydra.main = lambda **_kwargs: (lambda fn: fn)
    hydra.core = types.SimpleNamespace(
        hydra_config=types.SimpleNamespace(
            HydraConfig=types.SimpleNamespace(get=lambda: None),
        ),
    )
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = dict
    omegaconf.OmegaConf = types.SimpleNamespace(
        to_yaml=lambda _cfg: "",
        to_container=lambda cfg, resolve=True: cfg,
    )
    model_module = types.ModuleType("pace.helpers.model")
    model_module.load_model = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    monkeypatch.setitem(sys.modules, "pace.helpers.model", model_module)
    path = PACE_SRC / "pace/helpers/train.py"
    spec = importlib.util.spec_from_file_location("test_pace_train_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_pace_train_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pace_optimizer_rejects_step_while_swapped_to_ema():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = PACE([p], lr=0.1, lambda_pullback=0.1, use_ema_eval=True, ema_kappa=0.5)
    for _ in range(2):
        p.grad = torch.tensor([1.0])
        opt.step()
    live = p.detach().clone()

    opt.eval()
    with pytest.raises(RuntimeError, match="swapped to EMA"):
        p.grad = torch.tensor([1.0])
        opt.step()
    opt.train()

    assert torch.equal(p.detach(), live)


def test_pace_optimizer_rejects_zero_ema_update_frequency():
    p = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="ema_update_freq"):
        PACE([p], ema_update_freq=0)


def test_pace_get_ema_beta_t_skips_plain_param_groups():
    p_plain = torch.nn.Parameter(torch.tensor([1.0]))
    p_ema = torch.nn.Parameter(torch.tensor([2.0]))
    opt = PACE(
        [
            {"params": [p_plain], "lambda_pullback": 0.0, "use_ema_eval": False},
            {"params": [p_ema], "lambda_pullback": 0.1, "use_ema_eval": True, "ema_kappa": 0.5},
        ],
        lr=0.1,
        ema_kappa=None,
    )
    p_plain.grad = torch.tensor([1.0])
    p_ema.grad = torch.tensor([1.0])

    opt.step()

    assert opt.get_ema_beta_t() == pytest.approx(2.0**-0.5)


def test_pace_log_stats_aggregate_across_param_groups():
    p_fast = torch.nn.Parameter(torch.tensor([1.0]))
    p_slow = torch.nn.Parameter(torch.tensor([2.0]))
    opt = PACE(
        [
            {"params": [p_fast], "lambda_pullback": 1.0, "use_ema_eval": True, "ema_kappa": 0.5},
            {"params": [p_slow], "lambda_pullback": 1.0, "use_ema_eval": True, "ema_kappa": 0.5},
        ],
        lr=0.1,
        log_stats=True,
    )
    p_fast.grad = torch.tensor([2.0])
    p_slow.grad = torch.tensor([0.5])

    opt.step()

    decay = 2.0**-0.5
    raw_fast = decay / (2.0 + 1e-8)
    raw_slow = decay / (0.5 + 1e-8)
    stats = opt.get_step_stats()
    assert stats["raw_lambda_mean"] == pytest.approx((raw_fast + raw_slow) / 2)
    assert stats["raw_lambda_min"] == pytest.approx(raw_fast)
    assert stats["raw_lambda_max"] == pytest.approx(raw_slow)
    assert stats["effective_lambda_mean"] == pytest.approx(0.1 * (raw_fast + raw_slow) / 2)
    assert stats["lambda_mean"] == pytest.approx(0.1 * (raw_fast + raw_slow) / 2)
    assert stats["adam_step_norm"] == pytest.approx((0.101**2 + 0.102**2) ** 0.5)


def test_eval_loss_respects_max_samples_inside_batch(monkeypatch):
    train = _load_train_module(monkeypatch)

    class CountingModel:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, input_ids, attention_mask, labels):
            self.batch_sizes.append(input_ids.size(0))
            return types.SimpleNamespace(loss=torch.tensor(2.0))

    batch = {
        "input_ids": torch.ones(4, 3, dtype=torch.long),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.ones(4, 3, dtype=torch.long),
    }
    model = CountingModel()

    loss = train._eval_loss(model, [batch], max_samples=1, ctx=contextlib.nullcontext())

    assert loss == pytest.approx(2.0)
    assert model.batch_sizes == [1]


def test_tokenize_conversation_normalizes_sharegpt_roles_to_chatml():
    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(ch) for ch in text]

    ids, _attention, labels = tokenize_conversation(
        [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "ok"},
        ],
        CharTokenizer(),
        200,
    )

    text = "".join(chr(i) for i in ids)
    supervised = "".join(chr(i) for i in labels if i != -100)
    assert "<|im_start|>user\n" in text
    assert "<|im_start|>assistant\n" in text
    assert "<|im_start|>human\n" not in text
    assert "<|im_start|>gpt\n" not in text
    assert supervised == "ok<|im_end|>\n"
