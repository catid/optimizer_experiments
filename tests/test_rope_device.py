import importlib.util
import sys
from pathlib import Path

import torch


def _load_rope():
    path = Path(__file__).resolve().parents[1] / "workers/codex_noradam_confidence/rope.py"
    spec = importlib.util.spec_from_file_location("test_rope_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_rope_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vision_rope_uses_input_device_on_cpu():
    rope = _load_rope()
    module = rope.VisionRotaryEmbedding(dim=4, pt_seq_len=2)
    x = torch.randn(4, 4, 8)

    out = module(x)

    assert out.device == x.device
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_vision_rope_rejects_non_square_patch_count():
    rope = _load_rope()
    module = rope.VisionRotaryEmbedding(dim=4, pt_seq_len=2)
    x = torch.randn(4, 5, 8)

    try:
        module(x)
    except ValueError as exc:
        assert "square patch grid" in str(exc)
    else:
        raise AssertionError("expected non-square patch count to fail clearly")
