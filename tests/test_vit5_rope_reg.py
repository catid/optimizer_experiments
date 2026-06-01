import importlib.util
import sys
from pathlib import Path

import pytest


def _load_models_vit5():
    root = Path(__file__).resolve().parents[1] / "workers/codex_noradam_confidence"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "models_vit5.py"
    spec = importlib.util.spec_from_file_location("test_models_vit5_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_models_vit5_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vit_models_rope_reg_flag_controls_register_rope():
    models = _load_models_vit5()

    with_rope = models.vit_models(
        img_size=32,
        patch_size=4,
        embed_dim=48,
        depth=1,
        num_heads=3,
        num_registers=4,
        rope=True,
        rope_reg=True,
    )
    without_rope = models.vit_models(
        img_size=32,
        patch_size=4,
        embed_dim=48,
        depth=1,
        num_heads=3,
        num_registers=4,
        rope=True,
        rope_reg=False,
    )

    assert with_rope.blocks[0].attn.rope_reg is not None
    assert without_rope.blocks[0].attn.rope_reg is None


def test_vit5_factory_accepts_rope_reg_false():
    models = _load_models_vit5()

    model = models.vit5_micro(rope_reg=False)

    assert model.blocks[0].attn.rope_reg is None


def test_vit_models_only_requires_square_registers_when_register_rope_enabled():
    models = _load_models_vit5()

    models.vit_models(
        img_size=32,
        patch_size=4,
        embed_dim=48,
        depth=1,
        num_heads=3,
        num_registers=3,
        rope=True,
        rope_reg=False,
    )
    with pytest.raises(AssertionError, match="rope_reg=True"):
        models.vit_models(
            img_size=32,
            patch_size=4,
            embed_dim=48,
            depth=1,
            num_heads=3,
            num_registers=3,
            rope=True,
            rope_reg=True,
        )
