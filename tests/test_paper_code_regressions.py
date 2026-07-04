import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_muown_vanilla_config_alias_builds_optimizer(monkeypatch):
    root = REPO / "papers/code/muown"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.modules.setdefault("wandb", types.SimpleNamespace(run=None))

    class FakeMuon(torch.optim.Optimizer):
        def __init__(self, params, **kwargs):
            super().__init__(params, {"lr": kwargs["lr"]})

    monkeypatch.setattr(torch.optim, "Muon", FakeMuon, raising=False)
    init_optim = importlib.import_module("optim.init_optim")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    cfg = SimpleNamespace(
        optim="muon_vanilla",
        lr=0.01,
        adjust_lr="match_adam",
        muon_weight_decay=0.0,
        muon_beta=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        adamw_weight_decay=0.0,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        fused_optim=False,
        eps=1e-8,
    )

    optimizers = init_optim.intialize_optimizer(model, cfg)

    assert isinstance(optimizers["muon"], FakeMuon)
    assert "adamw" in optimizers


def test_muowndp_matches_muown_single_rank_with_weight_decay(tmp_path):
    root = REPO / "papers/code/muown"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.modules.setdefault("wandb", types.SimpleNamespace(run=None))
    muown = importlib.import_module("optim.muown")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")

    initialized_here = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'dist_init'}",
            rank=0,
            world_size=1,
        )
        initialized_here = True
    try:
        torch.manual_seed(11)
        p_ref = torch.nn.Parameter(torch.randn(4, 4))
        p_dp = torch.nn.Parameter(p_ref.detach().clone())
        grad = torch.randn_like(p_ref)
        opt_ref = muown.Muown([p_ref], lr=0.05, weight_decay=0.2, backend="svd")
        opt_dp = muown.MuownDP([p_dp], lr=0.05, weight_decay=0.2, backend="svd")

        p_ref.grad = grad.clone()
        p_dp.grad = grad.clone()
        opt_ref.step()
        opt_dp.step()

        assert torch.allclose(p_dp, p_ref, atol=1e-6, rtol=1e-6)
    finally:
        if initialized_here:
            torch.distributed.destroy_process_group()


def test_normuown_weight_decay_resyncs_magnitude_state():
    root = REPO / "papers/code/muown"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.modules.setdefault("wandb", types.SimpleNamespace(run=None))
    normuown = importlib.import_module("optim.normuown")

    torch.manual_seed(17)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = normuown.NorMuown(
        [p],
        lr=0.01,
        weight_decay=0.2,
        backend="svd",
        use_normuon=False,
        shape_scaling_type="muon",
    )
    p.grad = torch.randn_like(p)

    opt.step()

    assert torch.allclose(
        opt.state[p]["g"],
        p.detach().norm(dim=1, keepdim=True),
        atol=1e-7,
        rtol=1e-7,
    )


def test_muown_concat_chunk_drops_short_batches():
    module = _load_module(
        "test_muown_data_prep_utils",
        REPO / "papers/code/muown/data/datasets/data_prep_utils.py",
    )

    out = module.concat_chunck({"input_ids": [[1, 2], [3]]}, max_seq_length=5)

    assert out["input_ids"] == []
    assert out["docs_lengths"] == []


def _load_llama_opt_utils():
    sys.modules.setdefault("optimizers", types.ModuleType("optimizers"))
    muon_module = types.ModuleType("optimizers.muon")
    muon_module.MuonWithAuxAdam = object
    sys.modules["optimizers.muon"] = muon_module
    return _load_module(
        "test_llama_opt_utils",
        REPO / "papers/code/ema-nesterov/llama/opt_utils.py",
    )


def test_llama_ema_nesterov_reads_scheduler_lr_from_wrapper():
    opt_utils = _load_llama_opt_utils()
    p = torch.nn.Parameter(torch.tensor([1.0]))
    inner = torch.optim.SGD([p], lr=0.1)
    wrapper = opt_utils.EMA_Nesterov(
        [{"params": [p], "lr": 0.1}],
        inner,
        lookahead_stepsize=0.2,
        use_scheduled_lookahead_stepsize=True,
        rest_step=10,
    )
    wrapper.param_groups[0]["initial_lr"] = 0.1
    wrapper.param_groups[0]["lr"] = 0.05

    assert wrapper.get_lr_lambda() == pytest.approx(0.5)
    wrapper.nesterov_step()


def test_llama_gradient_clipping_helper_flattens_param_groups():
    opt_utils = _load_llama_opt_utils()
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([2.0])

    torch.nn.utils.clip_grad_norm_(list(opt_utils.trainable_tensors([{"params": [p]}])), 1.0)

    assert p.grad.abs().item() <= 1.0


def test_nanogpt_ema_nesterov_eval_lookahead_accepts_alpha():
    module = _load_module(
        "test_nanogpt_opt_utils",
        REPO / "papers/code/ema-nesterov/nanogpt/opt_utils.py",
    )
    p = torch.nn.Parameter(torch.tensor([1.0]))
    inner = torch.optim.SGD([p], lr=0.1)
    wrapper = module.EMA_Nesterov([{"params": [p], "lr": 0.1}], inner, rest_step=10)
    wrapper.state[p]["lookahead_buffer"] = (torch.ones_like(p), -1)

    wrapper.lookahead_step(0.5)

    assert torch.allclose(p.detach(), torch.tensor([1.5]))


def test_nanogpt_muon_step_runs_under_grad_mode(tmp_path):
    module = _load_module(
        "test_nanogpt_muon_optimizer",
        REPO / "papers/code/ema-nesterov/nanogpt/optimizers/muon_nanogpt.py",
    )
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")

    initialized_here = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'nanogpt_muon_dist_init'}",
            rank=0,
            world_size=1,
        )
        initialized_here = True
    try:
        p = torch.nn.Parameter(torch.ones(2, 2))
        p.grad = torch.ones_like(p)
        opt = module.Muon([p], backend_steps=1)

        opt.step()

        assert torch.isfinite(p).all()
        assert not torch.equal(p.detach(), torch.ones_like(p))
    finally:
        if initialized_here:
            torch.distributed.destroy_process_group()


def test_nanogpt_muon_step_handles_missing_grad(tmp_path):
    module = _load_module(
        "test_nanogpt_muon_optimizer_missing_grad",
        REPO / "papers/code/ema-nesterov/nanogpt/optimizers/muon_nanogpt.py",
    )
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")

    initialized_here = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'nanogpt_muon_dist_init'}",
            rank=0,
            world_size=1,
        )
        initialized_here = True
    try:
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = module.Muon([p], backend_steps=1)

        opt.step()

        assert torch.equal(p.detach(), torch.ones_like(p))
        assert torch.equal(p.grad, torch.zeros_like(p))
    finally:
        if initialized_here:
            torch.distributed.destroy_process_group()


def test_pace_reproduce_uses_check_true(monkeypatch, tmp_path):
    hydra = types.ModuleType("hydra")

    def hydra_main(**_kwargs):
        def decorate(fn):
            fn.__wrapped__ = fn
            return fn

        return decorate

    hydra.main = hydra_main
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = dict
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    module = _load_module(
        "test_pace_reproduce",
        REPO / "papers/code/pace/src/pace_figures/reproduce.py",
    )
    calls = []

    monkeypatch.setattr(
        module,
        "expand_figure",
        lambda _fig: [
            {
                "model": "tiny",
                "method": "pace",
                "schedule": "constant",
                "seed": 1,
                "optimizer": {"name": "adamw", "lr": 0.001},
            }
        ],
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    class AttrDict(dict):
        def __getattr__(self, name):
            return self[name]

    cfg = AttrDict(
        fig=AttrDict(id="figx", paper="test", desc="test", dataset="demo"),
        run=True,
        limit=1,
        out=str(tmp_path),
    )

    module.main.__wrapped__(cfg)

    assert calls
    assert calls[0][1]["check"] is True
