from __future__ import annotations

from io import BytesIO

import torch

from equimuse_normuon import EquiMuseNorMuon, build_equimuse_normuon_param_groups, normuon_precondition


def _make_params() -> list[torch.nn.Parameter]:
    torch.manual_seed(9403)
    return [
        torch.nn.Parameter(torch.randn(4, 8) * 0.1),
        torch.nn.Parameter(torch.randn(8, 4) * 0.1),
        torch.nn.Parameter(torch.randn(5) * 0.1),
    ]


def _clone_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(p.detach().clone()) for p in params]


def _assign_grads(params_a: list[torch.nn.Parameter], params_b: list[torch.nn.Parameter], step: int) -> None:
    for i, (pa, pb) in enumerate(zip(params_a, params_b, strict=True)):
        grad = torch.arange(pa.numel(), dtype=pa.dtype).reshape_as(pa)
        grad = (grad + 1.0 + 0.17 * i) * (0.011 + 0.004 * step)
        pa.grad = grad.clone()
        pb.grad = grad.clone()


def _groups(params: list[torch.nn.Parameter], *, batch_muon: bool) -> list[dict]:
    return [
        {
            "params": params[:2],
            "lr": 0.04,
            "momentum": 0.85,
            "use_muon": True,
            "weight_decay": 0.02,
            "soda_anchor_scale": 0.5,
            "pmuoneq_beta": 0.95,
            "pmuoneq_row_gamma": 0.15,
            "pmuoneq_col_gamma": 0.15,
            "normuon_beta2": 0.9,
            "batch_muon": batch_muon,
        },
        {
            "params": params[2:],
            "lr": 0.04,
            "beta2": 0.91,
            "eps": 1e-10,
            "use_muon": False,
            "weight_decay": 0.02,
            "soda_anchor_scale": 0.5,
        },
    ]


def test_batch_mode_matches_loop_mode() -> None:
    loop_params = _make_params()
    batch_params = _clone_params(loop_params)
    loop = EquiMuseNorMuon(
        _groups(loop_params, batch_muon=False),
        beta1=0.6,
        rho=0.5,
        warmup_steps=2,
        soda_anchor_scale=0.5,
        batch_muon=False,
        ns_dtype=torch.float32,
    )
    batch = EquiMuseNorMuon(
        _groups(batch_params, batch_muon=True),
        beta1=0.6,
        rho=0.5,
        warmup_steps=2,
        soda_anchor_scale=0.5,
        batch_muon=True,
        ns_dtype=torch.float32,
    )
    loop.train()
    batch.train()
    for step in range(5):
        _assign_grads(loop_params, batch_params, step)
        loop.step()
        batch.step()
    for a, b in zip(loop_params, batch_params, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_param_group_helper_splits_matrix_and_fallback_params() -> None:
    embed = torch.nn.Parameter(torch.zeros(8, 16))
    head = torch.nn.Parameter(torch.zeros(8, 16))
    bias = torch.nn.Parameter(torch.zeros(16))
    hidden = torch.nn.Parameter(torch.zeros(16, 16))
    named = [
        ("token_embed.weight", embed),
        ("lm_head.weight", head),
        ("blocks.0.bias", bias),
        ("blocks.0.mlp.weight", hidden),
    ]
    groups = build_equimuse_normuon_param_groups(named, lr=0.01, weight_decay=0.05, normuon_beta2=0.9)
    assert [group["use_muon"] for group in groups] == [False, True]
    assert {id(p) for p in groups[0]["params"]} == {id(embed), id(head), id(bias)}
    assert {id(p) for p in groups[1]["params"]} == {id(hidden)}
    assert all("mimuon_enabled" not in group for group in groups)
    assert all("normuon_enabled" not in group for group in groups)
    assert groups[1]["normuon_beta2"] == 0.9


def test_normuon_precondition_preserves_frobenius_norm() -> None:
    update = torch.tensor([[1.0, 2.0, 3.0], [0.25, 0.5, 0.75]])
    moment = torch.zeros(2, 1)
    out = normuon_precondition(update, moment, beta2=0.9)
    assert torch.all(moment > 0)
    assert torch.allclose(out.norm(), update.norm(), atol=1e-6, rtol=1e-6)


def test_state_dict_roundtrip() -> None:
    params = _make_params()
    opt = EquiMuseNorMuon(_groups(params, batch_muon=True), beta1=0.6, warmup_steps=2)
    opt.train()
    for step in range(2):
        for i, p in enumerate(params):
            p.grad = torch.ones_like(p) * (0.02 + i * 0.01 + step * 0.005)
        opt.step()
    buffer = BytesIO()
    torch.save(opt.state_dict(), buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    other = EquiMuseNorMuon(_groups(_clone_params(params), batch_muon=True), beta1=0.6, warmup_steps=2)
    other.load_state_dict(loaded)
    assert other.train_mode is True


def test_train_eval_return_self_and_checkpoint_roundtrips() -> None:
    params = _make_params()
    opt = EquiMuseNorMuon(_groups(params, batch_muon=True), beta1=0.6, warmup_steps=2)
    assert opt.train() is opt
    for step in range(3):
        for i, p in enumerate(params):
            p.grad = torch.ones_like(p) * (0.03 + i * 0.01 + step * 0.007)
        opt.step()

    assert opt.eval() is opt
    eval_params = [p.detach().clone() for p in params]
    assert opt.eval() is opt
    for p, expected in zip(params, eval_params, strict=True):
        assert torch.allclose(p, expected, atol=0.0, rtol=0.0)

    assert opt.train() is opt
    train_params = [p.detach().clone() for p in params]
    opt.eval()
    for p, expected in zip(params, eval_params, strict=True):
        assert torch.allclose(p, expected, atol=1e-6, rtol=1e-6)

    eval_checkpoint = BytesIO()
    torch.save({"params": eval_params, "optimizer": opt.state_dict()}, eval_checkpoint)
    eval_checkpoint.seek(0)
    eval_loaded = torch.load(eval_checkpoint, weights_only=False)
    eval_clone_params = [torch.nn.Parameter(p.clone()) for p in eval_loaded["params"]]
    eval_clone = EquiMuseNorMuon(_groups(eval_clone_params, batch_muon=True), beta1=0.6, warmup_steps=2)
    eval_clone.load_state_dict(eval_loaded["optimizer"])
    assert eval_clone.train_mode is False
    eval_clone.train()
    for p, expected in zip(eval_clone_params, train_params, strict=True):
        assert torch.allclose(p, expected, atol=1e-6, rtol=1e-6)

    opt.train()
    train_checkpoint = BytesIO()
    torch.save({"params": [p.detach().clone() for p in params], "optimizer": opt.state_dict()}, train_checkpoint)
    train_checkpoint.seek(0)
    train_loaded = torch.load(train_checkpoint, weights_only=False)
    train_clone_params = [torch.nn.Parameter(p.clone()) for p in train_loaded["params"]]
    train_clone = EquiMuseNorMuon(_groups(train_clone_params, batch_muon=True), beta1=0.6, warmup_steps=2)
    train_clone.load_state_dict(train_loaded["optimizer"])
    assert train_clone.train_mode is True
    train_clone.eval()
    for p, expected in zip(train_clone_params, eval_params, strict=True):
        assert torch.allclose(p, expected, atol=1e-6, rtol=1e-6)
