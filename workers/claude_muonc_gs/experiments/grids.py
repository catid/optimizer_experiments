"""Trial grids for the MuonC x MuonGS combination study.

Every stage is a pure function of its name, so a result bundle can be
reproduced (or resumed) with ``run_muonc_gs_lm.py --stage <name>``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from run_muonc_gs_lm import Trial

_DEFAULTS = {f.name: f.default for f in fields(Trial)}
_NAME_FIELDS = (
    "opt", "base", "gs", "gs_mode", "decay", "norm_mode", "emb", "unprojected_decay",
    "fallback_mode", "fallback_lr_mult", "fallback_wd", "fallback_decay",
    "row_gamma", "normuon_beta2", "soda", "warmup_steps", "final_lr_scale", "decay_frac",
    # appended later; every name above is unchanged for trials that leave these at their defaults
    "pmuoneq_beta", "norm_target", "init_resid_scale", "emulate_gs_theta0", "ns_dtype",
    "init_scale", "lr_profile", "radial_lr",
    "batch_size", "block_size", "eval_every", "eval_batches", "final_eval_batches", "diag_every",
)
_SPECIAL = {"name", "stage", "data", "schedule", "steps", "lr", "wd", "seed", "gs", "decay"}
assert set(_DEFAULTS) == set(_NAME_FIELDS) | _SPECIAL, set(_DEFAULTS) ^ (set(_NAME_FIELDS) | _SPECIAL)
_SHORT = {
    "opt": "", "base": "", "gs": "gs", "gs_mode": "", "decay": "", "norm_mode": "norm", "emb": "emb",
    "unprojected_decay": "udec", "fallback_mode": "fb", "fallback_lr_mult": "flr", "fallback_wd": "fwd",
    "fallback_decay": "fdec", "row_gamma": "rg", "normuon_beta2": "nb", "soda": "soda",
    "warmup_steps": "wu", "final_lr_scale": "fin", "decay_frac": "df",
    "pmuoneq_beta": "pb", "norm_target": "kappa", "init_resid_scale": "rinit", "emulate_gs_theta0": "emu",
    "ns_dtype": "ns", "batch_size": "bs", "block_size": "blk", "eval_every": "ev", "eval_batches": "eb",
    "final_eval_batches": "feb", "diag_every": "dg", "init_scale": "iscale", "lr_profile": "prof-",
    "radial_lr": "rad",
}


def _fmt(value: Any, key: str = "") -> str:
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value)
    if key == "lr_profile":  # a reference trial name: drop its protocol prefix, keep the rest readable
        text = "-".join(text.split("_")[1:])
    return text


def mk(stage: str, **kw: Any) -> Trial:
    values = {**_DEFAULTS, **kw}
    parts = [f"{values['data']}-{values['schedule']}{values['steps'] // 1000}k"]
    for key in _NAME_FIELDS:
        if key in ("gs", "decay"):
            continue
        if values[key] != _DEFAULTS[key]:
            parts.append(f"{_SHORT[key]}{_fmt(values[key], key)}")
    if values["opt"] == "muonc":
        gs = values["gs"]
        parts.append("gs" if gs == 1.0 else ("nogs" if gs == 0.0 else f"gs{_fmt(gs)}"))
        parts.append(values["decay"])
    parts.append(f"lr{_fmt(values['lr'])}")
    if values["opt"] != "muonc" or values["decay"] != "none":
        parts.append(f"wd{_fmt(values['wd'])}")
    parts.append(f"s{values['seed']}")
    return Trial(name="_".join(parts), stage=stage, **kw)


SEEDS3 = (123, 124, 125)
SEEDS5 = (123, 124, 125, 126, 127)
LRS = (0.25e-3, 0.5e-3, 1.0e-3, 2.0e-3, 4.0e-3)  # coarse x2 grid; refined per cell in stage 2
WDS = (0.0125, 0.05, 0.2)

# The four cells of the 2x2 that matter plus the two decay-free references.
MUONW = dict(gs=0.0, decay="standard")
MUONC = dict(gs=0.0, decay="corrected")
MUONGS = dict(gs=1.0, decay="none")
GS_W = dict(gs=1.0, decay="standard")
GS_C = dict(gs=1.0, decay="corrected")
MUON = dict(gs=0.0, decay="none")
# MuonH-style hard constraint: tangent step, then rescale to the initial Frobenius norm.
GS_FIXED = dict(gs=1.0, decay="none", norm_mode="fixed")

PROTOCOLS = {
    # repo-standard: 8M-token cache (~4 passes), warmup-stable-decay
    "std": dict(data="std", schedule="wsd", steps=8000),
    # single-pass regime with the hnet trainer's schedule (warmup, cosine to 10%)
    "sp": dict(data="sp", schedule="cosine", steps=8000),
}


def factorial(stage: str, protocol: str, lrs=LRS, wds=WDS, seeds=(123,)) -> list[Trial]:
    trials: list[Trial] = []
    proto = PROTOCOLS[protocol]
    for seed in seeds:
        for lr in lrs:
            for cell in (MUON, MUONGS, GS_FIXED):
                trials.append(mk(stage, **proto, **cell, lr=lr, seed=seed))
            for wd in wds:
                for cell in (MUONW, MUONC, GS_W, GS_C):
                    trials.append(mk(stage, **proto, **cell, lr=lr, wd=wd, seed=seed))
    return trials


SP = PROTOCOLS["sp"]
STD = PROTOCOLS["std"]
# The H-Net trainer's optimizer stack around the Muon direction.
HNET = dict(base="pmuoneq+normuon+soda", fallback_mode="atan2", fallback_lr_mult=0.5, ns_dtype="bf16")
# Best unprojected run of stage 1 on the single-pass protocol; its norm growth
# defines the effective-LR trajectory that the emulation trials reproduce.
SP_BEST_UNPROJECTED = "sp-cosine8k_nogs_standard_lr0.001_wd0.2_s123"


def _smoke() -> list[Trial]:
    stage = "smoke"
    base = dict(data="std", steps=40, eval_every=20, diag_every=10, eval_batches=2, final_eval_batches=4, warmup_steps=5)
    return [
        mk(stage, **base, opt="repo_muon"),
        mk(stage, **base, **MUONW),
        mk(stage, **base, **GS_C),
        mk(stage, **base, **GS_C, **HNET),
        mk(stage, **base, **MUONGS, norm_mode="fixed"),
        mk(stage, **base, **GS_C, emb="fallback", fallback_wd=0.05),
        mk(stage, **base, **GS_C, schedule="cosine"),
        mk(stage, **base, **MUONGS, norm_target=1.5, init_resid_scale=0.25),
        mk(stage, **base, **MUONGS, init_scale=2.0, emulate_gs_theta0=0.02),
        mk(stage, **base, gs=0.5, gs_mode="outward", decay="standard", emb="matrix_nogs", unprojected_decay="corrected"),
    ]


def _s0_parity() -> list[Trial]:
    # (a) the repo's own PlainMuon / AdamW inside this loop, to compare with the README;
    # (b) MuonC configured as plain MuonW, to show the port reproduces PlainMuon;
    # (c) five seeds of each, which is also the seed-noise estimate.
    stage = "s0_parity"
    trials = [mk(stage, **STD, opt="repo_muon", lr=1.8e-3, wd=0.05, seed=s) for s in SEEDS5]
    trials += [mk(stage, **STD, **MUONW, lr=1.8e-3, wd=0.05, seed=s) for s in SEEDS5]
    trials += [mk(stage, **STD, opt="repo_adamw", lr=5e-4, wd=0.05, seed=123)]
    return trials


def _s2_diag() -> list[Trial]:
    """Why does pinning the norm cost loss in the single-pass regime?

    Three hypotheses, each with its own intervention:
    * embedding: the tied embedding/unembedding should not be projected;
    * scale: GS freezes every matrix at its initial scale, and that scale is
      not where the loss wants the network to operate;
    * effective LR: the unprojected run's norm growth is an LR anneal that the
      projected run lacks.
    """

    stage = "s2_diag"
    t: list[Trial] = []
    for lr in (5e-4, 1e-3):
        t.append(mk(stage, **SP, **MUONGS, emb="matrix_nogs", lr=lr))
        t.append(mk(stage, **SP, **GS_C, wd=0.05, emb="matrix_nogs", lr=lr))
        t.append(mk(stage, **SP, **MUONGS, emb="fallback", lr=lr))
    for lr in (1e-3, 2e-3):
        t.append(mk(stage, **SP, **MUONW, wd=0.2, emb="fallback", lr=lr))
    # Same angular step theta_0 = 10 * lr / init_scale at a larger operating scale.
    for scale, lrs in ((1.5, (7.5e-4, 1.5e-3)), (2.0, (1e-3, 2e-3)), (3.0, (1.5e-3, 3e-3))):
        for lr in lrs:
            t.append(mk(stage, **SP, **MUONGS, init_scale=scale, lr=lr))
    for lr in (1e-3, 2e-3):
        t.append(mk(stage, **SP, **GS_FIXED, init_scale=2.0, lr=lr))
        t.append(mk(stage, **SP, **MUONW, wd=0.2, init_scale=2.0, lr=lr))
    # Partial and one-sided projection.
    for strength in (0.5, 0.9):
        for lr in (5e-4, 1e-3):
            t.append(mk(stage, **SP, gs=strength, decay="none", lr=lr))
    t.append(mk(stage, **SP, gs=0.5, decay="standard", wd=0.2, lr=1e-3))
    t.append(mk(stage, **SP, gs=0.5, decay="corrected", wd=0.2, lr=1e-3))
    for lr in (5e-4, 1e-3):
        t.append(mk(stage, **SP, gs=1.0, gs_mode="outward", decay="none", lr=lr))
    t.append(mk(stage, **SP, gs=1.0, gs_mode="outward", decay="standard", wd=0.2, lr=1e-3))
    # Reproduce the best unprojected run's effective LR with the norm pinned:
    # at the initial scale, and at the scale that run ends at.
    t.append(mk(stage, **SP, **GS_FIXED, lr=1e-3, lr_profile=SP_BEST_UNPROJECTED))
    t.append(mk(stage, **SP, **MUONGS, lr=1e-3, lr_profile=SP_BEST_UNPROJECTED))
    t.append(mk(stage, **SP, **GS_FIXED, lr=2e-3, init_scale=2.0, lr_profile=SP_BEST_UNPROJECTED))
    # Spot checks on the multi-pass protocol.
    t.append(mk(stage, **STD, **MUONGS, emb="matrix_nogs", lr=5e-4))
    t.append(mk(stage, **STD, **MUONGS, emb="fallback", lr=5e-4))
    t.append(mk(stage, **STD, **MUONGS, init_scale=2.0, lr=1e-3))
    return t


def _s2_refine() -> list[Trial]:
    """sqrt(2) LR points around the projected optimum; stronger decay for the unprojected cells."""

    stage = "s2_refine"
    t: list[Trial] = []
    for proto in (SP, STD):
        for lr in (3.5e-4, 7e-4):
            t.append(mk(stage, **proto, **MUONGS, lr=lr))
            t.append(mk(stage, **proto, **GS_FIXED, lr=lr))
            t.append(mk(stage, **proto, **GS_C, wd=0.0125, lr=lr))
            t.append(mk(stage, **proto, **GS_C, wd=0.05, lr=lr))
            t.append(mk(stage, **proto, **GS_W, wd=0.05, lr=lr))
        for cell in (MUONW, MUONC):
            for wd in (0.4, 0.8):
                for lr in (1e-3, 2e-3):
                    t.append(mk(stage, **proto, **cell, wd=wd, lr=lr))
            t.append(mk(stage, **proto, **cell, wd=0.2, lr=1.4e-3))
    return t


def _s2_hnet() -> list[Trial]:
    """The H-Net stack (PMuonEq + NorMuon + SODA, AdamATan2 at half LR, BF16 GramNS)."""

    stage = "s2_hnet"
    t: list[Trial] = []
    for lr in (5e-4, 1e-3, 2e-3):
        t.append(mk(stage, **SP, **HNET, **GS_C, wd=0.05, lr=lr))  # hnet's MuonC
        t.append(mk(stage, **SP, **HNET, **MUONW, wd=0.05, lr=lr))  # hnet's previous "gramns" path
        t.append(mk(stage, **SP, **HNET, **MUONW, wd=0.2, lr=lr))
    return t


FRESH_SEEDS = (124, 125, 126)
HALF = dict(gs=0.5)  # half-strength projection


def _finalists(proto: dict, cells: list[dict], seeds) -> list[Trial]:
    return [mk("s3_seeds", **proto, **cell, seed=seed) for cell in cells for seed in seeds]


def _s3_seeds() -> list[Trial]:
    """Fresh seeds for every finalist.

    Each configuration was selected on seed 123; scoring it on seeds 124-126
    removes the winner's-curse bias of picking the minimum of a noisy grid.
    """

    sp_cells = [
        dict(**MUONW, wd=0.2, lr=1e-3),
        dict(**MUONC, wd=0.4, lr=1e-3),
        dict(**MUON, lr=1e-3),
        dict(**MUONGS, lr=7e-4),
        dict(**GS_C, wd=0.05, lr=5e-4),
        dict(**GS_FIXED, lr=7e-4),
        dict(**MUONGS, init_scale=2.0, lr=1e-3),
        dict(**HALF, decay="corrected", wd=0.2, lr=1e-3),
        dict(**HALF, decay="none", lr=1e-3),
        dict(**GS_FIXED, init_scale=2.0, lr=2e-3, lr_profile=SP_BEST_UNPROJECTED),
        dict(**MUONGS, emb="matrix_nogs", lr=1e-3),
    ]
    std_cells = [
        dict(**MUONW, wd=0.4, lr=1e-3),
        dict(**MUONGS, lr=7e-4),
        dict(**GS_C, wd=0.0125, lr=7e-4),
        dict(**GS_FIXED, lr=5e-4),
    ]
    trials = _finalists(SP, sp_cells, FRESH_SEEDS) + _finalists(STD, std_cells, FRESH_SEEDS)
    # GS + C configured to *hold* the better scale: equilibrium RMS 0.2*sqrt(lr/(2 wd)) = 0.04
    # = the 2x initial scale.  New configuration, so it also needs the selection seed.
    trials += _finalists(SP, [dict(**GS_C, wd=0.0125, init_scale=2.0, lr=1e-3)], (123, *FRESH_SEEDS))
    return trials


def _s3_long() -> list[Trial]:
    """Four times the horizon (32K steps, single-pass, cosine): does the ranking move?"""

    stage = "s3_long"
    long = dict(data="sp", schedule="cosine", steps=32000)
    t: list[Trial] = []
    for lr in (7e-4, 1e-3, 1.4e-3):
        t.append(mk(stage, **long, **MUONW, wd=0.2, lr=lr))
        t.append(mk(stage, **long, **MUONGS, init_scale=2.0, lr=lr))
    for lr in (7e-4, 1e-3):
        t.append(mk(stage, **long, **MUONC, wd=0.4, lr=lr))
        t.append(mk(stage, **long, **HALF, decay="corrected", wd=0.2, lr=lr))
    for lr in (5e-4, 7e-4):
        t.append(mk(stage, **long, **MUONGS, lr=lr))
    t.append(mk(stage, **long, **MUON, lr=1e-3))
    t.append(mk(stage, **long, **GS_C, wd=0.05, lr=5e-4))
    t.append(mk(stage, **long, **HALF, decay="none", lr=1e-3))
    t.append(mk(stage, **long, **GS_C, wd=0.0125, init_scale=2.0, lr=1e-3))
    t.append(mk(stage, **long, **GS_W, wd=0.0125, init_scale=2.0, lr=1e-3))
    t.append(mk(stage, **long, **GS_FIXED, init_scale=2.0, lr=1e-3))
    # Equilibrium regime: the norm window 1/(2 wd lr) = 10K steps is well inside the run and the
    # equilibrium RMS 0.2*sqrt(lr/(2 wd)) = 0.04 equals the starting scale, so the two decay laws
    # start at their common fixed point and only the schedule can pull them apart.
    t.append(mk(stage, **long, **GS_C, wd=0.025, init_scale=2.0, lr=2e-3))
    t.append(mk(stage, **long, **GS_W, wd=0.025, init_scale=2.0, lr=2e-3))
    return t


def _s3_hnet() -> list[Trial]:
    """Take the H-Net stack's projected-vs-unprojected gap apart."""

    stage = "s3_hnet"
    t: list[Trial] = []
    # Fresh seeds for the two headline cells of s2_hnet.
    for seed in (124, 125):
        t.append(mk(stage, **SP, **HNET, **GS_C, wd=0.05, lr=1e-3, seed=seed))
        t.append(mk(stage, **SP, **HNET, **MUONW, wd=0.05, lr=1e-3, seed=seed))
    # One change at a time from the unprojected stack.
    t.append(mk(stage, **SP, **HNET, **MUONC, wd=0.05, lr=1e-3))  # corrected decay only
    t.append(mk(stage, **SP, **HNET, **GS_W, wd=0.05, lr=1e-3))  # projection only
    t.append(mk(stage, **SP, **HNET, **MUONGS, lr=1e-3))  # projection, no decay
    # Remedies suggested by the plain-base diagnosis.
    for lr in (1e-3, 2e-3):
        t.append(mk(stage, **SP, **HNET, **GS_C, wd=0.05, emb="matrix_nogs", lr=lr))
        t.append(mk(stage, **SP, **HNET, **GS_C, wd=0.0125, init_scale=2.0, lr=lr))
    t.append(mk(stage, **SP, **HNET, **HALF, decay="corrected", wd=0.05, lr=1e-3))
    t.append(mk(stage, **SP, **{**HNET, "fallback_lr_mult": 1.0}, **GS_C, wd=0.05, lr=1e-3))  # faster gains
    return t


def _s4_radial() -> list[Trial]:
    """MuonGS direction plus an adaptive radial step: does giving the scale back to the loss help?"""

    stage = "s4_radial"
    t: list[Trial] = []
    for c in (0.25, 0.5, 1.0, 2.0):
        for lr in (7e-4, 1e-3):
            t.append(mk(stage, **SP, **MUONGS, radial_lr=c, lr=lr))
    for wd in (0.05, 0.2):
        t.append(mk(stage, **SP, **GS_C, wd=wd, radial_lr=1.0, lr=1e-3))
    for lr in (5e-4, 7e-4):
        t.append(mk(stage, **STD, **MUONGS, radial_lr=1.0, lr=lr))
        t.append(mk(stage, **STD, **GS_C, wd=0.2, radial_lr=1.0, lr=lr))
    return t


def _s5_headfree() -> list[Trial]:
    """Projection on hidden matrices only (embedding / output matrix left free).

    This is the routing closest to the H-Net trainer, whose output head is on
    the fallback path and which has no token embedding.  With the output scale
    free, how much does the decay under the projection still matter?
    """

    stage = "s5_headfree"
    free = dict(emb="matrix_nogs")
    t: list[Trial] = []
    for base in (dict(), HNET):
        for wd in (0.0125, 0.2):
            t.append(mk(stage, **SP, **base, **free, **GS_C, wd=wd, lr=1e-3))
        t.append(mk(stage, **SP, **base, **free, **GS_W, wd=0.05, lr=1e-3))
        t.append(mk(stage, **SP, **base, **free, **GS_FIXED, lr=1e-3))
        t.append(mk(stage, **SP, **base, **free, **GS_C, wd=0.05, lr=1.4e-3))
        t.append(mk(stage, **SP, **base, **free, **GS_C, wd=0.05, lr=7e-4))
    # Not yet run for either base at these points.
    t.append(mk(stage, **SP, **free, **MUONGS, lr=1.4e-3))
    t.append(mk(stage, **SP, **HNET, **free, **MUONGS, lr=1e-3))
    t.append(mk(stage, **SP, **HNET, **free, **GS_C, wd=0.05, lr=1e-3, seed=124))
    t.append(mk(stage, **SP, **HNET, **HALF, decay="corrected", wd=0.05, lr=1e-3, seed=124))
    t.append(mk(stage, **SP, **free, **MUONGS, init_scale=2.0, lr=2e-3))
    t.append(mk(stage, **SP, **free, **GS_C, wd=0.05, radial_lr=1.0, lr=1e-3))
    return t


def _s6_accumulator() -> list[Trial]:
    """"The weight norm is a free accumulator": projected Muon without decay or LR schedule.

    For a normalized direction the accumulator is a step counter, so the free
    norm gives theta_t = theta_0 / sqrt(1 + theta_0^2 t): an inverse-sqrt anneal
    whose time scale is set by the LR itself.  A-arms let the norm grow; C-arms
    pin the norm and put the same anneal into the matrix LR instead.
    """

    stage = "s6_accumulator"
    const = dict(data="sp", schedule="cosine", steps=8000, final_lr_scale=1.0)  # constant after warmup
    wsd = dict(data="sp", schedule="wsd", steps=8000)  # constant, then 20% linear decay to 10%
    t: list[Trial] = []
    for lr in (1e-3, 2e-3, 4e-3, 8e-3):
        t.append(mk(stage, **const, **MUONGS, lr=lr))
    for lr in (2e-3, 4e-3, 8e-3):
        t.append(mk(stage, **wsd, **MUONGS, lr=lr))
        t.append(mk(stage, **const, **GS_FIXED, emulate_gs_theta0=10.0 * lr, lr=lr))
    t.append(mk(stage, **wsd, **GS_FIXED, emulate_gs_theta0=0.04, lr=4e-3))
    # Controls: the cosine arm of these two already exists (4.3884 at wd 0.0125).
    t.append(mk(stage, **const, **GS_C, wd=0.0125, lr=7e-4))
    t.append(mk(stage, **wsd, **GS_C, wd=0.0125, lr=7e-4))
    # Unprojected references under the same two schedules.
    t.append(mk(stage, **const, **MUONW, wd=0.2, lr=1e-3))
    t.append(mk(stage, **wsd, **MUONW, wd=0.2, lr=1e-3))
    return t


def _s7_long_refine() -> list[Trial]:
    """Bracket the 32K-step LR optima (most were at the low edge of s3_long) and add a second seed."""

    stage = "s7_long_refine"
    long = dict(data="sp", schedule="cosine", steps=32000)
    t: list[Trial] = []
    # Lower learning rates for every family whose best s3_long point was its lowest LR.
    t.append(mk(stage, **long, **MUONW, wd=0.2, lr=5e-4))
    t.append(mk(stage, **long, **MUONC, wd=0.4, lr=5e-4))
    for lr in (5e-4, 7e-4):
        t.append(mk(stage, **long, **MUON, lr=lr))
        t.append(mk(stage, **long, **HALF, decay="none", lr=lr))
    t.append(mk(stage, **long, **HALF, decay="corrected", wd=0.2, lr=5e-4))
    t.append(mk(stage, **long, **MUONGS, lr=3.5e-4))
    t.append(mk(stage, **long, **MUONGS, init_scale=2.0, lr=5e-4))
    t.append(mk(stage, **long, **GS_FIXED, init_scale=2.0, lr=7e-4))
    t.append(mk(stage, **long, **GS_C, wd=0.0125, init_scale=2.0, lr=7e-4))
    # Second seed for the leading configuration of each family.
    for cell in (
        dict(**HALF, decay="none", lr=1e-3),
        dict(**MUONGS, init_scale=2.0, lr=7e-4),
        dict(**MUON, lr=1e-3),
        dict(**MUONC, wd=0.4, lr=7e-4),
        dict(**MUONW, wd=0.2, lr=7e-4),
        dict(**MUONGS, lr=5e-4),
    ):
        t.append(mk(stage, **long, **cell, seed=124))
    return t


def _timing() -> list[Trial]:
    """Step-time comparison.  Run with ONE job per GPU on otherwise idle GPUs."""

    stage = "timing"
    quick = dict(data="sp", schedule="cosine", steps=800, eval_every=800, eval_batches=1, final_eval_batches=1, diag_every=100000)
    return [
        mk(stage, **quick, opt="repo_muon", lr=1e-3, wd=0.2),
        mk(stage, **quick, **MUONW, wd=0.2, lr=1e-3),
        mk(stage, **quick, **GS_C, wd=0.05, lr=1e-3),
        mk(stage, **quick, **HALF, decay="corrected", wd=0.05, lr=1e-3),
        mk(stage, **quick, **HNET, **MUONW, wd=0.05, lr=1e-3),
        mk(stage, **quick, **HNET, **GS_C, wd=0.05, lr=1e-3),
        mk(stage, **quick, opt="repo_adamw", lr=5e-4, wd=0.05),
    ]


STAGES = {
    "smoke": _smoke,
    "s0_parity": _s0_parity,
    "s1_std": lambda: factorial("s1_std", "std"),
    "s1_sp": lambda: factorial("s1_sp", "sp"),
    "s2_diag": _s2_diag,
    "s2_refine": _s2_refine,
    "s2_hnet": _s2_hnet,
    "s3_seeds": _s3_seeds,
    "s3_long": _s3_long,
    "s3_hnet": _s3_hnet,
    "s4_radial": _s4_radial,
    "s5_headfree": _s5_headfree,
    "s6_accumulator": _s6_accumulator,
    "s7_long_refine": _s7_long_refine,
    "timing": _timing,
}


def stage_trials(stage: str) -> list[Trial]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    return STAGES[stage]()
