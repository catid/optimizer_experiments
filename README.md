# Optimizer Experiments Monorepo

## Current Best Result

The current best shippable optimizer is the root [`optimizer.py`](optimizer.py)
implementation:

**Best version:** `AnchorMuon` from root `optimizer.py`

**Algorithm:** SODA anchor updates on all parameter groups, row-only PMuonEq,
five-step Gram Newton-Schulz, and NorMuon. Matrix weights use SODA instead of
ordinary weight decay. Fallback tensors also receive the SODA anchor update and
do not have a separate decay knob. AMUSE, MiMuon, full PMuon, column PMuonEq,
and post-NorMuon aspect scaling are disabled.

**Latest schedule validation:** ViT-5 micro on CIFAR-10, 45k train / 5k
validation split from the official training set, official 10k test split
evaluated only at the end, batch size 512, 50 epochs, seed `123`, BF16
autocast, channels-last tensors, 16 dataloader workers. Learning-rate schedules
were owned by the trainer. A 12-epoch HPO picked the best LR for each schedule,
then only those schedule winners were replayed for 50 epochs.

The best observed recipe is root `AnchorMuon` with trainer-side 80-step warmup
and WSD schedule: `lr=0.012`, `lr_final_scale=0.1`,
`wsd_decay_frac=0.2`, `row_gamma=0.35`, `pmuoneq_beta=0.90`,
`normuon_beta2=0.93`.

| Recipe | Schedule | Official test acc | Official test loss | Final val acc | Final val loss | Best val acc | Best val loss | Step time | Examples/sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnchorMuon | WSD | 87.67% | 0.4111 | 88.36% | 0.4005 | 88.36% | 0.3716 | 16.23 ms | 31.5k |
| AnchorMuon | constant | 85.74% | 0.4317 | 86.06% | 0.4036 | 86.54% | 0.4036 | 17.19 ms | 29.8k |
| AnchorMuon | cosine | 86.90% | 0.4588 | 87.40% | 0.4511 | 87.48% | 0.4097 | 16.96 ms | 30.2k |
| AnchorMuon | linear | 86.75% | 0.4597 | 87.18% | 0.4434 | 87.58% | 0.4151 | 17.02 ms | 30.1k |
| AdamW baseline | cosine | 79.55% | 0.6240 | 79.62% | 0.6133 | 79.64% | 0.5998 | 11.39 ms | 45.0k |

The 12-epoch HPO stage selected `lr=0.012` for every AnchorMuon schedule:

| Schedule | Best 12-epoch HPO trial | LR | 12-epoch val loss | 12-epoch val acc | Step time |
|---|---|---:|---:|---:|---:|
| constant | `root_named_constant_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.6865 | 75.36% | 16.94 ms |
| cosine | `root_named_cosine_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5946 | 79.30% | 17.09 ms |
| linear | `root_named_linear_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5820 | 79.46% | 16.86 ms |
| WSD | `root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5632 | 80.08% | 16.06 ms |
| AdamW baseline | `adamw_cosine_lr0.004_wd0.001` | 0.004 | 0.9095 | 67.56% | 11.04 ms |

Result bundle:
`workers/codex_noradam_confidence/results/root_lr_schedule_sweep_20260529/`.
The final 50-epoch schedule-winner plots are in
`final50_schedule_winners/val_loss.png`, `val_acc.png`, and
`step_time_ms_bar.png`.

![Validation accuracy curves for the 50-epoch AnchorMuon schedule winners](workers/codex_noradam_confidence/results/root_lr_schedule_sweep_20260529/final50_schedule_winners/val_acc.png)

**Previous three-seed direct root validation:** same ViT-5 micro CIFAR-10 split
and official test protocol, but using the earlier constant-LR direct root
recipe at `lr=8e-3` over seeds `123,456,789`.

| Recipe | Official test acc | Official test loss | Final val acc | Final val loss | Best val acc | Best val loss | Step time | Examples/sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Root `optimizer.py` default | 84.94% | 0.4501 | 85.72% | 0.4286 | 86.03% | 0.4129 | 16.57 ms | 30.9k |

The strongest multi-seed evidence lives in `workers/codex_noradam_confidence`
and used the same core recipe in the research runner. That package is useful
for confidence intervals and ablation context:

| Recipe | Official test acc | Official test loss | Final val acc | Final val loss | Step time |
|---|---:|---:|---:|---:|---:|
| AnchorMuon + SODA + PMuonEq + GramNS + NorMuon | 84.77% +/- 0.65 | 0.4607 +/- 0.0196 | 84.97% +/- 0.35 | 0.4414 +/- 0.0228 | 19.95 ms |
| AnchorMuon + NorMuon + aspect scale | 84.55% +/- 0.40 | 0.4595 +/- 0.0074 | 85.25% +/- 0.32 | 0.4445 +/- 0.0130 | 19.92 ms |
| AdamW cosine baseline | 79.28% +/- 0.23 | 0.6338 +/- 0.0125 | 79.69% +/- 0.22 | 0.6180 +/- 0.0043 | 11.59 ms |

Use `workers/codex_noradam_confidence/ALGORITHM_RESULTS.md` for the full
self-contained algorithm description, hyperparameters, protocol, and caveats.
The runner also exposes a `--preset best_cifar10` preset containing the winning
configuration, the closest aspect-scaled near miss, and the AdamW baseline.

## Standalone Optimizer

The root [`optimizer.py`](optimizer.py) is the shippable single-file optimizer
for reuse in other projects. It exposes:

```python
from optimizer import AnchorMuon

optimizer = AnchorMuon(model)
print(optimizer.group_summary())
```

The root file implements only the focused winner: SODA anchor updates,
row-only PMuonEq, Gram Newton-Schulz, and NorMuon. It intentionally does not
include AMUSE, MiMuon, full PMuon, column PMuonEq, or aspect-scaling ablations.
The direct validation above used the normal public API, not a trainer-side
adapter. The param-group builder remains available for advanced custom routing,
but normal training code should not need to construct optimizer groups by hand.
Routing is based on effective tensor shape only: tensors with at least two
non-singleton dimensions go through the matrix path, and scalar/vector tensors
go through the fallback path. Names are retained only for `group_summary()` and
diagnostics. Sparse gradients are not supported; use dense embeddings or a
separate sparse optimizer for those parameters.

Trainer integration should be boring:

```python
optimizer = AnchorMuon(
    model,
    lr=8e-3,
    fallback_lr=None,  # defaults to lr
    row_gamma=0.35,
    normuon_beta2=0.93,
)

# Learning-rate schedules live in the trainer, not inside AnchorMuon.
for group in optimizer.param_groups:
    group["lr"] = scheduled_lr
```

Do not scale gradients, override the matrix grouping to reproduce old ablation
paths, or subclass the optimizer for aspect/column-gamma behavior when testing
the root file. Those are research ablations, not the shippable root optimizer.

### Default Hyperparameters

The constructor defaults are the recommended starting recipe from the current
experiments:

| Parameter | Default | Start by tuning? | Notes |
|---|---:|---|---|
| `lr` | `8e-3` | Yes | Conservative starting LR consumed from the param group. For this ViT-5 CIFAR-10 harness, 12-epoch HPO selected `0.012` for the best WSD/constant/cosine/linear replays. |
| `row_gamma` | `0.35` | Yes | Row-only PMuonEq scaling strength before GramNS. Try `0.25`, `0.35`, `0.45`. |
| `fallback_lr` | same as `lr` | Later | LR for scalar/vector/fallback tensors. Leave as `None` first. |
| `normuon_beta2` | `0.93` | Later | Row second-moment smoothing after GramNS. Try `0.90`, `0.93`, `0.95`. |
| `min_matrix_dim` | `2` | Rarely | Keeps tiny effective matrices out of the spectral path. |
| `momentum` | `0.95` | Usually no | Momentum for the matrix source update. |
| `pmuoneq_beta` | `0.90` | Usually no | EMA for row gradient-power estimates. |
| `fallback_betas` | `(0.9, 0.95)` | Usually no | RMS/AdamW-style fallback moments. |
| `soda_lambda_scale`, `soda_lambda_power` | `1.0`, `1.0` | Usually no | SODA anchor schedule; changing this changes the regularizer. |
| `eps` values and `ns_compute_dtype` | internal defaults | No | Numerical and profiling knobs. |

The optimizer no longer has `matrix_lr`, `warmup_steps`, `base_lr`, or
`use_external_lr` knobs. Warmup, WSD, linear decay, cosine decay, and constant
LR are standard trainer-side schedules that update each param group's `lr`.

Practical tuning order: start with the defaults, tune `lr` and the trainer-side
schedule first, then tune `row_gamma`. In the latest schedule study, the useful
workflow was 12-epoch HPO over constant/cosine/linear/WSD schedules and LR
candidates, followed by a 50-epoch replay of the best config per schedule.
Only revisit `fallback_lr`/`normuon_beta2` if the result is close.

## References

- SODA anchor regularization: [Optimistic Dual Averaging Unifies Modern Optimizers](https://arxiv.org/abs/2605.11172). AnchorMuon uses this as an always-on initialization-anchor pull instead of ordinary weight decay.
- Gram Newton-Schulz / polar update: [Dao-AILab gram-newton-schulz reference implementation](https://github.com/Dao-AILab/gram-newton-schulz/blob/main/gram_newton_schulz/gram_newton_schulz.py). AnchorMuon uses this family for matrix orthogonalization.
- NorMuon / HTMuon lineage: [HTMuon: Improving Muon via Heavy-Tailed Spectral Correction](https://arxiv.org/abs/2603.10067) and the [HTMuon reference code](https://github.com/TDCSZ327/HTmuon). AnchorMuon uses the post-Gram row-normalization idea, not the full HTMuon optimizer.
- PMuon lineage: the [PMuon track-3 implementation notes](https://github.com/zzp1012/modded-nanogpt/tree/pmuon-track3-3225/records/track_3_optimization/results/20260507_pmuon) motivated the pre-polar preconditioning idea. AnchorMuon implements only a cheap row-only PMuonEq approximation, not dense two-sided PMuon.
- WSD schedule context: [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192). WSD is implemented in the training harness, not in `optimizer.py`.
