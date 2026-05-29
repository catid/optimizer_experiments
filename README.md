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

**Direct root validation:** ViT-5 micro on CIFAR-10, 45k train / 5k validation
split from the official training set, official 10k test split evaluated only at
the end, batch size 512, 50 epochs, seeds `123,456,789`, BF16 autocast,
channels-last tensors, 16 dataloader workers. The trainer called the root
optimizer directly as `AnchorMuon(model.named_parameters(), ...)`. No trainer-side custom
parameter groups, matrix filters, root subclasses, column-gamma adapters, or
aspect-scaling adapters were used.

| Recipe | Official test acc | Official test loss | Final val acc | Final val loss | Best val acc | Best val loss | Step time | Examples/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
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
Prefer passing a module or `model.named_parameters()` rather than
`model.parameters()`, because names are needed to route embeddings, heads,
normalization weights, and tied tensors safely. Sparse gradients are not
supported; use dense embeddings or a separate sparse optimizer for those
parameters.

Trainer integration should be boring:

```python
optimizer = AnchorMuon(
    model.named_parameters(),
    matrix_lr=8e-3,
    fallback_lr=8e-3,
    row_gamma=0.35,
    normuon_beta2=0.93,
    warmup_steps=80,
)
```

Do not scale gradients, override the matrix grouping to reproduce old ablation
paths, or subclass the optimizer for aspect/column-gamma behavior when testing
the root file. Those are research ablations, not the shippable root optimizer.

### Default Hyperparameters

The constructor defaults are the recommended starting recipe from the current
experiments:

| Parameter | Default | Start by tuning? | Notes |
|---|---:|---|---|
| `matrix_lr` | `8e-3` | Yes | Main LR for matrix weights. Try `0.004`, `0.006`, `0.008` first. |
| `row_gamma` | `0.35` | Yes | PMuonEq row scaling strength. Try `0.25`, `0.35`, `0.45`. |
| `fallback_lr` | same as `matrix_lr` | Later | LR for embeddings, heads, norms, biases, scalars, vectors. |
| `normuon_beta2` | `0.93` | Later | Row second-moment smoothing after GramNS. Try `0.90`, `0.93`, `0.95`. |
| `warmup_steps` | `80` | Rarely | Increase if early steps are unstable; shorten only by ablation. |
| `min_matrix_dim` | `2` | Rarely | Keeps tiny 2D tensors out of the spectral path unless a custom filter routes them. |
| `momentum` | `0.95` | Usually no | Momentum for the matrix source update. |
| `pmuoneq_beta` | `0.90` | Usually no | EMA for row gradient-power estimates. |
| `fallback_betas` | `(0.9, 0.95)` | Usually no | RMS/AdamW-style fallback moments. |
| `soda_lambda_scale`, `soda_lambda_power` | `1.0`, `1.0` | Usually no | SODA anchor schedule; changing this changes the regularizer. |
| `eps` values and `ns_compute_dtype` | internal defaults | No | Numerical and profiling knobs. |

Practical tuning order: start with the defaults, tune only `matrix_lr` and
`row_gamma`, then revisit `fallback_lr`/`normuon_beta2` if the result is close.
