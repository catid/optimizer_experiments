# Optimizer Experiments Monorepo

## Current Best Result

The clearest current winner lives in
`workers/codex_noradam_confidence`.

**Best version:** `normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93`

**Algorithm:** AnchorMuon with SODA on all parameter groups, row-only PMuonEq,
five-step Gram Newton-Schulz, and NorMuon. AMUSE, MiMuon, and post-NorMuon
aspect scaling are disabled.

**Exact setting:** ViT-5 micro on CIFAR-10, 45k train / 5k validation split from
the official training set, official 10k test split evaluated only at the end,
batch size 512, 50 epochs, three seeds, BF16 autocast, channels-last tensors,
16 dataloader workers, and one trial per visible GPU.

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
from optimizer import SodaPmuonEqNorMuon

optimizer = SodaPmuonEqNorMuon(model)
```

The root file implements only the focused winner: SODA anchor updates,
row-only PMuonEq, Gram Newton-Schulz, and NorMuon. It intentionally does not
include AMUSE, MiMuon, full PMuon, column PMuonEq, or aspect-scaling ablations.
The param-group builder remains available for advanced custom routing, but
normal training code should not need to construct optimizer groups by hand.

### Default Hyperparameters

The constructor defaults are the recommended starting recipe from the current
experiments:

| Parameter | Default | Start by tuning? | Notes |
|---|---:|---|---|
| `matrix_lr` | `8e-3` | Yes | Main LR for matrix weights. Try `0.004`, `0.006`, `0.008` first. |
| `row_gamma` | `0.35` | Yes | PMuonEq row scaling strength. Try `0.25`, `0.35`, `0.45`. |
| `fallback_lr` | same as `matrix_lr` | Later | LR for embeddings, heads, norms, biases, scalars, vectors. |
| `normuon_beta2` | `0.93` | Later | Row second-moment smoothing after GramNS. Try `0.90`, `0.93`, `0.95`. |
| `fallback_weight_decay` | `0.05` | Later | Applies only to fallback parameters; matrix params use SODA instead. |
| `warmup_steps` | `80` | Rarely | Increase if early steps are unstable; shorten only by ablation. |
| `min_matrix_dim` | `2` | Rarely | Keeps tiny 2D tensors out of the spectral path unless a custom filter routes them. |
| `momentum` | `0.95` | Usually no | Momentum for the matrix source update. |
| `pmuoneq_beta` | `0.90` | Usually no | EMA for row gradient-power estimates. |
| `fallback_betas` | `(0.9, 0.95)` | Usually no | RMS/AdamW-style fallback moments. |
| `soda_lambda_scale`, `soda_lambda_power` | `1.0`, `1.0` | Usually no | SODA anchor schedule; changing this changes the regularizer. |
| `eps` values and `ns_compute_dtype` | internal defaults | No | Numerical and profiling knobs. |

Practical tuning order: start with the defaults, tune only `matrix_lr` and
`row_gamma`, then revisit `fallback_lr`/`normuon_beta2` if the result is close.
