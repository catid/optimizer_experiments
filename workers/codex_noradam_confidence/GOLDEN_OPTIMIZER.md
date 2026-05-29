# Golden Optimizer

Updated: 2026-05-29

This folder now has a stripped standalone implementation of the current winning
optimizer family:

```text
golden_soda_pmuoneq_normuon.py
```

## Algorithm

The golden optimizer implements only:

```text
SODA + row-only PMuonEq + Gram Newton-Schulz + NorMuon
```

The following older switches are intentionally removed:

- AMUSE / schedule-free iterate averaging
- MiMuon branch mixing
- full/stale PMuon
- PMuonEq disable path
- NorMuon disable path
- post-NorMuon aspect scaling

Default recipe:

```text
lr = 8e-3
warmup_steps = 80
momentum = 0.95
pmuoneq_beta = 0.90
row_gamma = 0.35
col_gamma = 0.0  # not implemented; row-only PMuonEq
normuon_beta = 0.93
ns_steps = 5
SODA = all parameter groups
```

For each matrix parameter:

```text
M_t = 0.95 M_{t-1} + 0.05 G_t
U_t = 0.05 G_t + 0.95 M_t

r_t = 0.90 r_{t-1} + 0.10 mean_cols(G_t^2)
A_t = normalize(r_t^-0.35) * U_t

Q_t = GramNS_5(A_t)

s_t = 0.93 s_{t-1} + 0.07 mean_cols_or_rows(Q_t^2)
D_t = Q_t / sqrt(s_t + eps)
D_t = D_t * ||Q_t||_F / ||D_t||_F

W_t <- W_t - lr_t * 0.2 * sqrt(max(rows, cols)) * D_t
W_t <- W_t + (W_init - W_before_step) / (t + 1)
```

Fallback parameters use the winning RMS-style second-moment fallback and the
same SODA anchor path.

## Reproduction Validation

I validated the golden file against the previous best proper-split CIFAR-10
result using the same runner, model, split, seeds, batch size, and official test
readout.

Command shape:

```bash
/home/catid/screen/.venv/bin/python \
  workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --preset best_cifar10 \
  --only '^golden_' \
  --output-dir workers/codex_noradam_confidence/results/cifar10_golden_repro_20260529/final50 \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 50 \
  --train-subset 0 --val-subset 0 \
  --val-source train_split \
  --train-val-size 5000 \
  --split-seed 12345 \
  --eval-test \
  --test-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --model vit5_micro \
  --seeds 123,456,789 \
  --no-sync-step-timing
```

Result: exact metric reproduction of the previous AnchorMuon winner at printed
CSV precision.

| metric | previous AnchorMuon | golden replay |
|---|---:|---:|
| final train loss | 0.354673770767 +/- 0.007306816049 | 0.354673770767 +/- 0.007306816049 |
| final val loss | 0.441353829002 +/- 0.022751204447 | 0.441353829002 +/- 0.022751204447 |
| final val acc | 84.9733333333 +/- 0.3523256070 | 84.9733333333 +/- 0.3523256070 |
| best val loss | 0.430994171588 +/- 0.011240487027 | 0.430994171588 +/- 0.011240487027 |
| best val acc | 85.2466666667 +/- 0.2663331247 | 85.2466666667 +/- 0.2663331247 |
| official test loss | 0.460712182077 +/- 0.019593584841 | 0.460712182077 +/- 0.019593584841 |
| official test acc | 84.7733333333 +/- 0.6474823035 | 84.7733333333 +/- 0.6474823035 |

The golden replay artifact is:

```text
workers/codex_noradam_confidence/results/cifar10_golden_repro_20260529/final50/
```

## Grouping Note

The CIFAR reproduction intentionally uses the same `_anchor_param_groups()` path
as the previous best run, so the optimizer path is identical for the benchmark.

For language-modeling work, prefer:

```python
from golden_soda_pmuoneq_normuon import (
    GoldenSodaPmuonEqNorMuon,
    build_golden_param_groups,
)

groups = build_golden_param_groups(model.named_parameters())
optimizer = GoldenSodaPmuonEqNorMuon(groups)
```

The named helper deduplicates tied/shared parameters and keeps embeddings, LM
heads, norms, biases, vectors, and scalars in the fallback path.
