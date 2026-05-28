# Codex Worker: SODA-PMuonEq-NorMuon

This folder contains my standalone optimizer contribution to the shared
`optimizer_experiments` monorepo.

## Contents

- `soda_pmuoneq_normuon.py` - a focused, copyable PyTorch optimizer file.
- `tests/test_soda_pmuoneq_normuon.py` - self-contained CPU tests for grouping,
  finite updates, state creation, no-op train/eval compatibility, state-dict
  resume parity, bucket parity, and a simple train-loss sanity check.
- `tests/ddp_smoke_soda_pmuoneq_normuon.py` - torchrun smoke test that checks
  model parameters and optimizer tensor state stay rank-identical after DDP
  gradient all-reduce.
- `results/cifar10_seed34000_summary.csv` - CIFAR-10 ViT-5 comparison against a
  tuned AdamW baseline.
- `results/cifar10_seed34000_curves.csv` - bin-level train/validation loss and
  speed metrics from the same run.

## Optimizer

The optimizer is the tuned local recipe:

```text
SODA anchor pull
+ PMuonEq row/column gradient-power scaling
+ Gram Newton-Schulz matrix orthogonalization
+ NorMuon row normalization
```

The standalone file intentionally removes the previous ablation switches:

- no AMUSE / schedule-free branch,
- no MiMuon branch,
- no optional PMuonEq disable path,
- no optional NorMuon disable path.

The matrix path is always the best local recipe. Biases, norms, embeddings,
heads, and other fallback parameters use the same RMS/AdamW-style
second-moment fallback that matched the prior winning implementation. This
fallback is intentionally not ordinary AdamW: it uses the current gradient
divided by a bias-corrected second-moment denominator, with no first-moment
EMA.

Use `build_soda_pmuoneq_normuon_param_groups(model.named_parameters())` for
normal training. Passing raw `model.parameters()` is supported, but it routes
all `ndim >= 2` tensors through the matrix path, including embeddings and output
heads. The named-parameter helper keeps common embeddings, heads, norms, and
biases in the fallback path.

The NorMuon step includes the tuned aspect-ratio multiplier
`sqrt(max(1, rows / cols))` after Frobenius-norm restoration. That is a real
layerwise step-size choice and is part of this recipe, so compare it separately
from NorMuon implementations that preserve only the Frobenius norm.

## Tuned Defaults

```python
SodaPmuonEqNorMuon(
    params,
    matrix_lr=8e-3,
    adam_lr=8e-4,
    momentum=0.95,
    pmuoneq_beta=0.90,
    row_gamma=0.35,
    col_gamma=0.05,
    normuon_beta2=0.93,
    adam_weight_decay=0.05,
    matrix_weight_decay=0.0,
    warmup_steps=10,
)
```

By default the optimizer owns a short internal linear warmup:

```text
lr_t = base_lr * min(1, t / warmup_steps)
```

Set `use_external_lr=True` on the optimizer or an individual parameter group if
an external scheduler should write `group["lr"]` before each `step()`.

## CIFAR-10 Check

ViT-5 tiny / CIFAR-10, 50 epochs, DDP across 4 GPUs, global batch 512, seed
`34000`:

| optimizer | best val loss | final val loss | best val acc | final val acc | examples/s | mean step |
|---|---:|---:|---:|---:|---:|---:|
| SODA-PMuonEq-NorMuon standalone | 0.4598 | 0.4950 | 85.77% | 85.70% | 20,260 | 23.75 ms |
| tuned AdamW baseline | 0.5855 | 0.7083 | 81.56% | 81.54% | 27,610 | 17.33 ms |

The standalone optimizer preserved the prior quality advantage after stripping
the ablation conditionals. It is slower than AdamW by about 1.37x step time on
this setup, but reached substantially better validation loss and accuracy.

## Quick Test

From this folder:

```bash
python -m pytest -q tests/test_soda_pmuoneq_normuon.py
```

Optional DDP smoke test from the repository root:

```bash
torchrun --standalone --nproc_per_node=2 \
  workers/codex_soda_pmuoneq_normuon/tests/ddp_smoke_soda_pmuoneq_normuon.py
```
