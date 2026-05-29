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
- `experiments/run_cifar10_normuon_aspect_ablation.py` - launches the latest
  AdamW vs NorMuon aspect-scaling ablation, one trial per visible GPU.
- `experiments/plot_cifar10_normuon_results.py` - regenerates the loss,
  accuracy, and speed figures from a result directory.
- `results/cifar10_normuon_aspect_ablation_seed34000/` - latest result bundle
  with metrics JSONL, summaries, logs, resolved commands, markdown report, and
  PNG figures.

## Optimizer

The optimizer is the tuned local recipe:

```text
SODA anchor pull
+ PMuonEq row/column gradient-power scaling
+ Gram Newton-Schulz matrix orthogonalization
+ NorMuon row normalization
```

The standalone file intentionally removes the old broad optimizer-family
ablation switches:

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

For peer-review compatibility, the optimizer exposes two narrow NorMuon
ablation controls without changing the rest of the recipe:

```python
normuon_mode="row"              # default; row statistics for every matrix
normuon_aspect_scale=True       # default; winning tuned aspect multiplier
```

The tested alternatives are `normuon_aspect_scale=False` and
`normuon_mode="orientation"` with `normuon_aspect_scale=False`. They are useful
for reproducing the ablation below, but the recommended/default recipe remains
row-wise NorMuon with aspect scaling.

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

## Latest NorMuon Aspect Ablation

ViT-5 tiny / CIFAR-10, 50 epochs, seed `34000`, batch size 512 per trial. The
runner launched one single-GPU trial per visible GPU, so all four GPUs were
used concurrently. Full artifacts are in
`results/cifar10_normuon_aspect_ablation_seed34000/`.

| rank | optimizer | best val loss | best val acc | examples/s | mean step |
|---:|---|---:|---:|---:|---:|
| 1 | SODA-PMuonEq-NorMuon row + aspect | 0.4036 | 87.16% | 14,997 | 34.14 ms |
| 2 | SODA-PMuonEq-NorMuon row | 0.4174 | 86.43% | 14,932 | 34.29 ms |
| 3 | SODA-PMuonEq-NorMuon orientation | 0.4212 | 86.98% | 14,990 | 34.16 ms |
| 4 | tuned AdamW baseline | 0.5476 | 83.03% | 27,075 | 18.91 ms |

This run supports keeping the aspect multiplier in the default recipe. Removing
it or switching to orientation-aware row/column statistics remained much better
than AdamW, but both were worse than the default on best validation loss.

Generated figures:

- `results/cifar10_normuon_aspect_ablation_seed34000/figures/loss_curves.png`
- `results/cifar10_normuon_aspect_ablation_seed34000/figures/accuracy_best_vs_adamw.png`
- `results/cifar10_normuon_aspect_ablation_seed34000/figures/iteration_speed.png`

Reproduce the run:

```bash
/home/catid/attractor/.venv/bin/python \
  workers/codex_soda_pmuoneq_normuon/experiments/run_cifar10_normuon_aspect_ablation.py \
  --epochs 50 --eval-bins 8 --batch-size 512 --num-workers 8 \
  --output-dir workers/codex_soda_pmuoneq_normuon/results/cifar10_normuon_aspect_ablation_seed34000
```

Regenerate figures:

```bash
/home/catid/attractor/.venv/bin/python \
  workers/codex_soda_pmuoneq_normuon/experiments/plot_cifar10_normuon_results.py \
  workers/codex_soda_pmuoneq_normuon/results/cifar10_normuon_aspect_ablation_seed34000
```

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
