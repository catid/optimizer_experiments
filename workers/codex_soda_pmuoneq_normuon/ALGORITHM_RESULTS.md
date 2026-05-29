# SODA-PMuonEq-NorMuon Summary

## Best Confirmed Version

The best confirmed version in this folder is exactly:

```text
trial id:  row_aspect_mlr0.008_rg0.35_cg0.05_nb0.93
optimizer: SodaPmuonEqNorMuon
mode:      row-wise NorMuon + aspect scaling
```

Use this recipe when referring to the reported best result:

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
    normuon_mode="row",
    normuon_aspect_scale=True,
    matrix_weight_decay=0.0,
    adam_weight_decay=0.05,
    warmup_steps=10,
)
```

It was the best by mean final validation loss on the following workload:

| field | value |
|---|---|
| model | `vit5_tiny` from the local ViT-5 registration |
| model size | 2,691,274 trainable parameters |
| architecture shape | 32x32 input, patch size 4, hidden size 192, 6 transformer blocks, MLP hidden size 768, 10-class head |
| dataset | CIFAR-10, 50,000 train images and 10,000 validation/test images |
| augmentation | train: random crop 32 with padding 4, random horizontal flip, CIFAR normalization; eval: CIFAR normalization |
| training length | 50 epochs, 4,850 optimizer steps |
| batch size | 512 images per single-GPU trial |
| run mode | one independent single-GPU trial per visible GPU; this final comparison used all four GPUs for parallel trials, not DDP |
| precision/layout | CUDA bf16 autocast, channels-last model/input layout, TF32 enabled |
| eval schedule | 8 evenly spaced validation bins plus final summary |
| seeds | `34000`, `456`, `789` |
| baseline | tuned fused AdamW, `lr=2.5e-3`, `weight_decay=0.005`, same model/data/steps/eval |

Aggregate result for that exact version:

```text
final val loss: 0.3975 +/- 0.0150
best val loss:  0.3935 +/- 0.0119
final val acc:  87.44% +/- 0.43%
throughput:     14,816 images/s
step time:      34.56 ms
```

The tuned AdamW baseline on the same workload reached:

```text
final val loss: 0.5868 +/- 0.0289
best val loss:  0.5614 +/- 0.0191
final val acc:  82.56% +/- 0.89%
throughput:     26,441 images/s
step time:      19.37 ms
```

Scope caveat: this is a CIFAR-10 / ViT-5 tiny result. It should not be quoted
as the best recipe for larger ViTs, language models, or different batch-size
regimes until those workloads are tuned and confirmed separately.

## Algorithm

The standalone optimizer in this folder is a quality-first matrix optimizer for
ViT-style models:

```text
SODA anchor pull
+ PMuonEq row/column gradient-power scaling
+ Gram Newton-Schulz matrix orthogonalization
+ NorMuon row normalization with aspect scaling
+ RMS/AdamW-style fallback for non-matrix parameters
```

For each matrix weight `W_t` with gradient `G_t`, it builds a Nesterov-style
momentum source:

```text
M_t = beta_m M_{t-1} + (1 - beta_m) G_t
U_t = (1 - beta_m) G_t + beta_m M_t
```

PMuonEq is the fast diagonal approximation to PMuon. It avoids dense covariance
matrices, QR/eigendecompositions, and inverse-power applications:

```text
r_t = beta_p r_{t-1} + (1 - beta_p) mean_cols(G_t^2)
c_t = beta_p c_{t-1} + (1 - beta_p) mean_rows(G_t^2)
A_t = normalize(r_t^-gamma_row) * U_t * normalize(c_t^-gamma_col)
```

The matrix direction is then produced by Gram Newton-Schulz:

```text
D_t = 0.2 * sqrt(max(rows, cols)) * GramNS(A_t)
```

NorMuon normalizes the post-Gram direction with a row-wise second-moment EMA,
restores the Frobenius norm, and applies the tuned tall-matrix aspect
multiplier:

```text
s_t = beta_n s_{t-1} + (1 - beta_n) mean_cols(D_t^2)
N_t = D_t / sqrt(s_t + eps)
N_t = N_t * ||D_t||_F / ||N_t||_F
N_t = N_t * sqrt(max(1, rows / cols))
```

SODA applies an anchor pull toward the initialization before the learned update:

```text
lambda_t = min(1, soda_lambda_scale / (t + 1)^soda_lambda_power)
W_t <- (1 - lambda_t) W_t + lambda_t W_0
W_{t+1} = W_t - lr_t N_t
```

By default, SODA disables decoupled matrix weight decay because SODA is the
matrix regularization path. Fallback parameters still use ordinary weight
decay.

## Tuned Recipe

This section repeats the best confirmed recipe for implementation convenience.

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
    normuon_mode="row",
    normuon_aspect_scale=True,
    adam_weight_decay=0.05,
    matrix_weight_decay=0.0,
    warmup_steps=10,
)
```

## Results

Latest peer-feedback comparison:

- Model/data: `vit5_tiny` on CIFAR-10.
- Model size: 2,691,274 trainable parameters.
- Training: 50 epochs, 4,850 steps, batch size 512 per single-GPU trial,
  bf16 autocast, channels-last layout, TF32 enabled.
- Hardware/run mode: one single-GPU trial per visible GPU, all four GPUs used
  concurrently for the final comparison. This was not DDP.
- Seeds: `34000`, `456`, `789`.
- Baseline: tuned AdamW, `lr=2.5e-3`, `weight_decay=0.005`.

| rank | optimizer | final val loss | best val loss | final val acc | examples/s | step ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | row + aspect, rg0.35 | 0.3975 +/- 0.0150 | 0.3935 +/- 0.0119 | 87.44% +/- 0.43% | 14,816 | 34.56 |
| 2 | orient + aspect, rg0.40 | 0.4087 +/- 0.0155 | 0.3988 +/- 0.0033 | 87.04% +/- 0.55% | 14,689 | 34.86 |
| 3 | orient + aspect, rg0.30/cg0 | 0.4115 +/- 0.0123 | 0.3996 +/- 0.0084 | 87.04% +/- 0.46% | 14,885 | 34.40 |
| 4 | row + aspect, rg0.40 | 0.4115 +/- 0.0140 | 0.4073 +/- 0.0174 | 87.05% +/- 0.59% | 14,580 | 35.12 |
| 5 | orient no aspect, rg0.40 | 0.4133 +/- 0.0194 | 0.3978 +/- 0.0099 | 87.05% +/- 0.41% | 14,602 | 35.06 |
| 6 | row no aspect, rg0.30/cg0 | 0.4146 +/- 0.0166 | 0.4105 +/- 0.0095 | 87.05% +/- 0.66% | 15,024 | 34.08 |
| 7 | AdamW baseline | 0.5868 +/- 0.0289 | 0.5614 +/- 0.0191 | 82.56% +/- 0.89% | 26,441 | 19.37 |

Conclusion: the peer-requested orientation variants were competitive, but the
three-seed final-loss aggregate still favors the original row-wise aspect
recipe. AdamW remains about `1.8x` faster per step on this small model, so this
optimizer is recommended when final quality matters more than raw iteration
speed.

## Artifacts

- Aggregate report:
  `results/cifar10_peer_feedback_final50_aggregate/report.md`
- Aggregate CSV:
  `results/cifar10_peer_feedback_final50_aggregate/aggregate_summary.csv`
- Mean validation-loss plot:
  `results/cifar10_peer_feedback_final50_aggregate/figures/mean_validation_loss.png`
- Experiment runner:
  `experiments/run_cifar10_normuon_aspect_ablation.py`
- Aggregate script:
  `experiments/aggregate_cifar10_peer_feedback.py`

## Validation

The pushed implementation was checked with:

```bash
python -m py_compile \
  workers/codex_soda_pmuoneq_normuon/soda_pmuoneq_normuon.py \
  workers/codex_soda_pmuoneq_normuon/experiments/run_cifar10_normuon_aspect_ablation.py \
  workers/codex_soda_pmuoneq_normuon/experiments/plot_cifar10_normuon_results.py \
  workers/codex_soda_pmuoneq_normuon/experiments/aggregate_cifar10_peer_feedback.py

python -m pytest -q workers/codex_soda_pmuoneq_normuon/tests/test_soda_pmuoneq_normuon.py

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  workers/codex_soda_pmuoneq_normuon/tests/ddp_smoke_soda_pmuoneq_normuon.py
```
