# Root SODA Scope Ablation, 50 Epochs

This run compares AdamW against the root optimizer with three SODA scopes:

- `all`: SODA anchor interpolation enabled for both matrix and fallback parameters.
- `matrix`: SODA enabled for matrix PMuonEq/NorMuon parameters and disabled for fallback/non-matrix parameters.
- `none`: SODA disabled for every parameter.

The implementation adds explicit group-level controls:

- `matrix_soda=True|False`
- `fallback_soda=True|False`

For `soda=matrix`, fallback parameters still use their ordinary fallback weight
decay. Only the SODA anchor pull is disabled for those parameters.

## Setup

- Dataset: CIFAR-10
- Model: `vit5_micro`
- Train/validation split: 45k/5k from train set, split seed 12345
- Test set: full 10k
- Seed: 123
- Epochs: 50
- Batch size: 512
- Warmup steps: 80
- Root optimizer: PMuonEq row gamma 0.35, beta 0.90, NorMuon beta 0.93
- Root grouping: named/effective-dimension routing
- GPU mode: 4 concurrent one-GPU trials across all visible GPUs
- PyTorch: 2.13.0.dev20260513+cu130

## Final Results

| rank | trial | SODA scope | final val loss | final val acc | best val loss | best val acc | test loss | test acc | avg step ms | examples/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `root_row_named_sodaall_mlr0.008_rg0.35_pb0.9_nb0.93` | all params | 0.4279 | 85.40 | 0.4095 | 85.84 | 0.4563 | 84.92 | 17.73 | 28,870 |
| 2 | `root_row_named_sodamatrix_mlr0.008_rg0.35_pb0.9_nb0.93` | matrix only | 0.4650 | 84.62 | 0.4390 | 85.54 | 0.4806 | 84.18 | 18.00 | 28,449 |
| 3 | `root_row_named_nosoda_mlr0.008_rg0.35_pb0.9_nb0.93` | none | 0.5491 | 83.76 | 0.4864 | 84.64 | 0.5579 | 83.61 | 17.99 | 28,459 |
| 4 | `adamw_cosine_lr0.004_wd0.001` | n/a | 0.5948 | 79.54 | 0.5887 | 80.14 | 0.6199 | 79.78 | 12.21 | 41,933 |

## Interpretation

SODA on all parameters was best in this 50-epoch seed. It improved test accuracy
by +5.14 points over AdamW and +1.31 points over no-SODA. Matrix-only SODA was
close to all-SODA on validation accuracy but lost 0.74 test-accuracy points,
suggesting that fallback/non-matrix SODA contributes useful regularization in
the longer run.

No-SODA learned fastest early but overfit more by the end: it reached high train
accuracy and weaker final validation/test loss. SODA-all had slower early
progress, then overtook the no-SODA variants during the second half of training.

Speed was essentially unchanged among root SODA scopes. AdamW was faster per
step, but substantially worse on final validation/test accuracy under this
configuration.

## Artifacts

- Full table: `e50/all_runs.csv`
- Per-trial metrics: `e50/*/metrics.jsonl`
- Loss curves: `e50/train_loss.png`, `e50/val_loss.png`
- Accuracy curve: `e50/val_acc.png`
- Speed charts: `e50/step_time_ms_bar.png`, `e50/examples_per_sec_bar.png`

