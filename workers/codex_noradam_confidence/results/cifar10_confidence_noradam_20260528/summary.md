# CIFAR-10 NorMuon+base vs AdamW Confidence Comparison

Date: 2026-05-28

## Scope

This is a narrowed comparison of only:

- AdamW baseline
- NorMuon+base: SODA + PMuonEq + Gram Newton-Schulz + NorMuon, AMUSE off

Model and data:

- Model: `vit5_micro`, `num_classes=10`, `img_size=32`, `drop_path_rate=0.05`
- Dataset: CIFAR-10 train/test via torchvision
- Augmentation: random crop with padding 4, random horizontal flip, CIFAR normalization
- Batch size: 512
- Runner: one trial per visible GPU
- Torch: `2.13.0.dev20260506+cu130`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition

## HPO

HPO used 12 epochs on full CIFAR-10.

AdamW search:

- LR: `0.002`, `0.003`, `0.004`
- Weight decay: `0.001`, `0.005`, `0.01`, `0.03`, `0.05`
- Schedule: cosine and constant after warmup

NorMuon+base search:

- Matrix LR: `0.006`, `0.007`, `0.008`
- Row gamma: `0.2`, `0.3`
- Column gamma: `0.0`, `0.1`
- NorMuon beta: `0.9`, `0.95`
- Fixed: SODA all, PMuonEq, PMuon beta `0.9`, momentum `0.95`, AMUSE off

Selected configs:

| Family | Selected config | 12-epoch val loss | 12-epoch val acc | Step time | Throughput |
|---|---:|---:|---:|---:|---:|
| AdamW | `lr=0.004`, `wd=0.001`, cosine | 0.8845 | 68.50% | 11.95 ms | 42.8k ex/s |
| NorMuon+base | `lr=0.008`, `row_gamma=0.3`, `col_gamma=0.0`, `normuon_beta=0.95` | 0.6273 | 78.39% | 22.07 ms | 23.2k ex/s |

## 20-Epoch Confirmation

Five seeds: `123`, `456`, `789`, `101112`, `131415`.

| Family | Final val loss | Final val acc | Best val loss | Best val acc | Step time | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 0.7469 +/- 0.0110 | 73.55% +/- 0.48 | 0.7467 +/- 0.0111 | 73.68% +/- 0.43 | 11.86 ms | 43.2k ex/s |
| NorMuon+base | 0.5647 +/- 0.0086 | 80.09% +/- 0.20 | 0.5523 +/- 0.0146 | 80.60% +/- 0.56 | 21.61 ms | 23.7k ex/s |

Delta at 20 epochs:

- NorMuon+base improved final validation loss by about `0.182`.
- NorMuon+base improved final validation accuracy by about `6.54` percentage points.
- NorMuon+base used about `1.82x` the AdamW step time.

## 50-Epoch Confirmation

Three seeds: `123`, `456`, `789`.

| Family | Final val loss | Final val acc | Best val loss | Best val acc | Step time | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 0.6167 +/- 0.0117 | 80.07% +/- 0.40 | 0.6061 +/- 0.0064 | 80.20% +/- 0.28 | 11.78 ms | 43.5k ex/s |
| NorMuon+base | 0.4284 +/- 0.0145 | 85.59% +/- 0.53 | 0.4167 +/- 0.0077 | 85.78% +/- 0.23 | 21.69 ms | 23.6k ex/s |

Delta at 50 epochs:

- NorMuon+base improved final validation loss by about `0.188`.
- NorMuon+base improved final validation accuracy by about `5.52` percentage points.
- NorMuon+base used about `1.84x` the AdamW step time.

## Interpretation

NorMuon+base is the better optimizer in this focused comparison by validation loss and accuracy at both 20 and 50 epochs. AdamW is substantially faster per step, but the quality gap is large enough that NorMuon+base also reaches AdamW's 50-epoch accuracy much earlier in the curve.

The result is consistent across HPO, five-seed 20-epoch confirmation, and three-seed 50-epoch confirmation. It also matches the direction of the external result table the user provided, though the absolute accuracy differs because this run uses this repo's `vit5_micro` harness, augmentation, and no EMA.

## Artifacts

- HPO CSV: `hpo/all_runs.csv`
- 20-epoch CSV: `final20/all_runs.csv`
- 50-epoch CSV: `final50/all_runs.csv`
- Loss curves: `hpo/val_loss.png`, `final20/val_loss.png`, `final50/val_loss.png`
- Accuracy curves: `hpo/val_acc.png`, `final20/val_acc.png`, `final50/val_acc.png`
- Speed bars: `hpo/step_time_ms_bar.png`, `final20/step_time_ms_bar.png`, `final50/step_time_ms_bar.png`

