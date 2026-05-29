# CIFAR-10 5-Seed Optimizer Confirmation

Dataset/model: CIFAR-10 fixed 45k/5k train-validation split, official 10k test evaluation, same CIFAR model/augmentation pipeline as the worker ablation runner. Each selected config was tuned by the preceding 12-epoch HPO pass and then rerun for 50 epochs on seeds 123, 456, 789, 101112, 131415 with batch size 512 and two visible RTX PRO 6000 Blackwell GPUs scheduled one trial per GPU.

| Rank | Method | Selected config | Val acc % | Test acc % | Val loss | Test loss | Step ms | Examples/s |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon RMS | AnchorMuon lr=0.012 WSD SODA+PMuonEq+NorMuon RMS fallback | 87.76 ± 0.61 | 87.61 ± 0.48 | 0.4069 ± 0.0336 | 0.4203 ± 0.0239 | 17.27 ± 0.07 | 29639 ± 115 |
| 2 | AnchorMuon AdamATan2 | AnchorMuon lr=0.014 WSD SODA+PMuonEq+NorMuon AdamATan2 fallback_lr_mult=0.5 | 87.96 ± 0.44 | 87.57 ± 0.39 | 0.3923 ± 0.0119 | 0.4147 ± 0.0152 | 17.51 ± 0.16 | 29240 ± 267 |
| 3 | Plain Muon WSD | Muon+Gram lr=0.014 wd=0.001 WSD | 86.29 ± 0.48 | 86.02 ± 0.42 | 0.5029 ± 0.0379 | 0.5230 ± 0.0317 | 17.24 ± 0.05 | 29701 ± 92 |
| 4 | AdamW cosine | AdamW lr=0.004 wd=0.001 cosine | 79.34 ± 0.47 | 78.97 ± 0.69 | 0.6221 ± 0.0190 | 0.6386 ± 0.0221 | 11.85 ± 0.12 | 43209 ± 429 |

Result: AnchorMuon with RMS fallback has the highest mean official test accuracy, while AnchorMuon with AdamATan2 fallback has the highest mean final validation accuracy and lowest mean validation loss. The RMS-vs-AdamATan2 gap is much smaller than seed-to-seed variance, so these are effectively tied on this CIFAR-10 confirmation. AdamW is materially faster per step but substantially lower accuracy at the same 50-epoch budget.

Generated plots:
- `val_acc_curve_mean.png`
- `val_loss_curve_mean.png`
- `train_loss_curve_mean.png`
- `iteration_step_time_ms_bar_mean.png`
- `iteration_examples_per_sec_bar_mean.png`
- `val_acc_epoch_mean.png`
- `val_loss_epoch_mean.png`
- `train_loss_epoch_mean.png`

Raw outputs:
- `all_runs.csv`: final per-seed rows
- `all_metrics_flat.csv`: all per-epoch and eight-bin metrics
- `aggregate_summary.csv`: mean/std table used above
