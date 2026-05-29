# CIFAR-10 Fallback-Mode Comparison

Date: 2026-05-29

## Question

Test whether scalar/vector fallback updates based on AdamATan2 or AdamC improve
the current root AnchorMuon CIFAR-10 baseline.

Only the fallback path changed. Matrix parameters still used the same root
AnchorMuon stack:

- SODA anchor update
- row-only PMuonEq
- Gram Newton-Schulz
- NorMuon
- trainer-side WSD schedule

## Protocol

- Model: `vit5_micro`
- Dataset: CIFAR-10
- Split: 45k train / 5k validation from the official training set
- Test: official 10k CIFAR-10 test set, evaluated once after each selected final run
- Batch size: 512
- Epochs: 12-epoch HPO, then 50-epoch replay of the best run per fallback family
- Seed: 123
- GPUs: 2 x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Torch: `2.13.0.dev20260506+cu130`
- Step timing: CUDA-synchronized

Command:

```bash
/home/catid/screen/.venv/bin/python \
  workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --preset root_fallback_modes \
  --two-stage \
  --hpo-epochs 12 \
  --final-epochs 50 \
  --train-subset 0 \
  --val-source train_split \
  --val-subset 0 \
  --batch-size 512 \
  --num-workers 12 \
  --eval-test \
  --output-dir workers/codex_noradam_confidence/results/fallback_modes_cifar10_20260529
```

## HPO Grid

Control:

- RMS fallback: `lr=0.012`, `fallback_lr_mult=1.0`, `beta2=0.95`

AdamATan2 fallback:

- `lr in {0.010, 0.012, 0.014}`
- `fallback_lr_mult in {0.5, 1.0, 2.0}`
- `fallback_beta2 in {0.95, 0.99}`

AdamC fallback:

- same LR/fallback-LR/beta2 grid as AdamATan2
- two extra AdamC decay probes: `fallback_weight_decay in {0.1, 1.0}` at
  `lr=0.012`, `fallback_lr_mult=1.0`, `beta2=0.95`

## 12-Epoch HPO Winners

| Family | Best trial | Val loss | Val acc | Step time | Examples/sec |
|---|---|---:|---:|---:|---:|
| AdamC fallback | `fallback_adamc_wsd_lr0.01_flr0.5_b20.99` | 0.5758 | 80.16% | 18.10 ms | 28.3k |
| RMS control | `fallback_rms_control_wsd_lr0.012_flr1_b2_0.95` | 0.5588 | 79.90% | 17.16 ms | 29.8k |
| AdamATan2 fallback | `fallback_atan2_wsd_lr0.014_flr0.5_b20.95` | 0.5751 | 79.72% | 18.02 ms | 28.4k |

The 12-epoch selector did not make AdamATan2 look like the best global run.
Its advantage appeared later in the 50-epoch replay.

## Final 50-Epoch Results

| Rank | Fallback | Final val loss | Final val acc | Best val loss | Best val acc | Test loss | Test acc | Step time | Examples/sec | Settings |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | AdamATan2 | 0.3782 | 88.52% | 0.3782 | 88.52% | 0.3996 | 88.10% | 17.51 ms | 29.2k | `lr=0.014`, `fallback_lr_mult=0.5`, `beta2=0.95` |
| 2 | RMS control | 0.4118 | 88.00% | 0.3856 | 88.00% | 0.4405 | 87.13% | 16.07 ms | 31.9k | `lr=0.012`, `fallback_lr_mult=1.0`, `beta2=0.95` |
| 3 | AdamC | 0.4164 | 87.94% | 0.4088 | 87.94% | 0.4231 | 87.24% | 17.53 ms | 29.2k | `lr=0.010`, `fallback_lr_mult=0.5`, `beta2=0.99` |

## Conclusion

AdamATan2 fallback is the only tested variant that beat the RMS control in the
50-epoch replay, improving final validation accuracy by `+0.52` points and
official test accuracy by `+0.97` points versus the same-run RMS control.

Against the previous WSD RMS result in the README, AdamATan2 also improves the
single-seed official test score (`88.10%` vs `87.67%`), but this is still a
single-seed result with stochastic augmentation. Treat it as the best observed
CIFAR-10 recipe, not as a fully settled new default.

AdamC did not beat the RMS control after 50 epochs. Its best 12-epoch HPO row
looked competitive, but it underperformed both validation and test in the final
replay.

## Artifacts

- HPO table: `hpo/all_runs.csv`
- Final table: `final_bins/all_runs.csv`
- Best run: `final_bins/best_run.json`
- Curves: `final_bins/train_loss.png`, `final_bins/val_loss.png`,
  `final_bins/val_acc.png`
- Speed charts: `final_bins/step_time_ms_bar.png`,
  `final_bins/examples_per_sec_bar.png`

![Validation accuracy](final_bins/val_acc.png)

