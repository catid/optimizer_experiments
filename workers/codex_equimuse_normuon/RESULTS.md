# EquiMuse Experimental Results

Generated: 2026-05-29T01:18:00Z

Scope: ViT-5-Small on CIFAR-10, img224, 1000 optimizer steps, 8 validation
bins, seed 67890, all 4 visible GPUs with DDP.

## Peer-Feedback Comparison

| method | final acc@1 | final val loss | final train loss | samples/s | optimizer s/bin | total train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct SODA-PMuonEq-NorMuon + aspect | 76.69 | 0.7269 | 1.2382 | 4009 | 4.445 | 124.5 |
| Direct SODA-PMuonEq-NorMuon | 75.16 | 0.7740 | 1.2579 | 3949 | 4.525 | 126.9 |
| EquiMuse-NorMuon row | 65.96 | 1.0144 | 1.4699 | 4053 | 3.448 | 124.6 |
| EquiMuse-NorMuon row + aspect | 65.85 | 1.0019 | 1.4519 | 4027 | 3.481 | 125.4 |
| EquiMuse-NorMuon auto + aspect | 65.72 | 1.0140 | 1.4547 | 4021 | 3.433 | 126.0 |
| EquiMuse-NorMuon peer-gamma + aspect | 65.11 | 1.0302 | 1.4516 | 4030 | 3.546 | 125.7 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4280 | 0.237 | 117.1 |

## Readout

- Best overall method: direct no-AMUSE SODA-PMuonEq-NorMuon + aspect at
  76.69% acc@1 and 0.7269 validation loss.
- Versus AdamW: +13.63 accuracy points and -0.3622 validation loss, at -6.3%
  final-interval samples/s.
- Within schedule-free EquiMuse, aspect scaling improved validation loss
  (1.0144 -> 1.0019) but did not improve final accuracy on this seed.
- Direct SODA without aspect reached 75.16% / 0.7740, so the main win is the
  no-AMUSE direct update path, with aspect adding another +1.53 points and
  -0.0471 validation loss.
- The direct SODA rows used the peer standalone implementation from
  `workers/codex_soda_pmuoneq_normuon/soda_pmuoneq_normuon.py`, wired into the
  same ViT-5 harness for an apples-to-apples run. That implementation is now
  mirrored here as `direct_soda_pmuoneq_normuon.py` so this folder contains the
  code path that produced the best result.
- This is still single-seed evidence. The next confidence step is a 3-seed
  confirmation of direct SODA-PMuonEq-NorMuon + aspect against AdamW and
  EquiMuse row.

## Tuning

The 64-step tuning pass was too short to discriminate because all trials were
still near 10% accuracy. A narrower 250-step pass over LR, PMuonEq gamma,
NorMuon beta2, and aspect scaling selected:

```text
lr/matrix_lr = 0.01
pmuoneq_row_gamma = 0.15
pmuoneq_col_gamma = 0.15
normuon_beta2 = 0.93
normuon_aspect_scale = true
```

At 250 steps, the best aspect-on EquiMuse row reached 43.78% / 1.6011, while
the best aspect-off row reached 41.66% / 1.6768. The final 1000-step run showed
that this tuning signal transferred to validation loss, but not to final
accuracy.

## Figures

- Validation loss curve: `results/figures/equimuse_feedback_val_loss_curve.png`
- Training loss curve: `results/figures/equimuse_feedback_train_loss_curve.png`
- Accuracy curve: `results/figures/equimuse_feedback_accuracy_curve.png`
- Iteration speed: `results/figures/equimuse_feedback_iteration_speed.png`
- Final accuracy bar chart: `results/figures/equimuse_feedback_final_accuracy_bar.png`

## Tables

- Final summary: `results/tables/equimuse_feedback_final_results.csv`
- Final 8-bin curves: `results/tables/equimuse_feedback_final_curves.csv`
- 250-step tuning summary: `results/tables/equimuse_aspect_tuning_250_summary.csv`
- 250-step tuning curves: `results/tables/equimuse_aspect_tuning_250_curves.csv`
