# Focused Optimizer Experimental Results

Source completed run: `/home/catid/screen/external/vit5-sodamuseeq/runs/focused_optimizer_confidence`.

The fresh rerun from the rebased monorepo branch is still active at `vit5/runs/focused_feedback_20260528_235818`; these diagrams use the completed focused run so the report includes full 10k and 20k curves.

## Diagrams

- Validation loss, best vs AdamW: `val_loss_best_vs_adamw.png`
- Training interval loss, best vs AdamW: `train_loss_best_vs_adamw.png`
- Validation accuracy, best vs AdamW: `val_acc_best_vs_adamw.png`
- Final test accuracy bar chart: `final_test_accuracy_bar.png`
- Iteration speed bar chart: `iteration_speed_steps_per_sec.png`
- All optimizer validation loss: `val_loss_all_optimizers.png`

## 10k Final Multi-Seed Summary

| rank | optimizer | seeds | best val loss | best val acc | test acc | steps/sec |
|---:|---|---:|---:|---:|---:|---:|
| 1 | SODA+PMuonEq+Gram | 3 | 0.4210 +/- 0.0097 | 87.19% +/- 0.24 | 86.90% +/- 0.35 | 42.84 +/- 0.20 |
| 2 | NorMuon+BaseGram | 3 | 0.5069 +/- 0.0100 | 86.21% +/- 0.49 | 86.05% +/- 0.36 | 47.93 +/- 0.67 |
| 3 | AdamW | 3 | 0.6030 +/- 0.0167 | 81.47% +/- 0.75 | 81.27% +/- 0.74 | 58.46 +/- 0.42 |

## 20k Seed-0 Check

| rank | optimizer | best val loss | best val acc | test acc | steps/sec |
|---:|---|---:|---:|---:|---:|
| 1 | SODA+PMuonEq+Gram | 0.4151 | 87.58% | 87.77% | 42.94 |
| 2 | NorMuon+BaseGram | 0.5001 | 86.10% | 86.42% | 49.22 |
| 3 | AdamW | 0.5967 | 82.46% | 82.25% | 59.19 |

## Interpretation

- `SODA+PMuonEq+Gram` is the quality winner: lower validation loss and higher accuracy than AdamW and NorMuon+BaseGram in both the 10k multi-seed final and 20k seed-0 check.
- AdamW remains the fastest per iteration, but gives substantially worse validation/test accuracy in this ViT-5/CIFAR-10 harness.
- NorMuon+BaseGram is the middle point: faster than SODA+PMuonEq+Gram and much better than AdamW, but worse than SODA+PMuonEq+Gram on loss and accuracy.
