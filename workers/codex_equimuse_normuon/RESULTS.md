# EquiMuse Experimental Results

Generated: 2026-05-29T00:08:18.641115+00:00

Scope: ViT-5-Small on CIFAR-10, img224, 1000 optimizer steps, 8 validation bins, seed 67890, all 4 visible GPUs with DDP.

## Summary

| method | final acc@1 | final val loss | final train loss | samples/s | optimizer s/bin | total train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EquiMuse-NorMuon row | 65.96 | 1.0145 | 1.4699 | 3918 | 4.241 | 128.7 |
| EquiMuse-NorMuon auto | 64.57 | 1.0387 | 1.4817 | 4214 | 3.312 | 121.2 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4167 | 0.212 | 120.2 |

## Readout

- Best method in this table: EquiMuse-NorMuon row at 65.96% acc@1 and 1.0145 validation loss.
- Versus AdamW: accuracy delta +2.90 points; validation-loss delta -0.0746.
- Speed delta versus AdamW: -6.0% samples/s.
- This is single-seed evidence; use the tracked multi-seed task before treating small deltas as robust.

## Figures

- Validation loss curve: `results/figures/equimuse_best_vs_adamw_val_loss_curve.png`
- Training loss curve: `results/figures/equimuse_best_vs_adamw_train_loss_curve.png`
- Accuracy curve: `results/figures/equimuse_best_vs_adamw_accuracy_curve.png`
- Iteration speed: `results/figures/equimuse_best_vs_adamw_iteration_speed.png`
