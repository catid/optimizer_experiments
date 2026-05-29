# Validation

Source workstation:

- 4x NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs
- Python 3.12
- PyTorch nightly `2.13.0.dev20260513+cu130`

Checks run before export:

```text
python -m py_compile equimuse_normuon.py test_equimuse_normuon.py
pytest -q test_equimuse_normuon.py
```

Results:

- Standalone worker tests: 6 passed.
- Source workspace unit/parity tests: 6 passed for
  `tests/test_equimuse_normuon_standalone.py`.
- Source workspace DDP consistency:
  `torchrun --standalone --nproc_per_node=4 check_equimuse_ddp.py --steps 3`
  passed with zero rank spread for model params, fast weights, and eval weights,
  including the row-wise standalone mode and the orientation-aware auto mode.

## Fixed ViT-5-Small/CIFAR-10 img224 Validation

All runs used all 4 visible GPUs with DDP, seed `67890`, batch 128 per GPU,
1000 optimizer steps, and 8 validation bins.

| method | acc@1 | val loss | train loss | samples/s | optimizer s/bin | total train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EquiMuse-NorMuon row | 65.96 | 1.0145 | 1.4699 | 3918 | 4.241 | 128.7 |
| EquiMuse-NorMuon auto | 64.57 | 1.0387 | 1.4817 | 4214 | 3.312 | 121.2 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4167 | 0.212 | 120.2 |

The row-wise selected recipe remains the quality winner: +2.90 accuracy points
and -0.0746 validation loss versus AdamW. The orientation-aware auto ablation
was faster than the row-wise recipe in this run, but lower quality.

Result artifacts in this folder:

- `results/tables/equimuse_best_vs_adamw_results.csv`
- `results/tables/equimuse_best_vs_adamw_curves.csv`
- `results/figures/equimuse_best_vs_adamw_val_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_train_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_accuracy_curve.png`
- `results/figures/equimuse_best_vs_adamw_iteration_speed.png`

The exact commands used for each training run are stored in
`results/tables/equimuse_best_vs_adamw_results.csv`.
