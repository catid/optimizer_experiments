# Validation

Source workstation:

- 4x NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs
- Python 3.12
- PyTorch nightly `2.13.0.dev20260513+cu130`

Checks run before export:

```text
/home/catid/attractor/.venv/bin/python -m py_compile \
  workers/codex_equimuse_normuon/equimuse_normuon.py \
  workers/codex_equimuse_normuon/test_equimuse_normuon.py
PYTHONPATH=workers/codex_equimuse_normuon \
  /home/catid/attractor/.venv/bin/python -m pytest -q \
  workers/codex_equimuse_normuon/test_equimuse_normuon.py
```

Results:

- Standalone worker tests: 9 passed.
- Added coverage for:
  - `load_state_dict()` not mutating the caller's state dict;
  - resume parity after optimizer load;
  - explicit NorMuon aspect-scale magnitude change;
  - batch-vs-loop parity with aspect scaling enabled.
- Source workspace DDP consistency:
  `torchrun --standalone --nproc_per_node=4 check_equimuse_ddp.py --steps 3 --output reports/equimuse_normuon_aspect_ddp_check.json`
  passed with zero rank spread for model params, fast weights, and eval weights,
  including row-wise standalone, auto, row+aspect, and auto+aspect standalone
  modes.

## Fixed ViT-5-Small/CIFAR-10 img224 Validation

All runs used all 4 visible GPUs with DDP, seed `67890`, batch 128 per GPU,
1000 optimizer steps, and 8 validation bins.

| method | acc@1 | val loss | train loss | samples/s | optimizer s/bin | total train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct SODA-PMuonEq-NorMuon + aspect | 76.69 | 0.7269 | 1.2382 | 4009 | 4.445 | 124.5 |
| Direct SODA-PMuonEq-NorMuon | 75.16 | 0.7740 | 1.2579 | 3949 | 4.525 | 126.9 |
| EquiMuse-NorMuon row | 65.96 | 1.0144 | 1.4699 | 4053 | 3.448 | 124.6 |
| EquiMuse-NorMuon row + aspect | 65.85 | 1.0019 | 1.4519 | 4027 | 3.481 | 125.4 |
| EquiMuse-NorMuon auto + aspect | 65.72 | 1.0140 | 1.4547 | 4021 | 3.433 | 126.0 |
| EquiMuse-NorMuon peer-gamma + aspect | 65.11 | 1.0302 | 1.4516 | 4030 | 3.546 | 125.7 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4280 | 0.237 | 117.1 |

The direct no-AMUSE SODA-PMuonEq-NorMuon + aspect recipe is the quality winner
in this single-seed run. Within schedule-free EquiMuse, the aspect multiplier
improved validation loss but not final accuracy.

Result artifacts in this folder:

- `results/tables/equimuse_best_vs_adamw_results.csv`
- `results/tables/equimuse_best_vs_adamw_curves.csv`
- `results/tables/equimuse_feedback_final_results.csv`
- `results/tables/equimuse_feedback_final_curves.csv`
- `results/figures/equimuse_best_vs_adamw_val_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_train_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_accuracy_curve.png`
- `results/figures/equimuse_best_vs_adamw_iteration_speed.png`
- `results/figures/equimuse_feedback_val_loss_curve.png`
- `results/figures/equimuse_feedback_accuracy_curve.png`
- `results/figures/equimuse_feedback_iteration_speed.png`

The exact commands used for each training run are stored in
`results/tables/equimuse_best_vs_adamw_results.csv`.
