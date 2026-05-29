# Codex SodaMuseEq ViT-5 Experiments

This folder is intentionally self-contained so it does not conflict with other
workers in the monorepo.

Contents:

- `sodamuseeq.py`: standalone optimizer module for reuse in other projects.
- `vit5/`: compact ViT-5 CIFAR-10 experiment fork with optimizer ablations.
- `vit5/optim_sodamuseeq.py`: ViT-integrated optimizer implementation.
- `vit5/experiments/sodamuseeq_ablation.py`: broad SodaMuseEq ablation runner.
- `vit5/experiments/focused_optimizer_confidence.py`: focused comparison runner
  for AdamW, NorMuon+BaseGram, and SODA+PMuonEq+Gram.
- `vit5/tests/`: unit and DDP smoke tests for optimizer modes.
- `results/focused_optimizer_confidence/`: committed result tables and plots
  for the completed focused comparison.

The optimizer supports optional switches for:

- SODA anchor correction.
- AMUSE schedule-free fast/eval iterate bookkeeping.
- PMuonEq row/column EMA equilibration before Gram-Newton-Schulz projection.
- MiMuon branch selection.
- NorMuon row second-moment normalization after projection.

## Validation

Validated in the source workspace before packaging:

```bash
python -m py_compile \
  sodamuseeq.py \
  vit5/optim_sodamuseeq.py \
  vit5/main.py \
  vit5/engine.py \
  vit5/experiments/sodamuseeq_ablation.py \
  vit5/experiments/focused_optimizer_confidence.py

python -m pytest \
  vit5/tests/test_sodamuseeq_modes.py \
  tests/test_sodamuseeq_standalone.py

torchrun --standalone --nproc-per-node=2 \
  vit5/tests/ddp_sodamuseeq_smoke.py
```

The DDP smoke checks train weights, eval weights, restored train weights, and
optimizer tensor state across ranks.

The tests include:

- all ablation modes run without NaNs,
- batch projection parity against individual projection,
- same-shape matrix bucket parity,
- repeated schedule-free `train()` / `eval()` roundtrips,
- train-mode and eval-mode checkpoint resume parity,
- standalone optimizer grouping and resume coverage.

## Checkpoint Recipe

For schedule-free/AMUSE modes, call `optimizer.eval()` before validation if you
want the averaged/eval weights, then call `optimizer.train()` before resuming
training. Checkpoints are safe in either mode as long as both the model state and
optimizer state are saved together. The optimizer stores its train/eval mode and
SODA step counter in `state_dict()` under `sodamuseeq_extra`.

SODA is configured to replace ordinary weight decay by default. When
`soda_replaces_weight_decay=True`, matrix and fallback weight decay are disabled
inside the optimizer and the anchor correction supplies the regularization path.

## Focused Confidence Runner

Run the focused comparison:

```bash
uv run python vit5/experiments/focused_optimizer_confidence.py \
  --out-dir vit5/runs/focused_optimizer_confidence \
  --data-dir data/cifar10 \
  --hpo-steps 1000 \
  --top-steps 3000 \
  --final-steps 10000 \
  --long-steps 20000 \
  --top-n 3 \
  --final-seeds 0,1,2 \
  --batch-size 256 \
  --eval-batch-size 1024 \
  --workers 8
```

The runner schedules one trial per visible GPU. It intentionally compares only:

- AdamW baseline.
- NorMuon+BaseGram: no SODA, no AMUSE, no PMuonEq.
- SODA+PMuonEq+Gram.

## Focused Results

The 3-seed 10k and seed-0 20k focused confidence pass found:

| Variant | Budget | Val loss | Val acc | Test acc | Steps/sec |
| --- | --- | ---: | ---: | ---: | ---: |
| SODA+PMuonEq+Gram | 10k, 3 seeds | 0.4210 +/- 0.0097 | 87.19% +/- 0.24 | 86.90% +/- 0.35 | 42.84 +/- 0.20 |
| NorMuon+BaseGram | 10k, 3 seeds | 0.5069 +/- 0.0100 | 86.21% +/- 0.49 | 86.05% +/- 0.36 | 47.93 +/- 0.67 |
| AdamW | 10k, 3 seeds | 0.6030 +/- 0.0167 | 81.47% +/- 0.75 | 81.27% +/- 0.74 | 58.46 +/- 0.42 |
| SODA+PMuonEq+Gram | 20k, seed 0 | 0.4151 | 87.58% | 87.77% | 42.94 |
| NorMuon+BaseGram | 20k, seed 0 | 0.5001 | 86.10% | 86.42% | 49.22 |
| AdamW | 20k, seed 0 | 0.5967 | 82.46% | 82.25% | 59.19 |

The result supports SODA+PMuonEq+Gram as the best quality recipe in this compact
ViT-5/CIFAR-10 harness. NorMuon+BaseGram is faster than SODA+PMuonEq+Gram and
better than AdamW, but it did not close the quality gap.

Committed result artifacts:

- Summary report: `results/focused_optimizer_confidence/experimental_results.md`
- Validation loss, best optimizer vs AdamW:
  `results/focused_optimizer_confidence/val_loss_best_vs_adamw.png`
- Train interval loss, best optimizer vs AdamW:
  `results/focused_optimizer_confidence/train_loss_best_vs_adamw.png`
- Validation accuracy, best optimizer vs AdamW:
  `results/focused_optimizer_confidence/val_acc_best_vs_adamw.png`
- Final test accuracy bar chart:
  `results/focused_optimizer_confidence/final_test_accuracy_bar.png`
- Iteration speed bar chart:
  `results/focused_optimizer_confidence/iteration_speed_steps_per_sec.png`
- Raw final tables:
  `results/focused_optimizer_confidence/final10k_all_runs.csv` and
  `results/focused_optimizer_confidence/long20k_all_runs.csv`

The optimizer implementation used for the completed result is
`vit5/optim_sodamuseeq.py`; the current branch adds checkpoint/test hardening on
top of that code path without changing the training update rule.
