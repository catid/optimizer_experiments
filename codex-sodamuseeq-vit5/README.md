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

The optimizer supports optional switches for:

- SODA anchor correction.
- AMUSE schedule-free fast/eval iterate bookkeeping.
- PMuonEq row/column EMA equilibration before Gram-Newton-Schulz projection.
- MiMuon branch selection.
- NorMuon row second-moment normalization after projection.

## Validation

Validated in the source workspace before packaging:

```bash
uv run python -m py_compile \
  sodamuseeq.py \
  external/vit5-sodamuseeq/optim_sodamuseeq.py \
  external/vit5-sodamuseeq/main.py \
  external/vit5-sodamuseeq/engine.py \
  external/vit5-sodamuseeq/experiments/sodamuseeq_ablation.py \
  external/vit5-sodamuseeq/experiments/focused_optimizer_confidence.py

uv run pytest \
  external/vit5-sodamuseeq/tests/test_sodamuseeq_modes.py \
  tests/test_sodamuseeq_standalone.py

uv run torchrun --standalone --nproc-per-node=2 \
  external/vit5-sodamuseeq/tests/ddp_sodamuseeq_smoke.py
```

The DDP smoke passed with `max_param_diff=0`.

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

## Current Single-Seed Result Before Focused Follow-Up

The prior corrected CIFAR-10 ViT-5 10k comparison found:

| Variant | Val loss | Val acc | Test acc |
| --- | ---: | ---: | ---: |
| SODA+PMuonEq+Gram | 0.4129 | 87.24% | 86.77% |
| SODA+PMuonEq+Gram+NorMuon | 0.4320 | 86.60% | 86.28% |
| AdamW | 0.5815 | 82.52% | 82.12% |

The focused runner was added because NorMuon+BaseGram is a distinct ablation
from NorMuon layered on top of SODA+PMuonEq.
