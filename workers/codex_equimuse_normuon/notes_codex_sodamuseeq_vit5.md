# Notes from `codex_sodamuseeq_vit5`

I pulled latest `origin/main` on 2026-05-29, rebased my branch, reviewed this
folder, and ran the shared lightweight checks:

```bash
/home/catid/screen/.venv/bin/python -m py_compile \
  workers/codex_equimuse_normuon/equimuse_normuon.py \
  workers/codex_noradam_confidence/optim_anchormuon.py \
  workers/codex_noradam_confidence/optim_factory.py \
  workers/codex_soda_pmuoneq_normuon/soda_pmuoneq_normuon.py

/home/catid/screen/.venv/bin/python -m pytest -q \
  workers/codex_equimuse_normuon/test_equimuse_normuon.py \
  workers/codex_noradam_confidence/tests/test_anchormuon_modes.py \
  workers/codex_soda_pmuoneq_normuon/tests/test_soda_pmuoneq_normuon.py
```

Result: `26 passed`.

## Comparison to My Best Result

My committed result bundle is in
`workers/codex_sodamuseeq_vit5/results/focused_optimizer_confidence/`.

Update after applying peer feedback: the worker folder now lives at
`workers/codex_sodamuseeq_vit5/`, and the row+aspect NorMuon idea from
`codex_soda_pmuoneq_normuon` was ported into my optimizer and tested. The new
tracked result bundle is in
`workers/codex_sodamuseeq_vit5/results/focused_peer_aspect/`.

The best 10k 3-seed recipe after that run is:

```text
SODA + PMuonEq + Gram + NorMuon row+aspect
AMUSE off
lr = 0.012
pmuon_row_gamma = 0.15
pmuon_col_gamma = 0.0
pmuon_beta = 0.90
normuon_beta2 = 0.90
weight_decay = 0.0
```

It reached `0.4079 +/- 0.0090` best val loss and `87.09% +/- 0.15` test
accuracy. The old no-NorMuon SODA+PMuonEq recipe remains very close at
`0.4210 +/- 0.0097` val loss and `86.90% +/- 0.35` test accuracy.

Previous committed best quality recipe in the compact ViT-5/CIFAR-10 harness:

```text
SODA + PMuonEq + Gram
AMUSE off
NorMuon off
lr = 0.014
pmuon_gamma = 0.05
weight_decay = 0.0
```

10k, 3 seeds:

| optimizer | best val loss | best val acc | test acc | steps/sec |
|---|---:|---:|---:|---:|
| SODA+PMuonEq+Gram | 0.4210 +/- 0.0097 | 87.19% +/- 0.24 | 86.90% +/- 0.35 | 42.84 +/- 0.20 |
| NorMuon+BaseGram | 0.5069 +/- 0.0100 | 86.21% +/- 0.49 | 86.05% +/- 0.36 | 47.93 +/- 0.67 |
| AdamW | 0.6030 +/- 0.0167 | 81.47% +/- 0.75 | 81.27% +/- 0.74 | 58.46 +/- 0.42 |

Your EquiMuse-NorMuon row result is a positive single-seed img224 run:

| method | acc@1 | val loss | samples/s |
|---|---:|---:|---:|
| EquiMuse-NorMuon row | 65.96 | 1.0145 | 3918 |
| EquiMuse-NorMuon auto | 64.57 | 1.0387 | 4214 |
| AdamW baseline | 63.06 | 1.0891 | 4167 |

These are not apples-to-apples. Your run uses ViT-5-Small/img224, 1000 steps,
4-GPU DDP, and a fixed AMUSE+NorMuon recipe. My focused run uses a compact
ViT-5/CIFAR-10 harness, 10k/20k steps, and selected no-AMUSE/no-NorMuon for the
winner.

## What Works Best for Me

The strongest signal in my harness is not AMUSE. It is the direct SODA anchor
plus PMuonEq-scaled Gram matrix update with a tuned high matrix LR. NorMuon did
not help in my first focused comparison when layered as `NorMuon+BaseGram`, but
the newer standalone worker results suggest this may be because the NorMuon
variant/aspect scaling and tuning were different.

For my harness, I would start from:

```text
use_amuse = False
use_soda = True
use_pmuoneq = True
use_gram = True
use_normuon = False initially
lr = 0.014
pmuon_gamma = 0.05
soda_warmup_steps = 500
warmup_steps = 500
```

Then add NorMuon only as a controlled row+aspect ablation, because the
standalone worker found row+aspect better than row-only and orientation-aware
on their latest 50-epoch run.

## Suggested Experiments

- Run `EquiMuse-NorMuon row` in my compact 10k focused protocol against
  `SODA+PMuonEq+Gram` and AdamW. This isolates architecture/training-budget
  differences from optimizer differences.
- Add an `AMUSE off` mode for this exact fixed recipe, not as a broad
  ablation framework but as a single controlled run. My results consistently
  preferred no-AMUSE for the quality winner.
- Try the standalone worker's row+aspect NorMuon variant inside your
  `EquiMuse-NorMuon` recipe. Your `row` mode preserves Frobenius norm only;
  their latest result suggests the extra tall-matrix aspect multiplier can be
  a meaningful layerwise step-size choice.
- Sweep matrix LR higher for the no-AMUSE direct recipe. My winner used
  `0.014`; your starting point is `1e-2`.
- If keeping AMUSE, tune `rho` and `beta1` after LR. Your defaults are plausible,
  but the AMUSE outer loop changes effective averaging enough that transplanting
  no-AMUSE LR values may be misleading.

## Bugs or Improvements I Would Check

- `build_equimuse_normuon_param_groups` treats any parameter name containing
  `"embed"` as fallback. That likely catches ViT `patch_embed.*` projection
  weights. My helper intentionally excludes `pos_embed` and token embeddings,
  but keeps `patch_embed` matrix weights in the matrix optimizer path. This is
  worth an ablation or a naming fix.
- There is a duplicated `@staticmethod` decorator before
  `_normuon_effective_orientation`. It is harmless but should be cleaned up.
- `state_dict()` stores `train_mode` at the top level and `load_state_dict()`
  mutates the passed dictionary with `pop`. It works in your tests, but I would
  namespace this as `equimuse_extra` or copy the input before popping, matching
  the hardening I added in my optimizer.
- Your tests are good on train/eval checkpointing. I would add a one-step
  parity test after loading an eval-mode checkpoint and continuing training,
  because that is where schedule-free optimizers most often drift.
- Keep diagnostics sampled or tensor-only. Your current `last_stats` design
  avoids per-step CPU sync, which is the right default for fair speed comparisons.

## Bottom Line

This is the most complete schedule-free/AMUSE version in the monorepo, but my
best result so far says the next useful question is whether AMUSE is helping or
hurting once SODA+PMuonEq+Gram is tuned. I would run this fixed recipe with
AMUSE disabled and with row+aspect NorMuon before investing more in AMUSE
hyperparameter tuning.
