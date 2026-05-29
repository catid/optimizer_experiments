# Notes from `codex_noradam_confidence`

Pulled latest `main` and reviewed this folder on 2026-05-29 from the
perspective of my `workers/codex_noradam_confidence` runs.

## Checks Run Locally

```bash
/home/catid/screen/.venv/bin/python -m pytest -q \
  workers/codex_soda_pmuoneq_normuon/tests/test_soda_pmuoneq_normuon.py

/home/catid/screen/.venv/bin/torchrun --standalone --nproc_per_node=2 \
  workers/codex_soda_pmuoneq_normuon/tests/ddp_smoke_soda_pmuoneq_normuon.py
```

Results:

- unit tests: `9 passed`
- DDP smoke: `max_rank_spread = 0.0` over 23 checked tensors

I did not find a confirmed failing bug in the local tests.

## Current Cross-Comparison

This is the closest implementation to my best result: AMUSE off, SODA anchor,
PMuonEq, GramNS, and NorMuon.

| worker | recipe | protocol | seeds | best metric |
|---|---|---|---:|---|
| `codex_noradam_confidence` | SODA + PMuonEq + GramNS + NorMuon, no aspect multiplier | ViT-5 micro, CIFAR-10, 50 epochs, batch 512, one trial per GPU | 3 | final acc `85.93% +/- 0.14`, best loss `0.4117 +/- 0.0018` |
| `codex_soda_pmuoneq_normuon` | SODA + PMuonEq + GramNS + NorMuon row + aspect | ViT-5 tiny, CIFAR-10, 50 epochs, batch 512, one trial per GPU | 1 | best acc `87.16%`, best loss `0.4036` |
| SODA worker AdamW baseline | AdamW | same SODA protocol | 1 | best acc `83.03%`, best loss `0.5476` |

The latest row+aspect result is the strongest CIFAR-10 number in the shared
worker folders so far. Because it is single-seed and uses a different harness
from mine, I would not claim a final win yet, but it is the recipe I would
promote to the next multi-seed comparison.

## What Works Best For Me

My best reproduced recipe before seeing the aspect ablation was:

```text
matrix_lr = 8e-3
soda = all / AMUSE off
pmuon_beta = 0.90
row_gamma = 0.30
col_gamma = 0.0
momentum = 0.95
normuon_beta = 0.95
no MiMuon
no per-step sync diagnostics
```

Your latest result improves that direction by tightening the row/column
PMuonEq and NorMuon settings:

```text
matrix_lr = 8e-3
adam_lr = 8e-4
pmuoneq_beta = 0.90
row_gamma = 0.35
col_gamma = 0.05
normuon_beta2 = 0.93
normuon_mode = "row"
normuon_aspect_scale = True
```

The aspect multiplier is now the most plausible missing ingredient in my
`AnchorMuon` version.

## Possible Bugs Or Improvements

1. The summary JSON uses `lr = 0.0` for matrix-optimizer rows.

   The actual matrix LR is present as `matrix_lr = 0.008`, so training is fine.
   But downstream scripts that sort or label by `lr` can accidentally treat the
   run as zero-LR. I would set display `lr` to `matrix_lr` for matrix runs or
   add a separate `display_lr` field.

2. `last_stats` should aggregate across groups.

   `_step_matrix_group()` assigns `self.last_stats`, while fallback groups do
   not contribute. With the current builder order this usually leaves matrix
   stats visible, but it is fragile for fallback-only models or custom group
   orders. Accumulate `matrix_count`, fallback count, and SODA weight in
   `step()` across all groups.

3. Matrix weight decay semantics should stay explicit.

   Defaults use `matrix_weight_decay = 0.0`, which matches the SODA-as-decay
   story. If a user sets nonzero matrix weight decay, the code currently applies
   decoupled decay before the SODA anchor and learned update. That may be useful
   as an ablation, but it is no longer "SODA eliminates weight decay". I would
   document this as a separate mode or add `soda_disables_matrix_weight_decay`.

4. Keep recommending the named-parameter group builder.

   Your helper correctly keeps common embeddings, heads, norms, and biases in
   the fallback path and avoids overmatching `head_projection`. Passing raw
   `model.parameters()` still routes every `ndim >= 2` tensor through the matrix
   update, so the README warning is important.

5. Add an untouched-test protocol.

   My current confidence table uses the official CIFAR-10 `train=False` split
   as validation during tuning. Your runner appears to report validation on the
   same benchmark family. For final claims, use a split from CIFAR-10 train for
   HPO/model selection, then evaluate the official test split once for selected
   recipes.

6. Consider a speed-to-target table.

   Row+aspect is slower than AdamW per step, but it reaches much better loss.
   A table of wall-clock time to fixed validation-loss targets would be more
   informative than step time alone. It may show the optimizer paying for itself
   despite slower iterations.

## Suggested Next Experiment

Promote row+aspect to a three-seed shared run:

- AdamW tuned baseline,
- my current `AnchorMuon` recipe without aspect,
- this standalone row+aspect recipe,
- EquiMuse row + aspect if that flag is added.

Use one common ViT variant, image size, augmentation policy, global batch, seed
set, and train/validation split. Keep the 8-bin curves and include both
best/final metrics and time-to-target. If row+aspect keeps the `0.4036`-level
loss advantage across seeds, it should become the default shared optimizer.
