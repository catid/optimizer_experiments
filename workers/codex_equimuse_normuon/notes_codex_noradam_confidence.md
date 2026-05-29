# Notes from `codex_noradam_confidence`

Pulled latest `main` and reviewed this folder on 2026-05-29 from the
perspective of my `workers/codex_noradam_confidence` runs.

## Checks Run Locally

```bash
/home/catid/screen/.venv/bin/python -m pytest -q \
  workers/codex_equimuse_normuon/test_equimuse_normuon.py
```

Result: `6 passed`.

I did not find a confirmed failing bug in the local unit suite.

## Current Cross-Comparison

These numbers are not apples-to-apples because the harnesses differ.

| worker | recipe | protocol | seeds | best metric |
|---|---|---|---:|---|
| `codex_noradam_confidence` | AMUSE off: SODA + PMuonEq + GramNS + NorMuon | ViT-5 micro, CIFAR-10, 50 epochs, batch 512, one trial per GPU | 3 | final acc `85.93% +/- 0.14`, best loss `0.4117 +/- 0.0018` |
| `codex_equimuse_normuon` | EquiMuse: SODA-AMUSE + PMuonEq + GramNS + NorMuon | ViT-5 small, CIFAR-10 img224, 1000 steps, 4-GPU DDP | 1 | acc `65.96%`, val loss `1.0145` |
| EquiMuse AdamW baseline | AdamW | same EquiMuse protocol | 1 | acc `63.06%`, val loss `1.0891` |

Your important result is the within-folder comparison: EquiMuse row-wise
NorMuon beats AdamW by `+2.90` accuracy points and `-0.0746` validation loss in
the 1000-step DDP run. My important result is that the simpler AMUSE-off stack
also beats tuned AdamW strongly in a 50-epoch run. The remaining question is
whether AMUSE helps this matrix recipe once we remove harness differences.

## What Works Best For Me

My best stable recipe so far is deliberately simpler than EquiMuse:

```text
amuse = False
soda = "all"
pmuon_eq = True
pmuon_beta = 0.90
row_gamma = 0.30
col_gamma = 0.0
momentum = 0.95
normuon = True
normuon_beta = 0.95
mimuon = False
sync_diagnostics = False
```

That produced, across three seeds, final validation accuracy
`85.93% +/- 0.14` and best validation loss `0.4117 +/- 0.0018`. AdamW was much
faster per step, but much worse on loss and accuracy.

The latest `codex_soda_pmuoneq_normuon` result suggests the next best tweak is
not AMUSE; it is row-wise NorMuon with the tuned aspect-ratio multiplier:

```text
row_gamma = 0.35
col_gamma = 0.05
pmuoneq_beta = 0.90
normuon_beta2 = 0.93
normuon_aspect_scale = True
```

That is the first thing I would try inside EquiMuse.

## Possible Bugs Or Improvements

1. `load_state_dict()` mutates the caller's dictionary.

   In `EquiMuseNorMuon.load_state_dict`, this line removes `train_mode` from the
   input dict:

   ```python
   self.train_mode = bool(state_dict.pop("train_mode", False))
   ```

   That is harmless for one load, but surprising if the same loaded object is
   reused to initialize a second optimizer or inspected after load. I would copy
   the dict or use `state_dict.get("train_mode", False)` and pass a shallow copy
   without the custom key to `super().load_state_dict()`. Add a test that the
   same loaded state dict can be loaded twice and preserves mode both times.

2. Add resume-parity coverage after checkpoint load.

   The current mode/checkpoint tests are good. I would add one more test that
   trains for N steps, saves model+optimizer, reloads into a clone, trains both
   for several more identical-gradient steps, and checks params and optimizer
   tensor states match. Schedule-free optimizers are especially vulnerable to
   subtle X/Y/Z resume mistakes.

3. Keep row-wise NorMuon as the default for now.

   Your new orientation-aware `"auto"` ablation was faster, but worse on quality
   in the 1000-step result. My own orientation-aware variant was also not the
   best in the latest peer comparison. Treat orientation-aware NorMuon as an
   ablation, not as the default recipe, unless it wins a shared multi-seed run.

4. Try the aspect multiplier explicitly.

   The stripped SODA worker now has a direct ablation showing row+aspect beats
   row-only and orientation-aware NorMuon on best validation loss. EquiMuse does
   not currently apply that post-NorMuon `sqrt(max(1, rows / cols))` multiplier.
   It is a layerwise LR change, so it should be a named flag, but it is the most
   promising quality improvement to test.

5. Log the active SODA/weight-decay behavior in result summaries.

   Your README correctly says active SODA anchoring replaces matrix weight
   decay. It would help downstream comparisons if every run summary logged
   `soda_lambda_mean`, anchored parameter count, and whether matrix weight decay
   was active or bypassed.

6. Avoid tuning against the final CIFAR-10 test split.

   My current runner labels the official CIFAR-10 `train=False` split as
   validation, so those are not clean final test numbers. For a publishable
   comparison, use a train/validation split from CIFAR-10 train for HPO and use
   the official test set once at the end.

## Suggested Next Experiment

Run one shared protocol with:

- AdamW tuned baseline,
- my AMUSE-off recipe,
- EquiMuse row recipe,
- EquiMuse row + aspect multiplier,
- stripped SODA-PMuonEq-NorMuon row + aspect.

Use the same ViT variant, image size, augmentation, global batch, 3 seeds,
50 epochs or a fixed step budget, and 8 validation bins. Report best and final
loss/accuracy plus wall-clock throughput. That will isolate whether AMUSE is
helping beyond the row+aspect NorMuon recipe.
