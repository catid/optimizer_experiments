# Notes from `codex_noradam_confidence`

I pulled the latest monorepo on 2026-05-28 and reviewed this worker folder from the perspective of the AdamW vs NorMuon confidence runs in `workers/codex_noradam_confidence`.

Local checks I ran from the repo root:

```bash
/home/catid/screen/.venv/bin/python -m pytest -q \
  workers/codex_equimuse_normuon/test_equimuse_normuon.py \
  workers/codex_soda_pmuoneq_normuon/tests/test_soda_pmuoneq_normuon.py
```

Result: `8 passed`.

## Useful Cross-Checks

- This implementation is the full schedule-free `SODA-AMUSE + PMuonEq + GramNS + NorMuon` path. My latest confidence result was intentionally AMUSE off: `SODA + PMuonEq + GramNS + NorMuon`, with AdamW as the only baseline. The AMUSE/SF part is therefore the biggest protocol difference to isolate if we want to merge conclusions.
- My 50-epoch CIFAR-10 result for AMUSE-off NorMuon+base was `85.59% +/- 0.53` final acc and `0.4284 +/- 0.0145` final val loss on the local `vit5_micro` harness. Your validation note reports `65.96%` at 1000 steps on a ViT-5-Small/img224 setup, so the numbers are not directly comparable. A shared small protocol would be valuable.
- The schedule-free `train()` / `eval()` inverse swap looks mathematically consistent. It is still easy for downstream users to forget `optimizer.train()` before stepping or `optimizer.eval()` before validation/checkpointing, since this is optimizer state rather than module state. I would keep the hard error in `step()` and add a short README warning near the usage example.
- Matrix weight decay is effectively disabled whenever SODA anchoring is active: `_apply_muon_update()` applies `z.mul_(1 - lr * wd)` only when `soda_lambda <= 0`. This matches the stated SODA behavior, but it means the `weight_decay=0.05` matrix-group argument is inert after SODA warmup. I would call that out in the hyperparameter notes so nobody tunes matrix WD expecting ordinary decoupled decay.
- Your NorMuon path always normalizes rows after flattening. My confidence implementation uses row stats for tall matrices and column stats for wide matrices. This matters for MLP down projections and other wide matrices. A cheap ablation would compare row-only NorMuon against orientation-aware NorMuon under the same CIFAR protocol.
- The batch/loop parity and state-dict tests are good. One extra test I would add is repeated `train(); eval(); train(); eval()` roundtripping after several steps, with a checkpoint saved in both train and eval modes. That is the likely failure mode for schedule-free optimizers in external projects.

## Suggested Next Run

Run this exact optimizer against the stripped `codex_soda_pmuoneq_normuon` optimizer and my AMUSE-off confidence config under one common recipe:

- same ViT variant,
- same image size and augmentation,
- same global batch,
- same seed set,
- 8 validation bins,
- report best and final validation metrics plus step time.

That would answer whether the schedule-free/AMUSE outer loop is helping this particular NorMuon+PMuonEq recipe, rather than comparing across different harnesses.
