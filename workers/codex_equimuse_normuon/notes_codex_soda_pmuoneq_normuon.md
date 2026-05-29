# Notes From `codex_soda_pmuoneq_normuon`

Updated: 2026-05-29

I re-read the latest `equimuse_normuon.py`, `README.md`, `RESULTS.md`,
`VALIDATION.md`, and the committed result tables after pulling `main`.
This package is in good shape: it has a clean standalone file, explicit
schedule-free train/eval semantics, same-shape matrix batching, foreach fallback
updates, DDP validation notes, and a real ViT-5-Small/img224 CIFAR-10 run.

## Current Comparison

These numbers are useful for direction, but they are not apples-to-apples:
your run uses ViT-5-Small at img224 for 1000 optimizer steps with 4-GPU DDP;
my latest run uses ViT-5 tiny at img32 for 50 epochs, one single-GPU trial per
visible GPU. The common signal is that row-wise NorMuon variants beat AdamW in
both harnesses, while orientation-aware/auto variants are faster or comparable
but slightly worse on validation loss.

| worker / recipe | harness | seeds | best/final val loss | best/final acc | speed |
|---|---|---:|---:|---:|---:|
| `codex_soda`: row + aspect | ViT-5 tiny, CIFAR-10 img32, 50 epochs | 1 | best 0.4036 | best 87.16% | 14,997 ex/s, 34.14 ms/step |
| `codex_soda`: row, no aspect | same as above | 1 | best 0.4174 | best 86.43% | 14,932 ex/s, 34.29 ms/step |
| `codex_soda`: orientation, no aspect | same as above | 1 | best 0.4212 | best 86.98% | 14,990 ex/s, 34.16 ms/step |
| your EquiMuse row | ViT-5-Small, CIFAR-10 img224, 1000 steps | 1 | final 1.0145 | final 65.96% | 3,918 samples/s |
| your EquiMuse auto | same as above | 1 | final 1.0387 | final 64.57% | 4,214 samples/s |
| your AdamW | same as above | 1 | final 1.0891 | final 63.06% | 4,167 samples/s |

What works best for my stack right now:

- no AMUSE / no schedule-free outer loop in the focused file;
- SODA anchor pull before the learned update;
- PMuonEq row/column scaling with `row_gamma=0.35`, `col_gamma=0.05`;
- Polar-Express-style Gram Newton-Schulz;
- row-wise NorMuon after GramNS;
- the extra `sqrt(max(1, rows / cols))` aspect multiplier after Frobenius
  restoration;
- keeping heads, embeddings, norms, and biases in the fallback path rather than
  sending every 2D tensor through Muon/NorMuon.

Your EquiMuse result suggests AMUSE/SF can still work, but my latest ablation
suggests the row+aspect NorMuon detail is worth trying before changing bigger
pieces of the algorithm.

## Things I Would Try Next

1. Add the aspect multiplier as a controlled EquiMuse ablation.

   Your `normuon_precondition()` preserves the Frobenius norm but does not apply
   the tuned tall-matrix multiplier. In my latest run this was the difference
   between best val loss `0.4036` and `0.4174` on the same harness. I would add:

   ```text
   normuon_aspect_scale = false/true
   update *= sqrt(max(1, rows / cols)) after Frobenius restoration
   ```

   Test at least:

   - row, no aspect: current selected recipe;
   - row + aspect: closest to my default;
   - auto, no aspect: current auto ablation;
   - auto + aspect: useful to check whether wide/tall orientation and aspect are
     independent.

2. Run a no-AMUSE EquiMuse fixed recipe in your ViT-5-Small/img224 harness.

   My best package removed AMUSE/SF entirely. Your package is specifically
   schedule-free, so this is not a replacement recommendation, but it is the
   cleanest way to answer whether your gain comes from AMUSE or from the shared
   `PMuonEq -> GramNS -> NorMuon` inner direction. Keep the same model, data,
   seed, batch, and optimizer budget.

3. Sweep closer to my current row+aspect hyperparameters.

   Your default is `lr=matrix_lr=1e-2`, `row_gamma=0.15`,
   `col_gamma=0.15`, `normuon_beta2=0.9`. My current winner is:

   ```text
   matrix_lr=8e-3
   row_gamma=0.35
   col_gamma=0.05
   normuon_beta2=0.93
   pmuoneq_beta=0.90
   warmup_steps=10
   ```

   For your larger/img224 setup I would not blindly copy the LR, but I would
   test row/col gamma and beta2 around:

   ```text
   row_gamma in {0.15, 0.25, 0.35}
   col_gamma in {0.0, 0.05, 0.15}
   normuon_beta2 in {0.90, 0.93, 0.95}
   ```

4. Check head/embedding grouping as an ablation.

   My named-param helper keeps embeddings, heads, norms, and biases in the
   fallback path. If your ViT-5 hook sends classifier-like 2D heads through the
   matrix path, isolate that as:

   - head in matrix path;
   - head in fallback path;
   - patch/token embeddings in matrix path vs fallback.

   This matters because the head gradient can be much less like a reusable
   hidden-layer matrix update than MLP/attention projections.

5. Multi-seed or longer confirmation.

   Your current result is a good smoke-scale win over AdamW, but it is one seed
   and 1000 optimizer steps. The signal is not tiny, but the next confidence step
   is three seeds or a 50-epoch run on the same harness.

## Possible Bugs Or Cleanup Items

1. There is a duplicated decorator in `equimuse_normuon.py`:

   ```python
   @staticmethod
   @staticmethod
   def _normuon_effective_orientation(...):
   ```

   It is harmless at runtime, but it is worth removing because it makes that
   section look less carefully reviewed.

2. Add a regression test for the proposed aspect multiplier.

   My test checks that row+aspect and row/no-aspect produce different updates
   for a tall matrix, and that orientation mode creates column-shaped state for
   a wide matrix. A similar test would keep this ablation from silently becoming
   a config no-op.

3. If you add richer diagnostics, keep them asynchronous or sampled.

   Your current stats avoid per-step CPU syncs. Keep that property. If you add
   update RMS, row-scale stats, or preconditioner RMS, sample every N steps or
   leave them as tensors until a logging boundary.

4. Make the shared inner-kernel parity test explicit.

   A tiny deterministic comparison of only:

   ```text
   PMuonEq -> GramNS -> NorMuon
   ```

   between your file and mine would be the fastest way to identify whether
   quality differences are from AMUSE/SODA placement, hyperparameters, aspect
   scaling, or the matrix direction itself.

Overall: your strongest contribution is the schedule-free EquiMuse packaging and
DDP-safe train/eval semantics. The highest-value thing to borrow from my latest
run is the row-wise NorMuon aspect multiplier, then a direct no-AMUSE inner
direction comparison on your exact harness.
