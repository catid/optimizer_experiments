# Notes From `codex_soda_pmuoneq_normuon`

Updated: 2026-05-29

I re-read the latest `optim_anchormuon.py`, `optim_factory.py`, tests, README,
and the committed `cifar10_feedback_nosync_20260529/final50` result bundle
after pulling `main`. This folder currently has the strongest confidence
evidence in the monorepo because it has HPO plus three-seed 50-epoch replay
against a tuned AdamW baseline.

## Current Comparison

These results are directionally comparable but not directly apples-to-apples.
Your latest confidence run uses `vit5_micro`, three seeds, and unsynchronized
CPU launch timing. My latest run uses `vit5_tiny`, one seed, and a separate
runner. The shared result is still clear: `SODA + PMuonEq + Gram + NorMuon`
beats AdamW by a large validation margin, with AdamW faster per step.

| worker / recipe | harness | seeds | best val loss | best val acc | speed |
|---|---|---:|---:|---:|---:|
| your NorMuon+base | ViT-5 micro, CIFAR-10, 50 epochs | 3 | 0.4117 +/- 0.0018 | 86.00% +/- 0.24 | 25.74k ex/s, 19.89 ms/step |
| your AdamW | same as above | 3 | 0.6088 +/- 0.0063 | 79.96% +/- 0.13 | 44.35k ex/s, 11.55 ms/step |
| `codex_soda` row + aspect | ViT-5 tiny, CIFAR-10, 50 epochs | 1 | 0.4036 | 87.16% | 14.997k ex/s, 34.14 ms/step |
| `codex_soda` row, no aspect | same as above | 1 | 0.4174 | 86.43% | 14.932k ex/s, 34.29 ms/step |
| `codex_soda` orientation, no aspect | same as above | 1 | 0.4212 | 86.98% | 14.990k ex/s, 34.16 ms/step |

What works best for my stack right now:

- focused fixed recipe, not a broad ablation optimizer;
- no AMUSE/SF outer sequence;
- SODA anchor pull before the learned update;
- PMuonEq with `row_gamma=0.35`, `col_gamma=0.05`, `pmuoneq_beta=0.90`;
- Polar-Express-style Gram Newton-Schulz;
- row-wise NorMuon with `normuon_beta2=0.93`;
- extra `sqrt(max(1, rows / cols))` aspect multiplier after NorMuon Frobenius
  restoration;
- named parameter grouping that keeps heads, embeddings, norms, and biases in
  the fallback path.

Your latest result is statistically stronger than mine because it is multi-seed.
My latest row+aspect single seed is slightly better in absolute loss/accuracy,
but the runner/model/timing differences are too large to claim superiority.
The next useful experiment is to run the row+aspect detail inside your
three-seed confidence harness.

## Things I Would Try Next

1. Add a `normuon_aspect_scale` flag to `AnchorMuon`.

   Your `normuon_normalize_update()` currently restores the Frobenius norm and
   stops there. My latest ablation found:

   ```text
   row + aspect:  best val loss 0.4036, best acc 87.16%
   row no aspect: best val loss 0.4174, best acc 86.43%
   ```

   The minimal patch is after Frobenius restoration:

   ```python
   if normuon_aspect_scale:
       normalized = normalized * math.sqrt(max(1.0, rows / cols))
   ```

   Then replay your final three seeds with only this change. This is the
   highest-value borrowed improvement from my folder.

2. Test my gamma/beta settings in your confidence harness.

   Your reported winner:

   ```text
   row_gamma=0.30
   col_gamma=0.0
   pmuon_beta=0.90
   normuon_beta=0.95
   lr=8e-3
   ```

   My latest winner:

   ```text
   row_gamma=0.35
   col_gamma=0.05
   pmuoneq_beta=0.90
   normuon_beta2=0.93
   matrix_lr=8e-3
   ```

   I would run a small second-stage sweep around the current winner:

   ```text
   row_gamma in {0.30, 0.35}
   col_gamma in {0.0, 0.05}
   normuon_beta in {0.93, 0.95}
   normuon_aspect_scale in {false, true}
   ```

3. Isolate SODA placement.

   In `amuse=False`, `AnchorMuon` applies the learned update to `z`, then adds
   the SODA anchor delta:

   ```text
   z <- z - lr * update
   z <- z + lambda_t * (init - old_base)
   ```

   My fixed file applies the anchor pull before the learned update:

   ```text
   z <- (1 - lambda_t) z + lambda_t init
   z <- z - lr * update
   ```

   These are close but not identical. Your result is strong, so this is not a
   correctness accusation; it is a hidden algorithmic difference worth a small
   deterministic parity test and one HPO-row ablation.

4. Compare head/fallback grouping.

   You documented that `_anchor_param_groups()` is intentionally name-free, so
   a 2D classifier/head matrix can go through Muon/NorMuon. My helper keeps
   heads and embeddings in fallback. The result can go either way:

   - Muon head may improve classifier adaptation on CIFAR.
   - fallback head may reduce noisy spectral updates and transfer better to LM
     or larger output spaces.

   A focused `head_in_muon` vs `head_in_fallback` switch would clarify this
   without changing the core optimizer.

5. Compare GramNS coefficient families.

   `AnchorMuon` uses the simple quintic Newton-Schulz recurrence. My file uses
   the five-stage Polar Express coefficient schedule. Since the matrix update is
   small enough to dominate quality but not pure runtime, this is a clean
   ablation:

   ```text
   simple NS vs Polar Express GramNS
   same PMuonEq/NorMuon/SODA settings
   ```

6. Try zero-initialized NorMuon second moment.

   Your `_ensure_normuon_state()` initializes the NorMuon second moment to
   `1.0`; my file initializes it to `0.0`. Because both implementations restore
   the Frobenius norm, this mostly affects early row/column allocation rather
   than total update norm. It is still worth testing:

   ```text
   normuon_second_init in {0.0, 1.0}
   ```

## Possible Bugs Or Improvement Checks

1. SODA strength is currently hardcoded as `1 / (t + 1)`.

   That may be the intended recipe, but it means the optimizer does not expose
   a SODA scale/power knob. If later result tables compare "SODA" variants, the
   exact schedule should be reported because it is part of the algorithm.

2. Keep `sync_diagnostics=False` as the default.

   This is fixed in the latest run and README. I would keep any RMS diagnostics
   opt-in or sampled. The old CPU-sync stats are useful for debugging but should
   not be in timing comparisons.

3. Add a regression test for the aspect-scale flag when you add it.

   The test should verify:

   - row+aspect and row/no-aspect differ on a tall matrix;
   - the flag scales only after GramNS/NorMuon, not raw gradients or momentum;
   - DDP parity still holds.

4. Preserve your multi-seed discipline.

   The best thing in this folder is not a single implementation detail; it is
   the HPO plus three-seed final replay. Keep that structure for any borrowed
   improvement so the combined table does not regress into one-off wins.

Overall: your confidence run is the best evidence that the no-AMUSE
`SODA + PMuonEq + Gram + NorMuon` family is real. The main improvement I would
borrow from my folder is the row-wise NorMuon aspect multiplier, followed by the
small `col_gamma=0.05` / `normuon_beta=0.93` retune.
