# Notes from `codex_equimuse_normuon`

I pulled the latest shared repo and reviewed this worker folder on 2026-05-28.
I ran:

```bash
python -m py_compile soda_pmuoneq_normuon.py tests/test_soda_pmuoneq_normuon.py
pytest -q tests/test_soda_pmuoneq_normuon.py
```

The focused tests passed: `4 passed`.

## Things That Look Useful

- This is a clean fixed-recipe standalone file, which matches the user's request
  better than an ablation-heavy optimizer.
- The README includes a concrete tuned default and a direct AdamW comparison.
- The no-op `train()`/`eval()` methods make this optimizer easy to plug into
  training harnesses that call schedule-free optimizers' mode swaps.
- Same-shape matrix bucketing is already present, which is the right throughput
  pattern for GramNS + NorMuon.

## Comparability Notes

This implementation is close to my `EquiMuseNorMuon` recipe but not identical:

- This file removes AMUSE/schedule-free X/Y/Z bookkeeping. My standalone keeps
  the schedule-free outer loop and validates by reproducing the earlier
  EquiMuse+NorMuon result. Your version is a simpler "SODA direct-on-weights"
  recipe, which may be preferable for some projects, but it should not be mixed
  casually with AMUSE/SF numbers.
- `_normuon_row_normalize` applies an extra `sqrt(max(1, rows / cols))` after
  already applying the Muon scale `0.2 * sqrt(max(rows, cols))`. My recipe does
  not add that second aspect-ratio factor after NorMuon. This is a meaningful
  per-layer LR change for tall matrices; I would either document it as a tuned
  part of the recipe or ablate it.
- Your best tuning uses stronger PMuonEq/NorMuon settings (`row_gamma=0.35`,
  `col_gamma=0.05`, `normuon_beta2=0.93`) than my ViT-5-Small/img224 result
  (`row_gamma=0.15`, `col_gamma=0.15`, `normuon_beta2=0.9`). The harnesses are
  different enough that this is expected, but it would be useful to run both
  recipes on one shared model/data setup.

## Suggested Next Fixes

- Add a DDP rank-spread smoke test. The README reports DDP runs, but the tests
  are CPU/local. A tiny 2-GPU check should verify params and optimizer state
  stay rank-identical after `optimizer.step()`.
- Add `state_dict` / `load_state_dict` coverage. This optimizer carries SODA
  anchors, PMuonEq EMAs, NorMuon EMAs, momentum buffers, and group step counters;
  resume correctness is worth testing explicitly.
- Add a batch-vs-single transformation parity test for two same-shaped matrix
  parameters. The code buckets same-shape matrices, so a parity test will catch
  future changes to `_transform_matrix_bucket`.
- Revisit the fallback grouping heuristic:
  `fallback_tokens` contains the broad substring `"head"`. That is fine for
  `classifier_head` and `lm_head`, but it can accidentally catch unrelated
  parameter names in other architectures. I would narrow it to exact/common
  output-head patterns such as `lm_head`, `classifier_head`, `head.weight`, and
  `unembed`.
- External LR scheduling is currently not supported: `step()` recomputes
  `lr = base_lr * min(1, t / warmup_steps)` every step. If this is intended,
  say so in the README. If not, add a `use_external_lr` option or detect
  externally written group LRs like the AMUSE-style implementations do.

