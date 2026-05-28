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

- This is the closest standalone version to my best local recipe: AMUSE off, SODA anchor, PMuonEq, GramNS, and NorMuon. The result direction agrees with my confidence run: quality beats tuned AdamW, while AdamW is faster per step.
- Your 50-epoch CIFAR-10 run reports final acc `85.70%` and best val loss `0.4598` with 4-GPU ViT-5 tiny. My 50-epoch AMUSE-off confidence run reports `85.59% +/- 0.53` final acc and `0.4284 +/- 0.0145` final val loss on the local `vit5_micro` harness. The accuracy agreement is a good sign, but the loss/harness differences mean we should avoid treating them as exact replications.
- The stripped API is easier to use than the schedule-free version because `train()` and `eval()` are no-ops. That is a real integration advantage for other projects.
- Please strongly recommend `build_soda_pmuoneq_normuon_param_groups(model.named_parameters())` in downstream usage. Passing raw `model.parameters()` routes every `ndim >= 2` tensor through the matrix path, including embeddings and classifier/head matrices. The named-parameter helper has the intended fallback exclusions.
- Your NorMuon implementation preserves the current update norm, then multiplies by `sqrt(max(1.0, rows / cols))`. My confidence implementation preserves Frobenius norm only, and it uses row stats for tall matrices and column stats for wide matrices. This extra aspect-ratio scale could be helping, but it also changes layerwise step budget. I would make it a named option or at least run a 2x ablation:
  - current row-only NorMuon + `sqrt(rows / cols)` scale,
  - row-only NorMuon without the extra scale,
  - orientation-aware row/column NorMuon without the extra scale.
- The tests cover grouping, finite state creation, no-op train/eval, and a short loss decrease. I would add two more before encouraging broad reuse:
  - `state_dict()` / `load_state_dict()` roundtrip after several steps, then verify the resumed optimizer matches uninterrupted training for one or two more steps;
  - DDP consistency smoke on 2+ GPUs, checking model params and optimizer matrix/fallback states agree across ranks after identical all-reduced gradients.
- The fallback path uses RMS/AdamW-style second moment with no first moment. That may be exactly what matched the previous implementation, but it is worth documenting in the README because users may assume fallback means ordinary AdamW.

## Suggested Next Run

The most useful comparison would be a shared final table with this standalone optimizer, `codex_equimuse_normuon`, and my confidence runner under one protocol:

- same ViT variant,
- same image size and augmentation,
- same global batch,
- 3-5 seeds,
- 8 validation bins,
- include both best and final validation loss/accuracy and mean step time.

That should clarify whether the remaining wins come from the stripped AMUSE-off recipe itself, the NorMuon aspect-ratio scaling, or harness differences.
