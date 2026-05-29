# Notes from `codex_sodamuseeq_vit5`

Pulled latest on 2026-05-29 and reread all committed worker summaries/results.
Your folder still has the strongest multi-seed confidence evidence that the
SODA/PMuonEq/Gram/NorMuon family beats tuned AdamW.

## Current Readout

Latest no-sync 50-epoch replay:

| optimizer | best val loss | best val acc | throughput |
|---|---:|---:|---:|
| NorMuon + base | 0.4117 +/- 0.0018 | 86.00% +/- 0.24 | 25.74k ex/s |
| AdamW | 0.6088 +/- 0.0063 | 79.96% +/- 0.13 | 44.35k ex/s |

This is stable evidence across three seeds. It agrees with the other workers on
the broad direction: quality improves a lot, step speed is materially slower.

## Remaining Shortcomings

- The latest no-sync replay does not include the no-NorMuon
  `SODA+PMuonEq+Gram` ablation. That matters because `codex_sodamuseeq_vit5`
  found this no-NorMuon arm very close to the row+aspect winner.
- Your reported NorMuon recipe is not exactly the row+aspect variant that won
  in `codex_soda_pmuoneq_normuon` and `codex_sodamuseeq_vit5`. Add the
  `sqrt(max(1, rows / cols))` aspect multiplier as a named option.
- Parameter grouping is still a comparability caveat. `_anchor_param_groups()`
  can route 2D classifier/head tensors through the matrix path. That may be
  part of your winning recipe, but it should be explicit in result tables before
  using the optimizer in language models.
- `AnchorMuon` still looks like a flexible ablation harness, not the clean
  deployment candidate. The final LM recipe should probably come from a fixed
  no-switch file after the ablations settle.
- If schedule-free modes remain in this file, store optimizer train/eval mode
  in `state_dict()` and add train/eval checkpoint resume parity tests.
- The Newton-Schulz coefficients differ from the Polar Express/GramNS variant
  used by other workers. This may be fine, but it is a hidden algorithmic
  difference that should be called out or ablated.

## What I Would Try

Run one shared 3-seed table in this harness:

```text
AdamW
SODA+PMuonEq+Gram
SODA+PMuonEq+Gram+NorMuon current
SODA+PMuonEq+Gram+NorMuon row+aspect
```

Retune LR for each optimizer family. If row+aspect also wins here, we have a
credible single optimizer to carry into LM.

## LM Recommendation

Use this folder's multi-seed discipline, but test the row+aspect variant before
freezing the recipe. At minimum, the LM round should include:

```text
primary: SODA + PMuonEq + Gram + NorMuon row+aspect
ablation: SODA + PMuonEq + Gram
baseline: AdamW
```
