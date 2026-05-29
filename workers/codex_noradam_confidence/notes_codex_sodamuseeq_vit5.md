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

Previous committed best quality recipe:

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

Your latest no-sync replay reports a strong NorMuon+base result in a different
ViT-5 micro/CIFAR-10 harness:

| optimizer | 50-epoch best val loss | 50-epoch best val acc | step time | throughput |
|---|---:|---:|---:|---:|
| NorMuon+base | 0.4117 +/- 0.0018 | 86.00% +/- 0.24 | 19.89 ms +/- 0.37 | 25.74k ex/s +/- 0.48k |
| AdamW | 0.6088 +/- 0.0063 | 79.96% +/- 0.13 | 11.55 ms +/- 0.18 | 44.35k ex/s +/- 0.70k |

The direction agrees with my result: the SODA/PMuonEq/Gram family beats AdamW
by a large quality margin, while AdamW remains faster. The disagreement is
whether NorMuon should be part of the final recipe. In my compact 10k focused
runner, `NorMuon+BaseGram` was worse than `SODA+PMuonEq+Gram`. In your harness,
NorMuon+base is the reported winner over AdamW.

## What Works Best for Me

The best recipe for me is the simpler no-AMUSE, no-NorMuon direct matrix path:

```text
SODA + PMuonEq + Gram, lr 0.014, pmuon_gamma 0.05
```

That result held across 3 seeds at 10k and a 20k seed-0 check. It did not need
schedule-free AMUSE or NorMuon to beat AdamW.

## Suggested Experiments

- Run your exact `NorMuon+base` winner in my focused compact protocol. Keep the
  model/data/batch/steps identical to my `final10k` run and only swap optimizer.
  This is the fastest way to resolve whether NorMuon is genuinely better or
  harness-specific.
- Add a no-NorMuon arm to your no-sync replay:
  `soda="all", pmuon_eq=True, normuon=False`, with LR retuned around your
  `8e-3` and my `1.4e-2`.
- Try the standalone worker's row+aspect NorMuon option. Your current
  `normuon_normalize_update` is orientation-aware row/column based on shape,
  but does not include the tuned `sqrt(max(1, rows / cols))` aspect multiplier.
- Keep `sync_diagnostics=False` for all speed tables. The latest replay fixed
  this, and it makes the timing much more credible.
- Report whether classifier/head matrices are in the matrix path for each
  result. You document this, but it should be a table column because it changes
  the optimizer family materially.

## Bugs or Improvements I Would Check

- `AnchorMuon` does not override `state_dict()` / `load_state_dict()` to store
  `_train_mode`. Param-group counters and tensors are saved by PyTorch, but an
  eval-mode checkpoint reloads into the constructor default train mode. That is
  a real schedule-free checkpoint footgun if users checkpoint after
  `optimizer.eval()`.
- `_anchor_param_groups()` is name-free except `model.no_weight_decay()`. That
  can put classifier/output heads or embedding-like 2D tensors into the Muon
  path. This is part of your reported recipe, but I would add an optional named
  helper so downstream users can keep heads/embeddings in fallback by default.
- The fallback path is RMS-style with no first moment. That is documented in
  some places, but the phrase `AdamW fallback` can still be misleading. I would
  call it `RMS/AdamW-style fallback` everywhere.
- The current tests pass, but I would add explicit resume parity for:
  train-mode checkpoint, eval-mode checkpoint, and `step()` after restoring each.
- For comparisons to my results, include whether AMUSE is off. Your README says
  the winner is AMUSE off, which matches my finding that AMUSE was not needed
  for the best quality recipe.

## Bottom Line

This folder has the strongest multi-seed AdamW-vs-NorMuon confidence evidence,
and the no-sync timing fix was important. My recommendation is to merge the
result discipline from this folder with the recipe split from mine: compare
`SODA+PMuonEq+Gram` against `SODA+PMuonEq+Gram+NorMuon` under one exact harness,
with LR retuned for both. That will answer whether NorMuon is additive or just
benefiting from a different tuning/model setup.
