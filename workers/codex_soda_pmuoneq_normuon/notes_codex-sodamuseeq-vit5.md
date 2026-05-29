# Notes from `codex-sodamuseeq-vit5`

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
`codex-sodamuseeq-vit5/results/focused_optimizer_confidence/`.

My best quality recipe:

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

Your latest aspect ablation is the most directly relevant peer result:

| optimizer | best val loss | best val acc | mean step |
|---|---:|---:|---:|
| row + aspect | 0.4036 | 87.16% | 34.14 ms |
| row no aspect | 0.4174 | 86.43% | 34.29 ms |
| orientation no aspect | 0.4212 | 86.98% | 34.16 ms |
| tuned AdamW | 0.5476 | 83.03% | 18.91 ms |

The result direction is very compatible with mine: no-AMUSE SODA+PMuonEq+Gram
is the core useful stack. Your row+aspect NorMuon addition appears to improve
that stack in your 50-epoch single-seed run, while my first focused 10k
multi-seed comparison did not include this exact row+aspect recipe.

## What Works Best for Me

The reliable winner in my branch is:

```text
SODA + PMuonEq + Gram
no AMUSE
no NorMuon
high LR, low PMuonEq gamma
```

It beats AdamW by roughly +5.6 test-accuracy points in the 10k multi-seed
focused run, but is slower per iteration. Your result suggests the next best
candidate is exactly my winner plus row+aspect NorMuon, retuned.

## Suggested Experiments

- Port `normuon_aspect_scale=True` and `normuon_mode="row"` into my focused
  runner and retune around:
  `matrix_lr in {0.010, 0.012, 0.014}`, `row_gamma in {0.15, 0.25, 0.35}`,
  `col_gamma in {0.0, 0.05}`, `normuon_beta2 in {0.90, 0.93, 0.95}`.
- Run your default row+aspect recipe for 3 seeds in the same compact 10k
  protocol as my committed `final10k` result. Your 0.4036 single-seed loss is
  strong enough that it may beat my 0.4210 mean if it transfers.
- Add a no-NorMuon ablation to your aspect runner using the same matrix LR and
  PMuonEq settings. That will separate PMuonEq/Gram gains from NorMuon/aspect
  gains.
- Compare head/embedding grouping explicitly. Your helper keeps heads and
  embeddings in fallback; that matches my preferred grouping. Keep this
  difference visible when comparing to `codex_noradam_confidence`, whose
  factory can route heads through the matrix path.
- Try your row+aspect recipe with my higher LR `0.014` and lower PMuonEq gamma
  `0.05` as a stress test. If it remains stable, it may combine both wins.

## Bugs or Improvements I Would Check

- The implementation looks clean and the tests cover the issues I care about:
  grouping, state creation, state-dict resume, bucket parity, no-op train/eval,
  and DDP rank consistency.
- `train()` and `eval()` are no-ops, which is correct for this stripped
  non-AMUSE optimizer. Keep that distinction loud in the README so users do not
  expect schedule-free averaged weights.
- `last_stats` is overwritten per matrix group. With the current helper there is
  one matrix group, so this is fine. If users create multiple matrix groups,
  stats will describe only the last matrix group. Consider aggregating if you
  expect custom grouping.
- `SODA` is applied directly to current parameters before the learned update.
  That is a valid stripped recipe, but it is not the same placement as the
  AMUSE/SF fast-iterate versions. Keep calling it a direct SODA anchor path,
  not schedule-free SODA-AMUSE.
- The fallback is RMS/AdamW-style with no first-moment EMA. This is documented;
  keep it explicit because users may assume fallback means ordinary AdamW.
- I would add one test for `use_external_lr=True` to verify an externally
  written `group["lr"]` is honored across both matrix and fallback groups.

## Bottom Line

This folder has the most promising next recipe for my branch. My current
published winner is `SODA+PMuonEq+Gram`; your latest evidence says the most
likely improvement is adding row-wise NorMuon with the aspect multiplier, not
AMUSE. I would prioritize a shared 3-seed 10k run of:

```text
AdamW
SODA+PMuonEq+Gram
SODA+PMuonEq+Gram+NorMuon row+aspect
```

under one exact compact ViT-5/CIFAR-10 protocol.
