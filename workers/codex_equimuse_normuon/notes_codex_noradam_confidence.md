# Notes from `codex_noradam_confidence`

Updated: 2026-05-29

I pulled latest `main`, read the feedback left for my folder, implemented the
post-NorMuon aspect multiplier suggestion, validated it, and ran a focused HPO
plus a 3-seed 50-epoch replay.

## What Changed In My Folder

The new optional flag is:

```text
normuon_aspect_scale
```

When enabled, it multiplies tall matrix updates by
`sqrt(max(1, rows / cols))` after Gram Newton-Schulz and NorMuon normalization.
It is a final update scale, not a gradient/momentum/PMuonEq preconditioner. The
default remains disabled.

I also fixed a runner bug where relative output paths broke subprocess result
collection after the worker changed `cwd` to the repo root. Worker launches now
use resolved `output_dir` and `data_path`.

## Validation

```bash
/home/catid/screen/.venv/bin/python -m py_compile \
  workers/codex_noradam_confidence/optim_anchormuon.py \
  workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  workers/codex_noradam_confidence/tests/test_anchormuon_modes.py \
  workers/codex_noradam_confidence/tests/ddp_smoke_anchormuon.py

/home/catid/screen/.venv/bin/python -m pytest -q \
  workers/codex_noradam_confidence/tests/test_anchormuon_modes.py

CUDA_VISIBLE_DEVICES=0,1 /home/catid/screen/.venv/bin/torchrun \
  --standalone --nproc-per-node=2 \
  workers/codex_noradam_confidence/tests/ddp_smoke_anchormuon.py
```

Results:

- compile passed;
- unit tests: `13 passed`;
- DDP smoke passed.

## Latest Three-Seed Result

Protocol:

- ViT-5 micro, CIFAR-10.
- Full train split, official test split used as validation.
- HPO: 12 epochs.
- Final replay: 50 epochs, seeds `123,456,789`.
- Batch size 512, 16 dataloader workers.
- All visible GPUs scheduled one trial per GPU.
- No synchronized step timing.

| rank | recipe | final val loss | final acc | best val loss | best acc | step | throughput |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon NorMuon, no aspect | 0.4225 +/- 0.0027 | 85.86% +/- 0.23 | 0.4198 +/- 0.0043 | 85.98% +/- 0.09 | 20.20 ms | 25.4k ex/s |
| 2 | AnchorMuon NorMuon, aspect | 0.4288 +/- 0.0103 | 85.43% +/- 0.29 | 0.4236 +/- 0.0083 | 85.60% +/- 0.17 | 19.56 ms | 26.2k ex/s |
| 3 | AdamW cosine | 0.6210 +/- 0.0105 | 79.66% +/- 0.12 | 0.6088 +/- 0.0063 | 79.96% +/- 0.13 | 11.54 ms | 44.4k ex/s |

The aspect multiplier did not beat the retuned no-aspect recipe in my harness,
though it remained much better than AdamW and was slightly faster than no-aspect.

Current best recipe:

```text
amuse = False
soda = "all"
pmuon_eq = True
pmuon_beta = 0.90
row_gamma = 0.35
col_gamma = 0.05
momentum = 0.95
normuon = True
normuon_beta = 0.95
normuon_aspect_scale = False
mimuon = False
```

## Notes For EquiMuse

Your EquiMuse result still shows a within-folder gain over AdamW in the shorter
DDP run. My latest result suggests that, before adding more outer-loop
complexity, it is worth testing whether the simple AMUSE-off row/column
PMuonEq+NorMuon recipe transfers into your harness.

Concretely, I would test these in one common EquiMuse runner:

1. EquiMuse row recipe as-is.
2. EquiMuse row + post-NorMuon aspect scale.
3. AMUSE-off recipe with `row_gamma=0.35`, `col_gamma=0.05`,
   `normuon_beta=0.95`.
4. Same AMUSE-off recipe with aspect enabled.

Keep the same seed set and image/model settings. The key question is whether
AMUSE/SF bookkeeping adds value once the row/column PMuonEq+NorMuon recipe is
retuned.

## Bugs Or Follow-Ups Still Worth Checking

- `EquiMuseNorMuon.load_state_dict()` should avoid mutating the caller's state
  dict if it still uses `pop("train_mode", ...)`.
- Add resume parity after checkpoint load: train, save, reload, then continue
  both copies and compare parameters plus optimizer state.
- Log active SODA behavior and anchored parameter counts in each run summary.
- Reserve a true untouched test split. My runner currently uses CIFAR-10
  `train=False` as validation during HPO, so these are protocol-validation
  numbers rather than clean final test numbers.

My best current conclusion is conservative: row/column PMuonEq + NorMuon + SODA
is clearly better than AdamW in my CIFAR-10 protocol, but the aspect multiplier
is not yet a confirmed universal improvement.
