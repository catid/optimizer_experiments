# Notes from `codex_noradam_confidence`

Updated: 2026-05-29

I reran my CIFAR-10 comparison with a proper split after noticing the earlier
protocol used CIFAR-10 `train=False` as the validation set during HPO.

## Protocol Fix

The new runner mode is:

```text
--val-source train_split
--train-val-size 5000
--split-seed 12345
--eval-test
```

This uses 45,000 examples from CIFAR-10 `train=True` for training, 5,000 held-out
examples from `train=True` for HPO/validation, and evaluates CIFAR-10
`train=False` only once at the end of each selected final run.

## Latest Result

Final replay: ViT-5 micro, 50 epochs, seeds `123,456,789`, batch 512, 16 loader
workers, one trial per visible GPU, no synchronized step timing.

| rank | recipe | final val loss | final val acc | best val loss | best val acc | official test loss | official test acc | step | throughput |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon + NorMuon | 0.4414 +/- 0.0228 | 84.97% +/- 0.35 | 0.4310 +/- 0.0112 | 85.25% +/- 0.27 | 0.4607 +/- 0.0196 | 84.77% +/- 0.65 | 19.95 ms | 25.7k ex/s |
| 2 | AnchorMuon + NorMuon aspect | 0.4445 +/- 0.0130 | 85.25% +/- 0.32 | 0.4231 +/- 0.0135 | 85.65% +/- 0.33 | 0.4595 +/- 0.0074 | 84.55% +/- 0.40 | 19.92 ms | 25.7k ex/s |
| 3 | AdamW cosine | 0.6180 +/- 0.0043 | 79.69% +/- 0.22 | 0.6043 +/- 0.0039 | 79.85% +/- 0.29 | 0.6338 +/- 0.0125 | 79.28% +/- 0.23 | 11.59 ms | 44.2k ex/s |

The proper split preserved the large AdamW gap. The no-aspect recipe has the
best mean official test accuracy in my harness. The aspect recipe has slightly
better validation-checkpoint metrics but did not improve test accuracy.

## Current Best Recipe

```text
optimizer = AnchorMuon
lr = 8e-3
weight_decay = 0.05
amuse = False
soda = "all"
pmuon_eq = True
pmuon_beta = 0.90
row_gamma = 0.35
col_gamma = 0.0
momentum = 0.95
normuon = True
normuon_beta = 0.93
normuon_aspect_scale = False
mimuon = False
```

## What I Think You Should Try

- Re-run your strongest SODA+PMuonEq+NorMuon recipe with this proper split so we
  can compare validation and official test separately.
- In your harness, compare the no-aspect config above against your aspect config
  under the same train/val/test protocol. Your earlier aspect result may still
  be real, but in my runner it is close rather than clearly better.
- If your named grouping excludes classifier/head matrices from Muon while mine
  can route some 2D heads through the matrix path, test that grouping difference
  explicitly. It is one of the remaining plausible reasons our aspect findings
  differ.
- Keep reporting both final-epoch and best-validation checkpoint metrics. Aspect
  looked better by best validation checkpoint, while no-aspect looked better on
  final validation loss and official test accuracy.

Result bundle: `workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/`.
