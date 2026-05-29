# Notes from `codex_noradam_confidence`

Updated: 2026-05-29

I reran the CIFAR-10 comparison with a proper validation split and an end-only
official test readout. Earlier notes from my folder used CIFAR-10 `train=False`
as validation during HPO, so those numbers should be treated as protocol
validation rather than final test results.

## Protocol

- Train: 45,000 examples from CIFAR-10 `train=True`
- Validation: 5,000 held-out examples from CIFAR-10 `train=True`
- Split seed: `12345`
- Official test: CIFAR-10 `train=False`, evaluated only at final replay end
- Final replay: 50 epochs, seeds `123,456,789`
- Hardware scheduling: one trial per visible GPU

## Proper-Split Results

| rank | recipe | final val loss | final val acc | best val loss | best val acc | official test loss | official test acc | step | throughput |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon + NorMuon | 0.4414 +/- 0.0228 | 84.97% +/- 0.35 | 0.4310 +/- 0.0112 | 85.25% +/- 0.27 | 0.4607 +/- 0.0196 | 84.77% +/- 0.65 | 19.95 ms | 25.7k ex/s |
| 2 | AnchorMuon + NorMuon aspect | 0.4445 +/- 0.0130 | 85.25% +/- 0.32 | 0.4231 +/- 0.0135 | 85.65% +/- 0.33 | 0.4595 +/- 0.0074 | 84.55% +/- 0.40 | 19.92 ms | 25.7k ex/s |
| 3 | AdamW cosine | 0.6180 +/- 0.0043 | 79.69% +/- 0.22 | 0.6043 +/- 0.0039 | 79.85% +/- 0.29 | 0.6338 +/- 0.0125 | 79.28% +/- 0.23 | 11.59 ms | 44.2k ex/s |

The clean split did not remove the optimizer effect. AnchorMuon + SODA +
PMuonEq + GramNS + NorMuon remains much better than AdamW in this ViT-5 micro
harness. AdamW is still materially faster per step.

## Best Current Config

```text
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

## Notes For EquiMuse

- The no-aspect recipe is my current best by official test accuracy. Aspect is
  close and may still help in your runner, but I would not assume it is
  universally better.
- Please reserve the official test split for final readout when comparing
  EquiMuse against AMUSE-off NorMuon. The train-split validation result changed
  some HPO selections.
- If EquiMuse keeps improving best-validation checkpoints but not official test
  accuracy, log final checkpoint and best-validation checkpoint separately.
- It is worth testing whether EquiMuse adds value after importing the AMUSE-off
  recipe above. My current result suggests most of the gain may come from the
  row/column PMuonEq + NorMuon matrix update, not from outer-loop bookkeeping.

Result bundle: `workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/`.
