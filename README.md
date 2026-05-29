# Optimizer Experiments Monorepo

## Current Best Result

The clearest current winner lives in
`workers/codex_noradam_confidence`.

**Best version:** `normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93`

**Algorithm:** AnchorMuon with SODA on all parameter groups, row-only PMuonEq,
five-step Gram Newton-Schulz, and NorMuon. AMUSE, MiMuon, and post-NorMuon
aspect scaling are disabled.

**Exact setting:** ViT-5 micro on CIFAR-10, 45k train / 5k validation split from
the official training set, official 10k test split evaluated only at the end,
batch size 512, 50 epochs, three seeds, BF16 autocast, channels-last tensors,
16 dataloader workers, and one trial per visible GPU.

| Recipe | Official test acc | Official test loss | Final val acc | Final val loss | Step time |
|---|---:|---:|---:|---:|---:|
| AnchorMuon + SODA + PMuonEq + GramNS + NorMuon | 84.77% +/- 0.65 | 0.4607 +/- 0.0196 | 84.97% +/- 0.35 | 0.4414 +/- 0.0228 | 19.95 ms |
| AnchorMuon + NorMuon + aspect scale | 84.55% +/- 0.40 | 0.4595 +/- 0.0074 | 85.25% +/- 0.32 | 0.4445 +/- 0.0130 | 19.92 ms |
| AdamW cosine baseline | 79.28% +/- 0.23 | 0.6338 +/- 0.0125 | 79.69% +/- 0.22 | 0.6180 +/- 0.0043 | 11.59 ms |

Use `workers/codex_noradam_confidence/ALGORITHM_RESULTS.md` for the full
self-contained algorithm description, hyperparameters, protocol, and caveats.
The runner also exposes a `--preset best_cifar10` preset containing the winning
configuration, the closest aspect-scaled near miss, and the AdamW baseline.
