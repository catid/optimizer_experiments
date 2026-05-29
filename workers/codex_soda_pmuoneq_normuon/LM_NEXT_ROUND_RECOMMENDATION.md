# LM Next Round Recommendation

I pulled the latest shared repo and compared the worker result sets on 2026-05-29.

## Recommendation

Use this as the primary optimizer candidate for the next language-modeling round:

```text
AnchorMuon / direct SODA-PMuonEq-NorMuon
SODA: all groups
AMUSE: off
MiMuon: off
PMuonEq: row-only
row_gamma: 0.35
col_gamma: 0.0
pmuon_beta: 0.90 or 0.95
Gram-Newton-Schulz: enabled
NorMuon: enabled after GramNS
normuon_beta: 0.93
aspect scaling: off for primary, on as ablation
```

The most specific current best row is from `workers/codex_noradam_confidence`:

```text
normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93
```

I would describe the winner as:

```text
SODA + row-only PMuonEq + GramNS + NorMuon, direct/no-AMUSE/no-MiMuon, no aspect
```

## Why This Is The Best Current Candidate

The strongest evidence is the proper-split CIFAR-10 run from `codex_noradam_confidence`:

```text
model: vit5_micro
params: 458,858
data: CIFAR-10 45k train / 5k validation, official 10k test only at end
batch: 512
epochs: 50
steps: 4,350
seeds: 123, 456, 789
precision/layout: bf16, channels-last
```

Results:

| optimizer | official test loss | official test acc | step time | throughput |
|---|---:|---:|---:|---:|
| AnchorMuon + NorMuon, no aspect | 0.4607 +/- 0.0196 | 84.77% +/- 0.65 | 19.95 ms | 25.7k ex/s |
| AnchorMuon + NorMuon, aspect | 0.4595 +/- 0.0074 | 84.55% +/- 0.40 | 19.92 ms | 25.7k ex/s |
| AdamW cosine | 0.6338 +/- 0.0125 | 79.28% +/- 0.23 | 11.59 ms | 44.2k ex/s |

This is the cleanest result because it has a proper validation split, official test-only final evaluation, and 3 seeds.

## Cross-Worker Stability

The broad optimizer family is stable:

```text
direct SODA + PMuonEq + GramNS + NorMuon beats AdamW in all worker result sets I found.
```

The exact aspect switch is not stable:

- `codex_noradam_confidence` proper split: no-aspect wins official test accuracy mean.
- `codex_noradam_confidence` earlier test-as-val run: no-aspect also wins final accuracy/loss.
- `codex_soda_pmuoneq_normuon` ViT-tiny proxy: aspect row was best.
- `codex_equimuse_normuon` ViT-small 1000-step proxy: aspect row was best.

Conclusion: aspect scaling should be a small LM ablation, not part of the primary claim.

## Not Recommended As Primary

I would not make these the primary LM candidate:

- EquiMuse/SF variants: the EquiMuse worker's own table shows the direct path was far better.
- AMUSE variants: useful historically, but current best evidence uses AMUSE off.
- MiMuon variants: useful ablation, but current best evidence uses MiMuon off.
- Full/stale PMuon: higher overhead and not necessary given PMuonEq results.

## LM-Specific Risks

CIFAR optimizer evidence does not automatically transfer to language modeling. Before a serious LM run, audit and report:

- Tied token embedding / LM head grouping.
- Whether embedding and unembedding are excluded from spectral updates or intentionally included.
- Norm/bias/vector exclusion.
- Small matrix exclusions.
- DDP parity for optimizer state.
- Memory overhead from row/column EMA state on large vocab/projection layers.
- Comparison against the actual LM baseline optimizer, including MuonAdamW if that is the baseline recipe.

## Suggested LM Sweep

Keep the search narrow:

| parameter | values |
|---|---|
| matrix_lr | 0.004, 0.006, 0.008 |
| row_gamma | 0.25, 0.35, 0.45 |
| col_gamma | 0.0, 0.05 |
| normuon_beta | 0.90, 0.93, 0.95 |
| pmuon_beta | 0.90, 0.95 |
| aspect | false, true |

Baselines for the LM round:

- Repo-default LM optimizer, especially MuonAdamW where applicable.
- Tuned AdamW/fused AdamW.
- Plain Muon/GramNS if already supported.

## Bottom Line

There is enough evidence to pick one primary candidate for the LM round:

```text
SODA + row-only PMuonEq + GramNS + NorMuon, direct/no-AMUSE/no-MiMuon, no aspect
```

There is not enough evidence to freeze every switch. Carry aspect scaling as the one required ablation, and treat LM parameter grouping as the main bug-risk area.
