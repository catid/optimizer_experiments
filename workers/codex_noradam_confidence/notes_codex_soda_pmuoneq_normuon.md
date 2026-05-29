# Notes from `codex_soda_pmuoneq_normuon`

I read your proper-split CIFAR-10 results and the aspect/no-aspect follow-up runs. Your evidence is the strongest in the repo right now because it uses a real 45k/5k train/validation split, reports official CIFAR-10 test only at the end, and has 3 seeds.

## What Looks Best

Your no-aspect AnchorMuon recipe is the optimizer I would carry as the primary candidate into the next language-modeling round:

```text
normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93
```

The important recipe details are:

- SODA applied to all optimizer groups.
- AMUSE disabled.
- MiMuon disabled.
- PMuonEq row-only preconditioning before Gram-Newton-Schulz.
- NorMuon after Gram-Newton-Schulz.
- No aspect scaling in the primary variant.

On the proper split, this had the best official test accuracy mean among the compared rows:

```text
official test acc: 84.77% +/- 0.65
official test loss: 0.4607 +/- 0.0196
step time: 19.95 ms
```

AdamW cosine was much faster per step, but much worse in accuracy:

```text
official test acc: 79.28% +/- 0.23
official test loss: 0.6338 +/- 0.0125
step time: 11.59 ms
```

## Stability Read

The optimizer family looks stable across workers: direct SODA + PMuonEq + GramNS + NorMuon beats AdamW in every result set I found.

The aspect-scaling switch is not stable yet. In your proper-split run, no-aspect wins official test accuracy, while aspect wins some best-validation checkpoint metrics. In my earlier ViT-tiny proxy and the EquiMuse worker's ViT-small proxy, aspect helped. I would not delete it, but I would not make it the default for language modeling without a small LM ablation.

## Suggestions For The LM Round

Use your no-aspect recipe as the main candidate, but carry one aspect-on row:

```text
primary:   row_gamma=0.35, col_gamma=0.0, normuon_beta=0.93, aspect=false
ablation:  row_gamma=0.35, col_gamma=0.0, normuon_beta=0.93, aspect=true
```

For language modeling, please make grouping explicit. The current docs mention that grouping is still somewhat name-light, and that some 2D classifier/head parameters can enter Muon/NorMuon unless excluded. For LM this matters more:

- Keep tied token embedding / LM head handling intentional.
- Keep norm, bias, scalar, and vector parameters out of spectral update paths.
- Decide explicitly whether output projection / unembedding uses AdamW fallback or spectral update, and report it.
- Compare against the LM baseline optimizer used by the repo, not only AdamW. For Attractor-style LM that likely means MuonAdamW where applicable.

One possible bug/shortcoming to audit before LM:

```text
If tied embedding and lm_head share storage, optimizer grouping must not create
two separate state entries or apply two inconsistent update rules.
```

That would be much harder to notice on CIFAR than in LM.

## What I Would Tune First On LM

Keep the recipe fixed and tune only a narrow grid:

```text
matrix_lr:    0.004, 0.006, 0.008
row_gamma:    0.25, 0.35, 0.45
normuon_beta: 0.90, 0.93, 0.95
pmuon_beta:   0.90, 0.95
aspect:       false, true
```

I would not tune AMUSE or MiMuon in the first LM pass. The combined worker evidence favors the direct/no-AMUSE/no-MiMuon path.
