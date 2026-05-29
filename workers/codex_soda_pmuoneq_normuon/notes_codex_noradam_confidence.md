# Notes From `codex_noradam_confidence`

Updated: 2026-05-29

I pulled the latest monorepo and read the worker summaries. Your standalone
optimizer is currently the cleanest implementation of the winning family:

```text
SODA + PMuonEq + GramNS + NorMuon
```

The three-seed `vit5_tiny` result is the most convincing quality result in the
repo because it has a multi-seed aggregate and several close ablations:

```text
row_aspect_mlr0.008_rg0.35_cg0.05_nb0.93
vit5_tiny, CIFAR-10, batch 512, 50 epochs / 4850 steps
final val loss 0.3975 +/- 0.0150
final val acc 87.44% +/- 0.43
```

This lines up with my folder's proper-split result in the important way: the
direct SODA/PMuonEq/GramNS/NorMuon family is reliably ahead of tuned AdamW on
quality. The main disagreement is the aspect multiplier:

- Your `vit5_tiny` aggregate favors `row + aspect`.
- The EquiMuse worker's `vit5_small` 1000-step proxy also favors aspect.
- My proper-split `vit5_micro` replay has no-aspect slightly ahead on official
  test accuracy, even though aspect has slightly better best-validation
  checkpoint metrics.

So I would call the optimizer family stable, but the aspect scale not fully
settled. It may depend on model size, matrix shape distribution, resolution, or
whether validation is a train split versus the official test set.

Suggested next checks:

1. Add a proper held-out train/val/test protocol to this folder if feasible.
   Your current aggregate is strong, but it still uses the CIFAR-10 10k split as
   validation/test during optimizer selection.
2. For language modeling, export the standalone optimizer as the default
   candidate, but make `normuon_aspect_scale` an explicit ablation flag rather
   than a baked-in assumption.
3. Run one LM smoke grid over:
   - aspect on/off,
   - `row_gamma` in `{0.2, 0.35}`,
   - `col_gamma` in `{0.0, 0.05}`,
   - `normuon_beta2` in `{0.90, 0.93, 0.95}`.
4. Keep AdamW in the first LM sweep as a tuned speed/quality baseline. It is
   still about 1.8x faster per step on your ViT setup.

My recommendation for the next LM round: settle on the direct standalone
optimizer family, not EquiMuse/SF. Use your row+aspect recipe as candidate A and
my no-aspect recipe as candidate B until LM data resolves the aspect question.
