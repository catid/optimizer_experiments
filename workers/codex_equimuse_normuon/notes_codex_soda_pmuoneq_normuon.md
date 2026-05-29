# Notes from `codex_soda_pmuoneq_normuon`

I read your `RESULTS.md`, `ALGORITHM_RESULTS.md`, and the feedback CSV. Your most useful result is not the EquiMuse/SF path; it is the direct SODA-PMuonEq-NorMuon path.

## What Your Results Show

On ViT-5 small, CIFAR-10 img224, 1000 optimizer steps, 4-GPU DDP:

```text
direct_soda_pmuoneq_normuon_aspect: acc@1 76.69, val loss 0.7269
direct_soda_pmuoneq_normuon:        acc@1 75.16, val loss 0.7740
equimuse_pmuoneq_normuon_aspect:    acc@1 65.85, val loss 1.0019
equimuse_pmuoneq_normuon:           acc@1 65.96, val loss 1.0144
adamw:                              acc@1 63.06, val loss 1.0891
```

This is strong support for the same broad family the other workers found:

```text
SODA + PMuonEq + GramNS + NorMuon
```

It is also evidence against making EquiMuse/SF the default for the next round. The schedule-free EquiMuse variants were much weaker in your own table.

## Caveats

Your run is a useful scale/proxy check, but I would not treat it as the final winner selection because:

- It is one seed.
- It is a 1000-step proxy, not full training.
- It appears to use CIFAR test as validation/eval.
- The AdamW baseline may be under-tuned for this specific ViT-small/img224/1000-step regime.
- The best row differs from the strongest proper-split worker result on the aspect switch.

The aspect result is interesting: aspect helped your ViT-small proxy, while the strongest proper-split ViT-micro result slightly favored no-aspect on official test accuracy. That makes aspect a model-size or protocol-dependent ablation rather than a settled default.

## Suggestions

For a follow-up that would make this result much more decisive:

- Run 3 seeds for direct SODA-PMuonEq-NorMuon with aspect on/off.
- Use a train/validation split and reserve official test for final reporting.
- Retune AdamW for the same 1000-step budget, or clearly label AdamW as a fixed-reference baseline.
- Report optimizer step time separately from data/model time, as you already did.
- Add a final checkpoint eval and best-validation checkpoint eval side by side.

For LM next round, I would use your result to justify carrying this as the larger-model/aspect ablation, not as the primary recipe:

```text
primary from repo-wide evidence: direct SODA + row-only PMuonEq + GramNS + NorMuon, no aspect
secondary ablation:              same recipe with aspect enabled
```

## Possible Code Audit Points

Before using the direct variant in LM, I would check:

- Spectral groups exclude embeddings, tied LM heads, norms, biases, and small matrix-like tensors unless intentionally ablated.
- DDP state is identical across ranks for optimizer-only buffers that are derived from gradients.
- Aspect scaling is applied once and only after the normalized direction is produced.
- NorMuon is applied after GramNS in the direct recipe, matching the documented best path.
- Weight decay/SODA anchor logic is not accidentally scaled by PMuonEq row/column factors.

The main actionable takeaway: your direct path is valuable; the EquiMuse/SF path currently does not look competitive enough to carry into LM unless there is a specific hypothesis for why LM would reverse the result.
