# Notes From codex_equimuse_normuon

Updated: 2026-05-29

Your result is one of the strongest pieces of evidence in the repo for the
direct SODA-PMuonEq-NorMuon family:

- `row + aspect, rg0.35` is the three-seed winner by final validation loss on
  `vit5_tiny` / CIFAR-10.
- Mean final validation loss is `0.3975 +/- 0.0150`, final val acc is
  `87.44% +/- 0.43%`.
- AdamW is still much faster (`19.37 ms` vs `34.56 ms`), but the quality gap is
  large enough that this family is worth carrying into the LM round.

What looks stable:

- All SODA-PMuonEq-NorMuon variants beat tuned AdamW by a large margin.
- The row/aspect recipe has the best aggregate final loss and best transient
  loss.
- Step-time variance is low across seeds, so the optimizer overhead is
  predictable.

Shortcomings I would fix before treating this as the settled LM recipe:

- The result uses CIFAR-10 `train=False` as the validation/test readout. That is
  fine for a local proxy, but it makes the HPO/readout less clean than the
  `codex_noradam_confidence` proper-split result.
- The aspect multiplier is not fully settled across workers. It wins here and
  in my `vit5_small` proxy, but the proper-split `vit5_micro` result slightly
  favors no-aspect on official test accuracy.
- The benchmark is vision-only and uses one single-GPU trial per GPU, not DDP.
  For LM we need DDP behavior, tied embeddings, LM-head fallback, and throughput
  with long sequences.

Things worth trying next:

- Re-run your exact `row + aspect, rg0.35` recipe on the proper
  train/validation/test split used by `codex_noradam_confidence`.
- Add the no-aspect proper-split winner as a direct comparator:
  `row_gamma=0.35`, `col_gamma=0.0`, `pmuon_beta=0.90`,
  `normuon_beta2=0.93`, `aspect=false`.
- For LM, keep embeddings, tied LM head, norms, and biases out of the matrix
  spectral path at first. Route them through the fallback optimizer.
- Compare against the repo's real LM baseline optimizer, not only AdamW. In the
  Attractor LM setting, MuonAdamW is part of the baseline recipe.
- Treat aspect scaling as an ablation in LM, not a default conclusion:
  `aspect=false` primary, `aspect=true` secondary, same data/budget.

My recommendation for the next LM round:

```text
Primary candidate:
  Direct SODA/Anchor + PMuonEq + GramNS + NorMuon
  row_gamma=0.35, col_gamma=0.0, pmuon_beta=0.90,
  normuon_beta2=0.93, aspect=false

Secondary candidate:
  Same recipe, but col_gamma=0.05 and aspect=true
```

This family is the current winner, but the exact aspect/column-gamma setting is
not stable enough yet to declare one universal optimizer for language modeling.
