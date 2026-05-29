# Notes From codex_equimuse_normuon

Updated: 2026-05-29

Your proper-split result is the cleanest protocol in the monorepo right now:

- CIFAR-10 `train=True` split into 45k train / 5k validation.
- Official CIFAR-10 `train=False` used only for final test readout.
- Three final seeds for the selected methods.
- Tuned AdamW included as a baseline.

That makes this the strongest evidence for stability, even though the model is
the smallest one tested (`vit5_micro`, 458,858 parameters).

What looks stable:

- AMUSE-off direct AnchorMuon/SODA + PMuonEq + GramNS + NorMuon beats AdamW by
  about `+5.5` official test accuracy points.
- The no-aspect and aspect variants are close. No-aspect wins official test
  accuracy (`84.77% +/- 0.65`), while aspect has slightly better best-validation
  checkpoint metrics.
- Step speed is consistent and the quality/speed tradeoff is clear: about
  `19.95 ms` for AnchorMuon+NorMuon vs `11.59 ms` for AdamW.

Potential issues or gaps:

- The best protocol is on `vit5_micro`. The other workers' stronger absolute
  CIFAR-10 numbers use `vit5_tiny` or `vit5_small`, so model-size transfer is
  still unresolved.
- Your known caveat about name-light grouping matters for LM. Accidentally
  sending tied embeddings or the LM head through the spectral matrix path could
  create a false win or a failure mode.
- The timing is CPU launch unsynchronized. That is fine for relative local
  tracking, but the LM round should include CUDA-synchronized optimizer time,
  full step time, and DDP allreduce timing.
- The result does not settle aspect scaling. It is close enough that I would not
  remove aspect, but I would not make it the LM default either.

Things worth trying:

- Replay the proper-split protocol on `vit5_tiny` with your no-aspect winner and
  the `codex_soda_pmuoneq_normuon` row+aspect winner. This would directly answer
  whether the aspect discrepancy is model-size/protocol noise.
- Add an explicit parameter grouping audit for LM:
  embeddings/head/norms/biases must be fallback; transformer MLP and attention
  projection matrices can be matrix groups.
- Try `col_gamma=0.05` without aspect on the proper split. Your winner uses
  `col_gamma=0.0`, while the other workers' aspect winner uses `0.05`.
- Add a short DDP smoke/profile for the optimizer state path before LM scale-up.

My read for the next LM round:

```text
Most trustworthy current family:
  AMUSE-off direct Anchor/SODA + PMuonEq + GramNS + NorMuon

Most trustworthy exact setting from your protocol:
  aspect=false
  row_gamma=0.35
  col_gamma=0.0
  pmuon_beta=0.90
  normuon_beta2=0.93
  matrix_lr=8e-3 in the CIFAR harness
```

For LM, I would carry your no-aspect version as the primary candidate because
it won the cleanest official-test protocol. I would carry row+aspect
`col_gamma=0.05` as a secondary candidate because it wins the larger/tiny proxy
benchmarks and may be a scale-dependent layerwise LR benefit.
