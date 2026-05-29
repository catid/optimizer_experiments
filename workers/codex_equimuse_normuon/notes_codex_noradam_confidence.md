# Notes From `codex_noradam_confidence`

Updated: 2026-05-29

I pulled the latest monorepo and read the result summaries from all worker
folders. My read is that your results are directionally consistent with the
other workers:

- The winning family is the direct no-AMUSE/no-SF path:
  `SODA + PMuonEq + GramNS + NorMuon`.
- The schedule-free EquiMuse path is not competitive in the current ViT/CIFAR
  comparisons. It is much better than a broken run would be, but it is far
  behind the direct optimizer in your 1000-step `vit5_small` proxy.
- AdamW remains the throughput winner, but all credible direct NorMuon variants
  beat it on loss/accuracy in these CIFAR runs.

Your strongest row is:

```text
Direct SODA-PMuonEq-NorMuon + aspect
vit5_small, CIFAR-10 img224, global batch 512, 1000 steps, seed 67890
final acc@1 76.69, final val loss 0.7269
```

This is useful evidence, but I would not treat the aspect multiplier as settled
from this folder alone. My proper-split `vit5_micro` replay had the no-aspect
variant slightly ahead on official test accuracy, while your `vit5_small` proxy
and the standalone worker both favor aspect. That looks workload/shape
dependent rather than a universal rule.

Suggested next checks before LM:

1. Run a proper 45k/5k/train=False split with 3 seeds for your exact
   `vit5_small` winner. Your current best uses CIFAR-10 `train=False` as the
   validation readout and one seed.
2. Retune AdamW for the 1000-step proxy if you keep using it. `lr=5e-4` may be
   a conservative baseline relative to the stronger AdamW settings seen in the
   other folder.
3. Keep the direct optimizer as the main line. Treat EquiMuse/SF as a separate
   research branch only if you want to debug why the outer averaging hurts.
4. For language modeling, start with direct `SODA + PMuonEq + GramNS + NorMuon`
   and run aspect/no-aspect as a first ablation. I would not enable aspect by
   default until it wins on one LM smoke grid.

Overall: your result supports the same conclusion as the other folders. We have
one winning optimizer family, but not yet one universal aspect/no-aspect switch.
