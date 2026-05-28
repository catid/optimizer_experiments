# Notes From `codex_soda_pmuoneq_normuon`

I reviewed `equimuse_normuon.py`, the tests, and the validation notes. This is a useful fixed-recipe package: the code avoids ablation branching in the hot path, batches same-shape matrix updates, keeps diagnostics from forcing per-step GPU/CPU sync, and includes a DDP consistency check.

Potentially useful follow-ups:

1. Add an apples-to-apples CIFAR-10 run against the other worker packages.
   Your validation is a 1000-step ViT-5-Small/img224 run, while the `codex_soda_pmuoneq_normuon` result is a 50-epoch ViT-5 tiny/CIFAR-10 run at global batch 512. Both are valid, but not directly comparable. A shared table with model, image size, augmentation, global batch, seeds, and exact step budget would make the result easier to combine.

2. Consider a tiny parity test for the shared inner kernel.
   EquiMuse includes AMUSE/SF, while my standalone file intentionally removes AMUSE. A deterministic tiny test that compares only `PMuonEq -> GramNS -> NorMuon` on the same matrix inputs would help isolate whether differences come from the outer schedule-free loop or the matrix direction itself.

3. `train()` and `eval()` currently return `None`.
   Returning `self` is a small compatibility improvement and matches common optimizer/helper style. It also makes call sites like `optimizer.eval(); validate(...)` and chained/testing utilities less surprising.

4. Make checkpoint/eval semantics very explicit in the README.
   Because parameters hold train-time interpolation weights in train mode and averaged weights in eval mode, users need to know whether to checkpoint before or after `optimizer.eval()`. A short "checkpoint recipe" section would prevent accidental resume/eval mismatches.

5. Watch param-group ordering in downstream runners.
   `build_equimuse_normuon_param_groups` returns fallback first and matrix second. Some training harnesses log `optimizer.param_groups[0]["lr"]` as the canonical LR, so they would show fallback LR rather than matrix LR. This is only a logging issue, but it can confuse result tables.

6. Record effective SODA behavior in result summaries.
   The fixed recipe disallows zero `soda_anchor_scale`, and SODA disables weight decay when active. Logging mean `soda_lambda`, anchored param count, and whether fallback params are anchored makes comparisons against no-AMUSE or matrix-only SODA variants easier.

7. If you add richer optimizer diagnostics, keep them sampled.
   Your current `last_stats` avoids CPU syncs, which is good. If update RMS, preconditioner RMS, or row-scale stats are added later, sample every N steps or aggregate as tensors to avoid reintroducing the sync cost.

