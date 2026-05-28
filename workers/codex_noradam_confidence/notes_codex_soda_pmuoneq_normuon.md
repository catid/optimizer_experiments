# Notes From `codex_soda_pmuoneq_normuon`

I reviewed `optim_anchormuon.py`, `optim_factory.py`, the CIFAR runner, tests, and the committed result bundle. The confidence package is valuable because it has multi-seed HPO/final runs and a stronger AdamW comparison than the quick standalone checks.

Potentially useful follow-ups:

1. Gate the per-step CPU-synchronizing diagnostics in `AnchorMuon.step()`.
   `mean_update_rms` and `mean_precond_matrix_rms` call `.detach().cpu()` inside the per-parameter step loop. That is useful for debugging, but it forces GPU synchronization every optimizer step and can materially slow the measured optimizer. My standalone path removed those syncs and only kept count/schedule stats. A `diagnostics_interval` or `enable_sync_stats=False` default would make speed comparisons cleaner.

2. The runner intentionally synchronizes CUDA around every step for timing.
   That gives accurate isolated step timing, but it depresses normal training throughput because it prevents natural overlap. I would report those numbers as "instrumented step time" and optionally add an unsynchronized throughput mode for end-to-end training comparisons.

3. Check whether classifier/head matrices should use Muon/NorMuon.
   `optim_factory._anchor_param_groups` does not pass names into `AnchorMuon`, and `_use_muon_for_param` treats most 2D tensors as Muon matrices. That likely includes classifier heads unless `no_weight_decay()` excludes them. In my focused CIFAR run I kept heads in the fallback group. It is worth an ablation or at least documenting which treatment produced the reported numbers.

4. SODA placement differs from my focused no-AMUSE file.
   In `amuse=False`, `AnchorMuon` applies the update to `z` and then applies the anchor delta after the learned update. My standalone file applies the SODA anchor pull before the learned update, matching the old local flexible path I validated against. The two are close but not identical. A tiny deterministic parity test for `amuse=False, soda=all/matrix, normuon=True` would clarify whether this difference matters.

5. The Newton-Schulz coefficients differ from the Polar Express/GramNS variant.
   `optim_anchormuon.py` uses the simple `(3.4445, -4.7750, 2.0315)` recurrence for `ns_steps`, while the other worker standalone files use the five-coefficient Polar Express GramNS path. This may be fine, but it is a hidden algorithmic difference in optimizer comparisons. A small ablation on the same HPO winner would tell whether it affects loss or speed.

6. Baseline comparisons are fair inside this harness, but not directly comparable to the other folders.
   Your AdamW winner uses cosine schedule with `lr=0.004, wd=0.001`; my standalone check used tuned AdamW `lr=0.0025, wd=0.005` in a different ViT runner. When making a combined table, include scheduler, warmup, model variant, batch size, image size, and seed count.

7. Consider adding a focused production optimizer file after the ablation work.
   `AnchorMuon` is useful as an ablation framework. For deployment or final benchmarking, a stripped fixed-recipe file like `SODA + PMuonEq + Gram + NorMuon` without `pmuon_eq`, `mimuon`, `normuon`, and `soda` conditionals would reduce surface area and make accidental config drift less likely.

8. Add one DDP parity result to the README.
   I saw `tests/ddp_smoke_anchormuon.py`, which is good. The README would be stronger if it listed whether DDP matched single-process behavior or at least that the smoke test passed on the source machine.

