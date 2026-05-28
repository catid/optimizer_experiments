# Notes From `codex-sodamuseeq-vit5`

I reviewed `optim_anchormuon.py`, `optim_factory.py`, the tests, and the committed confidence report.

Useful things to keep:

- This is the strongest confidence package among the folders I reviewed: it has HPO, 5-seed 20-epoch confirmation, 3-seed 50-epoch confirmation, plots, and speed summaries.
- The optimizer state handling is mostly aligned with what I would do: `z`, SODA anchors, matrix momentum, PMuonEq EMAs, and fallback second moments are stored in FP32.
- The `step()` guard that raises if the optimizer is still in eval/x mode is good. It avoids the common schedule-free bug where validation leaves parameters in averaged weights and the next forward/backward happens at the wrong point.

Things I would clarify or ablate:

- The README calls the target `NorMuon+base`, but the described run is not just base Gram/Muon plus NorMuon. It includes SODA + PMuonEq + GramNS + NorMuon with AMUSE off. In my focused runner I used `NorMuon+BaseGram` to mean no SODA, no AMUSE, no PMuonEq, only base Gram/Muon plus NorMuon. These are different hypotheses. I suggest renaming yours to something like `SODA+PMuonEq+Gram+NorMuon (AMUSE off)` or adding the pure `NorMuon+BaseGram` arm.
- `optim_factory._anchor_param_groups()` sends all decay-eligible matrices to `AnchorMuon` by shape. That likely includes the ViT classifier head unless `model.no_weight_decay()` excludes it. My local ViT-5 grouping keeps classifier head, embeddings, norms, and biases in fallback. This grouping difference can materially change CIFAR accuracy, so I would add a `head_fallback=True` ablation.
- The confidence comparison is against AdamW only. Given my latest focused run, the more relevant next comparison is a three-way one: AdamW, pure `SODA+PMuonEq+Gram`, and your best `SODA+PMuonEq+Gram+NorMuon`. Otherwise NorMuon may look like the winner when the gain is mostly SODA/PMuonEq.
- The SODA correction is applied after the learned update in the no-AMUSE path via `z.add_(init - old_base, alpha=lam)`. That is a reasonable anchor correction, but it differs from implementations that apply the anchor pull before the update. If results differ from other workers, this ordering is one likely reason.

Suggested tests:

- Add a param-group test that asserts classifier head and embeddings go to the intended path for the actual ViT-5 model, not just the synthetic `TinyNet`.
- Add a no-SODA/no-PMuonEq NorMuon+BaseGram arm:
  `amuse=False`, `soda="none"`, `pmuon_eq=False`, `normuon=True`.
- Add an apples-to-apples final table against `SODA+PMuonEq+Gram` without NorMuon. That will isolate whether NorMuon itself improves the already-strong matrix stack.
