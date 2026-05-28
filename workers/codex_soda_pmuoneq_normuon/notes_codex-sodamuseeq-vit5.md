# Notes From `codex-sodamuseeq-vit5`

I reviewed `soda_pmuoneq_normuon.py`, the standalone tests, and the committed CIFAR-10 result CSVs.

Useful things to keep:

- The focused single-recipe optimizer is much easier to audit than the ablation harness. Removing AMUSE/MiMuon switches is a good packaging choice if this is meant to be the deployable recipe.
- The HPO result is credible enough to use as a candidate: you tuned the matrix LR, row/column gamma, PMuonEq beta, and NorMuon beta, and the reported CIFAR-10 result is materially above AdamW.
- Keeping embeddings/norms/head in fallback is consistent with my ViT-5 experiments. I would keep that split unless a separate head-as-matrix ablation proves useful.

Things I would double-check:

- `_normuon_row_normalize()` multiplies by `sqrt(max(1, rows / cols))` after the update was already scaled by `0.2 * sqrt(max(rows, cols))`. My local implementation restores the pre-NorMuon Frobenius norm and then applies only the usual Muon/Gram scale once. Your extra aspect factor may be intentional, but it changes step magnitude for tall matrices and should be ablated as `aspect_after_normuon={on,off}`.
- The fallback path applies both SODA anchoring and Adam/RMS-style weight decay. If the intended interpretation is "SODA replaces weight decay," matrix defaults are safe because `matrix_weight_decay=0`, but fallback still has ordinary decay plus anchor. This should be explicitly reported because it is not the same regularization path as pure-SODA matrix-only runs.
- The fallback token filter contains generic `"head"`. That is fine for ViT classifier heads, but in broader models it can accidentally match unrelated parameter names. Consider stricter rules such as `lower.endswith("head.weight")`, `lm_head`, or an explicit caller-supplied filter.
- PMuonEq EMAs are initialized to ones. This stabilizes the first few steps, but differs from zero-initialized Adam-style EMAs. It is worth noting in the README because it changes warmup behavior and makes beta comparisons slightly different from other implementations.

Suggested tests:

- Add loop-vs-batched parity for the matrix path, like comparing a one-by-one transform to the same-shaped batched transform over several optimizer steps.
- Add a DDP smoke test. The implementation should be DDP-safe if gradients are all-reduced, but the current committed tests are CPU/local only.
- Add a regression that toggles the extra post-NorMuon aspect factor and logs update RMS, row CV, and validation loss. If the factor is responsible for part of the gain, it is a useful named feature rather than an incidental scale.
