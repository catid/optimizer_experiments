# Notes From `codex-sodamuseeq-vit5`

I reviewed `equimuse_normuon.py`, the tests, and the validation note.

Useful things to keep:

- The standalone fixed-recipe packaging is clean. It is easier to reuse than an ablation-heavy optimizer class.
- The explicit `train()` / `eval()` schedule-free state API is the right choice. I would keep the README warning that validation/checkpointing should call `optimizer.eval()`.
- Batching same-shaped matrix updates is a good production detail. My local ViT-5 path saw the same pattern: batching the Gram path matters more than micro-optimizing Python loops elsewhere.

Things I would double-check:

- Optimizer states are documented as FP32, but several states appear to inherit parameter dtype. Examples: `_get_z()` clones `p.detach()` without `.float()`, fallback `exp_avg_sq` uses `torch.zeros_like(p)`, and `momentum_buffer` uses `torch.zeros_like(p)`. If the model runs BF16/FP16, those states will also be BF16/FP16. My implementation keeps `z`, momentum, PMuonEq EMAs, and Adam/RMS states in FP32 and casts only when copying back to params. Add a BF16 unit test that asserts state dtypes.
- PMuonEq row/column factors are applied as `ema.pow(-gamma)` without RMS/geometric normalization. Because GramNS is finite-step, global/input scale still changes numerical behavior even if the ideal polar direction is scale-invariant. Compare against normalized factors, e.g. factor divided by its RMS, especially for early training.
- Your SODA lambda is `scale / (t - warmup + 1)` after warmup. That is a valid warmup-reset variant, but it is not the same as the global-index SODA schedule. I would name/report it explicitly so runs are not compared against global-index SODA by accident.
- The fixed recipe includes AMUSE. In my focused CIFAR-10 runs, AMUSE looked good early but tended to overfit or degrade later compared with no-AMUSE SODA+PMuonEq. If you keep AMUSE, a longer 10k/20k step curve against no-AMUSE is worth running.

Suggested tests:

- Add a BF16/mixed-precision test that checks all long-lived optimizer states are FP32.
- Add a train/eval misuse regression: after `eval()`, the next forward should either require `train()` first or the training loop should call `train()` before the forward/backward, not just before `step()`.
- Add a DDP smoke for the exported standalone file itself if the current DDP check only covered the source harness.
