# Root Optimizer vs AdamW, 50 Epochs

Fresh run after pulling `main` on 2026-05-29. This validates the monorepo root
`optimizer.py` through the CIFAR-10 ViT-5 harness.

## Setup

- Model: `vit5_micro`
- Dataset: CIFAR-10
- Train/validation split: 45k/5k split from train set, split seed `12345`
- Test set: official 10k CIFAR-10 test split, evaluated once at the end
- Epochs: 50
- Batch size: 512
- Seed: 123
- GPUs: 2 x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Torch: `2.13.0.dev20260506+cu130`
- Step timing: CPU launch timing via `--no-sync-step-timing`

## Runner Fix

Latest `main` had a stale benchmark adapter call:

```python
fallback_weight_decay=...
```

Current root `AnchorMuon` no longer accepts that keyword because the standalone
winner uses SODA instead of an ordinary fallback weight-decay path. The runner
adapter was updated to remove that obsolete argument. Root `optimizer.py` was
not changed.

## Results

| optimizer | final val loss | final val acc | best val loss | best val acc | test loss | test acc | train loss | avg step ms | examples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| root SODA+PMuonEq+GramNS+NorMuon | 0.4145 | 85.84% | 0.3998 | 86.30% | 0.4389 | 85.73% | 0.3447 | 16.85 | 30,380 |
| AdamW cosine | 0.6133 | 79.62% | 0.5998 | 79.64% | 0.6240 | 79.55% | 0.4207 | 11.46 | 44,680 |

Root optimizer improved final test accuracy by `+6.18` points over AdamW in
this single-seed 50-epoch check. AdamW remained faster per step, at about `1.47x`
the root optimizer's examples/second.

## Artifacts

- Full table: `all_runs.csv`
- Per-trial metrics: `*/metrics.jsonl`
- Curves: `train_loss.png`, `val_loss.png`, `val_acc.png`
- Speed charts: `step_time_ms_bar.png`, `examples_per_sec_bar.png`
