# Root Optimizer Trainer Compatibility and Ablations

Date: 2026-05-29

These runs validate the root `optimizer.py` from this monorepo through the ViT-5
CIFAR-10 ablation trainer without modifying `optimizer.py`.

## Setup

- Model: `vit5_micro`
- Dataset: CIFAR-10
- Split: 45k train / 5k validation from CIFAR-10 train split, official 10k test at end
- Batch size: 512
- Epochs: 20
- Seed: 123
- GPUs: 4 visible RTX PRO 6000 Blackwell Max-Q GPUs, one trial per GPU
- Timing: `--no-sync-step-timing`, so step time is CPU-launch timing; interval throughput is the more stable speed metric
- Root optimizer hyperparameters: `matrix_lr=8e-3`, `pmuoneq_beta=0.90`, `row_gamma=0.35`, `momentum=0.95`, `normuon_beta2=0.93`, warmup 80 steps

Artifacts:

- NorMuon-axis pass: `workers/codex_noradam_confidence/results/root_normuon_ablation_20260529/e20/`
- SODA/grouping pass: `workers/codex_noradam_confidence/results/root_soda_ablation_20260529/e20/`
- Each pass includes `all_runs.csv`, per-trial `metrics.jsonl`, and plots for train loss, validation loss, validation accuracy, examples/sec, and step time.

## Trainer Changes

- Added root optimizer import from repository root while preserving existing worker-local optimizer imports.
- Added `optimizer="root"` trial support.
- Added root API adapter mapping:
  - `lr` -> `matrix_lr` and `fallback_lr`
  - `normuon_beta` -> `normuon_beta2`
  - `soda="all"` -> `soda_lambda_scale=1.0` and fallback weight decay disabled
  - `soda="none"` -> `soda_lambda_scale=0.0`
- Added `root_grouping`:
  - `anchor`: uses the historical `_anchor_param_groups` split, then allows root to run the matrix path on matrix tensors inside both groups.
  - `named`: uses root optimizer's safer name-aware grouping.
- Added `root_normuon_mode`:
  - `row`: root optimizer default row-only NorMuon.
  - `orientation`: trainer-side subclass that keeps root Polar Express / PMuonEq / SODA but switches only NorMuon's second-moment axis to rows for tall matrices and columns for wide matrices.
- Added supervisor-side CIFAR-10 pre-download to avoid parallel worker download/extract corruption.

## NorMuon Axis Ablation

This pass isolates row-only vs orientation-aware NorMuon with root optimizer and anchor grouping. AdamW and worker-local golden are included for context.

| Trial | Optimizer | NorMuon mode | SODA | Final val loss | Final val acc | Best val acc | Test acc | Avg step ms | Examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `golden_orient_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93` | worker golden | orientation | all | 0.5744 | 79.92 | 79.92 | 79.60 | 20.79 | 24.6k |
| `root_orient_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93` | root | orientation | all | 0.5744 | 79.56 | 80.14 | 79.43 | 19.27 | 26.6k |
| `root_row_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93` | root | row | all | 0.5829 | 79.58 | 80.46 | 79.11 | 19.17 | 26.7k |
| `adamw_cosine_lr0.004_wd0.001` | AdamW | n/a | n/a | 0.7597 | 72.96 | 73.04 | 72.76 | 12.39 | 41.3k |

Interpretation:

- Root row-only had the highest best validation accuracy in this pass, `80.46%`.
- Root orientation-aware had slightly better official test accuracy than root row-only, `79.43%` vs `79.11%`, and essentially the same final validation loss as worker golden.
- The orientation change itself is not a large speed cost: root row-only was `19.17 ms`, root orientation-aware was `19.27 ms`.
- Worker golden was slightly better on official test in this pass, but slower than both root variants.

## SODA and Root Grouping Ablation

This pass keeps root NorMuon row-only fixed and tests SODA on/off plus root grouping mode separately from the NorMuon-axis ablation.

| Trial | Optimizer | Grouping | SODA | Final val loss | Final val acc | Best val acc | Test acc | Avg step ms | Examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `root_row_anchor_nosoda_mlr0.008_rg0.35_pb0.9_nb0.93` | root | anchor | none | 0.5191 | 81.44 | 82.64 | 81.70 | 19.13 | 26.8k |
| `root_row_named_nosoda_mlr0.008_rg0.35_pb0.9_nb0.93` | root | named | none | 0.5785 | 80.72 | 81.14 | 80.18 | 16.84 | 30.4k |
| `golden_orient_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93` | worker golden | anchor | all | 0.5744 | 79.92 | 79.92 | 79.60 | 19.48 | 26.3k |
| `root_row_anchor_soda_mlr0.008_rg0.35_pb0.9_nb0.93` | root | anchor | all | 0.5829 | 79.58 | 80.46 | 79.11 | 18.97 | 27.0k |
| `root_row_named_soda_mlr0.008_rg0.35_pb0.9_nb0.93` | root | named | all | 0.6117 | 78.54 | 79.86 | 77.35 | 16.96 | 30.2k |
| `adamw_cosine_lr0.004_wd0.001` | AdamW | n/a | n/a | 0.7597 | 72.96 | 73.04 | 72.76 | 12.42 | 41.2k |

Interpretation:

- On this 20-epoch CIFAR-10 horizon, root anchor grouping with SODA disabled was best by a clear margin:
  - best validation accuracy: `82.64%`
  - official test accuracy: `81.70%`
- Root named/safe grouping was faster because it routes fewer matrices through the spectral path (`17` matrix tensors vs `21` for anchor grouping), but it lost accuracy.
- SODA hurt this short-horizon ViT-5 run for both grouping modes.
- Caveat: root `soda="none"` disables the SODA anchor (`soda_lambda_scale=0.0`) but root matrix-path parameters still do not receive ordinary matrix weight decay. Fallback parameters use the configured fallback weight decay. Treat this as a root no-anchor ablation, not a full AdamW-style weight-decay ablation.

## Current Recommendation

For this specific benchmark:

```text
root optimizer.py
root_grouping = "anchor"
root_normuon_mode = "row"
soda = "none"
matrix_lr = 8e-3
pmuoneq_beta = 0.90
row_gamma = 0.35
momentum = 0.95
normuon_beta2 = 0.93
warmup_steps = 80
```

This is the best result from these ablations, but it should be confirmed with
multiple seeds and a longer 50-epoch run before replacing the previous golden
recipe.

