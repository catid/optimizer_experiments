# HRM WSD Optimizer Comparison

Date: 2026-05-29

## Setup

- Repo: `sapientinc/HRM` at `ac15626f8db096a63c775b84c9dc868776a6feda`, patched by `workers/codex_noradam_confidence/hrm_anchormuon.patch`
- Dataset: `data/sudoku-extreme-1k-noaug-test1k`
- Model: default HRM ACT model, 27.3M parameters
- GPUs: 2 visible CUDA GPUs, DDP via `torch.distributed.run`
- Batch: `global_batch_size=384`
- Schedule: WSD, `lr_warmup_steps=50`, `lr_min_ratio=0.1`, `wsd_decay_frac=0.2`
- Decay: `weight_decay=1.0`, `puzzle_emb_weight_decay=1.0`
- Environment flags: `DISABLE_COMPILE=1`, `HRM_ALLOW_SDPA_FALLBACK=1`, `WANDB_MODE=offline`

The local `adam_atan2` package is missing `adam_atan2_backend`, so the baseline used the exact torch `AdamATan2Reference` fallback added in the patch. This preserves the update equation for quality comparison, but it is not the fused speed baseline.

## 300-Epoch HPO

| Optimizer | LR | Token Acc | Exact Acc | LM Loss | Q Halt Acc | Q Halt Loss | Wall Sec | Iter/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnchorMuon | `1.25e-5` | 21.20% | 0.00% | 2.1727 | 100.0% | 0.7803 | 63 | 12.40 |
| AnchorMuon | `2.5e-5` | 40.84% | 0.00% | 1.6619 | 100.0% | 0.7837 | 63 | 12.40 |
| AnchorMuon | `5e-5` | 43.16% | 0.00% | 1.4845 | 100.0% | 0.7867 | 63 | 12.40 |
| AnchorMuon | `1e-4` | 43.18% | 0.00% | 1.6946 | 100.0% | 0.8183 | 63 | 12.40 |
| AnchorMuon | `2e-4` | 43.12% | 0.00% | 2.4483 | 100.0% | 0.7945 | 63 | 12.40 |
| AdamATan2Reference | `6.25e-6` | 43.09% | 0.00% | 1.4580 | 100.0% | 0.7973 | 61 | 12.80 |
| AdamATan2Reference | `1.25e-5` | 44.50% | 0.00% | 1.4088 | 100.0% | 0.7957 | 61 | 12.80 |
| AdamATan2Reference | `2.5e-5` | 45.52% | 0.00% | 1.4295 | 100.0% | 0.9224 | 61 | 12.80 |
| AdamATan2Reference | `5e-5` | 45.03% | 0.00% | 1.8416 | 100.0% | 1.1545 | 61 | 12.80 |
| AdamATan2Reference | `1e-4` | 43.79% | 0.00% | 2.6899 | 16.5% | 1.9371 | 61 | 12.80 |
| AdamATan2Reference | `2e-4` | 44.23% | 0.00% | 2.8339 | 97.4% | 1.2006 | 61 | 12.80 |

Selector:

- AnchorMuon best by LM loss: `lr=5e-5`
- AdamATan2Reference best by LM loss: `lr=1.25e-5`
- AdamATan2Reference best by token accuracy: `lr=2.5e-5`

## 1000-Epoch Final Replay

| Optimizer | Selected By | LR | Token Acc | Exact Acc | LM Loss | Q Halt Acc | Q Halt Loss | Wall Sec | Iter/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamATan2Reference | 300-epoch LM loss | `1.25e-5` | 44.67% | 0.00% | 2.0961 | 100.0% | 0.7849 | 188 | 13.85 |
| AnchorMuon | 300-epoch LM loss | `5e-5` | 42.77% | 0.00% | 2.6036 | 100.0% | 0.7652 | 196 | 13.29 |
| AdamATan2Reference | 300-epoch token accuracy | `2.5e-5` | 44.50% | 0.00% | 2.8958 | 0.0% | 4.7208 | 189 | 13.78 |

## Conclusion

For this bounded no-augmentation Sudoku comparison, tuned AdamATan2Reference + WSD is the better choice. It beat AnchorMuon on final token accuracy, LM loss, and iteration speed. AnchorMuon did not show a useful advantage on this HRM task at the tuned WSD settings.

This is not a full HRM reproduction. Exact accuracy remained 0% for all finalists, and the full augmented Sudoku/Maze recipes still need separate tuning if HRM quality reproduction is the goal.

## Notable Run Artifacts

- `hpo/`: 300-epoch LR sweep logs.
- `final1000_anchormuon_loss_pick_wsd_lr0p00005_e1000/`: AnchorMuon final run.
- `final1000_adam_atan2_ref_loss_pick_wsd_lr0p0000125_e1000/`: best final run by LM loss.
- `final1000_adam_atan2_ref_acc_pick_wsd_lr0p000025_e1000/`: accuracy-selected baseline replay; overfit/halt behavior degraded by 1000 epochs.
