# HRM AnchorMuon Sudoku No-Aug Run

This was a bounded integration run of root `AnchorMuon` inside the official HRM
training stack. It is not a tuned reproduction of the HRM paper recipe.

## Setup

| Field | Value |
|---|---|
| HRM repo | `sapientinc/HRM` |
| HRM commit | `ac15626f8db096a63c775b84c9dc868776a6feda` |
| Patch | `workers/codex_noradam_confidence/hrm_anchormuon.patch` |
| Optimizer | `optimizer=anchormuon` |
| Model | Default HRM ACT config, 27,275,266 params |
| Dataset | Sudoku-Extreme, 1k train examples, no augmentation |
| Eval split | Trimmed 1k examples from Sudoku-Extreme test split |
| GPUs | 2x RTX PRO 6000 Blackwell |
| PyTorch | `2.13.0.dev20260506+cu130` |
| Attention | SDPA fallback via `HRM_ALLOW_SDPA_FALLBACK=1` |
| Compile | Disabled via `DISABLE_COMPILE=1` |
| Batch | Global batch size 384 |
| Schedule | HRM cosine helper with `lr_min_ratio=1.0`, so effectively constant LR |
| LR | `lr=1e-4`, `puzzle_emb_lr=1e-4` |
| Epochs | 1000 |
| Steps | 2600 completed optimizer steps |

## Command

```bash
OMP_NUM_THREADS=16 WANDB_MODE=offline DISABLE_COMPILE=1 HRM_ALLOW_SDPA_FALLBACK=1 \
  /home/catid/screen/.venv/bin/python -m torch.distributed.run --nproc-per-node 2 pretrain.py \
  data_path=data/sudoku-extreme-1k-noaug-test1k \
  optimizer=anchormuon \
  epochs=1000 \
  eval_interval=100 \
  checkpoint_every_eval=False \
  global_batch_size=384 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  lr_warmup_steps=0 \
  +project_name=HRM-anchor-run \
  +run_name=anchormuon-sudoku-noaug-e1000 \
  +checkpoint_path=/tmp/hrm_anchormuon_sudoku_noaug_e1000
```

Final checkpoint:

```text
/tmp/hrm_anchormuon_sudoku_noaug_e1000/step_2600
```

## Results

The run completed without NaNs or optimizer crashes. Training overfit signal was
strong on the no-augmentation train stream, but held-out generalization was poor
after this short/limited-data run.

| Metric | Value |
|---|---:|
| Final logged train token accuracy | 99.154% |
| Final logged train exact accuracy | 61.006% |
| Final logged train LM loss | 0.09511 |
| Final eval token accuracy, trimmed 1k test | 43.180% |
| Final eval exact accuracy, trimmed 1k test | 0.000% |
| Final eval LM loss, trimmed 1k test | 2.9123 |
| Final eval Q halt accuracy | 40.200% |
| Final eval Q halt loss | 1.5786 |
| Eval mean steps | 16.0 |
| End-to-end progress-bar throughput | ~12.9 steps/s including eval stalls |
| Training-step throughput after warmup | ~13.7 steps/s |
| Approx global examples/s after warmup | ~5.3k examples/s |

## Interpretation

AnchorMuon is wired correctly enough to run HRM's full model and sparse puzzle
embedding path on both GPUs. This run should not be treated as an optimizer
quality comparison:

- It used no Sudoku augmentation, while HRM's reported small-sample recipe uses
  large augmentation.
- It used only 1000 epochs instead of the README's 20000-epoch Sudoku-Extreme
  recipe.
- AdamATan2 baseline reproduction is currently blocked in this venv because
  `adam_atan2_backend` is missing.
- FlashAttention is not installed, so this used the explicit SDPA validation
  fallback.

The useful takeaway is operational: full HRM + AnchorMuon trains stably and can
overfit the tiny no-augmentation train set. The next real experiment should fix
the AdamATan2 backend, install FlashAttention if possible, then run matched
AnchorMuon/AdamATan2 sweeps on the augmented Sudoku-Extreme recipe.
