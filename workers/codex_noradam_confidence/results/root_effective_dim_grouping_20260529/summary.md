# Root Effective-Dimension Grouping Check

This run validates the root `optimizer.py` change that routes matrix updates by
effective non-singleton tensor dimensions instead of broad embedding/head/token
name exclusions.

Effective-shape examples for ViT-5 micro:

- `cls_token` with shape `[1, 1, 96]` is vector-like and uses the fallback path.
- `pos_embed` with shape `[1, 64, 96]` is matrix-like and uses the matrix path.
- `reg_token` with shape `[1, 4, 96]` is matrix-like and uses the matrix path.
- `head.weight` with shape `[10, 96]` is matrix-like and uses the matrix path.

All root no-SODA runs reported `opt_soda_weight = 0.0`, so the SODA anchor pull
was disabled for every parameter. Fallback parameters still use the configured
ordinary fallback weight decay in no-SODA mode.

## Setup

- Dataset: CIFAR-10
- Model: `vit5_micro`
- Train/validation split: 45k/5k from train set, split seed 12345
- Test set: full 10k
- Seed: 123
- Epochs: 20
- Batch size: 512
- Warmup steps: 80
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- PyTorch: 2.13.0.dev20260513+cu130

## Results

| trial | SODA | grouping | matrix/fallback | val loss | val acc | best val acc | test loss | test acc | avg step ms | examples/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `root_row_named_nosoda_mlr0.008_rg0.35_pb0.9_nb0.93` | none | named | 20 / 40 | 0.5351 | 81.10 | 81.36 | 0.5484 | 80.95 | 18.20 | 28,129 |
| `root_row_anchor_nosoda_mlr0.008_rg0.35_pb0.9_nb0.93` | none | anchor | 20 / 40 | 0.5219 | 82.00 | 82.00 | 0.5417 | 80.83 | 17.91 | 28,589 |
| `adamw_cosine_lr0.004_wd0.001` | n/a | n/a | n/a | 0.7609 | 72.54 | 72.96 | 0.7647 | 73.13 | 12.16 | 42,106 |

The corrected named grouping now matches the anchor path's matrix/fallback
counts and closes the earlier routing gap. On this single 20-epoch seed, the two
root no-SODA variants are effectively tied: anchor is slightly better on
validation, named is slightly better on test.

