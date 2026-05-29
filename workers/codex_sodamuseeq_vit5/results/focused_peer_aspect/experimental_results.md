# Focused Optimizer Confidence Comparison

Compared the focused arms: AdamW baseline, NorMuon+BaseGram, SODA+PMuonEq+Gram, and the peer-suggested SODA+PMuonEq+Gram+NorMuon row+aspect recipe.

## Best Specific Version

The best version in this comparison is:

```text
soda_pmuoneq_normuon_aspect
SODA+PMuonEq+Gram+NorMuon row+aspect
```

This is not the full AMUSE path. The winning config uses `use_amuse=False`,
`use_soda=True`, `use_pmuoneq=True`, `use_gram=True`, and `use_normuon=True`.
PMuonEq is row-only before Gram projection (`pmuon_row_gamma=0.15`,
`pmuon_col_gamma=0.0`), then NorMuon applies row normalization with
`normuon_aspect_scale=True` after the Gram/Newton-Schulz update direction is
formed.

## Model, Data, And Training Params

| Item | Value |
|---|---|
| Model | compact ViT-5 CIFAR model |
| Trainable parameters | 2,691,274 |
| Image size | 32x32 |
| Patch size | 4 |
| Embed dim / depth / heads | 192 / 6 / 3 |
| MLP ratio | 4 |
| Architecture details | RMSNorm, RoPE, q/k norm, layer scale, 4 register tokens |
| Dataset | CIFAR-10 |
| Split | 45k train / 5k validation / 10k test |
| Split seed | 12345 |
| Training augmentation | random crop with padding 4, random horizontal flip, normalize |
| Final steps | 10,000 |
| Final seeds | 0, 1, 2 |
| Batch size | 256 |
| Eval batch size | 1024 |
| Workers | 8 |
| Eval bins | 8 |
| Precision | CUDA BF16 autocast |

## Winning Optimizer Params

| Parameter | Value |
|---|---:|
| `lr` | `0.012` |
| `weight_decay` | `0.0` |
| `momentum` | `0.95` |
| `beta1` | `0.6` |
| `beta2` | `0.999` |
| `rho` | `0.8` |
| `warmup_steps` | `500` |
| `soda_warmup_steps` | `500` |
| `pmuon_beta` | `0.90` |
| `pmuon_row_gamma` | `0.15` |
| `pmuon_col_gamma` | `0.0` |
| `normuon_beta2` | `0.90` |
| `normuon_mode` | `row` |
| `normuon_aspect_scale` | `True` |

## Artifacts

- HPO table: `hpo1k_all_runs.csv`
- Top-selection table: `top3k_all_runs.csv`
- Final table: `final10k_all_runs.csv`
- Final bin table: `final10k_bins.csv`

## Final 10k Multi-Seed Summary

| rank | family | label | seeds | val loss | val acc | test acc | steps/sec |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `soda_pmuoneq_normuon_aspect` | SODA+PMuonEq+Gram+NorMuon row+aspect | 3 | 0.4079 +/- 0.0090 | 0.8715 +/- 0.0008 | 0.8709 +/- 0.0015 | 42.8398 +/- 0.0594 |
| 2 | `soda_pmuoneq` | SODA+PMuonEq+Gram | 3 | 0.4210 +/- 0.0097 | 0.8719 +/- 0.0024 | 0.8690 +/- 0.0035 | 43.0520 +/- 0.2749 |
| 3 | `normuon_base` | NorMuon+BaseGram | 3 | 0.5069 +/- 0.0100 | 0.8621 +/- 0.0049 | 0.8605 +/- 0.0036 | 47.4430 +/- 0.3441 |
| 4 | `adamw` | AdamW | 3 | 0.6030 +/- 0.0167 | 0.8147 +/- 0.0075 | 0.8127 +/- 0.0074 | 58.4073 +/- 0.2447 |

## Long 20k Seed-0 Check

| rank | family | label | val loss | val acc | test acc | steps/sec |
|---:|---|---|---:|---:|---:|---:|

## Selected Configs

| phase | family | run | lr | wd | pmuon row/col gamma | normuon beta2 | aspect | seed | best val loss |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| top3k | `adamw` | `top3k_adamw_rank2` | 0.003 | 0.02 | 0.2/0.2 | 0.95 | False | 0 | 0.6955 |
| top3k | `adamw` | `top3k_adamw_rank1` | 0.003 | 0.01 | 0.2/0.2 | 0.95 | False | 0 | 0.7000 |
| top3k | `normuon_base` | `top3k_normuon_base_rank2` | 0.006 | 0 | 0/0 | 0.98 | False | 0 | 0.5208 |
| top3k | `normuon_base` | `top3k_normuon_base_rank1` | 0.006 | 0.02 | 0/0 | 0.9 | False | 0 | 0.5330 |
| top3k | `soda_pmuoneq` | `top3k_soda_pmuoneq_rank1` | 0.014 | 0 | 0.05/0.05 | 0.95 | False | 0 | 0.5386 |
| top3k | `soda_pmuoneq` | `top3k_soda_pmuoneq_rank2` | 0.016 | 0 | 0.025/0.025 | 0.95 | False | 0 | 0.5752 |
| top3k | `soda_pmuoneq_normuon_aspect` | `top3k_soda_pmuoneq_normuon_aspect_rank2` | 0.012 | 0 | 0.15/0 | 0.9 | True | 0 | 0.5155 |
| top3k | `soda_pmuoneq_normuon_aspect` | `top3k_soda_pmuoneq_normuon_aspect_rank1` | 0.014 | 0 | 0.35/0 | 0.95 | True | 0 | 0.5555 |
| final | `adamw` | `final10k_adamw_rank1` | 0.003 | 0.02 | 0.2/0.2 | 0.95 | False | 0 | 0.5815 |
| final | `adamw` | `final10k_adamw_rank1` | 0.003 | 0.02 | 0.2/0.2 | 0.95 | False | 1 | 0.6221 |
| final | `adamw` | `final10k_adamw_rank1` | 0.003 | 0.02 | 0.2/0.2 | 0.95 | False | 2 | 0.6055 |
| final | `normuon_base` | `final10k_normuon_base_rank1` | 0.006 | 0 | 0/0 | 0.98 | False | 0 | 0.5123 |
| final | `normuon_base` | `final10k_normuon_base_rank1` | 0.006 | 0 | 0/0 | 0.98 | False | 1 | 0.5155 |
| final | `normuon_base` | `final10k_normuon_base_rank1` | 0.006 | 0 | 0/0 | 0.98 | False | 2 | 0.4928 |
| final | `soda_pmuoneq` | `final10k_soda_pmuoneq_rank1` | 0.014 | 0 | 0.05/0.05 | 0.95 | False | 0 | 0.4129 |
| final | `soda_pmuoneq` | `final10k_soda_pmuoneq_rank1` | 0.014 | 0 | 0.05/0.05 | 0.95 | False | 1 | 0.4346 |
| final | `soda_pmuoneq` | `final10k_soda_pmuoneq_rank1` | 0.014 | 0 | 0.05/0.05 | 0.95 | False | 2 | 0.4155 |
| final | `soda_pmuoneq_normuon_aspect` | `final10k_soda_pmuoneq_normuon_aspect_rank1` | 0.012 | 0 | 0.15/0 | 0.9 | True | 0 | 0.4203 |
| final | `soda_pmuoneq_normuon_aspect` | `final10k_soda_pmuoneq_normuon_aspect_rank1` | 0.012 | 0 | 0.15/0 | 0.9 | True | 1 | 0.3995 |
| final | `soda_pmuoneq_normuon_aspect` | `final10k_soda_pmuoneq_normuon_aspect_rank1` | 0.012 | 0 | 0.15/0 | 0.9 | True | 2 | 0.4040 |

## Final Plots

- Validation loss: `plots/val_loss.png`
- Validation accuracy: `plots/val_acc.png`
- Train interval loss: `plots/train_interval_loss.png`
- Step speed bar chart: `plots/step_speed_bar.png`
