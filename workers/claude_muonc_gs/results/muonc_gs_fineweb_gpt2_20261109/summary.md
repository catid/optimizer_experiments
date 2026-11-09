# MuonC x MuonGS on the GPT-2-tokenized 50M LM proxy: generated tables

389 finished trials (0 diverged, 0 unfinished). Metric: validation loss over 256 sequential batches (`final_full_val_loss`). Configurations were selected on seed 123; the fresh-seed column is the unbiased estimate.

Trials per stage: `s0_parity` 11, `s1_sp` 75, `s1_std` 75, `s2_diag` 33, `s2_hnet` 9, `s2_refine` 40, `s3_hnet` 13, `s3_long` 20, `s3_seeds` 49, `s4_radial` 14, `s5_headfree` 18, `s6_accumulator` 15, `s7_long_refine` 17

## Single-pass protocol, 8K steps: one row per approach at its best (lr, wd)

| Configuration | LR | wd | Seed 123 | Fresh seeds 124+ | Final norm ratio hidden / emb |
|---|---:|---:|---:|---:|---:|
| Unprojected, corrected decay | 0.001 | 0.4 | 4.3305 | 4.3315 ± 0.0030 (n=3) | 1.70 / 1.86 |
| Unprojected, standard decay (MuonW) | 0.001 | 0.2 | 4.3396 | 4.3392 ± 0.0014 (n=3) | 1.81 / 1.95 |
| Unprojected, no decay (Muon) | 0.001 | 0 | 4.3675 | 4.3665 ± 0.0013 (n=3) | 2.96 / 3.13 |
| Half projection, corrected decay | 0.001 | 0.2 | 4.3401 | 4.3412 ± 0.0034 (n=3) | 1.57 / 1.73 |
| Half projection, no decay | 0.001 | 0 | 4.3412 | 4.3419 ± 0.0034 (n=3) | 2.21 / 2.38 |
| Projected, no decay (MuonGS) | 0.0007 | 0 | 4.3848 | 4.3894 ± 0.0036 (n=3) | 1.07 / 1.07 |
| Projected, fixed norm (MuonH-style) | 0.0007 | 0 | 4.3935 | 4.3984 ± 0.0037 (n=3) | 1.00 / 1.00 |
| Projected, corrected decay wd 0.0125 | 0.0007 | 0.0125 | 4.3884 |  | 1.04 / 1.04 |
| Projected, corrected decay wd 0.05 | 0.0005 | 0.05 | 4.4117 | 4.4119 ± 0.0026 (n=3) | 0.96 / 0.96 |
| Projected, standard decay wd 0.05 | 0.0007 | 0.05 | 4.4083 |  | 0.92 / 0.93 |

## Single-pass protocol: what closes the projection's gap

| Configuration | LR | wd | Seed 123 | Fresh seeds 124+ | Final norm ratio hidden / emb |
|---|---:|---:|---:|---:|---:|
| Projected, default init (reference) | 0.0007 | 0 | 4.3848 | 4.3894 ± 0.0036 (n=3) | 1.07 / 1.07 |
| + embeddings exempt from projection | 0.001 | 0 | 4.3781 | 4.3823 ± 0.0083 (n=3) | 1.13 / 2.99 |
| + replay unprojected run's effective LR (norm pinned) | 0.001 | 0 | 4.3761 |  | 1.00 / 1.00 |
| + adaptive radial step, c=2 | 0.001 | 0 | 4.3453 |  | 1.36 / 2.19 |
| + adaptive radial step, c=1 | 0.001 | 0 | 4.3476 |  | 1.25 / 2.10 |
| + start at 2x scale | 0.001 | 0 | 4.3460 | 4.3502 ± 0.0037 (n=3) | 1.04 / 1.04 |
| + start at 3x scale | 0.0015 | 0 | 4.3460 |  | 1.04 / 1.04 |
| + 2x scale, corrected decay holding it (wd 0.0125) | 0.001 | 0.0125 | 4.3464 | 4.3488 ± 0.0029 (n=3) | 1.00 / 1.00 |
| + 2x scale and replayed effective LR (norm pinned) | 0.002 | 0 | 4.3320 | 4.3353 ± 0.0008 (n=3) | 1.00 / 1.00 |
| Half projection, corrected decay | 0.001 | 0.2 | 4.3401 | 4.3412 ± 0.0034 (n=3) | 1.57 / 1.73 |
| Unprojected, standard decay (target) | 0.001 | 0.2 | 4.3396 | 4.3392 ± 0.0014 (n=3) | 1.81 / 1.95 |

## Repo-standard protocol (8M-token cache, about 4 passes), 8K steps

| Configuration | LR | wd | Seed 123 | Fresh seeds 124+ | Final norm ratio hidden / emb |
|---|---:|---:|---:|---:|---:|
| Unprojected, standard decay wd 0.4 | 0.001 | 0.4 | 4.7660 | 4.7682 ± 0.0054 (n=3) | 1.58 / 1.85 |
| Unprojected, corrected decay wd 0.4 | 0.001 | 0.4 | 4.7704 |  | 1.75 / 2.04 |
| Projected, no decay (MuonGS) | 0.0007 | 0 | 4.7671 | 4.7667 ± 0.0075 (n=3) | 1.14 / 1.15 |
| Projected, corrected decay wd 0.0125 | 0.0007 | 0.0125 | 4.7717 | 4.7703 ± 0.0043 (n=3) | 1.08 / 1.09 |
| Projected, fixed norm (MuonH-style) | 0.0005 | 0 | 4.7735 | 4.7678 ± 0.0091 (n=3) | 1.00 / 1.00 |
| Projected, embeddings exempt | 0.0005 | 0 | 4.8835 |  | 1.08 / 2.59 |
| Projected + adaptive radial step c=1 | 0.0007 | 0 | 4.9208 |  | 1.36 / 2.45 |
| Unprojected, no decay (Muon) | 0.002 | 0 | 4.9572 |  | 8.51 / 9.22 |
| This port as MuonW at the README settings (wd 0.05) | 0.0018 | 0.05 | 4.8914 | 4.8823 ± 0.0053 (n=4) | 5.64 / 6.18 |
| Repo PlainMuon at the README settings (wd 0.05) | 0.0018 | 0.05 | 4.8976 | 4.8875 ± 0.0104 (n=4) | 5.63 / 6.17 |
| Repo AdamW at the README settings | 0.0005 | 0.05 | 5.0961 |  | 1.90 / 6.40 |

## H-Net optimizer stack on the single-pass protocol (LR 1e-3)

| Configuration | LR | wd | Seed 123 | Fresh seeds 124+ | Final norm ratio hidden / emb |
|---|---:|---:|---:|---:|---:|
| Unprojected, standard decay wd 0.05 (previous path) | 0.001 | 0.05 | 4.5234 | 4.5200 ± 0.0025 (n=2) | 2.54 / 4.97 |
| Unprojected, corrected decay wd 0.05 | 0.001 | 0.05 | 4.5240 |  | 2.67 / 5.20 |
| Projected, corrected decay wd 0.05 (MuonC as configured) | 0.001 | 0.05 | 4.7398 | 4.7419 ± 0.0018 (n=2) | 0.98 / 0.98 |
| Projected, standard decay wd 0.05 | 0.001 | 0.05 | 4.7577 |  | 0.93 / 0.92 |
| Projected, no decay | 0.001 | 0 | 4.6923 |  | 1.13 / 1.13 |
| Projected + C, embeddings exempt | 0.001 | 0.05 | 4.5175 | 4.5232 (n=1) | 0.98 / 4.50 |
| Half projection, corrected decay wd 0.05 | 0.001 | 0.05 | 4.5109 | 4.5094 (n=1) | 2.00 / 3.66 |
| Projected + C wd 0.0125 at 2x scale | 0.001 | 0.0125 | 4.5947 |  | 0.99 / 0.99 |

## Projection on hidden matrices only (embedding/output matrix exempt), single-pass, 8K steps

| Configuration | LR | wd | Seed 123 | Fresh seeds 124+ | Final norm ratio hidden / emb |
|---|---:|---:|---:|---:|---:|
| Plain base: unprojected, corrected decay (reference) | 0.001 | 0.4 | 4.3305 | 4.3315 ± 0.0030 (n=3) | 1.70 / 1.86 |
| Plain base: everything projected, no decay (reference) | 0.0007 | 0 | 4.3848 | 4.3894 ± 0.0036 (n=3) | 1.07 / 1.07 |
| Plain base: hidden projected, no decay | 0.001 | 0 | 4.3781 | 4.3823 ± 0.0083 (n=3) | 1.13 / 2.99 |
| Plain base: hidden projected, corrected decay wd 0.0125 | 0.001 | 0.0125 | 4.3822 |  | 1.09 / 2.92 |
| Plain base: hidden projected, corrected decay wd 0.05 | 0.0007 | 0.05 | 4.3786 |  | 0.96 / 2.14 |
| Plain base: hidden projected, corrected decay wd 0.05 | 0.001 | 0.05 | 4.3919 |  | 0.98 / 2.73 |
| Plain base: hidden projected, standard decay wd 0.05 | 0.001 | 0.05 | 4.3950 |  | 0.93 / 2.60 |
| Plain base: hidden projected, corrected decay wd 0.2 | 0.001 | 0.2 | 4.4497 |  | 0.65 / 2.16 |
| Plain base: hidden projected, fixed norm | 0.001 | 0 | 4.3930 |  | 1.00 / 2.96 |
| Plain base: hidden projected + C wd 0.05 + radial step c=1 | 0.001 | 0.05 | 4.3672 |  | 1.08 / 2.72 |
| H-Net stack: unprojected, standard decay wd 0.05 (reference) | 0.001 | 0.05 | 4.5234 | 4.5200 ± 0.0025 (n=2) | 2.54 / 4.97 |
| H-Net stack: everything projected, corrected decay wd 0.05 (reference) | 0.001 | 0.05 | 4.7398 | 4.7419 ± 0.0018 (n=2) | 0.98 / 0.98 |
| H-Net stack: hidden projected, no decay | 0.001 | 0 | 4.5055 |  | 1.13 / 4.90 |
| H-Net stack: hidden projected, corrected decay wd 0.0125 | 0.001 | 0.0125 | 4.5099 |  | 1.09 / 4.82 |
| H-Net stack: hidden projected, corrected decay wd 0.05 | 0.0007 | 0.05 | 4.5114 |  | 0.96 / 4.01 |
| H-Net stack: hidden projected, corrected decay wd 0.05 | 0.001 | 0.05 | 4.5175 | 4.5232 (n=1) | 0.98 / 4.50 |
| H-Net stack: hidden projected, standard decay wd 0.05 | 0.001 | 0.05 | 4.5255 |  | 0.93 / 4.35 |
| H-Net stack: hidden projected, corrected decay wd 0.2 | 0.001 | 0.2 | 4.5818 |  | 0.66 / 3.69 |
| H-Net stack: hidden projected, fixed norm | 0.001 | 0 | 4.5160 |  | 1.00 / 4.79 |
| H-Net stack: half projection (all matrices), corrected decay wd 0.05 | 0.001 | 0.05 | 4.5109 | 4.5094 (n=1) | 2.00 / 3.66 |

## LR schedule and the free-norm accumulator, single-pass, 8K steps

| Configuration | LR | wd | Seed 123 | Fresh seeds 124+ | Final norm ratio hidden / emb |
|---|---:|---:|---:|---:|---:|
| Unprojected MuonW wd 0.2: cosine to 10% | 0.001 | 0.2 | 4.3396 | 4.3392 ± 0.0014 (n=3) | 1.81 / 1.95 |
| Unprojected MuonW wd 0.2: constant, last 20% linear to 10% | 0.001 | 0.2 | 4.3979 |  | 2.26 / 2.42 |
| Unprojected MuonW wd 0.2: constant LR | 0.001 | 0.2 | 4.7487 |  | 2.40 / 2.57 |
| Projected + C wd 0.0125: cosine to 10% | 0.0007 | 0.0125 | 4.3884 |  | 1.04 / 1.04 |
| Projected + C wd 0.0125: constant, last 20% linear to 10% | 0.0007 | 0.0125 | 4.5481 |  | 1.08 / 1.09 |
| Projected + C wd 0.0125: constant LR | 0.0007 | 0.0125 | 5.0582 |  | 1.09 / 1.10 |
| Free-norm accumulator (projected, no decay, constant LR) | 0.001 | 0 | 5.2190 |  | 1.30 / 1.32 |
| Free-norm accumulator (projected, no decay, constant LR) | 0.002 | 0 | 5.5470 |  | 1.91 / 1.98 |
| Free-norm accumulator (projected, no decay, constant LR) | 0.004 | 0 | 5.6502 |  | 3.28 / 3.53 |
| Free-norm accumulator (projected, no decay, constant LR) | 0.008 | 0 | 5.7867 |  | 6.08 / 6.73 |
| Inverse-sqrt LR on a pinned norm | 0.002 | 0 | 5.4431 |  | 1.00 / 1.00 |
| Inverse-sqrt LR on a pinned norm | 0.004 | 0 | 5.5127 |  | 1.00 / 1.00 |
| Inverse-sqrt LR on a pinned norm | 0.008 | 0 | 5.5486 |  | 1.00 / 1.00 |
| Free-norm accumulator + terminal decay | 0.002 | 0 | 4.8038 |  | 1.82 / 1.88 |
| Free-norm accumulator + terminal decay | 0.004 | 0 | 4.8536 |  | 3.08 / 3.32 |
| Free-norm accumulator + terminal decay | 0.008 | 0 | 4.9424 |  | 5.67 / 6.29 |
| Inverse-sqrt LR on a pinned norm + terminal decay | 0.004 | 0 | 4.7957 |  | 1.00 / 1.00 |

## Long horizon: 32K steps, single-pass (every configuration run, best first)

| Cell | LR | wd | Seed 123 | Fresh seeds 124+ | Mean of all seeds | Final norm ratio hidden / emb |
|---|---:|---:|---:|---:|---:|---:|
| GS0.5+0 | 0.0007 | 0 | 3.9134 |  | 3.9134 | 3.01 / 3.18 |
| GS0.5+0 | 0.0005 | 0 | 3.9157 |  | 3.9157 | 2.27 / 2.42 |
| GS0.5+0 | 0.001 | 0 | 3.9125 | 3.9191 (n=1) | 3.9158 | 4.15 / 4.36 |
| GS+0+fixednorm [init_scale=2] | 0.0007 | 0 | 3.9199 |  | 3.9199 | 1.00 / 1.00 |
| GS+0 [init_scale=2] | 0.0007 | 0 | 3.9189 | 3.9239 (n=1) | 3.9214 | 1.07 / 1.07 |
| GS+C [init_scale=2] | 0.0007 | 0.0125 | 3.9228 |  | 3.9228 | 0.96 / 0.97 |
| noGS+C | 0.0005 | 0.4 | 3.9255 |  | 3.9255 | 1.28 / 1.43 |
| GS0.5+C | 0.0005 | 0.2 | 3.9270 |  | 3.9270 | 1.27 / 1.40 |
| GS+0 [init_scale=2] | 0.001 | 0 | 3.9300 |  | 3.9300 | 1.14 / 1.14 |
| noGS+0 | 0.0007 | 0 | 3.9304 |  | 3.9304 | 4.10 / 4.24 |
| noGS+W | 0.0005 | 0.2 | 3.9323 |  | 3.9323 | 1.28 / 1.42 |
| noGS+0 | 0.0005 | 0 | 3.9350 |  | 3.9350 | 3.02 / 3.17 |
| noGS+C | 0.0007 | 0.4 | 3.9349 | 3.9380 (n=1) | 3.9364 | 1.49 / 1.65 |
| noGS+0 | 0.001 | 0 | 3.9322 | 3.9409 (n=1) | 3.9366 | 5.76 / 5.89 |
| GS+0 [init_scale=2] | 0.0005 | 0 | 3.9366 |  | 3.9366 | 1.04 / 1.04 |
| GS0.5+C | 0.0007 | 0.2 | 3.9373 |  | 3.9373 | 1.49 / 1.64 |
| noGS+W | 0.0007 | 0.2 | 3.9391 | 3.9424 (n=1) | 3.9408 | 1.40 / 1.55 |
| GS+0 | 0.00035 | 0 | 3.9423 |  | 3.9423 | 1.07 / 1.07 |
| GS+C [init_scale=2] | 0.001 | 0.0125 | 3.9446 |  | 3.9446 | 0.99 / 0.99 |
| GS+0+fixednorm [init_scale=2] | 0.001 | 0 | 3.9453 |  | 3.9453 | 1.00 / 1.00 |
| GS+W [init_scale=2] | 0.001 | 0.0125 | 3.9503 |  | 3.9503 | 0.94 / 0.94 |
| noGS+C | 0.001 | 0.4 | 3.9537 |  | 3.9537 | 1.75 / 1.95 |
| GS+0 | 0.0005 | 0 | 3.9542 | 3.9559 (n=1) | 3.9551 | 1.14 / 1.14 |
| GS0.5+C | 0.001 | 0.2 | 3.9559 |  | 3.9559 | 1.77 / 1.95 |
| GS+0 [init_scale=2] | 0.0014 | 0 | 3.9600 |  | 3.9600 | 1.26 / 1.26 |
| noGS+W | 0.001 | 0.2 | 3.9615 |  | 3.9615 | 1.52 / 1.68 |
| GS+0 | 0.0007 | 0 | 3.9898 |  | 3.9898 | 1.26 / 1.27 |
| GS+C | 0.0005 | 0.05 | 3.9984 |  | 3.9984 | 0.86 / 0.87 |
| noGS+W | 0.0014 | 0.2 | 3.9992 |  | 3.9992 | 1.62 / 1.80 |
| GS+C [init_scale=2] | 0.002 | 0.025 | 4.0930 |  | 4.0930 | 0.96 / 0.98 |
| GS+W [init_scale=2] | 0.002 | 0.025 | 4.1365 |  | 4.1365 | 0.78 / 0.80 |

## Every configuration, by learning rate (mean over all seeds run)

### sp-cosine-32k: final_full_val_loss by lr (rows: cell, wd; mean over seeds)

| cell | wd | 0.00035 | 0.0005 | 0.0007 | 0.001 | 0.0014 | 0.002 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GS+0 | 0 | 3.9423 | 3.9551±0.001 (n2) | 3.9898 |  |  |  |
| GS+0 [init_scale=2] | 0 |  | 3.9366 | 3.9214±0.004 (n2) | 3.9300 | 3.9600 |  |
| GS+0+fixednorm [init_scale=2] | 0 |  |  | 3.9199 | 3.9453 |  |  |
| GS+C | 0.05 |  | 3.9984 |  |  |  |  |
| GS+C [init_scale=2] | 0.0125 |  |  | 3.9228 | 3.9446 |  |  |
| GS+C [init_scale=2] | 0.025 |  |  |  |  |  | 4.0930 |
| GS+W [init_scale=2] | 0.0125 |  |  |  | 3.9503 |  |  |
| GS+W [init_scale=2] | 0.025 |  |  |  |  |  | 4.1365 |
| GS0.5+0 | 0 |  | 3.9157 | 3.9134 | 3.9158±0.005 (n2) |  |  |
| GS0.5+C | 0.2 |  | 3.9270 | 3.9373 | 3.9559 |  |  |
| noGS+0 | 0 |  | 3.9350 | 3.9304 | 3.9366±0.006 (n2) |  |  |
| noGS+C | 0.4 |  | 3.9255 | 3.9364±0.002 (n2) | 3.9537 |  |  |
| noGS+W | 0.2 |  | 3.9323 | 3.9408±0.002 (n2) | 3.9615 | 3.9992 |  |

### sp-cosine-8k: final_full_val_loss by lr (rows: cell, wd; mean over seeds)

| cell | wd | 0.00025 | 0.00035 | 0.0005 | 0.0007 | 0.00075 | 0.001 | 0.0014 | 0.0015 | 0.002 | 0.003 | 0.004 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GS(outward)+0 | 0 |  |  | 4.4032 |  |  | 4.4117 |  |  |  |  |  |
| GS(outward)+W | 0.2 |  |  |  |  |  | 4.6402 |  |  |  |  |  |
| GS+0 | 0 | 4.6435 | 4.5012 | 4.4063 | 4.3882±0.004 (n4) |  | 4.4126 |  |  | 4.5245 |  | 4.5674 |
| GS+0 [emb=fallback] | 0 |  |  | 4.7471 |  |  | 4.6983 |  |  |  |  |  |
| GS+0 [emb=matrix_nogs, init_scale=2] | 0 |  |  |  |  |  |  |  |  | 4.3724 |  |  |
| GS+0 [emb=matrix_nogs] | 0 |  |  | 4.4077 |  |  | 4.3812±0.007 (n4) | 4.4086 |  |  |  |  |
| GS+0 [init_scale=1.5] | 0 |  |  |  |  | 4.3727 |  |  | 4.3917 |  |  |  |
| GS+0 [init_scale=2] | 0 |  |  |  |  |  | 4.3492±0.004 (n4) |  |  | 4.3637 |  |  |
| GS+0 [init_scale=3] | 0 |  |  |  |  |  |  |  | 4.3460 |  | 4.3643 |  |
| GS+0 [lr_profile] | 0 |  |  |  |  |  | 4.3736 |  |  |  |  |  |
| GS+0 [pmuoneq+normuon+soda, emb=matrix_nogs, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0 |  |  |  |  |  | 4.5055 |  |  |  |  |  |
| GS+0 [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0 |  |  |  |  |  | 4.6923 |  |  |  |  |  |
| GS+0 [radial_lr=0.25] | 0 |  |  |  | 4.3501 |  | 4.3633 |  |  |  |  |  |
| GS+0 [radial_lr=0.5] | 0 |  |  |  | 4.3465 |  | 4.3615 |  |  |  |  |  |
| GS+0 [radial_lr=1] | 0 |  |  |  | 4.3525 |  | 4.3476 |  |  |  |  |  |
| GS+0 [radial_lr=2] | 0 |  |  |  | 4.3559 |  | 4.3453 |  |  |  |  |  |
| GS+0+fixednorm | 0 | 4.6420 | 4.5005 | 4.4089 | 4.3972±0.004 (n4) |  | 4.4429 |  |  | 4.6362 |  | 4.9320 |
| GS+0+fixednorm [emb=matrix_nogs] | 0 |  |  |  |  |  | 4.3930 |  |  |  |  |  |
| GS+0+fixednorm [init_scale=2, lr_profile] | 0 |  |  |  |  |  |  |  |  | 4.3344±0.002 (n4) |  |  |
| GS+0+fixednorm [init_scale=2] | 0 |  |  |  |  |  | 4.3462 |  |  | 4.3823 |  |  |
| GS+0+fixednorm [lr_profile] | 0 |  |  |  |  |  | 4.3761 |  |  |  |  |  |
| GS+0+fixednorm [pmuoneq+normuon+soda, emb=matrix_nogs, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0 |  |  |  |  |  | 4.5160 |  |  |  |  |  |
| GS+C | 0.0125 | 4.6421 | 4.5016 | 4.4080 | 4.3884 |  | 4.4194 |  |  | 4.5356 |  | 4.5958 |
| GS+C | 0.05 | 4.6407 | 4.5000 | 4.4118±0.002 (n4) | 4.4017 |  | 4.4447 |  |  | 4.5896 |  | 4.6689 |
| GS+C | 0.2 | 4.6351 |  | 4.4428 |  |  | 4.5651 |  |  | 4.7982 |  | 5.0151 |
| GS+C [emb=matrix_nogs, radial_lr=1] | 0.05 |  |  |  |  |  | 4.3672 |  |  |  |  |  |
| GS+C [emb=matrix_nogs] | 0.0125 |  |  |  |  |  | 4.3822 |  |  |  |  |  |
| GS+C [emb=matrix_nogs] | 0.05 |  |  | 4.4053 | 4.3786 |  | 4.3919 | 4.4302 |  |  |  |  |
| GS+C [emb=matrix_nogs] | 0.2 |  |  |  |  |  | 4.4497 |  |  |  |  |  |
| GS+C [init_scale=2] | 0.0125 |  |  |  |  |  | 4.3482±0.003 (n4) |  |  |  |  |  |
| GS+C [pmuoneq+normuon+soda, emb=matrix_nogs, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.0125 |  |  |  |  |  | 4.5099 |  |  |  |  |  |
| GS+C [pmuoneq+normuon+soda, emb=matrix_nogs, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.05 |  |  |  | 4.5114 |  | 4.5203±0.004 (n2) | 4.5553 |  | 4.6018 |  |  |
| GS+C [pmuoneq+normuon+soda, emb=matrix_nogs, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.2 |  |  |  |  |  | 4.5818 |  |  |  |  |  |
| GS+C [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16, init_scale=2] | 0.0125 |  |  |  |  |  | 4.5947 |  |  | 4.5863 |  |  |
| GS+C [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.05 |  |  | 4.8279 |  |  | 4.7412±0.002 (n3) |  |  | 4.8024 |  |  |
| GS+C [pmuoneq+normuon+soda, fallback_mode=atan2, ns_dtype=bf16] | 0.05 |  |  |  |  |  | 4.6755 |  |  |  |  |  |
| GS+C [radial_lr=1] | 0.05 |  |  |  |  |  | 4.3596 |  |  |  |  |  |
| GS+C [radial_lr=1] | 0.2 |  |  |  |  |  | 4.3975 |  |  |  |  |  |
| GS+W | 0.0125 | 4.6436 |  | 4.4082 |  |  | 4.4226 |  |  | 4.5440 |  | 4.6060 |
| GS+W | 0.05 | 4.6411 | 4.5034 | 4.4155 | 4.4083 |  | 4.4557 |  |  | 4.6069 |  | 4.7338 |
| GS+W | 0.2 | 4.6475 |  | 4.4739 |  |  | 4.6408 |  |  | 4.9777 |  | 5.3660 |
| GS+W [emb=matrix_nogs] | 0.05 |  |  |  |  |  | 4.3950 |  |  |  |  |  |
| GS+W [pmuoneq+normuon+soda, emb=matrix_nogs, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.05 |  |  |  |  |  | 4.5255 |  |  |  |  |  |
| GS+W [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.05 |  |  |  |  |  | 4.7577 |  |  |  |  |  |
| GS0.5+0 | 0 |  |  | 4.4014 |  |  | 4.3417±0.003 (n4) |  |  |  |  |  |
| GS0.5+C | 0.2 |  |  |  |  |  | 4.3410±0.003 (n4) |  |  |  |  |  |
| GS0.5+C [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.05 |  |  |  |  |  | 4.5101±0.001 (n2) |  |  |  |  |  |
| GS0.5+W | 0.2 |  |  |  |  |  | 4.3581 |  |  |  |  |  |
| GS0.9+0 | 0 |  |  | 4.3989 |  |  | 4.3677 |  |  |  |  |  |
| noGS+0 | 0 | 4.6639 |  | 4.4214 |  |  | 4.3668±0.001 (n4) |  |  | 4.3891 |  | 4.4536 |
| noGS+C | 0.0125 | 4.6635 |  | 4.4192 |  |  | 4.3650 |  |  | 4.3876 |  | 4.4370 |
| noGS+C | 0.05 | 4.6588 |  | 4.4137 |  |  | 4.3565 |  |  | 4.3760 |  | 4.4031 |
| noGS+C | 0.2 | 4.6414 |  | 4.3965 |  |  | 4.3381 | 4.3424 |  | 4.3526 |  | 4.3843 |
| noGS+C | 0.4 |  |  |  |  |  | 4.3312±0.003 (n4) |  |  | 4.3594 |  |  |
| noGS+C | 0.8 |  |  |  |  |  | 4.3563 |  |  | 4.4222 |  |  |
| noGS+C [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.05 |  |  |  |  |  | 4.5240 |  |  |  |  |  |
| noGS+W | 0.0125 | 4.6623 |  | 4.4196 |  |  | 4.3643 |  |  | 4.3857 |  | 4.4270 |
| noGS+W | 0.05 | 4.6583 |  | 4.4138 |  |  | 4.3555 |  |  | 4.3687 |  | 4.3857 |
| noGS+W | 0.2 | 4.6434 |  | 4.3990 |  |  | 4.3393±0.001 (n4) | 4.3498 |  | 4.3613 |  | 4.4267 |
| noGS+W | 0.4 |  |  |  |  |  | 4.3566 |  |  | 4.4168 |  |  |
| noGS+W | 0.8 |  |  |  |  |  | 4.4445 |  |  | 4.5646 |  |  |
| noGS+W [emb=fallback] | 0.2 |  |  |  |  |  | 4.6829 |  |  | 4.6741 |  |  |
| noGS+W [init_scale=2] | 0.2 |  |  |  |  |  | 4.3643 |  |  | 4.3636 |  |  |
| noGS+W [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.05 |  |  | 4.5794 |  |  | 4.5211±0.003 (n3) |  |  | 4.5387 |  |  |
| noGS+W [pmuoneq+normuon+soda, fallback_mode=atan2, fallback_lr_mult=0.5, ns_dtype=bf16] | 0.2 |  |  | 4.5736 |  |  | 4.5182 |  |  | 4.5311 |  |  |

### sp-cosine-8k-final_lr_scale=1: final_full_val_loss by lr (rows: cell, wd; mean over seeds)

| cell | wd | 0.0007 | 0.001 | 0.002 | 0.004 | 0.008 |
|---|---:|---:|---:|---:|---:|---:|
| GS+0 | 0 |  | 5.2190 | 5.5470 | 5.6502 | 5.7867 |
| GS+0+fixednorm [emulate_gs_theta0=0.02] | 0 |  |  | 5.4431 |  |  |
| GS+0+fixednorm [emulate_gs_theta0=0.04] | 0 |  |  |  | 5.5127 |  |
| GS+0+fixednorm [emulate_gs_theta0=0.08] | 0 |  |  |  |  | 5.5486 |
| GS+C | 0.0125 | 5.0582 |  |  |  |  |
| noGS+W | 0.2 |  | 4.7487 |  |  |  |

### sp-wsd-8k: final_full_val_loss by lr (rows: cell, wd; mean over seeds)

| cell | wd | 0.0007 | 0.001 | 0.002 | 0.004 | 0.008 |
|---|---:|---:|---:|---:|---:|---:|
| GS+0 | 0 |  |  | 4.8038 | 4.8536 | 4.9424 |
| GS+0+fixednorm [emulate_gs_theta0=0.04] | 0 |  |  |  | 4.7957 |  |
| GS+C | 0.0125 | 4.5481 |  |  |  |  |
| noGS+W | 0.2 |  | 4.3979 |  |  |  |

### std-wsd-8k: final_full_val_loss by lr (rows: cell, wd; mean over seeds)

| cell | wd | 0.00025 | 0.00035 | 0.0005 | 0.0007 | 0.001 | 0.0014 | 0.0018 | 0.002 | 0.004 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GS+0 | 0 | 4.9284 | 4.8552 | 4.7764 | 4.7668±0.006 (n4) | 4.8036 |  |  | 4.9166 | 4.9663 |
| GS+0 [emb=fallback] | 0 |  |  | 4.9711 |  |  |  |  |  |  |
| GS+0 [emb=matrix_nogs] | 0 |  |  | 4.8835 |  |  |  |  |  |  |
| GS+0 [init_scale=2] | 0 |  |  |  |  | 4.7737 |  |  |  |  |
| GS+0 [radial_lr=1] | 0 |  |  | 4.9957 | 4.9208 |  |  |  |  |  |
| GS+0+fixednorm | 0 | 4.9256 | 4.8488 | 4.7692±0.008 (n4) | 4.7828 | 4.8606 |  |  | 5.1679 | 5.5738 |
| GS+C | 0.0125 | 4.9259 | 4.8495 | 4.7763 | 4.7707±0.004 (n4) | 4.8178 |  |  | 4.9445 | 5.0172 |
| GS+C | 0.05 | 4.9156 | 4.8372 | 4.7766 | 4.7927 | 4.8670 |  |  | 5.0728 | 5.2124 |
| GS+C | 0.2 | 4.9033 |  | 4.8451 |  | 5.1430 |  |  | 5.4451 | 5.7493 |
| GS+C [radial_lr=1] | 0.2 |  |  | 4.9176 | 4.8583 |  |  |  |  |  |
| GS+W | 0.0125 | 4.9251 |  | 4.7770 |  | 4.8167 |  |  | 4.9657 | 5.0233 |
| GS+W | 0.05 | 4.9160 | 4.8365 | 4.7732 | 4.7959 | 4.8679 |  |  | 5.0703 | 5.2320 |
| GS+W | 0.2 | 4.9031 |  | 4.8543 |  | 5.1594 |  |  | 5.4918 | 5.8223 |
| noGS+0 | 0 | 5.0631 |  | 5.0250 |  | 4.9793 |  |  | 4.9572 | 4.9774 |
| noGS+C | 0.0125 | 5.0571 |  | 5.0151 |  | 4.9680 |  |  | 4.9473 | 4.9486 |
| noGS+C | 0.05 | 5.0445 |  | 4.9931 |  | 4.9321 |  |  | 4.8917 | 4.8634 |
| noGS+C | 0.2 | 4.9961 |  | 4.9158 |  | 4.8302 | 4.8059 |  | 4.7766 | 4.7959 |
| noGS+C | 0.4 |  |  |  |  | 4.7704 |  |  | 4.7886 |  |
| noGS+C | 0.8 |  |  |  |  | 4.7818 |  |  | 4.8735 |  |
| noGS+W | 0.0125 | 5.0571 |  | 5.0213 |  | 4.9704 |  |  | 4.9444 | 4.9539 |
| noGS+W | 0.05 | 5.0421 |  | 4.9928 |  | 4.9272 |  | 4.8841±0.006 (n5) | 4.8878 | 4.8620 |
| noGS+W | 0.2 | 4.9900 |  | 4.9078 |  | 4.8215 | 4.7952 |  | 4.7745 | 4.7997 |
| noGS+W | 0.4 |  |  |  |  | 4.7676±0.005 (n4) |  |  | 4.7940 |  |
| noGS+W | 0.8 |  |  |  |  | 4.7938 |  |  | 4.9050 |  |
| repo AdamW | 0.05 |  |  | 5.0961 |  |  |  |  |  |  |
| repo PlainMuon | 0.05 |  |  |  |  |  |  | 4.8895±0.010 (n5) |  |  |
