# Golden Optimizer Validation

Updated: 2026-05-29

## What Was Validated

The standalone golden optimizer:

```text
workers/codex_equimuse_normuon/golden_soda_pmuoneq_normuon.py
```

was validated against this worker's stored best result:

```text
Direct SODA-PMuonEq-NorMuon + aspect
vit5_small / CIFAR-10 img224 / 4-GPU DDP / seed 67890 / 1000 steps
```

The golden file implements only that best-result path:

```text
direct SODA/Anchor
+ row/column PMuonEq
+ Gram Newton-Schulz
+ row-wise NorMuon
+ post-NorMuon aspect scale
+ RMS/AdamW fallback with fallback weight decay
```

No AMUSE/SF, MiMuon, aspect toggle, or no-column ablation is present.

## Unit And Parity Validation

Commands:

```bash
/home/catid/attractor/.venv/bin/python -m py_compile \
  workers/codex_equimuse_normuon/golden_soda_pmuoneq_normuon.py \
  workers/codex_equimuse_normuon/direct_soda_pmuoneq_normuon.py

PYTHONPATH=workers/codex_equimuse_normuon \
  /home/catid/attractor/.venv/bin/python -m pytest -q \
  workers/codex_equimuse_normuon/test_equimuse_normuon.py \
  workers/codex_equimuse_normuon/test_golden_soda_pmuoneq_normuon.py
```

Result:

```text
14 passed
```

Additional deterministic optimizer-step parity check:

```text
golden_vs_direct_best_max_abs_diff=0.000e+00
golden_stats={
  'matrix_count': 2.0,
  'fallback_count': 1.0,
  'aspect_scale_enabled': 1.0,
  'column_preconditioning_enabled': 1.0,
  'soda_weight': 0.09090909090909091
}
```

This confirms the golden optimizer matches the older direct implementation
configured as the best-result row.

## Full 1000-Step Reproduction

The ViT-5 harness was run with a temporary import alias so that
`--opt soda_pmuoneq_normuon` resolved to the new golden optimizer.

Hardware and protocol:

| field | value |
|---|---|
| GPUs | 4 visible NVIDIA GPUs |
| launch | `torchrun --standalone --nproc_per_node=4` |
| model | `vit5_small`, 21,657,994 trainable params |
| dataset | CIFAR-10, img224 |
| seed | `67890` |
| per-GPU batch | 128 |
| effective global batch | 512 |
| gradient accumulation | 1 |
| epochs / steps | 8 epochs, 125 steps/epoch, 1000 optimizer steps |

Golden optimizer settings:

| setting | value |
|---|---:|
| matrix LR | `8e-3` |
| fallback LR | `8e-4` |
| momentum | `0.90` |
| fallback epsilon | `1e-8` |
| fallback weight decay | `0.05` |
| PMuonEq beta | `0.90` |
| row gamma | `0.35` |
| column gamma | `0.05` |
| NorMuon beta2 | `0.93` |
| aspect scale | enabled |

Final reproduced metrics:

| metric | golden validation | stored best |
|---|---:|---:|
| train loss | `1.2382302932739258` | `1.2382302932739258` |
| validation/test loss | `0.7268316862838609` | `0.7268840021320752` |
| acc@1 | `76.69000268554687` | `76.69000268554687` |
| acc@5 | `98.61000258789062` | not recorded in summary |
| final samples/s | `4189.526761187844` | `4008.82859236227` |
| final optimizer time/bin | `4.485048532485962` | `4.445104122161865` |

The accuracy and train loss reproduce exactly; validation loss differs by
`5.2e-5`, which is below the logging/eval noise expected from distributed data
loading and floating-point reduction details.

Run artifact path on the validation workstation:

```text
/home/catid/attractor/runs/golden_validate_direct_soda_pmuoneq_normuon_aspect_seed67890_img224_1000/log.txt
```

The final log row records:

```json
{
  "train_loss": 1.2382302932739258,
  "train_aspect_scale_enabled": 1.0,
  "train_column_preconditioning_enabled": 1.0,
  "test_loss": 0.7268316862838609,
  "test_acc1": 76.69000268554687,
  "test_acc5": 98.61000258789062,
  "epoch": 7,
  "n_parameters": 21657994
}
```
