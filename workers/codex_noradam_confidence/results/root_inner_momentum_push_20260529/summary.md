# Fallback Inner Momentum Push

Question: does adding first-moment momentum only to AnchorMuon fallback
scalar/vector parameters improve the current best ViT-5 CIFAR-10 recipe?

Short answer: no. It produced a tiny 12-epoch HPO win, but the improvement did
not survive the 3-seed 50-epoch replay. The existing no-fallback-inner-momentum
WSD recipe remains the best option.

## Protocol

- Model: `vit5_micro`
- Dataset: CIFAR-10 with a deterministic 45k/5k train/validation split from
  `train=True`
- Final test: official CIFAR-10 `train=False`, evaluated only at the end
- Batch size: 512
- Dataloader workers: 16
- Precision/layout: BF16 autocast, channels-last
- GPUs: two visible GPUs, one trial per GPU
- HPO: 12 epochs, seed `123`
- Final replay: 50 epochs, seeds `123,456,789`
- Schedule: trainer-side 80-step warmup and WSD with final scale `0.1` and
  decay fraction `0.2`

## Implementation

The root `optimizer.py` now exposes:

```python
AnchorMuon(..., fallback_inner_momentum=False)
```

When enabled, fallback parameters maintain an FP32 first moment `exp_avg` using
`fallback_betas[0]`, bias-correct it, and divide it by the existing
bias-corrected fallback RMS denominator. Matrix parameters are unchanged; they
already use the Muon/Nesterov-style inner momentum source.

Default remains `False`.

## 12-Epoch HPO

Top HPO rows:

| Rank | Trial | Fallback inner momentum | LR | Row gamma | Matrix momentum | Fallback beta1 | Val acc | Val loss | Step |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `root_fim_wsd_lr0.014_rg0.35_pb0.9_nb0.93_fb0.85` | yes | 0.014 | 0.35 | 0.95 | 0.85 | 80.18% | 0.5569 | 17.16 ms |
| 2 | `root_fim_wsd_lr0.014_rg0.35_pb0.9_nb0.93_fb0.95` | yes | 0.014 | 0.35 | 0.95 | 0.95 | 80.18% | 0.5569 | 16.91 ms |
| 3 | `root_fim_wsd_lr0.014_rg0.3_pb0.9_nb0.93_fb0.85` | yes | 0.014 | 0.30 | 0.95 | 0.85 | 80.16% | 0.5648 | 17.03 ms |
| 4 | `root_fim_wsd_lr0.014_rg0.3_pb0.9_nb0.93_fb0.9` | yes | 0.014 | 0.30 | 0.95 | 0.90 | 80.16% | 0.5648 | 17.10 ms |
| 5 | `root_fim_wsd_lr0.014_rg0.3_pb0.9_nb0.93_fb0.95` | yes | 0.014 | 0.30 | 0.95 | 0.95 | 80.16% | 0.5648 | 17.19 ms |
| 13 | `root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93_fim0` | no | 0.012 | 0.35 | 0.95 | n/a | 80.08% | 0.5632 | 17.00 ms |

Interpretation: fallback inner momentum was worth about +0.10 validation
accuracy at 12 epochs in this seed, but the margin was small.

## 50-Epoch Replay

Mean and population standard deviation over seeds `123,456,789`:

| Recipe | Final val acc | Best val acc | Final val loss | Best val loss | Official test acc | Official test loss | Step | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnchorMuon WSD, no fallback inner momentum | 88.18% +/- 0.41 | 88.35% +/- 0.29 | 0.3886 +/- 0.0122 | 0.3747 +/- 0.0025 | 87.80% +/- 0.21 | 0.4084 +/- 0.0035 | 16.57 ms | 30.9k ex/s |
| AnchorMuon WSD + fallback inner momentum | 88.17% +/- 0.14 | 88.23% +/- 0.19 | 0.3975 +/- 0.0021 | 0.3766 +/- 0.0093 | 87.73% +/- 0.43 | 0.4160 +/- 0.0061 | 16.97 ms | 30.2k ex/s |
| AdamW cosine baseline | 79.71% +/- 0.20 | 79.96% +/- 0.38 | 0.6164 +/- 0.0024 | 0.6027 +/- 0.0024 | 79.32% +/- 0.17 | 0.6301 +/- 0.0052 | 11.43 ms | 44.8k ex/s |

## Conclusion

Fallback inner momentum is not a win for this recipe. It slightly helped the
12-epoch proxy but was marginally worse at 50 epochs on validation loss, best
validation accuracy, official test loss, official test accuracy, and speed.

Use the existing AnchorMuon WSD recipe:

```text
lr=0.012
schedule=wsd
warmup_steps=80
lr_final_scale=0.1
wsd_decay_frac=0.2
row_gamma=0.35
pmuoneq_beta=0.90
momentum=0.95
normuon_beta2=0.93
fallback_inner_momentum=False
```

Plots:

- `final50_multiseed/val_acc.png`
- `final50_multiseed/val_loss.png`
- `final50_multiseed/train_loss.png`
- `final50_multiseed/step_time_ms_bar.png`
- `final50_multiseed/examples_per_sec_bar.png`

