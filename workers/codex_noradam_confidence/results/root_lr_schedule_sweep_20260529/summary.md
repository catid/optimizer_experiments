# Root AnchorMuon LR Schedule Sweep

Updated: 2026-05-29

This result bundle compares trainer-side LR schedules for the shippable root
`optimizer.py` `AnchorMuon` implementation.

## Protocol

- Model: `vit5_micro`
- Dataset: CIFAR-10
- Split: 45,000 train examples and 5,000 validation examples from CIFAR-10
  `train=True`
- Test: official CIFAR-10 `train=False`, evaluated only after selected final
  runs
- HPO: 12 epochs, seed `123`, one trial per visible GPU
- Final replay: 50 epochs, seed `123`, best HPO config per schedule
- Batch size: 512
- Workers: 16
- Precision/layout: BF16 autocast, channels-last tensors
- Hardware: two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs
- PyTorch: `2.13.0.dev20260506+cu130`

## 12-Epoch HPO Winners

| Schedule | Trial | LR | Val loss | Val acc | Step |
|---|---|---:|---:|---:|---:|
| constant | `root_named_constant_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.6865 | 75.36% | 16.94 ms |
| cosine | `root_named_cosine_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5946 | 79.30% | 17.09 ms |
| linear | `root_named_linear_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5820 | 79.46% | 16.86 ms |
| WSD | `root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5632 | 80.08% | 16.06 ms |
| AdamW cosine | `adamw_cosine_lr0.004_wd0.001` | 0.004 | 0.9095 | 67.56% | 11.04 ms |

## 50-Epoch Schedule-Winner Replay

| Rank | Recipe | Schedule | Final val loss | Best val loss | Final val acc | Best val acc | Test loss | Test acc | Step | Throughput |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon | WSD | 0.4005 | 0.3716 | 88.36% | 88.36% | 0.4111 | 87.67% | 16.23 ms | 31.5k ex/s |
| 2 | AnchorMuon | constant | 0.4036 | 0.4036 | 86.06% | 86.54% | 0.4317 | 85.74% | 17.19 ms | 29.8k ex/s |
| 3 | AnchorMuon | cosine | 0.4511 | 0.4097 | 87.40% | 87.48% | 0.4588 | 86.90% | 16.96 ms | 30.2k ex/s |
| 4 | AnchorMuon | linear | 0.4434 | 0.4151 | 87.18% | 87.58% | 0.4597 | 86.75% | 17.02 ms | 30.1k ex/s |
| 5 | AdamW | cosine | 0.6133 | 0.5998 | 79.62% | 79.64% | 0.6240 | 79.55% | 11.39 ms | 45.0k ex/s |

## Conclusion

WSD is the best trainer-side schedule for AnchorMuon in this single-seed
schedule study. It wins on final validation loss, best validation loss, final
validation accuracy, best validation accuracy, and official test accuracy.
AdamW remains the fastest per step.

The best observed run is:

```text
root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93
```

with:

```text
lr = 0.012
schedule = 80-step warmup + WSD
lr_final_scale = 0.1
wsd_decay_frac = 0.2
row_gamma = 0.35
pmuoneq_beta = 0.90
normuon_beta = 0.93
```

## Artifacts

- HPO CSV: `hpo/all_runs.csv`
- Final replay CSV: `final50_schedule_winners/all_runs.csv`
- Loss curves: `final50_schedule_winners/val_loss.png`
- Accuracy curves: `final50_schedule_winners/val_acc.png`
- Step speed bar chart: `final50_schedule_winners/step_time_ms_bar.png`
- Throughput bar chart: `final50_schedule_winners/examples_per_sec_bar.png`

## Commands

```bash
OUT=workers/codex_noradam_confidence/results/root_lr_schedule_sweep_20260529/hpo
uv run --with torch --with torchvision --with timm --with matplotlib \
  python workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --preset root_lr_schedule_sweep \
  --output-dir "$OUT" \
  --epochs 12 \
  --train-subset 0 --val-subset 0 \
  --val-source train_split --train-val-size 5000 \
  --batch-size 512 \
  --num-workers 16 \
  --seed 123 \
  --warmup-steps 80 \
  --lr-final-scale 0.1 \
  --wsd-decay-frac 0.2 \
  --eval-bins 8 \
  --log-every 200 \
  --no-sync-step-timing

OUT=workers/codex_noradam_confidence/results/root_lr_schedule_sweep_20260529/final50_schedule_winners
uv run --with torch --with torchvision --with timm --with matplotlib \
  python workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --preset root_lr_schedule_sweep \
  --only '^(adamw_cosine_lr0\.004_wd0\.001|root_named_(constant|cosine|linear|wsd)_lr0\.012_rg0\.35_pb0\.9_nb0\.93)$' \
  --output-dir "$OUT" \
  --epochs 50 \
  --train-subset 0 --val-subset 0 \
  --val-source train_split --train-val-size 5000 \
  --eval-test --test-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --seed 123 \
  --warmup-steps 80 \
  --lr-final-scale 0.1 \
  --wsd-decay-frac 0.2 \
  --eval-bins 8 \
  --log-every 200 \
  --no-sync-step-timing
```
