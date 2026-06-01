# CIFAR-10 Muown / EMA-Nesterov Transfer Check

Date: 2026-06-01

This run tested whether the Muown and EMA-Nesterov ideas that looked useful in
the 50M byte-LM proxy also transfer to the existing ViT-5 CIFAR-10 harness.

## Protocol

- Model: `vit5_micro`
- Dataset: CIFAR-10 with a proper split: 45,000 train examples, 5,000
  validation examples held out from `train=True`, and official 10,000-image
  test set evaluated only at the end of final selected runs.
- Batch: 512
- Precision/layout: BF16 autocast, channels-last tensors
- Workers: 16 dataloader workers
- Hardware: two RTX PRO 6000 Blackwell GPUs, one trial per GPU
- HPO: 12 epochs, seed `123`
- Final replay: best HPO config per optimizer family, 50 epochs, seed `123`
- LR schedule: WSD for Muon-family runs, cosine for AdamW baseline
- Warmup: 80 steps

## Compared Families

- AdamW cosine baseline
- Existing root AnchorMuon recipe:
  SODA + PMuonEq + GramNS + NorMuon + AdamAtan2 fallback
- Plain Muon + GramNS
- Muown
- EMA-Nesterov + plain Muon
- EMA-Nesterov + Muown

## Final 50-Epoch Results

Sorted by official test accuracy.

| Rank | Family | Selected run | Final val acc | Best val acc | Official test acc | Final val loss | Official test loss | Step | Throughput |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon root | `lr=0.016`, WSD, `row_gamma=0.35`, `pmuon_beta=0.90`, `normuon_beta=0.93`, AdamAtan2 fallback at `0.5x` | 88.32% | 88.32% | 87.98% | 0.3787 | 0.4065 | 17.11 ms | 29.9k ex/s |
| 2 | EMA-Nesterov + Muon | `lr=0.014`, `wd=0.001`, `ema_beta=0.1`, `ema_gamma=0.99` | 86.94% | 86.94% | 86.08% | 0.4754 | 0.5023 | 17.79 ms | 28.8k ex/s |
| 3 | Plain Muon | `lr=0.014`, `wd=0.001`, GramNS | 86.72% | 86.82% | 85.40% | 0.5289 | 0.5582 | 17.08 ms | 30.0k ex/s |
| 4 | Muown | `lr=0.012`, `wd=0`, WSD | 83.36% | 83.36% | 83.56% | 0.4845 | 0.4902 | 20.10 ms | 25.5k ex/s |
| 5 | AdamW | `lr=0.004`, `wd=0.001`, cosine | 79.22% | 79.60% | 79.76% | 0.6158 | 0.6352 | 11.62 ms | 44.1k ex/s |
| 6 | EMA-Nesterov + Muown | `lr=0.012`, `wd=0.001`, `ema_beta=0.1`, `ema_gamma=0.99` | 76.54% | 76.54% | 75.40% | 0.6649 | 0.6954 | 21.18 ms | 24.2k ex/s |

## HPO Notes

- The root AnchorMuon family won the 12-epoch HPO stage and the final 50-epoch
  replay.
- Plain Muon was close in 12-epoch HPO but fell behind root AnchorMuon by the
  end of the 50-epoch replay.
- EMA-Nesterov on plain Muon was stable and beat plain Muon on official test,
  but did not beat the existing root AnchorMuon recipe.
- Muown underperformed on this CIFAR-10 ViT-5 setup despite helping the
  byte-LM proxy. Weight decay was consistently harmful for Muown during HPO.
- EMA-Nesterov + Muown was not competitive; several no-weight-decay HPO trials
  became NaN, and the best stable weighted-decay final run still trailed Muown.

## Takeaway

For this CIFAR-10 ViT-5 benchmark, the existing root AnchorMuon recipe remains
the best option. Muown and EMA-Nesterov are not general replacements for the
CIFAR optimizer stack. EMA-Nesterov + Muon is worth keeping as a small ablation
candidate, but the stronger recipe for image classification is still:

```text
SODA + PMuonEq + GramNS + NorMuon + AdamAtan2 fallback, trainer-side WSD LR
```

## Plots

- Validation accuracy: `final_bins/val_acc.png`
- Validation loss: `final_bins/val_loss.png`
- Training loss: `final_bins/train_loss.png`
- Step time: `final_bins/step_time_ms_bar.png`
- Throughput: `final_bins/examples_per_sec_bar.png`

## Reproduction Command

```bash
/home/catid/screen/.venv/bin/python workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --two-stage \
  --preset muown_ema_cifar10 \
  --output-dir workers/codex_noradam_confidence/results/cifar10_muown_ema_20260601 \
  --val-source train_split \
  --train-subset 45000 --train-val-size 5000 --val-subset 5000 \
  --eval-test \
  --batch-size 512 --num-workers 16 \
  --hpo-epochs 12 --final-epochs 50 \
  --eval-bins 8 --warmup-steps 80 \
  --lr-final-scale 0.1 --wsd-decay-frac 0.2
```
