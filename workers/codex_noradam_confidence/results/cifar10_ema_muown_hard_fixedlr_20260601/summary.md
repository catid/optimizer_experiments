# CIFAR-10 EMA-Muown Hard HPO

Date: 2026-06-02

This resumed and completed the harder fixed-scheduler-ratio CIFAR-10 HPO for
EMA-Nesterov + Muown. The previous partial run was paused at 96/268 HPO
candidates; this pass completed all HPO rows and then ran the selected
50-epoch final replay with official CIFAR-10 test evaluation.

## Protocol

- Model: `vit5_micro`
- Dataset: CIFAR-10 with 45,000 training examples, 5,000 validation examples
  held out from the official training split, and the official 10,000-image
  test split evaluated only at the end of selected final runs
- Batch: 512
- Precision/layout: BF16 autocast, channels-last tensors
- Workers: 16 dataloader workers
- Hardware: two RTX PRO 6000 Blackwell GPUs, one trial per GPU
- HPO: 268 candidates, 12 epochs, seed `123`
- Final replay: best HPO config per optimizer family, 50 epochs, seed `123`
- LR schedules: WSD for Muon-family runs and cosine for AdamW
- Warmup: 80 steps

Compared families:

- AdamW cosine baseline
- Root AnchorMuon: SODA + PMuonEq + GramNS + NorMuon + AdamAtan2 fallback
- Muown
- EMA-Nesterov + Muon
- EMA-Nesterov + Muown

## Final 50-Epoch Results

Sorted by final validation accuracy.

| Rank | Family | Selected config | Final val acc | Official test acc | Final val loss | Official test loss | Step | Throughput |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon root | `lr=0.016`, WSD, SODA+PMuonEq+NorMuon, AdamAtan2 fallback `0.5x` | 88.48% | 87.85% | 0.3695 | 0.3972 | 16.25 ms | 31.5k ex/s |
| 2 | EMA-Nesterov + Muon | `lr=0.014`, WSD, `wd=0.001`, `ema_beta=0.03`, `ema_gamma=0.995` | 87.72% | 86.99% | 0.4231 | 0.4886 | 18.36 ms | 27.9k ex/s |
| 3 | EMA-Nesterov + Muown | `lr=0.014`, WSD, `wd=0`, `fallback=1.0x`, `mag_lr=1.0x`, `ema_beta=0.15`, `ema_gamma=0.995` | 85.34% | 84.85% | 0.4498 | 0.4657 | 20.79 ms | 24.6k ex/s |
| 4 | Muown | `lr=0.014`, WSD, `wd=0`, `fallback=0.5x`, `mag_lr=1.0x` | 83.24% | 82.61% | 0.5062 | 0.5197 | 19.55 ms | 26.2k ex/s |
| 5 | AdamW | `lr=0.004`, cosine, `wd=0.001` | 79.64% | 79.55% | 0.5998 | 0.6240 | 11.26 ms | 45.5k ex/s |

## HPO Notes

- The best 12-epoch HPO row was EMA-Nesterov + Muon at `80.86%` validation
  accuracy, slightly ahead of root AnchorMuon at `78.46%`.
- Root AnchorMuon still won the 50-epoch replay and official test evaluation.
  Its final validation/test accuracies were `88.48%` and `87.85%`.
- Harder tuning substantially improved EMA-Nesterov + Muown versus the earlier
  narrow pass (`85.34%` final validation accuracy versus `76.54%`), but it
  remained behind EMA-Nesterov + Muon and root AnchorMuon.
- Plain Muown remained behind the tuned spectral baselines on CIFAR-10.
- Tiny nonzero weight decay on EMA-Muown did not rescue the method; the best
  selected EMA-Muown config used `wd=0`.

## Takeaway

For this CIFAR-10 ViT-5 benchmark, the image-classification default remains:

```text
Root AnchorMuon: SODA + PMuonEq + GramNS + NorMuon + AdamAtan2 fallback,
trainer-side WSD LR schedule
```

EMA-Nesterov + Muown is now a stronger ablation than the earlier narrow run
suggested, but it is slower and less accurate than root AnchorMuon. It should
not replace the CIFAR optimizer stack.

## Artifacts

- HPO raw rows: `hpo/all_runs.csv`
- Final replay raw rows: `final_bins/all_runs.csv`
- HPO selected configs: `hpo/best_by_family.json`
- Final selected configs: `final_bins/best_by_family.json`
- Validation accuracy: `final_bins/val_acc.png`
- Validation loss: `final_bins/val_loss.png`
- Training loss: `final_bins/train_loss.png`
- Step time: `final_bins/step_time_ms_bar.png`
- Throughput: `final_bins/examples_per_sec_bar.png`

## Reproduction Command

```bash
/home/catid/screen/.venv/bin/python workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --two-stage \
  --preset ema_muown_cifar10_hard \
  --output-dir workers/codex_noradam_confidence/results/cifar10_ema_muown_hard_fixedlr_20260601 \
  --val-source train_split \
  --train-subset 45000 --train-val-size 5000 --val-subset 5000 \
  --eval-test \
  --batch-size 512 --num-workers 16 \
  --hpo-epochs 12 --final-epochs 50 \
  --eval-bins 8 --warmup-steps 80 \
  --lr-final-scale 0.1 --wsd-decay-frac 0.2 \
  --skip-completed
```
