# CIFAR-10 Final 50-Epoch Confidence Run

Run date: 2026-05-29 UTC

Command:

```bash
OUT=results/cifar10_feedback_nosync_20260529/final50
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --final-from-hpo results/cifar10_confidence_noradam_20260528/hpo/best_by_family.json \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --final-epochs 50 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 200 \
  --model vit5_micro \
  --seeds 123,456,789 \
  --no-sync-step-timing
```

Environment:

- Torch: `2.13.0.dev20260506+cu130`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Timing mode: `cpu_launch_unsynchronized`
- NorMuon optimizer diagnostics synchronization: disabled (`opt_sync_diagnostics = 0.0`)

## Aggregate Results

| Optimizer | Seeds | Final val loss | Best val loss | Final val acc | Best val acc | Avg step | Overall throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| NorMuon + base | 3 | 0.4193 +/- 0.0044 | 0.4117 +/- 0.0018 | 85.93% +/- 0.14 | 86.00% +/- 0.24 | 19.89 ms +/- 0.37 | 25.74k ex/s +/- 0.48k |
| AdamW | 3 | 0.6210 +/- 0.0105 | 0.6088 +/- 0.0063 | 79.66% +/- 0.12 | 79.96% +/- 0.13 | 11.55 ms +/- 0.18 | 44.35k ex/s +/- 0.70k |

## Diagrams

- Loss curves: `comparison_loss_curves.png`
- Validation accuracy: `comparison_accuracy.png`
- Iteration speed: `comparison_iteration_speed.png`

## Per-Seed Results

| Optimizer | Seed | Final val loss | Best val loss | Final val acc | Best val acc | Avg step | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 123 | 0.6286 | 0.6136 | 79.52% | 80.03% | 11.64 ms | 43.99k ex/s |
| AdamW | 456 | 0.6253 | 0.6112 | 79.70% | 80.05% | 11.66 ms | 43.91k ex/s |
| AdamW | 789 | 0.6091 | 0.6016 | 79.76% | 79.81% | 11.34 ms | 45.16k ex/s |
| NorMuon + base | 123 | 0.4163 | 0.4108 | 86.01% | 86.01% | 20.23 ms | 25.31k ex/s |
| NorMuon + base | 456 | 0.4173 | 0.4105 | 86.01% | 86.23% | 19.96 ms | 25.66k ex/s |
| NorMuon + base | 789 | 0.4244 | 0.4138 | 85.76% | 85.76% | 19.50 ms | 26.26k ex/s |

## Notes

- This run compares the narrowed result from the feedback loop: tuned AdamW versus tuned `SODA + PMuonEq + Gram + NorMuon`, with AMUSE disabled.
- NorMuon + base remains clearly better on CIFAR-10 validation loss and accuracy across all three seeds.
- AdamW remains faster per iteration. NorMuon + base costs about 1.72x more per step here, but improves final validation accuracy by about 6.27 percentage points.
- The speed numbers use unsynchronized CPU launch timing to avoid injecting optimizer-step synchronizations. They are useful for relative launcher overhead in this runner, but wall-clock throughput is the safer end-to-end speed metric.
- Full run artifacts are in this directory: `all_runs.csv`, per-trial `metrics.jsonl`, per-trial `summary.json`, and generated plots.
