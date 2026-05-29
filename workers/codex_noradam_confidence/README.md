# Codex NorMuon Confidence Worker

This folder contains the Codex worker copy of the ViT-5 CIFAR-10 optimizer
comparison focused on:

- AdamW baseline
- NorMuon+base: SODA + PMuonEq + Gram Newton-Schulz + NorMuon, AMUSE off

The source was copied from local repo:

```text
/home/catid/screen/repos/ViT-5-AnchorMuon
commit b009f03 Compare NorMuon base against AdamW
```

## Contents

- `optim_anchormuon.py`: AnchorMuon/NorMuon optimizer implementation.
- `optim_factory.py`: ViT-5 optimizer factory hook.
- `models_vit5.py`, `rope.py`: minimal ViT-5 model code needed by the runner.
- `experiments/run_cifar10_ablation.py`: CIFAR-10 HPO/final comparison runner.
- `tests/`: optimizer unit tests and DDP smoke test.
- `results/cifar10_confidence_noradam_20260528/`: committed result bundle, curves, and summary.
- `results/cifar10_feedback_nosync_20260529/final50/`: latest committed
  feedback run with synchronization diagnostics disabled, comparison diagrams,
  curves, CSV, and per-trial JSONL.

## Main Result

Latest result summary:
`results/cifar10_feedback_nosync_20260529/final50/summary.md`.

Latest diagrams:

- Loss curves: `results/cifar10_feedback_nosync_20260529/final50/comparison_loss_curves.png`
- Validation accuracy: `results/cifar10_feedback_nosync_20260529/final50/comparison_accuracy.png`
- Iteration speed: `results/cifar10_feedback_nosync_20260529/final50/comparison_iteration_speed.png`

### Latest 50-Epoch Replay

This replay uses the same HPO-selected AdamW and NorMuon+base configs, but with
`AnchorMuon.sync_diagnostics=False` and `--no-sync-step-timing` so optimizer
diagnostics do not force per-step GPU synchronization.

| Run | AdamW | NorMuon+base |
|---|---:|---:|
| 50 epoch final val loss | 0.6210 +/- 0.0105 | 0.4193 +/- 0.0044 |
| 50 epoch best val loss | 0.6088 +/- 0.0063 | 0.4117 +/- 0.0018 |
| 50 epoch final val acc | 79.66% +/- 0.12 | 85.93% +/- 0.14 |
| 50 epoch best val acc | 79.96% +/- 0.13 | 86.00% +/- 0.24 |
| 50 epoch step time | 11.55 ms +/- 0.18 | 19.89 ms +/- 0.37 |
| 50 epoch throughput | 44.35k ex/s +/- 0.70k | 25.74k ex/s +/- 0.48k |

NorMuon+base is again better on validation loss and accuracy across all three
seeds. AdamW remains faster per step.

### Historical Synchronized-Diagnostics Run

See `results/cifar10_confidence_noradam_20260528/summary.md`.

| Run | AdamW | NorMuon+base |
|---|---:|---:|
| 20 epoch val loss | 0.7469 +/- 0.0110 | 0.5647 +/- 0.0086 |
| 20 epoch val acc | 73.55% +/- 0.48 | 80.09% +/- 0.20 |
| 20 epoch step time | 11.86 ms | 21.61 ms |
| 50 epoch val loss | 0.6167 +/- 0.0117 | 0.4284 +/- 0.0145 |
| 50 epoch val acc | 80.07% +/- 0.40 | 85.59% +/- 0.53 |
| 50 epoch step time | 11.78 ms | 21.69 ms |

NorMuon+base is better on validation loss and accuracy in this harness, while
AdamW is faster per step.

## Exact Reported Recipe

The reported NorMuon+base rows use this explicit AnchorMuon configuration:

```text
lr = 8e-3
warmup_steps = 80
amuse = False
soda = "all"
soda_disables_weight_decay = True
pmuon_eq = True
pmuon_beta = 0.90
row_gamma = 0.30
col_gamma = 0.0
momentum = 0.95
normuon = True
normuon_beta = 0.95
mimuon = False
ns_steps = 5
sync_diagnostics = False for the latest run
```

`_anchor_param_groups()` is intentionally the grouping used by the committed
confidence runs. It is name-free except for the model's `no_weight_decay()`
hook, so a 2D classifier/head matrix can be routed through the Muon/NorMuon
path unless the model excludes it. Treat that as part of the reported recipe
rather than an implied general recommendation.

`AnchorMuon.sync_diagnostics` now defaults to `False`. The committed historical
metrics include `mean_update_rms` and `mean_precond_matrix_rms`, but those values
required GPU-to-CPU synchronization inside `step()`. Enable
`sync_diagnostics=True` only for debugging runs where that timing perturbation is
acceptable.

## Reproduction Commands

From this folder, with the repo environment active:

```bash
python -m py_compile experiments/run_cifar10_ablation.py optim_anchormuon.py
python -m pytest -q tests/test_anchormuon_modes.py
python -m torch.distributed.run --standalone --nproc-per-node=<num_gpus> tests/ddp_smoke_anchormuon.py
```

The HPO run used:

```bash
python experiments/run_cifar10_ablation.py \
  --preset confidence \
  --output-dir results/cifar10_confidence_noradam_20260528/hpo \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 12 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 100 \
  --model vit5_micro
```

Final 20-epoch and 50-epoch runs used `--final-from-hpo` with the HPO
`best_by_family.json`, seeds as recorded in the result summaries, and all visible
GPUs scheduled one trial per GPU.

The latest no-sync final replay used:

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
