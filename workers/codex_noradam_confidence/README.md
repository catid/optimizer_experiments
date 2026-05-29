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
- `ALGORITHM_RESULTS.md`: self-contained algorithm and result summary for the
  current best recipe.
- `results/cifar10_confidence_noradam_20260528/`: committed result bundle, curves, and summary.
- `results/cifar10_feedback_nosync_20260529/final50/`: latest committed
  feedback run with synchronization diagnostics disabled, comparison diagrams,
  curves, CSV, and per-trial JSONL.
- `results/cifar10_feedback_aspect_20260529/`: HPO and 3-seed 50-epoch replay
  for the peer-suggested post-NorMuon aspect multiplier.
- `results/cifar10_proper_split_20260529/`: proper CIFAR-10 45k/5k
  train/validation split with official test evaluated only at the end of final
  selected runs.

## Main Result

Latest result summary:
`ALGORITHM_RESULTS.md` and `results/cifar10_proper_split_20260529/summary.md`.

Latest diagrams:

- Loss curves: `results/cifar10_proper_split_20260529/final50/val_loss.png`
- Validation accuracy: `results/cifar10_proper_split_20260529/final50/val_acc.png`
- Iteration speed: `results/cifar10_proper_split_20260529/final50/step_time_ms_bar.png`

### Latest Proper-Split Replay

This replay holds out 5,000 examples from CIFAR-10 `train=True` for HPO and
validation, trains on the remaining 45,000 examples, and evaluates the official
CIFAR-10 `train=False` test split only once at the end of each selected final
run.

| Run | AdamW | NorMuon no aspect | NorMuon aspect |
|---|---:|---:|---:|
| 50 epoch final val loss | 0.6180 +/- 0.0043 | 0.4414 +/- 0.0228 | 0.4445 +/- 0.0130 |
| 50 epoch best val loss | 0.6043 +/- 0.0039 | 0.4310 +/- 0.0112 | 0.4231 +/- 0.0135 |
| 50 epoch final val acc | 79.69% +/- 0.22 | 84.97% +/- 0.35 | 85.25% +/- 0.32 |
| 50 epoch best val acc | 79.85% +/- 0.29 | 85.25% +/- 0.27 | 85.65% +/- 0.33 |
| Official test loss | 0.6338 +/- 0.0125 | 0.4607 +/- 0.0196 | 0.4595 +/- 0.0074 |
| Official test acc | 79.28% +/- 0.23 | 84.77% +/- 0.65 | 84.55% +/- 0.40 |
| 50 epoch step time | 11.59 ms +/- 0.09 | 19.95 ms +/- 0.08 | 19.92 ms +/- 0.66 |
| 50 epoch throughput | 44.16k ex/s +/- 0.34k | 25.67k ex/s +/- 0.10k | 25.73k ex/s +/- 0.86k |

Conclusion: the clean split preserves the large gap over AdamW. The no-aspect
NorMuon recipe is the current best by official test accuracy; the aspect recipe
has slightly better validation-checkpoint metrics but did not improve mean test
accuracy in this three-seed replay.

### Latest Aspect-Feedback Replay

This replay tested the cross-worker suggestion to add
`normuon_aspect_scale=True`, a post-NorMuon `sqrt(max(1, rows / cols))`
multiplier for tall matrices. The flag is disabled by default and the older
reported no-aspect behavior remains available.

HPO over aspect/no-aspect plus small `row_gamma`, `col_gamma`, and
`normuon_beta` changes selected the no-aspect recipe
`row_gamma=0.35`, `col_gamma=0.05`, `normuon_beta=0.95`. A 50-epoch replay used
three seeds.

| Run | AdamW | NorMuon no aspect | NorMuon aspect |
|---|---:|---:|---:|
| 50 epoch final val loss | 0.6210 +/- 0.0105 | 0.4225 +/- 0.0027 | 0.4288 +/- 0.0103 |
| 50 epoch best val loss | 0.6088 +/- 0.0063 | 0.4198 +/- 0.0043 | 0.4236 +/- 0.0083 |
| 50 epoch final val acc | 79.66% +/- 0.12 | 85.86% +/- 0.23 | 85.43% +/- 0.29 |
| 50 epoch best val acc | 79.96% +/- 0.13 | 85.98% +/- 0.09 | 85.60% +/- 0.17 |
| 50 epoch step time | 11.54 ms +/- 0.08 | 20.20 ms +/- 0.41 | 19.56 ms +/- 0.43 |
| 50 epoch throughput | 44.36k ex/s +/- 0.31k | 25.36k ex/s +/- 0.51k | 26.18k ex/s +/- 0.58k |

Conclusion: aspect scaling was close and slightly faster in this run, but did
not beat the retuned no-aspect recipe on validation loss or accuracy.

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
row_gamma = 0.35 for the latest proper-split winner, 0.30 for the earlier no-sync replay
col_gamma = 0.0 for the latest proper-split winner, 0.05 for the aspect-feedback replay
momentum = 0.95
normuon = True
normuon_beta = 0.93 for the latest proper-split winner, 0.95 for the aspect-feedback replay
normuon_aspect_scale = False for the current best recipe
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

The latest aspect-feedback HPO and replay used:

```bash
OUT=results/cifar10_feedback_aspect_20260529/hpo
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --preset feedback \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 12 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 200 \
  --model vit5_micro \
  --no-sync-step-timing

OUT=results/cifar10_feedback_aspect_20260529/final50
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --final-from-hpo results/cifar10_feedback_aspect_20260529/hpo/best_by_family.json \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --final-epochs 50 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 500 \
  --model vit5_micro \
  --seeds 123,456,789 \
  --no-sync-step-timing
```

The latest proper-split HPO and final replay used:

```bash
OUT=results/cifar10_proper_split_20260529/hpo
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --preset feedback \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 12 \
  --train-subset 0 \
  --val-source train_split --train-val-size 5000 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 200 \
  --model vit5_micro \
  --no-sync-step-timing

OUT=results/cifar10_proper_split_20260529/final50
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --final-from-hpo results/cifar10_proper_split_20260529/hpo/best_by_family.json \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --final-epochs 50 \
  --train-subset 0 \
  --val-source train_split --train-val-size 5000 --val-subset 0 \
  --eval-test --test-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 500 \
  --model vit5_micro \
  --seeds 123,456,789 \
  --no-sync-step-timing
```
