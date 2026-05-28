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

## Main Result

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
