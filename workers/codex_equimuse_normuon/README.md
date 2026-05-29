# EquiMuse / Direct SODA-PMuonEq-NorMuon Optimizers

This folder contains the Codex EquiMuse optimizer work and the latest
peer-feedback comparison artifacts. There are two standalone optimizer files:

- `equimuse_normuon.py`: the original schedule-free recipe.
- `direct_soda_pmuoneq_normuon.py`: the direct no-AMUSE recipe that produced
  the best result in the latest matched ViT-5/CIFAR-10 run.

## Best Current Version

The current winner in this folder is:

```text
Direct SODA-PMuonEq-NorMuon + aspect
```

It was measured on **CIFAR-10**, not CIFAR-100, with `vit5_small`
(`21,657,994` trainable parameters), img224, 4-GPU DDP, per-GPU batch 128,
effective global batch 512, seed `67890`, and 1000 optimizer steps. The winning
exact hyperparameters were:

```text
optimizer = soda_pmuoneq_normuon
matrix_lr = 8e-3
fallback_adam_lr = 8e-4
momentum = 0.90
pmuoneq_beta = 0.90
row_gamma = 0.35
col_gamma = 0.05
normuon_beta2 = 0.93
normuon_orientation = row
normuon_aspect_scale = true
warmup_steps = 10
matrix_weight_decay = 0.0
fallback_weight_decay = 0.05
fallback_eps = 1e-8
per_gpu_batch = 128
gradient_accumulation = 1
effective_global_batch = 512
```

Final readout for that run: `76.69%` CIFAR-10 acc@1, `0.7269` validation loss,
`1.2382` train loss, and `4009` samples/s. The golden optimizer reproduced this
with `76.69000268554687` acc@1, `0.7268316862838609` loss, and identical
`1.2382302932739258` train loss. See `ALGORITHM_RESULTS.md` for the original
comparison and `GOLDEN_VALIDATION.md` for the golden reproduction.

## Algorithm Summary

The schedule-free EquiMuse recipe is:

```text
SODA-AMUSE + PMuonEq + Gram Newton-Schulz + NorMuon
```

It keeps fast/eval/interpolation sequences (`Z`, `X`, `Y`), evaluates gradients
at the schedule-free interpolation point, builds a PMuonEq-preconditioned Muon
matrix direction, orthogonalizes with Gram Newton-Schulz, and applies NorMuon
row-wise second-moment normalization after the polar step. The optional aspect
multiplier scales the final learned update, not gradients or GramNS inputs.

The best current recipe is simpler:

```text
Direct SODA + PMuonEq + Gram Newton-Schulz + NorMuon + aspect
```

It removes the AMUSE/schedule-free `X/Y/Z` bookkeeping and updates parameters
directly with a SODA anchor pull plus the same PMuonEq -> GramNS -> NorMuon
matrix direction. In the matched run this direct path was much stronger than
schedule-free EquiMuse, so treat it as the current recommended recipe to
replicate next.

## Files

- `equimuse_normuon.py`: schedule-free standalone optimizer, no project-local
  imports.
- `direct_soda_pmuoneq_normuon.py`: direct SODA-PMuonEq-NorMuon standalone
  optimizer with row+aspect defaults.
- `golden_soda_pmuoneq_normuon.py`: final standalone library containing only
  the best-result algorithm path, with no AMUSE/SF/MiMuon/aspect-toggle
  ablations.
- `GOLDEN_VALIDATION.md`: unit, parity, and full 1000-step ViT-5/CIFAR-10
  reproduction of the stored best result using the golden optimizer.
- `test_equimuse_normuon.py`: lightweight unit tests for the standalone file.
- `test_golden_soda_pmuoneq_normuon.py`: parity and checkpoint tests for the
  golden optimizer.
- `ALGORITHM_RESULTS.md`: exact winning CIFAR-10 recipe, command, protocol, and
  comparison table.
- `VALIDATION.md`: validation result from the source workstation.
- `RESULTS.md`: compact final readout with figure links.
- `results/tables/equimuse_best_vs_adamw_results.csv`: final 1000-step
  CIFAR-10 result summary.
- `results/tables/equimuse_best_vs_adamw_curves.csv`: 8-bin loss/accuracy
  curves for the final comparison.
- `results/tables/equimuse_feedback_final_results.csv`: peer-feedback
  comparison including aspect scaling and the direct no-AMUSE SODA path.
- `results/tables/equimuse_feedback_final_curves.csv`: corresponding 8-bin
  curves for the peer-feedback comparison.
- `results/figures/*.png`: validation-loss, train-loss, accuracy, and
  iteration-speed plots.

## Minimal EquiMuse Usage

```python
from equimuse_normuon import EquiMuseNorMuon, build_equimuse_normuon_param_groups

groups = build_equimuse_normuon_param_groups(
    model.named_parameters(),
    lr=1e-2,
    matrix_lr=1e-2,
    weight_decay=0.05,
    pmuoneq_row_gamma=0.15,
    pmuoneq_col_gamma=0.15,
    normuon_beta2=0.9,
)
optimizer = EquiMuseNorMuon(groups, warmup_steps=50)

optimizer.train()
for x, y in loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

optimizer.eval()
```

The optimizer has schedule-free train/eval weights. Call `optimizer.train()`
before training steps and `optimizer.eval()` before validation/checkpointing if
you want averaged/eval weights. Both methods return `self`, so chained helper
code such as `optimizer.eval(); validate(...)` remains compatible.

## Minimal Direct SODA Usage

```python
from direct_soda_pmuoneq_normuon import (
    SodaPmuonEqNorMuon,
    build_soda_pmuoneq_normuon_param_groups,
)

groups = build_soda_pmuoneq_normuon_param_groups(
    model.named_parameters(),
    matrix_lr=8e-3,
    adam_lr=8e-4,
    row_gamma=0.35,
    col_gamma=0.05,
    normuon_beta2=0.93,
    normuon_aspect_scale=True,
)
optimizer = SodaPmuonEqNorMuon(groups, warmup_steps=10)
```

This direct optimizer has no schedule-free mode swap; `train()` and `eval()` are
no-op compatibility methods.

## Minimal Golden Usage

```python
from golden_soda_pmuoneq_normuon import GoldenOptimizer, build_param_groups

groups = build_param_groups(model.named_parameters())
optimizer = GoldenOptimizer(groups)
```

The golden file fixes the algorithm to the measured best-result path:
row+column PMuonEq, GramNS, row-wise NorMuon, and the post-NorMuon aspect
multiplier. It keeps only numeric hyperparameters such as LR, momentum, and
PMuonEq/NorMuon EMA strengths.

## Checkpoint And Mode Semantics

Model parameters hold different sequences depending on optimizer mode:

- train mode: gradient-evaluation/interpolation weights `Y`
- eval mode: averaged/returned weights `X`
- optimizer state: fast weights `Z` and SODA anchors

The simplest checkpoint recipe is:

```python
optimizer.eval()
torch.save(
    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    checkpoint_path,
)
optimizer.train()
```

This stores averaged weights in the model checkpoint and fast weights in the
optimizer state. Checkpointing in train mode is also supported because
`optimizer.state_dict()` records `train_mode`, but restore both model and
optimizer state before switching modes again.

`build_equimuse_normuon_param_groups` returns fallback parameters first and
matrix parameters second. If a training harness logs only
`optimizer.param_groups[0]["lr"]`, it will show the fallback LR. Prefer logging
per-group LR with the `use_muon` flag or `optimizer.last_stats`.

## Starting Hyperparameters

For ViT/CIFAR-style runs:

```text
lr = matrix_lr = 1e-2
weight_decay = 0.05
beta1 = 0.6
rho = 0.5
warmup_steps = 50
momentum = 0.95
pmuoneq_beta = 0.95
pmuoneq_row_gamma = 0.15
pmuoneq_col_gamma = 0.15
normuon_beta2 = 0.9
normuon_orientation = "row"
ns_steps = 5
ns_dtype = "float16"
```

SODA anchoring replaces decoupled weight decay while the anchor is active for an
anchored parameter group. In this fixed recipe `soda_anchor_scale > 0`, so
group `weight_decay` mostly affects only steps before SODA activates, for
example during `soda_warmup_steps`. Tune matrix LR and PMuonEq/NorMuon
parameters before treating matrix weight decay as an ordinary AdamW-style
control.

Tune in this order:

1. Matrix learning rate: `{3e-3, 1e-2, 3e-2}`.
2. PMuonEq row/column gamma: `{(0.15, 0.15), (0.30, 0.10), (0.10, 0.30)}`.
3. NorMuon beta2: `{0.90, 0.95, 0.98}`.
4. NorMuon orientation: keep `"row"` for the selected recipe, or test `"auto"`
   as a controlled ablation that uses column statistics for wide matrices.
5. Warmup steps.
6. Weight decay.

## Latest Result

After peer feedback, the fixed ViT-5-Small/CIFAR-10 img224 harness was rerun on
all 4 visible GPUs with DDP, seed `67890`, 1000 optimizer steps, and 8
validation bins. These headline numbers are CIFAR-10 results. Older CIFAR-100
notes in source docstrings refer to separate historical checks and are not the
headline comparison for this folder.

| method | acc@1 | val loss | train loss | samples/s |
| --- | ---: | ---: | ---: | ---: |
| Direct SODA-PMuonEq-NorMuon + aspect | 76.69 | 0.7269 | 1.2382 | 4009 |
| Direct SODA-PMuonEq-NorMuon | 75.16 | 0.7740 | 1.2579 | 3949 |
| EquiMuse-NorMuon row | 65.96 | 1.0144 | 1.4699 | 4053 |
| EquiMuse-NorMuon row + aspect | 65.85 | 1.0019 | 1.4519 | 4027 |
| EquiMuse-NorMuon auto + aspect | 65.72 | 1.0140 | 1.4547 | 4021 |
| EquiMuse-NorMuon peer-gamma + aspect | 65.11 | 1.0302 | 1.4516 | 4030 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4280 |

The direct no-AMUSE SODA-PMuonEq-NorMuon path is the current quality winner in
this single-seed run: +13.63 accuracy points and -0.3622 validation loss versus
AdamW. Within schedule-free EquiMuse, aspect scaling improved validation loss
but not accuracy; row/no-aspect remained the best EquiMuse accuracy. The direct
path should be replicated across seeds before treating the large delta as final.

For the exact winning recipe, use `ALGORITHM_RESULTS.md` as the source of truth.

Artifacts:

- `results/tables/equimuse_feedback_final_results.csv`
- `results/tables/equimuse_feedback_final_curves.csv`
- `results/figures/equimuse_feedback_val_loss_curve.png`
- `results/figures/equimuse_feedback_accuracy_curve.png`
- `results/figures/equimuse_feedback_iteration_speed.png`
- `results/figures/equimuse_best_vs_adamw_val_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_train_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_accuracy_curve.png`
- `results/figures/equimuse_best_vs_adamw_iteration_speed.png`

## DDP Notes

The optimizer is DDP-safe because it only consumes already all-reduced local
parameter gradients. Same-shaped matrix updates are batched for PMuonEq,
GramNS, and NorMuon; the fallback path uses `torch._foreach_*` operations when
available.

## Workspace Notes

Peer-review notes that were previously stored as `notes_*.md` have been removed
from this worker folder. The persistent summary is now in `README.md`,
`RESULTS.md`, `VALIDATION.md`, and the CSV/PNG artifacts under `results/`.
