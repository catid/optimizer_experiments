# EquiMuse-NorMuon Standalone Optimizer

This folder contains a single-file PyTorch optimizer recipe extracted from the
Attractor optimizer experiments:

```text
SODA-AMUSE + PMuonEq + Gram Newton-Schulz + NorMuon
```

The implementation is intentionally not an ablation framework. It keeps the
selected recipe fixed and exposes only tuning hyperparameters.

## Files

- `equimuse_normuon.py`: standalone optimizer, no project-local imports.
- `test_equimuse_normuon.py`: lightweight unit tests for the standalone file.
- `VALIDATION.md`: validation result from the source workstation.
- `RESULTS.md`: compact final readout with figure links.
- `results/tables/equimuse_best_vs_adamw_results.csv`: final 1000-step
  CIFAR-10 result summary.
- `results/tables/equimuse_best_vs_adamw_curves.csv`: 8-bin loss/accuracy
  curves for the final comparison.
- `results/figures/*.png`: validation-loss, train-loss, accuracy, and
  iteration-speed plots.

## Minimal Usage

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

On the source workstation, the fixed ViT-5-Small/CIFAR-10 img224 harness used
all 4 visible GPUs with DDP, seed `67890`, 1000 optimizer steps, and 8
validation bins.

| method | acc@1 | val loss | train loss | samples/s |
| --- | ---: | ---: | ---: | ---: |
| EquiMuse-NorMuon row | 65.96 | 1.0145 | 1.4699 | 3918 |
| EquiMuse-NorMuon auto | 64.57 | 1.0387 | 1.4817 | 4214 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4167 |

The row-wise selected recipe is the current quality winner in this single-seed
run: +2.90 accuracy points and -0.0746 validation loss versus AdamW. The
orientation-aware `"auto"` ablation is faster than row-wise NorMuon on this
run, but gave up quality.

Artifacts:

- `results/figures/equimuse_best_vs_adamw_val_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_train_loss_curve.png`
- `results/figures/equimuse_best_vs_adamw_accuracy_curve.png`
- `results/figures/equimuse_best_vs_adamw_iteration_speed.png`

## DDP Notes

The optimizer is DDP-safe because it only consumes already all-reduced local
parameter gradients. Same-shaped matrix updates are batched for PMuonEq,
GramNS, and NorMuon; the fallback path uses `torch._foreach_*` operations when
available.
