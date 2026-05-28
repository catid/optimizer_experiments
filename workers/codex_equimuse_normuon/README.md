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
4. Warmup steps.
5. Weight decay.

## DDP Notes

The optimizer is DDP-safe because it only consumes already all-reduced local
parameter gradients. Same-shaped matrix updates are batched for PMuonEq,
GramNS, and NorMuon; the fallback path uses `torch._foreach_*` operations when
available.
