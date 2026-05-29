# Codex SodaMuseEq ViT-5 Experiments

This folder is intentionally self-contained so it does not conflict with other
workers in the monorepo.

Contents:

- `sodamuseeq.py`: standalone optimizer module for reuse in other projects.
- `golden_soda_pmuoneq_normuon.py`: fixed final optimizer library for the
  current repo-wide LM candidate.
- `vit5/`: compact ViT-5 CIFAR-10 experiment fork with optimizer ablations.
- `vit5/optim_sodamuseeq.py`: ViT-integrated optimizer implementation.
- `vit5/experiments/sodamuseeq_ablation.py`: broad SodaMuseEq ablation runner.
- `vit5/experiments/focused_optimizer_confidence.py`: focused comparison runner
  for AdamW, NorMuon+BaseGram, SODA+PMuonEq+Gram, and the peer-suggested
  SODA+PMuonEq+Gram+NorMuon row+aspect recipe.
- `vit5/tests/`: unit and DDP smoke tests for optimizer modes.
- `results/focused_optimizer_confidence/`: committed result tables and plots
  for the completed focused comparison.

The optimizer supports optional switches for:

- SODA anchor correction.
- AMUSE schedule-free fast/eval iterate bookkeeping.
- PMuonEq row/column EMA equilibration before Gram-Newton-Schulz projection.
- MiMuon branch selection.
- NorMuon row second-moment normalization after projection.
- Optional NorMuon orientation mode and tall-matrix aspect scaling, ported from
  `workers/codex_soda_pmuoneq_normuon`.

## Golden Optimizer Library

Use `golden_soda_pmuoneq_normuon.py` when you want the fixed version of my best
reported optimizer without ablation switches:

```text
SODA + row-only PMuonEq + Gram/Newton-Schulz + row-wise NorMuon with aspect
AMUSE off / MiMuon off
```

This is the exact row+aspect recipe that produced my best compact
ViT-5/CIFAR-10 result. The latest cross-worker LM recommendation still keeps
aspect as an ablation because one proper-split worker result slightly preferred
no-aspect on official test accuracy. This golden file is therefore the
reproducible version of my best result, not a claim that aspect is settled for
language modeling.

Typical usage:

```python
from golden_soda_pmuoneq_normuon import (
    GoldenSodaPmuonEqNorMuon,
    build_golden_param_groups,
)

optimizer = GoldenSodaPmuonEqNorMuon(
    build_golden_param_groups(
        model,
        lr=0.012,
        weight_decay=0.0,
    ),
    warmup_steps=500,
    soda_warmup_steps=500,
)
```

The named grouping helper keeps embeddings, tied LM heads, norms, biases, and
small non-matrix tensors in the fallback path by default. Hidden 2D/4D matrix
weights go through the spectral path. The golden tests verify deterministic
update parity against the flexible `SodaMuseEq` implementation configured with
the best row+aspect flags.

## Current Algorithm

The best current recipe is a no-AMUSE matrix optimizer:

```text
SODA + PMuonEq + Gram-Newton-Schulz + NorMuon row+aspect
```

For matrix parameters, the update is:

1. Keep hidden 2D/4D matrix weights in the matrix optimizer path. Biases,
   norms, embeddings, and heads use the fallback RMS/AdamW-style path.
2. Apply SODA as an anchor correction after warmup, replacing ordinary weight
   decay in the selected recipe.
3. Build a Nesterov-style momentum matrix from the current gradient.
4. Apply PMuonEq before projection with cheap row/column EMA gradient-power
   scales. The winning setting uses row scaling only:
   `pmuon_row_gamma=0.15`, `pmuon_col_gamma=0.0`.
5. Apply Gram/Newton-Schulz projection to get a spectral matrix update.
6. Apply NorMuon after projection with row second-moment normalization and the
   tall-matrix aspect multiplier enabled.
7. Apply the final learned update directly to the parameter.

The important boundary is that PMuonEq acts before Gram projection, while
NorMuon row+aspect acts after projection as a final learned-update rescale.
AMUSE remains implemented and test-covered, but it was not part of the best
compact ViT-5/CIFAR-10 result.

## Best Version And Setup

The best specific version found in this workspace is:

```text
SODA+PMuonEq+Gram+NorMuon row+aspect
```

It is the `soda_pmuoneq_normuon_aspect` family in the focused runner. The
winning run disables AMUSE and MiMuon; it uses SODA, row-only PMuonEq before
Gram projection, and NorMuon row normalization with the tall-matrix aspect
multiplier after projection.

Model and data:

| Item | Value |
| --- | --- |
| Model | compact ViT-5 CIFAR model |
| Trainable parameters | 2,691,274 |
| Input | CIFAR images, 32x32 |
| Patch size | 4 |
| Embedding dim | 192 |
| Depth | 6 |
| Heads | 3 |
| MLP ratio | 4 |
| Extras | RMSNorm, RoPE, q/k norm, layer scale, 4 register tokens |
| Dataset | CIFAR-10 |
| Split | 45k train / 5k validation / 10k test |
| Split seed | 12345 |
| Train augmentation | random crop with padding 4, random horizontal flip, normalize |

Training and tuning:

| Item | Value |
| --- | --- |
| HPO budget | 1k-step sweep, seed 0 |
| Top-selection budget | 3k-step rerun of top candidates, seed 0 |
| Final budget | 10k steps, seeds 0/1/2 |
| Final batch size | 256 |
| Eval batch size | 1024 |
| Data workers | 8 |
| Eval bins | 8 |
| Precision | CUDA BF16 autocast |
| Scheduling | one trial per visible GPU |

Best optimizer parameters:

| Parameter | Value |
| --- | --- |
| `lr` | `0.012` |
| `weight_decay` | `0.0` |
| `momentum` | `0.95` |
| `beta1` | `0.6` |
| `beta2` | `0.999` |
| `rho` | `0.8` |
| `warmup_steps` | `500` |
| `soda_warmup_steps` | `500` |
| `use_soda` | `True` |
| `use_amuse` | `False` |
| `use_pmuoneq` | `True` |
| `use_gram` | `True` |
| `use_normuon` | `True` |
| `normuon_mode` | `row` |
| `normuon_aspect_scale` | `True` |
| `pmuon_beta` | `0.90` |
| `pmuon_row_gamma` | `0.15` |
| `pmuon_col_gamma` | `0.0` |
| `normuon_beta2` | `0.90` |

Final 10k, 3-seed result:

| Metric | Value |
| --- | ---: |
| Best validation loss | `0.4079 +/- 0.0090` |
| Best validation accuracy | `87.15% +/- 0.08` |
| Test accuracy | `87.09% +/- 0.15` |
| Steps/sec | `42.84 +/- 0.06` |

This is the version to reproduce first. The closest competitor was
`SODA+PMuonEq+Gram` at `0.4210 +/- 0.0097` validation loss and
`86.90% +/- 0.35` test accuracy; AdamW was faster but substantially worse at
`0.6030 +/- 0.0167` validation loss and `81.27% +/- 0.74` test accuracy.

## Validation

Validated in the source workspace before packaging:

```bash
python -m py_compile \
  sodamuseeq.py \
  vit5/optim_sodamuseeq.py \
  vit5/main.py \
  vit5/engine.py \
  vit5/experiments/sodamuseeq_ablation.py \
  vit5/experiments/focused_optimizer_confidence.py

python -m pytest \
  tests/test_golden_soda_pmuoneq_normuon.py \
  vit5/tests/test_sodamuseeq_modes.py \
  tests/test_sodamuseeq_standalone.py

torchrun --standalone --nproc-per-node=2 \
  vit5/tests/ddp_sodamuseeq_smoke.py
```

The DDP smoke checks train weights, eval weights, restored train weights, and
optimizer tensor state across ranks.

The tests include:

- golden optimizer update parity against flexible `SodaMuseEq` configured with
  the best row+aspect flags,
- all ablation modes run without NaNs,
- named grouping keeps ViT `patch_embed` matrices in the matrix optimizer while
  keeping heads/norms in fallback,
- NorMuon row/aspect and orientation controls have shape and update tests,
- batch projection parity against individual projection,
- same-shape matrix bucket parity,
- repeated schedule-free `train()` / `eval()` roundtrips,
- train-mode and eval-mode checkpoint resume parity,
- standalone optimizer grouping and resume coverage.

## Checkpoint Recipe

For schedule-free/AMUSE modes, call `optimizer.eval()` before validation if you
want the averaged/eval weights, then call `optimizer.train()` before resuming
training. Checkpoints are safe in either mode as long as both the model state and
optimizer state are saved together. The optimizer stores its train/eval mode and
SODA step counter in `state_dict()` under `sodamuseeq_extra`.

SODA is configured to replace ordinary weight decay by default. When
`soda_replaces_weight_decay=True`, matrix and fallback weight decay are disabled
inside the optimizer and the anchor correction supplies the regularization path.

## Focused Confidence Runner

Run the focused comparison:

```bash
uv run python vit5/experiments/focused_optimizer_confidence.py \
  --out-dir vit5/runs/focused_optimizer_confidence \
  --data-dir data/cifar10 \
  --hpo-steps 1000 \
  --top-steps 3000 \
  --final-steps 10000 \
  --long-steps 20000 \
  --top-n 3 \
  --final-seeds 0,1,2 \
  --batch-size 256 \
  --eval-batch-size 1024 \
  --workers 8
```

The runner schedules one trial per visible GPU. It compares:

- AdamW baseline.
- NorMuon+BaseGram: no SODA, no AMUSE, no PMuonEq.
- SODA+PMuonEq+Gram.
- SODA+PMuonEq+Gram+NorMuon row+aspect.

## Focused Results

Peer-feedback rerun, after adding row+aspect NorMuon support:

| Variant | Budget | Val loss | Val acc | Test acc | Steps/sec |
| --- | --- | ---: | ---: | ---: | ---: |
| SODA+PMuonEq+Gram+NorMuon row+aspect | 10k, 3 seeds | 0.4079 +/- 0.0090 | 87.15% +/- 0.08 | 87.09% +/- 0.15 | 42.84 +/- 0.06 |
| SODA+PMuonEq+Gram | 10k, 3 seeds | 0.4210 +/- 0.0097 | 87.19% +/- 0.24 | 86.90% +/- 0.35 | 43.05 +/- 0.27 |
| NorMuon+BaseGram | 10k, 3 seeds | 0.5069 +/- 0.0100 | 86.21% +/- 0.49 | 86.05% +/- 0.36 | 47.44 +/- 0.34 |
| AdamW | 10k, 3 seeds | 0.6030 +/- 0.0167 | 81.47% +/- 0.75 | 81.27% +/- 0.74 | 58.41 +/- 0.24 |

Best row+aspect config:

```text
use_amuse = False
use_soda = True
use_pmuoneq = True
use_gram = True
use_normuon = True
normuon_mode = row
normuon_aspect_scale = True
lr = 0.012
pmuon_beta = 0.90
pmuon_row_gamma = 0.15
pmuon_col_gamma = 0.0
normuon_beta2 = 0.90
weight_decay = 0.0
```

Artifacts: `results/focused_peer_aspect/`.

Previous focused confidence pass:

The 3-seed 10k and seed-0 20k focused confidence pass found:

| Variant | Budget | Val loss | Val acc | Test acc | Steps/sec |
| --- | --- | ---: | ---: | ---: | ---: |
| SODA+PMuonEq+Gram | 10k, 3 seeds | 0.4210 +/- 0.0097 | 87.19% +/- 0.24 | 86.90% +/- 0.35 | 42.84 +/- 0.20 |
| NorMuon+BaseGram | 10k, 3 seeds | 0.5069 +/- 0.0100 | 86.21% +/- 0.49 | 86.05% +/- 0.36 | 47.93 +/- 0.67 |
| AdamW | 10k, 3 seeds | 0.6030 +/- 0.0167 | 81.47% +/- 0.75 | 81.27% +/- 0.74 | 58.46 +/- 0.42 |
| SODA+PMuonEq+Gram | 20k, seed 0 | 0.4151 | 87.58% | 87.77% | 42.94 |
| NorMuon+BaseGram | 20k, seed 0 | 0.5001 | 86.10% | 86.42% | 49.22 |
| AdamW | 20k, seed 0 | 0.5967 | 82.46% | 82.25% | 59.19 |

The current best recipe is SODA+PMuonEq+Gram+NorMuon row+aspect. The gain over
SODA+PMuonEq+Gram is modest but repeatable in the 10k 3-seed run:
`0.4079` versus `0.4210` mean best validation loss. Validation accuracy is
essentially tied, while test accuracy improves slightly. AdamW remains much
faster per step but materially worse on loss and accuracy.

Committed result artifacts:

- Summary report: `results/focused_optimizer_confidence/experimental_results.md`
- Validation loss, best optimizer vs AdamW:
  `results/focused_optimizer_confidence/val_loss_best_vs_adamw.png`
- Train interval loss, best optimizer vs AdamW:
  `results/focused_optimizer_confidence/train_loss_best_vs_adamw.png`
- Validation accuracy, best optimizer vs AdamW:
  `results/focused_optimizer_confidence/val_acc_best_vs_adamw.png`
- Final test accuracy bar chart:
  `results/focused_optimizer_confidence/final_test_accuracy_bar.png`
- Iteration speed bar chart:
  `results/focused_optimizer_confidence/iteration_speed_steps_per_sec.png`
- Raw final tables:
  `results/focused_optimizer_confidence/final10k_all_runs.csv` and
  `results/focused_optimizer_confidence/long20k_all_runs.csv`

The optimizer implementation used for the completed result is
`vit5/optim_sodamuseeq.py`; the current branch adds checkpoint/test hardening on
top of that code path without changing the training update rule.
