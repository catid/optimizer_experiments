# HRM AnchorMuon Integration

Target repo: `https://github.com/sapientinc/HRM`

Cloned commit used for this patch: `ac15626f8db096a63c775b84c9dc868776a6feda`

## What Changed

The patch in `hrm_anchormuon.patch` adds an optional model optimizer path in
`pretrain.py` and declares the corresponding Hydra keys in
`config/cfg_pretrain.yaml`:

```bash
optimizer=anchormuon
```

HRM's sparse puzzle embedding optimizer is left unchanged:

```python
CastedSparseEmbeddingSignSGD_Distributed
```

The normal model optimizer changes from `AdamATan2(model.parameters())` to
root `optimizer.py`:

```python
AnchorMuon(
    model.named_parameters(),
    lr=0,
    fallback_lr=0,
    row_gamma=config.anchormuon_row_gamma,
    pmuoneq_beta=config.anchormuon_pmuoneq_beta,
    normuon_beta2=config.anchormuon_normuon_beta2,
    soda_lambda_scale=config.anchormuon_soda_lambda_scale,
    soda_lambda_power=config.anchormuon_soda_lambda_power,
    min_matrix_dim=config.anchormuon_min_matrix_dim,
)
```

The HRM trainer still owns learning rates. The patch adds a trainer-side
`lr_schedule` switch with `cosine`, `constant`, and `wsd` options, then writes
the scheduled LR into every optimizer param group each step. AnchorMuon does
not contain an LR scheduler.

The `adam_atan2` import is lazy now, so `optimizer=anchormuon` does not require
the AdamATan2 backend to import cleanly. If the Python package is installed but
the fused `adam_atan2_backend` extension is unavailable, the patch uses
`AdamATan2Reference`, an exact torch implementation of the AdamATan2 update
equation. That fallback is suitable for quality comparisons, but it is not a
fused speed baseline.

The patch also adds an explicit local-validation escape hatch for environments
without FlashAttention:

```bash
HRM_ALLOW_SDPA_FALLBACK=1
```

Without that environment variable, HRM keeps its original behavior and raises
if neither FlashAttention 3 nor FlashAttention 2 can be imported.

## Config Knobs

Defaults are aligned with the current AnchorMuon CIFAR-10 recipe, but HRM will
need real tuning because its baseline LR is much smaller.

| Config | Default | Meaning |
|---|---:|---|
| `optimizer` | `adam_atan2` | Set to `anchormuon` to use AnchorMuon for model weights. |
| `lr_schedule` | `cosine` | Trainer LR schedule: `cosine`, `constant`, or `wsd`. |
| `wsd_decay_frac` | `0.2` | Fraction of total steps used for the final WSD decay phase. |
| `anchormuon_row_gamma` | `0.35` | Row-only PMuonEq strength before GramNS. |
| `anchormuon_pmuoneq_beta` | `0.90` | Row gradient-power EMA. |
| `anchormuon_normuon_beta2` | `0.93` | NorMuon post-Gram row second moment. |
| `anchormuon_fallback_lr_scale` | `1.0` | LR multiplier for scalar/vector fallback params. |
| `anchormuon_soda_lambda_scale` | `1.0` | SODA anchor schedule scale. |
| `anchormuon_soda_lambda_power` | `1.0` | SODA anchor schedule power. |
| `anchormuon_min_matrix_dim` | `2` | Minimum effective matrix side for spectral path. |

## Reproduce The Local Smoke

From this monorepo root:

```bash
mkdir -p workers/codex_noradam_confidence/external
git clone https://github.com/sapientinc/HRM.git workers/codex_noradam_confidence/external/HRM
git -C workers/codex_noradam_confidence/external/HRM checkout ac15626f8db096a63c775b84c9dc868776a6feda
git -C workers/codex_noradam_confidence/external/HRM apply ../../hrm_anchormuon.patch

DISABLE_COMPILE=1 HRM_ALLOW_SDPA_FALLBACK=1 \
  .venv/bin/python workers/codex_noradam_confidence/hrm_anchormuon_smoke.py
```

Validated output in this workspace:

```text
HRM AnchorMuon smoke passed
optimizers ['CastedSparseEmbeddingSignSGD_Distributed', 'AnchorMuon']
anchormuon_stats {'matrix_count': 11.0, 'fallback_count': 1.0, 'group_count': 2.0, 'soda_weight': 0.5}
```

I also ran a tiny two-GPU `torchrun` path through HRM's real `pretrain.py` on a
16-example Sudoku training subset and 32-example trimmed test subset, with a
reduced 132k-parameter HRM config:

```bash
OMP_NUM_THREADS=16 WANDB_MODE=offline DISABLE_COMPILE=1 HRM_ALLOW_SDPA_FALLBACK=1 \
  /home/catid/screen/.venv/bin/python -m torch.distributed.run --nproc-per-node 2 pretrain.py \
  data_path=data/sudoku-smoke-16 \
  optimizer=anchormuon \
  epochs=2 \
  eval_interval=1 \
  checkpoint_every_eval=False \
  global_batch_size=8 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  lr_warmup_steps=0 \
  arch.H_layers=1 \
  arch.L_layers=1 \
  arch.hidden_size=64 \
  arch.num_heads=4 \
  arch.expansion=4 \
  arch.H_cycles=1 \
  arch.L_cycles=1 \
  arch.halt_max_steps=1 \
  arch.halt_exploration_prob=0.0 \
  arch.puzzle_emb_ndim=64
```

That run completed 4 optimizer steps across both visible GPUs. This is only an
integration check, not a quality comparison.

## WSD Sudoku Comparison

I compared AnchorMuon against the AdamATan2 baseline path on the compact local
Sudoku-Extreme no-augmentation split:

- Data: `data/sudoku-extreme-1k-noaug-test1k`
- Model: HRM default `HierarchicalReasoningModel_ACTV1`, 27.3M parameters
- GPUs: 2 visible CUDA GPUs via `torch.distributed.run`
- Batch: `global_batch_size=384`
- Schedule: `lr_schedule=wsd`, `lr_warmup_steps=50`, `lr_min_ratio=0.1`,
  `wsd_decay_frac=0.2`
- Decay settings: `weight_decay=1.0`, `puzzle_emb_weight_decay=1.0`
- HPO selector: 300 epochs; final replay: 1000 epochs

The AdamATan2 package in this venv lacked `adam_atan2_backend`, so the baseline
used the exact torch `AdamATan2Reference` fallback. The fused package may be
faster, but the update equation is the same.

### 300-Epoch WSD LR Sweep

| Optimizer | LR | Token Acc | Exact Acc | LM Loss | Notes |
|---|---:|---:|---:|---:|---|
| AnchorMuon | `1.25e-5` | 21.20% | 0.00% | 2.1727 | Too small. |
| AnchorMuon | `2.5e-5` | 40.84% | 0.00% | 1.6619 | Improved but behind. |
| AnchorMuon | `5e-5` | 43.16% | 0.00% | 1.4845 | Best AnchorMuon by LM loss. |
| AnchorMuon | `1e-4` | 43.18% | 0.00% | 1.6946 | Similar token acc, worse loss. |
| AnchorMuon | `2e-4` | 43.12% | 0.00% | 2.4483 | Too high. |
| AdamATan2Reference | `6.25e-6` | 43.09% | 0.00% | 1.4580 | Lower bracket. |
| AdamATan2Reference | `1.25e-5` | 44.50% | 0.00% | 1.4088 | Best baseline by LM loss. |
| AdamATan2Reference | `2.5e-5` | 45.52% | 0.00% | 1.4295 | Best baseline by token acc. |
| AdamATan2Reference | `5e-5` | 45.03% | 0.00% | 1.8416 | Worse loss. |
| AdamATan2Reference | `1e-4` | 43.79% | 0.00% | 2.6899 | Halt quality degraded. |
| AdamATan2Reference | `2e-4` | 44.23% | 0.00% | 2.8339 | Too high. |

### 1000-Epoch Final Replay

| Optimizer | Selected By | LR | Token Acc | Exact Acc | LM Loss | Wall Time | Iter/s |
|---|---|---:|---:|---:|---:|---:|---:|
| AdamATan2Reference | 300-epoch loss | `1.25e-5` | 44.67% | 0.00% | 2.0961 | 188s | 13.85 |
| AnchorMuon | 300-epoch loss | `5e-5` | 42.77% | 0.00% | 2.6036 | 196s | 13.29 |
| AdamATan2Reference | 300-epoch token acc | `2.5e-5` | 44.50% | 0.00% | 2.8958 | 189s | 13.78 |

Conclusion for this specific setup: tuned AdamATan2Reference + WSD is the better
baseline. AnchorMuon did not improve HRM Sudoku validation quality here and was
about 4% slower than the torch AdamATan2 fallback. None of these short local
no-augmentation runs solved exact puzzles, so this is an optimizer sanity
comparison, not a reproduction of HRM's reported Sudoku result.

Full logs and commands are under
`workers/codex_noradam_confidence/results/hrm_wsd_baseline_compare_20260529/`.

## Example Training Command

Use all visible GPUs with `torchrun`. This leaves HRM's puzzle embedding
optimizer and cosine/warmup schedule intact, and only swaps the model optimizer:

```bash
NUM_GPUS=$(.venv/bin/python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

cd workers/codex_noradam_confidence/external/HRM
OMP_NUM_THREADS=16 WANDB_MODE=offline HRM_ALLOW_SDPA_FALLBACK=1 \
torchrun --nproc-per-node "$NUM_GPUS" pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  optimizer=anchormuon \
  epochs=20000 \
  eval_interval=2000 \
  global_batch_size=384 \
  lr=5e-5 \
  puzzle_emb_lr=1e-4 \
  lr_schedule=wsd \
  lr_warmup_steps=50 \
  lr_min_ratio=0.1 \
  wsd_decay_frac=0.2 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  anchormuon_row_gamma=0.35 \
  anchormuon_pmuoneq_beta=0.90 \
  anchormuon_normuon_beta2=0.93
```

## Caveats

- This is smoke-tested and locally tuned on a no-augmentation Sudoku split, but
  not validated on the full augmented HRM Sudoku/Maze recipes.
- AnchorMuon replaces ordinary model weight decay with its SODA anchor path.
  HRM's `weight_decay` is ignored by AnchorMuon for model weights, but
  `puzzle_emb_weight_decay` still applies to sparse puzzle embeddings.
- Full HRM runs should install the proper FlashAttention package for speed.
  `HRM_ALLOW_SDPA_FALLBACK=1` is only for explicit local validation when
  FlashAttention is unavailable.
- The baseline AdamATan2 package import currently fails in this venv because
  `adam_atan2_backend` is missing. The patch falls back to a torch reference
  implementation for quality comparisons, but fused speed should be measured
  after installing the real backend.
