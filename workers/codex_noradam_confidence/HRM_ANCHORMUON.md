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

The existing HRM cosine/warmup trainer schedule still owns learning rates. The
training loop writes the scheduled LR into every optimizer param group each
step, with an optional fallback scale for scalar/vector parameters.

The `adam_atan2` import is lazy now, so `optimizer=anchormuon` does not require
the AdamATan2 backend to import cleanly.

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
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  anchormuon_row_gamma=0.35 \
  anchormuon_pmuoneq_beta=0.90 \
  anchormuon_normuon_beta2=0.93
```

## Caveats

- This is smoke-tested, not tuned on Sudoku/Maze yet.
- AnchorMuon replaces ordinary model weight decay with its SODA anchor path.
  HRM's `weight_decay` is ignored by AnchorMuon for model weights, but
  `puzzle_emb_weight_decay` still applies to sparse puzzle embeddings.
- Full HRM runs should install the proper FlashAttention package for speed.
  `HRM_ALLOW_SDPA_FALLBACK=1` is only for explicit local validation when
  FlashAttention is unavailable.
- The baseline AdamATan2 package import currently fails in this venv because
  `adam_atan2_backend` is missing. The lazy import lets AnchorMuon runs proceed,
  but AdamATan2 baseline reproduction needs that dependency fixed separately.
