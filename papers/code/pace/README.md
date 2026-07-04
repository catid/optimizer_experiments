# PACE

A drop-in PyTorch optimizer for iterate-averaged language-model training, and the code
companion to the paper:

> **Training for the Model You Return: Improving Optimization for Iterate-Averaged
> Language Models** — Au & Block, 2026 ([arXiv:2606.25086](https://arxiv.org/abs/2606.25086))

**PACE** is AdamW plus a small, per-coordinate pullback toward an exponential moving
average (EMA) of recent weights. When the model you ultimately return is the EMA of the
iterates (a common, robust choice), training that explicitly accounts for this averaging
reaches a better returned model. In the paper PACE improves on AdamW and EMA-evaluated
AdamW for 1–2B-parameter fine-tuning and GPT-2 / FineWeb pretraining, across a range of
learning rates and schedules.

## Installation

```bash
pip install pace-optimizer          # the optimizer only (depends on PyTorch)
```

To also run training and reproduce the paper, install the extras with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra experiments         # adds transformers, datasets, hydra, etc.
```

## Using the optimizer

`PACE` is a standard `torch.optim.Optimizer`:

```python
from pace import PACE

opt = PACE(model.parameters(), lr=1e-3,
           lambda_pullback=0.15,    # c: pullback strength
           ema_kappa=0.2,           # kappa: EMA decay exponent
           use_ema_eval=True)       # evaluate and return the EMA weights

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad()

opt.eval()       # swap in the EMA weights for validation
validate(model)
opt.train()      # restore the training weights
```

### Update rule

Each step `t`, per coordinate `i`:

```
theta'        = theta_t - AdamW_step(theta_t)                       # 1. AdamW step
decay_t       = max((1 + gamma * max(0, t - rho))^(-kappa), 1e-4)   #    rho=0 -> (1+t)^(-kappa)
lambda_{t,i}  = clamp( lr * c * decay_t / (sqrt(v_hat_i) + eps),  max=1 )
theta_{t+1}   = theta' + lambda_{t,i} * (ema_{t-1} - theta_t)       # 2. pullback toward the EMA
ema_t         = (1 - decay_t) * ema_{t-1} + decay_t * theta_{t+1}   # 3. EMA update (every uf steps)
```

`v_hat` is Adam's bias-corrected second moment. The pullback (step 2) is applied every
step; only the EMA update (step 3) honours `ema_update_freq`.

### Recovering the baselines

Two arguments place AdamW, EMA-evaluated AdamW, and PACE on one axis:

| `lambda_pullback` (c) | `use_ema_eval` | method |
|-----------------------|----------------|--------|
| `0` | `False` | plain AdamW (matches `torch.optim.AdamW` to floating-point ordering) |
| `0` | `True`  | EMA baseline (AdamW training, EMA returned) |
| `> 0` | `True` | PACE |

### Parameters

| argument | symbol | default | description |
|----------|--------|---------|-------------|
| `params` | – | – | parameters or parameter groups (standard `torch.optim` API) |
| `lr` | η | `1e-4` | AdamW learning rate |
| `betas` | β₁,β₂ | `(0.9, 0.999)` | AdamW moment decay rates |
| `eps` | ε | `1e-8` | AdamW epsilon (also the floor of the pullback denominator) |
| `weight_decay` | – | `0.01` | decoupled (AdamW) weight decay |
| `lambda_pullback` | c | `0.0` | pullback strength; `0` disables it (AdamW / EMA baseline) |
| `clamp_pullback` | – | `True` | clamp each per-coordinate gain to `≤ 1` (interpolate toward the EMA, never past it) |
| `beta_ema` | – | `0.01` | fixed EMA decay, used only when `ema_kappa is None` |
| `use_ema_eval` | – | `False` | if `True`, `eval()` returns the EMA weights and `train()` restores the iterate |
| `ema_kappa` | κ | `None` | exponent of the decaying EMA schedule; `None` uses the fixed `beta_ema` |
| `ema_rho` | ρ | `0.0` | steps held before the schedule decays; `0` gives the paper's `(1+t)^(-κ)` |
| `ema_gamma` | γ | `1.0` | schedule rate |
| `ema_update_freq` | uf | `1` | update the EMA every `uf` steps (the pullback is applied every step) |
| `log_stats` | – | `False` | record per-step telemetry (λ statistics, clip fraction, update norms) via `get_step_stats()`; off by default to avoid a per-step GPU synchronization |

## Training

Training is configured with [Hydra](https://hydra.cc). Config groups live in
`src/pace/helpers/conf/`: `model/`, `dataset/`, `optimizer/` (`adamw` | `ema` | `pace`), and
`schedule/` (`const` | `cos` | `wsd`). Prepare a tokenized dataset once, then train:

```bash
pace-data --out data/smoltalk --tokenizer HuggingFaceTB/SmolLM2-1.7B

pace-train model=smollm2_1.7b optimizer=pace schedule=const \
    optimizer.lr=1e-3 optimizer.lambda_pullback=0.15 optimizer.ema_kappa=0.2 \
    dataset.path=data/smoltalk
```

Each run writes `{config, summary, eval_logs, step_logs}` as JSON under `runs/`.
`summary.final_val_loss` is the reported metric (`final_train_loss` is recorded too).
Following the paper, `val_loss`/`ema_val_loss` are measured on the returned (EMA) weights
and `vanilla_val_loss` on the final iterate. W&B logging is off by default; training is a
single pass with no checkpoint/resume, so set `training.max_steps` accordingly.

### Data

`prepare_sft` formats each conversation in ChatML and trains only on the assistant
replies — system, user, and the role headers are masked out of the loss. Sequences are
padded to `max_seq_len` (1280 in the paper). It uses the whole dataset by default; set
`subset_ratio < 1` for a smaller seeded sample, or pass `revision=` to pin a dataset
version.

## Reproducing a figure number

To reproduce a figure, prepare the data once and run the figure's training runs:

```bash
pace-data --out data/smoltalk --tokenizer HuggingFaceTB/SmolLM2-135M

pace-figure fig=fig3_135m_grid                       # print the pace-train commands
pace-figure fig=fig3_135m_grid run=true data=data/smoltalk   # train them
```

Each run writes a result JSON under `runs/`; **`summary.final_val_loss` is the plotted
number.** `pace-figure fig=list` lists every figure, and `pace-figure fig=<id>` prints its
exact commands. Figure 1, for example, is these runs (one result number each):

| model | PACE lr | PACE c | PACE κ | baselines |
|-------|---------|--------|--------|-----------|
| SmolLM2-1.7B | 1e-3 | 0.15 | 0.2 | AdamW / EMA on cosine decay |
| Qwen3-1.7B | 5e-4 | 0.25 | 0.5 | AdamW / EMA on cosine decay |
| Gemma3-1B | 1e-3 | 0.15 | 0.5 | AdamW / EMA on cosine decay |

Two reference runs are committed under `tests/fixtures/results/`: the SmolLM2-135M Fig 3
cell (config, per-step train log, losses), which `tests/test_pipeline.py` reproduces
end-to-end, and the SmolLM2-1.7B Fig 1 headline run (config, losses).

## Tests

```bash
uv run pytest -q                    # quick mode (needs network + a GPU)
PACE_TEST_FULL=1 uv run pytest -q   # also reproduce the figure number (full run, hours)
```

**`tests/test_pipeline.py`** runs the pipeline by import: it walks the repo's code step by
step on the small model — prepare data (`pace.helpers.data`), read a figure yaml and pick a
run (`pace_figures.reproduce`), build its config from `conf/`, and train it
(`pace.helpers.train` → `pace.optimizer` / `pace.helpers.schedules` / `pace.helpers.model`).
Default: a small slice and a few steps
(checks it trains to a sane loss). With `PACE_TEST_FULL=1`: the full dataset, and it asserts
the reproduced `final_val_loss` matches the committed reference (`tests/fixtures/results/`)
within `PACE_REPRO_TOL`.

## Citation

```bibtex
@article{au2026pace,
  title  = {Training for the Model You Return: Improving Optimization for
            Iterate-Averaged Language Models},
  author = {Au, Kwok Chun and Block, Adam},
  journal = {arXiv preprint arXiv:2606.25086},
  year   = {2026},
  url    = {https://arxiv.org/abs/2606.25086}
}
```

## License

MIT — see [LICENSE](LICENSE).
