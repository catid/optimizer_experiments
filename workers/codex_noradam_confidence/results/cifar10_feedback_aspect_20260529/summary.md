# CIFAR-10 Feedback Aspect Ablation - 2026-05-29

This run tested the cross-worker suggestion from `codex_soda_pmuoneq_normuon`:
add an optional post-NorMuon aspect-ratio multiplier,
`sqrt(max(1, rows / cols))`, after NorMuon restores the update Frobenius norm.

The patch keeps the previous reported recipe unchanged by default:

- `normuon_aspect_scale=False` is the default.
- The multiplier is applied only after Gram Newton-Schulz and NorMuon update
  normalization.
- Raw gradients, momentum, PMuonEq row/column statistics, SODA anchoring, and
  AdamW fallback updates are not changed.

## Validation

Commands run from the repository root:

```bash
/home/catid/screen/.venv/bin/python -m py_compile \
  workers/codex_noradam_confidence/optim_anchormuon.py \
  workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  workers/codex_noradam_confidence/tests/test_anchormuon_modes.py \
  workers/codex_noradam_confidence/tests/ddp_smoke_anchormuon.py

/home/catid/screen/.venv/bin/python -m pytest -q \
  workers/codex_noradam_confidence/tests/test_anchormuon_modes.py

CUDA_VISIBLE_DEVICES=0,1 /home/catid/screen/.venv/bin/torchrun \
  --standalone --nproc-per-node=2 \
  workers/codex_noradam_confidence/tests/ddp_smoke_anchormuon.py
```

Results:

- Python compile: passed.
- Unit tests: `13 passed`.
- DDP smoke: passed with parameter/state parity.
- Runner smoke: passed for AdamW, no-aspect NorMuon, and aspect NorMuon.

The runner smoke also caught a bug: worker subprocesses changed `cwd` to the repo
root while trial JSON and output paths could be relative. `launch_trials()` now
resolves `output_dir` and `data_path` before spawning workers.

## HPO

Protocol:

- Model: `vit5_micro`
- Dataset: CIFAR-10, full train split and official test split used as validation
- Epochs: 12
- Batch size: 512
- Workers: 16
- GPUs: all visible GPUs, one trial per GPU
- Seeds: one HPO seed
- Timing: `--no-sync-step-timing`

Grid:

- AdamW: `lr=0.004`, `weight_decay=0.001`, cosine schedule
- AnchorMuon/NorMuon:
  - `lr=0.008`
  - `soda="all"`
  - `pmuon_beta=0.90`
  - `momentum=0.95`
  - `row_gamma in {0.30, 0.35}`
  - `col_gamma in {0.0, 0.05}`
  - `normuon_beta in {0.93, 0.95}`
  - `normuon_aspect_scale in {false, true}`
  - `amuse=False`

Best HPO rows:

| rank | recipe | aspect | best val loss | best val acc | step | throughput |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `row_gamma=0.35 col_gamma=0.05 normuon_beta=0.95` | no | 0.6397 | 77.81% | 20.33 ms | 25.2k ex/s |
| 2 | `row_gamma=0.30 col_gamma=0.00 normuon_beta=0.95` | no | 0.6431 | 77.56% | 19.88 ms | 25.8k ex/s |
| 3 | `row_gamma=0.35 col_gamma=0.00 normuon_beta=0.95` | yes | 0.6440 | 77.59% | 19.96 ms | 25.7k ex/s |
| 4 | AdamW cosine | n/a | 0.8845 | 68.50% | 11.55 ms | 44.3k ex/s |

HPO conclusion: in this implementation and protocol, retuned no-aspect NorMuon
won. Aspect scaling was close, but did not beat no-aspect after retuning.

## Final 50-Epoch Replay

Protocol:

- Best HPO config per family.
- Three seeds: `123,456,789`.
- 50 epochs.
- Same model, dataset, batch size, workers, and no-sync timing mode as HPO.

| rank | recipe | final val loss | final val acc | best val loss | best val acc | mean step | throughput |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon NorMuon, no aspect | 0.4225 +/- 0.0027 | 85.86% +/- 0.23 | 0.4198 +/- 0.0043 | 85.98% +/- 0.09 | 20.20 ms | 25.4k ex/s |
| 2 | AnchorMuon NorMuon, aspect | 0.4288 +/- 0.0103 | 85.43% +/- 0.29 | 0.4236 +/- 0.0083 | 85.60% +/- 0.17 | 19.56 ms | 26.2k ex/s |
| 3 | AdamW cosine | 0.6210 +/- 0.0105 | 79.66% +/- 0.12 | 0.6088 +/- 0.0063 | 79.96% +/- 0.13 | 11.54 ms | 44.4k ex/s |

The no-aspect recipe remained best on validation loss and accuracy. The aspect
variant was slightly faster in this replay, but worse on mean final and best
validation metrics. Both AnchorMuon/NorMuon variants strongly beat tuned AdamW
on quality, while AdamW remains much faster per step.

Important caveat: the runner uses the official CIFAR-10 `train=False` split as
the validation split during HPO and final replay. Treat these as validation
numbers from the shared protocol, not as an untouched final test-set claim.

## Artifacts

- HPO CSV: `hpo/all_runs.csv`
- HPO best-by-family: `hpo/best_by_family.json`
- HPO plots: `hpo/val_loss.png`, `hpo/val_acc.png`, `hpo/step_time_ms_bar.png`,
  `hpo/examples_per_sec_bar.png`
- Final CSV: `final50/all_runs.csv`
- Final best-by-family: `final50/best_by_family.json`
- Final plots: `final50/val_loss.png`, `final50/val_acc.png`,
  `final50/train_loss.png`, `final50/step_time_ms_bar.png`,
  `final50/examples_per_sec_bar.png`

## Current Best Recipe

For this worker folder, the best validated CIFAR-10 recipe is:

```text
optimizer = AnchorMuon
lr = 8e-3
warmup_steps = 80
amuse = False
soda = "all"
soda_disables_weight_decay = True
pmuon_eq = True
pmuon_beta = 0.90
row_gamma = 0.35
col_gamma = 0.05
momentum = 0.95
normuon = True
normuon_beta = 0.95
normuon_aspect_scale = False
mimuon = False
ns_steps = 5
sync_diagnostics = False
```

Recommended next controlled tests:

- isolate SODA placement before/after learned update;
- compare classifier/head in matrix path vs fallback path;
- compare simple Newton-Schulz vs Polar Express GramNS coefficients;
- add a clean train/validation split from the CIFAR-10 train set and reserve
  the official test split for final-only reporting.
