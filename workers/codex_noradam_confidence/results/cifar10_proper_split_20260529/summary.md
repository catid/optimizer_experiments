# CIFAR-10 Proper Split Result

Updated: 2026-05-29

This run fixes the earlier protocol issue where CIFAR-10 `train=False` was used
as the validation set during HPO. The new protocol is:

- training pool: CIFAR-10 `train=True`
- deterministic split seed: `12345`
- validation: 5,000 held-out examples from the training pool
- training: remaining 45,000 examples
- official test: CIFAR-10 `train=False`, evaluated once at the end of each
  selected final run with `--eval-test`

The runner now supports both historical behavior and the proper split through
`--val-source {test,train_split}`. Existing runs remain reproducible because the
default is still `--val-source test`; new reported final numbers here use
`--val-source train_split --eval-test`.

## Protocol

- Model: `vit5_micro`
- Dataset: CIFAR-10
- Batch size: 512
- Workers: 16
- Warmup: 80 steps
- HPO: 12 epochs, 17 trials, one seed
- Final replay: 50 epochs, seeds `123,456,789`
- Scheduling: one trial per visible GPU
- Step timing: CPU launch timing, no per-step CUDA synchronization
- Torch: `2.13.0.dev20260506+cu130`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition

## Final Results

Mean and standard deviation over three final seeds. Validation is the 5,000
example train-split validation set; test is the official CIFAR-10 test set.

| rank | recipe | final val loss | final val acc | best val loss | best val acc | test loss | test acc | step | throughput |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon + NorMuon | 0.4414 +/- 0.0228 | 84.97% +/- 0.35 | 0.4310 +/- 0.0112 | 85.25% +/- 0.27 | 0.4607 +/- 0.0196 | 84.77% +/- 0.65 | 19.95 ms | 25.7k ex/s |
| 2 | AnchorMuon + NorMuon aspect | 0.4445 +/- 0.0130 | 85.25% +/- 0.32 | 0.4231 +/- 0.0135 | 85.65% +/- 0.33 | 0.4595 +/- 0.0074 | 84.55% +/- 0.40 | 19.92 ms | 25.7k ex/s |
| 3 | AdamW cosine | 0.6180 +/- 0.0043 | 79.69% +/- 0.22 | 0.6043 +/- 0.0039 | 79.85% +/- 0.29 | 0.6338 +/- 0.0125 | 79.28% +/- 0.23 | 11.59 ms | 44.2k ex/s |

Per-seed rows are in `final50/all_runs.csv`.

## Selected Configs

The HPO selected these family winners:

| family | selected run |
|---|---|
| AdamW | `adamw_cosine_lr0.004_wd0.001` |
| AnchorMuon + NorMuon | `normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93` |
| AnchorMuon + NorMuon aspect | `normuon_aspect_mlr0.008_rg0.3_cg0.05_mom0.95_pb0.9_nb0.95` |

Best current recipe by official test accuracy:

```text
optimizer = AnchorMuon
lr = 8e-3
weight_decay = 0.05
amuse = False
soda = "all"
pmuon_eq = True
pmuon_beta = 0.90
row_gamma = 0.35
col_gamma = 0.0
momentum = 0.95
normuon = True
normuon_beta = 0.93
normuon_aspect_scale = False
mimuon = False
```

The aspect-scaled variant had the best mean validation checkpoint accuracy, but
the no-aspect variant had the best mean official test accuracy and slightly
better final validation loss. Treat them as close; both remain far ahead of
AdamW in this harness.

## Commands

Smoke check:

```bash
/home/catid/screen/.venv/bin/python workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --preset feedback \
  --only 'adamw_cosine|normuon_mlr0.008_rg0.35_cg0.05_mom0.95_pb0.9_nb0.95' \
  --output-dir workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/smoke \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 1 --max-steps 8 --train-subset 1024 \
  --val-source train_split --train-val-size 5000 --val-subset 512 \
  --eval-test --test-subset 512 \
  --batch-size 128 --num-workers 4 --warmup-steps 4 \
  --eval-bins 2 --log-every 4 --model vit5_micro \
  --no-sync-step-timing --no-plots
```

HPO:

```bash
/home/catid/screen/.venv/bin/python workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --preset feedback \
  --output-dir workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/hpo \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 12 \
  --train-subset 0 \
  --val-source train_split --train-val-size 5000 --val-subset 0 \
  --batch-size 512 --num-workers 16 \
  --warmup-steps 80 --eval-bins 8 --log-every 200 \
  --model vit5_micro --no-sync-step-timing
```

Final replay:

```bash
/home/catid/screen/.venv/bin/python workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  --final-from-hpo workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/hpo/best_by_family.json \
  --output-dir workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/final50 \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --final-epochs 50 \
  --train-subset 0 \
  --val-source train_split --train-val-size 5000 --val-subset 0 \
  --eval-test --test-subset 0 \
  --batch-size 512 --num-workers 16 \
  --warmup-steps 80 --eval-bins 8 --log-every 500 \
  --model vit5_micro --no-sync-step-timing \
  --seeds 123,456,789
```

## Artifacts

- `hpo/all_runs.csv`
- `hpo/best_by_family.json`
- `final50/all_runs.csv`
- `final50/val_loss.png`
- `final50/val_acc.png`
- `final50/train_loss.png`
- `final50/step_time_ms_bar.png`
- `final50/examples_per_sec_bar.png`

## Conclusion

The proper split did not invalidate the main qualitative result: AnchorMuon with
SODA + PMuonEq + GramNS + NorMuon still beats tuned AdamW by about 5.5
percentage points of official CIFAR-10 test accuracy in this ViT-5 micro
harness. AdamW remains about 1.7x faster per optimizer step, so the optimizer
win is quality/sample-efficiency rather than raw iteration speed.
