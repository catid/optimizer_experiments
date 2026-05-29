# Best Algorithm And Exact Recipe

Updated: 2026-05-29

## Winner

The best result in this worker folder is:

```text
Direct SODA-PMuonEq-NorMuon + aspect
```

Use this implementation:

```text
workers/codex_equimuse_normuon/direct_soda_pmuoneq_normuon.py
```

This is not the schedule-free EquiMuse implementation. It is the direct,
no-AMUSE/no-SF update path:

```text
SODA anchor pull
+ PMuonEq row/column diagonal preconditioning
+ Gram Newton-Schulz polar update
+ NorMuon row normalization
+ tall-matrix aspect multiplier
+ RMS/AdamW fallback for non-matrix parameters
```

The older schedule-free optimizer remains in:

```text
workers/codex_equimuse_normuon/equimuse_normuon.py
```

but it was not the best performer in the matched ViT-5/CIFAR-10 run.

## Exact Model, Data, And Batch

| field | value |
|---|---|
| model | `vit5_small` |
| parameter count | 21,657,994 trainable parameters |
| dataset | CIFAR-10 |
| image size | 224 |
| data path used | `/home/catid/attractor/data/cifar` |
| train/eval harness | ViT-5 fork harness, from-scratch training |
| validation set in this proxy run | CIFAR-10 `train=False` |
| seed | `67890` |
| GPUs | 4 NVIDIA RTX PRO 6000 Blackwell Max-Q |
| launch mode | DDP, `torchrun --standalone --nproc_per_node=4` |
| per-GPU batch | 128 |
| gradient accumulation | 1 |
| effective global batch | 512 |
| optimizer steps | 1000 |
| validation bins | 8 |
| steps per bin | 125 |
| trained examples | 512,000 |
| run label | `controlled_comparison` |

## Exact Optimizer Settings

| setting | value |
|---|---:|
| optimizer | `soda_pmuoneq_normuon` |
| fallback Adam/RMS LR | `8e-4` |
| matrix LR | `8e-3` |
| warmup steps | `10` |
| momentum | `0.95` |
| PMuonEq beta | `0.90` |
| PMuonEq row gamma | `0.35` |
| PMuonEq column gamma | `0.05` |
| NorMuon beta2 | `0.93` |
| NorMuon orientation | `row` |
| aspect multiplier | enabled |
| matrix weight decay | `0.0` |
| fallback weight decay | `0.05` |
| SODA lambda scale | `1.0` |
| SODA lambda power | `1.0` |
| GramNS steps | `5` |
| GramNS compute dtype | `float16` in the harness |

Other training settings:

| setting | value |
|---|---|
| base scheduler | cosine |
| warmup epochs | 1 |
| min LR | `1e-5` |
| `--unscale-lr` | enabled |
| label smoothing | `0.1` |
| mixup/cutmix | disabled |
| stochastic depth | `0.0` |
| random erasing | disabled |
| model EMA | disabled |
| repeated augmentation | disabled |
| DDP eval | enabled |
| dataloader workers | 8 |

## Result

| method | final acc@1 | final val loss | final train loss | samples/s | optimizer s/bin | total train s |
|---|---:|---:|---:|---:|---:|---:|
| Direct SODA-PMuonEq-NorMuon + aspect | **76.69** | **0.7269** | **1.2382** | 4009 | 4.445 | 124.5 |
| Direct SODA-PMuonEq-NorMuon | 75.16 | 0.7740 | 1.2579 | 3949 | 4.525 | 126.9 |
| EquiMuse-NorMuon row | 65.96 | 1.0144 | 1.4699 | 4053 | 3.448 | 124.6 |
| EquiMuse-NorMuon row + aspect | 65.85 | 1.0019 | 1.4519 | 4027 | 3.481 | 125.4 |
| EquiMuse-NorMuon auto + aspect | 65.72 | 1.0140 | 1.4547 | 4021 | 3.433 | 126.0 |
| EquiMuse-NorMuon peer-gamma + aspect | 65.11 | 1.0302 | 1.4516 | 4030 | 3.546 | 125.7 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | **4280** | **0.237** | **117.1** |

Direct SODA-PMuonEq-NorMuon + aspect beat the matched AdamW baseline by
`+13.63` acc@1 points and `-0.3622` validation loss in this 1000-step proxy
run. AdamW remained faster per step.

## Reproduction Command

```bash
/home/catid/attractor/.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  --model vit5_small \
  --input-size 224 \
  --data-set CIFAR10 \
  --data-path /home/catid/attractor/data/cifar \
  --output_dir runs/equimuse_feedback_final_1000/direct_soda_pmuoneq_normuon_lr0p008_rg0p35_cg0p05_nb0p93_aspect-seed67890-img224-ep8-g4-steps125 \
  --batch-size 128 \
  --epochs 8 \
  --seed 67890 \
  --weight-decay 0.05 \
  --sched cosine \
  --warmup-epochs 1 \
  --min-lr 1e-05 \
  --unscale-lr \
  --mixup 0 \
  --cutmix 0 \
  --smoothing 0.1 \
  --drop-path 0.0 \
  --reprob 0 \
  --num_workers 8 \
  --accum_iter 1 \
  --no-model-ema \
  --disable_wandb \
  --dist-eval \
  --no-repeated-aug \
  --opt soda_pmuoneq_normuon \
  --lr 0.0008 \
  --equimuse-matrix-lr 0.008 \
  --equimuse-warmup-steps 10 \
  --equimuse-pmuoneq-beta 0.90 \
  --equimuse-pmuoneq-row-gamma 0.35 \
  --equimuse-pmuoneq-col-gamma 0.05 \
  --equimuse-normuon-beta2 0.93 \
  --equimuse-normuon-orientation row \
  --equimuse-normuon-aspect-scale \
  --max-train-steps-per-epoch 125
```

## What Did Not Win

- Schedule-free EquiMuse-NorMuon trained, but reached only `65.96%` acc@1 in
  the matched 1000-step run.
- Aspect scaling improved EquiMuse validation loss but not EquiMuse final
  accuracy on this seed.
- The direct no-AMUSE path was the main jump. The aspect multiplier improved the
  direct path further, from `75.16%` to `76.69%`.
- AdamW was the fastest method, but had the weakest final validation metric in
  this comparison.

## Caveats

- This folder's best matched comparison is single-seed evidence.
- The run used CIFAR-10 `train=False` as the validation readout. Treat this as a
  controlled local proxy, not a publishable final test protocol.
- The strongest next check is a 3-seed confirmation of the exact winning recipe
  against tuned AdamW and the best schedule-free EquiMuse variant on a proper
  train/validation/test split.

## Primary Artifacts

```text
workers/codex_equimuse_normuon/results/tables/equimuse_feedback_final_results.csv
workers/codex_equimuse_normuon/results/tables/equimuse_feedback_final_curves.csv
workers/codex_equimuse_normuon/results/figures/equimuse_feedback_val_loss_curve.png
workers/codex_equimuse_normuon/results/figures/equimuse_feedback_train_loss_curve.png
workers/codex_equimuse_normuon/results/figures/equimuse_feedback_accuracy_curve.png
workers/codex_equimuse_normuon/results/figures/equimuse_feedback_iteration_speed.png
workers/codex_equimuse_normuon/results/figures/equimuse_feedback_final_accuracy_bar.png
```
