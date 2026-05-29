# AnchorMuon NorMuon Summary

Updated: 2026-05-29

This worker's best optimizer family is the AMUSE-off AnchorMuon variant:

```text
AnchorMuon + SODA + PMuonEq + Gram Newton-Schulz + NorMuon
```

The best observed single-seed recipe in this folder disables AMUSE/SF iterate
averaging and uses AnchorMuon as a SODA-regularized Muon-family optimizer with
row-wise PMuonEq and NorMuon normalization under a trainer-side WSD schedule.
The strongest multi-seed evidence uses the same core optimizer with a constant
LR schedule.

## Current Monorepo Winner

The best observed single-seed version to cite is:

```text
root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93
```

It is **not** the aspect-scaled variant and it does not change the root
optimizer internals. The improvement is from trainer-side LR scheduling:
80-step warmup followed by WSD with `lr=0.012`, `lr_final_scale=0.1`, and
`wsd_decay_frac=0.2`.

Exact scope of the claim:

- model: `vit5_micro`
- parameters: 458,858 trainable parameters
- dataset: CIFAR-10, 32x32 images
- split: 45,000 train examples and 5,000 validation examples from
  CIFAR-10 `train=True`
- final test: official CIFAR-10 `train=False` 10,000-example test split,
  evaluated only at the end of selected final runs
- final training: 50 epochs, 4,350 optimizer steps, seed `123` for the latest
  schedule study; seeds `123,456,789` for the older multi-seed constant-LR
  confidence replay
- batch size: 512
- dataloader workers: 16
- precision/layout: BF16 autocast with channels-last tensors
- hardware used: two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs,
  scheduled as one trial per visible GPU
- software used: PyTorch `2.13.0.dev20260506+cu130`
- selection protocol: 12-epoch HPO over constant/cosine/linear/WSD schedules
  and LR candidates for root `AnchorMuon`, followed by a 50-epoch replay of the
  best config per schedule. Older confidence intervals used 17-trial HPO over
  AdamW, no-aspect NorMuon, and aspect-scaled NorMuon followed by a three-seed
  50-epoch replay.

The exact optimizer settings for the winner are:

```text
optimizer = AnchorMuon
lr = 0.012
lr schedule = trainer-side 80-step warmup + WSD
lr_final_scale = 0.1
wsd_decay_frac = 0.2
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
ns_steps = 5
sync_diagnostics = False
```

## Algorithm

For each matrix-like parameter, flatten tensors to a matrix
`W in R^{rows x cols}`. Non-matrix parameters use the optimizer's fallback
adaptive path.

Given gradient `G_t`, momentum coefficient `mu`, PMuonEq EMA coefficient `beta`,
and row adaptivity exponent `gamma_r` for the current root optimizer:

```text
M_t = mu M_{t-1} + (1 - mu) G_t
U_t = (1 - mu) G_t + mu M_t

r_t = beta r_{t-1} + (1 - beta) mean_cols(G_t^2)

P_t = r_t^{-gamma_r} * U_t
Q_t = GramNewtonSchulz(P_t)
```

This is row-only PMuonEq. Older research ablations in this folder also tested
column scaling, but the shippable root optimizer does not maintain column EMA
state and has no `col_gamma` knob. `GramNewtonSchulz` is the standard Muon
quintic Newton-Schulz polar/zero-power approximation. In this worker it uses
five iterations by default.

NorMuon is applied after the polar approximation, not to the raw gradient:

```text
if rows >= cols:
    n_t = beta_n n_{t-1} + (1 - beta_n) mean_cols(Q_t^2)
    D_t = Q_t / sqrt(n_t + eps)
else:
    n_t = beta_n n_{t-1} + (1 - beta_n) mean_rows(Q_t^2)
    D_t = Q_t / sqrt(n_t + eps)

D_t = D_t * ||Q_t||_F / ||D_t||_F
```

The optional aspect ablation then multiplies `D_t` by
`sqrt(max(1, rows / cols))`. It is treated as a layerwise step scale, not as
part of the direction generator.

For the AMUSE-off winner, the fast parameter is updated directly:

```text
W <- W - lr * scale * D_t
```

SODA then applies an initialization-anchor pull to SODA-enabled parameter
groups. With `soda="all"`, both matrix and fallback groups use the SODA anchor
path; ordinary weight decay is disabled for those anchored parameters.

## Best Current Recipe

```text
optimizer = AnchorMuon
lr = 0.012
lr schedule = trainer-side 80-step warmup + WSD
lr_final_scale = 0.1
wsd_decay_frac = 0.2
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
ns_steps = 5
sync_diagnostics = False
```

## Root LR Schedule Result

Protocol:

- model: `vit5_micro`
- train: 45,000 examples from CIFAR-10 `train=True`
- validation: 5,000 held-out examples from CIFAR-10 `train=True`
- official test: CIFAR-10 `train=False`, evaluated only after each final run
- HPO: 12 epochs, seed `123`, best LR selected separately for each schedule
- final replay: 50 epochs, seed `123`, one best config per schedule
- batch size: 512
- loader workers: 16
- scheduling: one trial per visible GPU

Best 12-epoch HPO config per schedule:

| Schedule | Trial | LR | 12-epoch val loss | 12-epoch val acc | Step |
|---|---|---:|---:|---:|---:|
| constant | `root_named_constant_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.6865 | 75.36% | 16.94 ms |
| cosine | `root_named_cosine_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5946 | 79.30% | 17.09 ms |
| linear | `root_named_linear_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5820 | 79.46% | 16.86 ms |
| WSD | `root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5632 | 80.08% | 16.06 ms |
| AdamW cosine | `adamw_cosine_lr0.004_wd0.001` | 0.004 | 0.9095 | 67.56% | 11.04 ms |

50-epoch schedule-winner replay:

| recipe | schedule | final val loss | best val loss | final val acc | best val acc | official test loss | official test acc | step | throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnchorMuon | WSD | 0.4005 | 0.3716 | 88.36% | 88.36% | 0.4111 | 87.67% | 16.23 ms | 31.5k ex/s |
| AnchorMuon | constant | 0.4036 | 0.4036 | 86.06% | 86.54% | 0.4317 | 85.74% | 17.19 ms | 29.8k ex/s |
| AnchorMuon | cosine | 0.4511 | 0.4097 | 87.40% | 87.48% | 0.4588 | 86.90% | 16.96 ms | 30.2k ex/s |
| AnchorMuon | linear | 0.4434 | 0.4151 | 87.18% | 87.58% | 0.4597 | 86.75% | 17.02 ms | 30.1k ex/s |
| AdamW cosine | cosine | 0.6133 | 0.5998 | 79.62% | 79.64% | 0.6240 | 79.55% | 11.39 ms | 45.0k ex/s |

Conclusion: WSD is the best schedule for root `AnchorMuon` in this single-seed
study. It gives the best final validation loss, best validation loss, final
validation accuracy, best validation accuracy, and official test accuracy.
AdamW remains substantially faster per step.

## Proper-Split CIFAR-10 Result

Protocol:

- model: `vit5_micro`
- train: 45,000 examples from CIFAR-10 `train=True`
- validation: 5,000 held-out examples from CIFAR-10 `train=True`
- official test: CIFAR-10 `train=False`, evaluated only after final runs
- HPO: 12 epochs, one seed, 17 trials
- final replay: 50 epochs, seeds `123,456,789`
- batch size: 512
- loader workers: 16
- scheduling: one trial per visible GPU

Mean and standard deviation over the three final seeds:

| recipe | final val loss | final val acc | best val loss | best val acc | official test loss | official test acc | step | throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnchorMuon + NorMuon | 0.4414 +/- 0.0228 | 84.97% +/- 0.35 | 0.4310 +/- 0.0112 | 85.25% +/- 0.27 | 0.4607 +/- 0.0196 | 84.77% +/- 0.65 | 19.95 ms | 25.7k ex/s |
| AnchorMuon + NorMuon aspect | 0.4445 +/- 0.0130 | 85.25% +/- 0.32 | 0.4231 +/- 0.0135 | 85.65% +/- 0.33 | 0.4595 +/- 0.0074 | 84.55% +/- 0.40 | 19.92 ms | 25.7k ex/s |
| AdamW cosine | 0.6180 +/- 0.0043 | 79.69% +/- 0.22 | 0.6043 +/- 0.0039 | 79.85% +/- 0.29 | 0.6338 +/- 0.0125 | 79.28% +/- 0.23 | 11.59 ms | 44.2k ex/s |

Conclusion: this older three-seed replay is the strongest confidence interval
for the optimizer family. The no-aspect AnchorMuon + NorMuon recipe is best by
mean official test accuracy. The aspect-scaled variant is close and has better
best-validation checkpoint metrics, but it did not improve mean official test
accuracy. AdamW remains much faster per step but substantially worse on loss and
accuracy.

To reproduce or rerun the directly comparable set without reconstructing the HPO
grid, use the runner's `--preset best_cifar10`. That preset contains the exact
winner above, the closest aspect-scaled near miss, and the tuned AdamW cosine
baseline.

Primary result bundle:

```text
workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/
```

The stripped golden implementation is:

```text
workers/codex_noradam_confidence/golden_soda_pmuoneq_normuon.py
```

It was replayed on the same proper split and reproduced the previous winner's
train/validation/test metrics exactly at CSV precision. See
`GOLDEN_OPTIMIZER.md` and
`results/cifar10_golden_repro_20260529/final50/`.

## Known Caveats

- The result is for the ViT-5 micro CIFAR-10 harness, not a general proof that
  the optimizer transfers to every architecture or scale.
- The root optimizer's matrix grouping is shape-only. Names are kept for
  summaries only, so any sufficiently large 2D classifier/head/embedding
  parameter goes through the matrix path unless the trainer supplies explicit
  param groups.
- The official test set is now reserved for final readout in the proper-split
  result, but historical result folders in this workspace used CIFAR-10
  `train=False` as validation during HPO.
- AdamW is still the speed winner. AnchorMuon's gain here is quality and
  sample-efficiency, not iteration speed.
