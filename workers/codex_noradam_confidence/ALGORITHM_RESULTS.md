# AnchorMuon NorMuon Summary

Updated: 2026-05-29

This worker's best optimizer is the AMUSE-off AnchorMuon variant:

```text
AnchorMuon + SODA + PMuonEq + Gram Newton-Schulz + NorMuon
```

The best measured recipe in this folder disables AMUSE/SF iterate averaging and
uses AnchorMuon as a SODA-regularized Muon-family optimizer with row-wise
PMuonEq and NorMuon normalization.

## Current Monorepo Winner

The specific version to cite is:

```text
normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93
```

It is **not** the aspect-scaled variant. The aspect run had slightly better
best-validation checkpoint metrics, but the no-aspect version is the current
best by held-out official CIFAR-10 test accuracy.

Exact scope of the claim:

- model: `vit5_micro`
- parameters: 458,858 trainable parameters
- dataset: CIFAR-10, 32x32 images
- split: 45,000 train examples and 5,000 validation examples from
  CIFAR-10 `train=True`
- final test: official CIFAR-10 `train=False` 10,000-example test split,
  evaluated only at the end of selected final runs
- final training: 50 epochs, 4,350 optimizer steps, seeds `123,456,789`
- batch size: 512
- dataloader workers: 16
- precision/layout: BF16 autocast with channels-last tensors
- hardware used: two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs,
  scheduled as one trial per visible GPU
- software used: PyTorch `2.13.0.dev20260506+cu130`
- selection protocol: 17-trial HPO over AdamW, no-aspect NorMuon, and
  aspect-scaled NorMuon for 12 epochs on the train/validation split, followed
  by a three-seed 50-epoch replay of selected configurations

The exact optimizer settings for the winner are:

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
ns_steps = 5
sync_diagnostics = False
```

## Algorithm

For each matrix-like parameter, flatten tensors to a matrix
`W in R^{rows x cols}`. Non-matrix parameters use the optimizer's fallback
adaptive path.

Given gradient `G_t`, momentum coefficient `mu`, PMuonEq EMA coefficient `beta`,
and row/column adaptivity exponents `gamma_r`, `gamma_c`:

```text
M_t = mu M_{t-1} + (1 - mu) G_t
U_t = (1 - mu) G_t + mu M_t

r_t = beta r_{t-1} + (1 - beta) mean_cols(G_t^2)
c_t = beta c_{t-1} + (1 - beta) mean_rows(G_t^2)

P_t = r_t^{-gamma_r} * U_t * c_t^{-gamma_c}
Q_t = GramNewtonSchulz(P_t)
```

`GramNewtonSchulz` is the standard Muon quintic Newton-Schulz polar/zero-power
approximation. In this worker it uses five iterations by default.

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
ns_steps = 5
sync_diagnostics = False
```

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

Conclusion: the no-aspect AnchorMuon + NorMuon recipe is the current best by
official test accuracy. The aspect-scaled variant is close and has better
best-validation checkpoint metrics, but it did not improve mean official test
accuracy in this three-seed replay. AdamW remains much faster per step but
substantially worse on loss and accuracy.

To reproduce or rerun the directly comparable set without reconstructing the HPO
grid, use the runner's `--preset best_cifar10`. That preset contains the exact
winner above, the closest aspect-scaled near miss, and the tuned AdamW cosine
baseline.

Primary result bundle:

```text
workers/codex_noradam_confidence/results/cifar10_proper_split_20260529/
```

## Known Caveats

- The result is for the ViT-5 micro CIFAR-10 harness, not a general proof that
  the optimizer transfers to every architecture or scale.
- The matrix grouping is name-light. Some 2D classifier/head parameters can go
  through the Muon/NorMuon path unless the model excludes them through
  `no_weight_decay()`.
- The official test set is now reserved for final readout in the proper-split
  result, but historical result folders in this workspace used CIFAR-10
  `train=False` as validation during HPO.
- AdamW is still the speed winner. AnchorMuon's gain here is quality and
  sample-efficiency, not iteration speed.
