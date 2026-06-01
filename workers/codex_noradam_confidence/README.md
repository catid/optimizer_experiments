# Codex NorMuon Confidence Worker

This folder contains the Codex worker copy of the ViT-5 CIFAR-10 optimizer
comparison focused on:

- AdamW baseline
- NorMuon+base: SODA + PMuonEq + Gram Newton-Schulz + NorMuon, AMUSE off

The source was copied from local repo:

```text
/home/catid/screen/repos/ViT-5-AnchorMuon
commit b009f03 Compare NorMuon base against AdamW
```

## Contents

- `optim_anchormuon.py`: AnchorMuon/NorMuon optimizer implementation.
- `optim_sfplus.py`: experimental ScheduleFree+ outer-loop wrapper used only
  for the exhaustive toggle sweep below.
- `golden_soda_pmuoneq_normuon.py`: stripped standalone golden optimizer
  implementing only the winning direct SODA + PMuonEq + GramNS + NorMuon path.
- `optim_factory.py`: ViT-5 optimizer factory hook.
- `models_vit5.py`, `rope.py`: minimal ViT-5 model code needed by the runner.
- `experiments/run_cifar10_ablation.py`: CIFAR-10 HPO/final comparison runner.
- `tests/`: optimizer unit tests and DDP smoke test.
- `ALGORITHM_RESULTS.md`: self-contained algorithm and result summary for the
  current best recipe.
- `GOLDEN_OPTIMIZER.md`: golden optimizer algorithm and exact reproduction
  validation against the previous best proper-split CIFAR-10 result.
- `results/cifar10_confidence_noradam_20260528/`: committed result bundle, curves, and summary.
- `results/cifar10_feedback_nosync_20260529/final50/`: latest committed
  feedback run with synchronization diagnostics disabled, comparison diagrams,
  curves, CSV, and per-trial JSONL.
- `results/cifar10_feedback_aspect_20260529/`: HPO and 3-seed 50-epoch replay
  for the peer-suggested post-NorMuon aspect multiplier.
- `results/cifar10_proper_split_20260529/`: proper CIFAR-10 45k/5k
  train/validation split with official test evaluated only at the end of final
  selected runs.
- `results/root_lr_schedule_sweep_20260529/`: trainer-side LR schedule study
  for root `optimizer.py`; 12-epoch HPO per schedule followed by 50-epoch
  schedule-winner replay.
- `results/sfplus_combo20_20260529/`: 20-epoch exhaustive ScheduleFree+
  mechanism-combination sweep around the AnchorMuon matrix direction.
- `results/finewebedu_llm50m_muown_ema_20260601/`: 50M byte-level FineWeb-Edu
  comparison of Muown and EMA-Nesterov wrappers against AnchorMuon, Muon, AdamW,
  and AdamW-Atan2 controls.
- `results/cifar10_muown_ema_20260601/`: CIFAR-10 transfer check for Muown and
  EMA-Nesterov against the current AnchorMuon image-classification recipe.

## Main Result

Latest result summary:
`ALGORITHM_RESULTS.md` and `results/cifar10_proper_split_20260529/summary.md`.

Best observed single-seed version to cite:

```text
root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93
```

This is root `AnchorMuon` with SODA + row-only PMuonEq + five-step GramNS +
NorMuon, AMUSE off, trainer-side 80-step warmup and WSD schedule. It was
evaluated on `vit5_micro` with CIFAR-10 45k/5k train/validation split, official
10k test evaluated at the end, batch size 512, 50 epochs, seed `123`, BF16
autocast, channels-last tensors, and 16 dataloader workers. It is currently the
best observed single-seed result: `87.67%` official test accuracy, compared
with `79.55%` for the tuned AdamW cosine baseline in the same replay.

The previous strongest multi-seed evidence is still the no-aspect constant-LR
recipe `normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93`, which reached
`84.77% +/- 0.65` official test accuracy over seeds `123,456,789`.

Latest diagrams:

- Loss curves: `results/cifar10_proper_split_20260529/final50/val_loss.png`
- Validation accuracy: `results/cifar10_proper_split_20260529/final50/val_acc.png`
- Iteration speed: `results/cifar10_proper_split_20260529/final50/step_time_ms_bar.png`
- Latest LR-schedule loss curves:
  `results/root_lr_schedule_sweep_20260529/final50_schedule_winners/val_loss.png`
- Latest LR-schedule validation accuracy:
  `results/root_lr_schedule_sweep_20260529/final50_schedule_winners/val_acc.png`
- Latest LR-schedule iteration speed:
  `results/root_lr_schedule_sweep_20260529/final50_schedule_winners/step_time_ms_bar.png`

### FineWeb-Edu Muown / EMA-Nesterov Result

The newest LM optimizer check adds two paper-derived variants in the local
byte-LM runner only:

- `Muown` from arXiv `2605.10797v1`, implemented as optimizer-side row-norm
  control for Muon.
- `EMA-Nesterov` from arXiv `2605.25395v1`, implemented as a pre-forward
  lookahead wrapper around Muon or Muown.

Protocol: 49.4M byte-level GPT, FineWeb-Edu byte cache, batch `32 x 128`,
BF16, WSD with 100-step warmup, 1.2K-step HPO and 6K-step final replay, two RTX
PRO 6000 Blackwell GPUs scheduled one trial per GPU.

| Rank | Optimizer | Selected config | Final val loss | Final byte acc | Step | Throughput |
|---:|---|---|---:|---:|---:|---:|
| 1 | EMA-Nesterov + Muown | `lr=0.0016`, `wd=0`, `ema_beta=0.3`, `ema_gamma=0.995` | 1.2131 | 63.30% | 21.91 ms | 187.0k byte/s |
| 2 | Muown | `lr=0.0016`, `wd=0` | 1.2156 | 63.23% | 20.69 ms | 198.0k byte/s |
| 3 | AnchorMuon | `lr=0.0012`, `row_gamma=0.45`, `soda=0.003`, `fallback=RMS@1.0x` | 1.2312 | 62.84% | 14.83 ms | 276.2k byte/s |
| 4 | Plain Muon | `lr=0.0011`, `wd=0.05` | 1.2338 | 62.76% | 16.89 ms | 242.5k byte/s |
| 5 | EMA-Nesterov + Muon | `lr=0.001`, `wd=0.05`, `ema_beta=0.5`, `ema_gamma=0.995` | 1.2340 | 62.72% | 18.51 ms | 221.3k byte/s |

Takeaway: on this 6K-step proxy, Muown is the best new direction by loss.
EMA-Nesterov adds a small loss improvement on top of Muown, but it is slower
because it keeps full-parameter EMA lookahead state. This is not a replacement
for the CIFAR-10 AnchorMuon winner; it is evidence for the next language-model
optimizer round.

### CIFAR-10 Muown / EMA-Nesterov Transfer Check

This follow-up tested the same Muown and EMA-Nesterov ideas in the established
ViT-5 CIFAR-10 harness.

Protocol: `vit5_micro`, CIFAR-10 45k/5k train/validation split, official
10k test evaluated only at the end, batch size 512, seed `123`, BF16 autocast,
channels-last tensors, 16 dataloader workers, two RTX PRO 6000 Blackwell GPUs
scheduled one trial per GPU. HPO used 12 epochs; the best config per optimizer
family was replayed for 50 epochs.

| Rank | Family | Selected config | Final val acc | Official test acc | Final val loss | Test loss | Step | Throughput |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon root | `lr=0.016`, WSD, `row_gamma=0.35`, `pmuon_beta=0.90`, `normuon_beta=0.93`, AdamAtan2 fallback at `0.5x` | 88.32% | 87.98% | 0.3787 | 0.4065 | 17.11 ms | 29.9k ex/s |
| 2 | EMA-Nesterov + Muon | `lr=0.014`, `wd=0.001`, `ema_beta=0.1`, `ema_gamma=0.99` | 86.94% | 86.08% | 0.4754 | 0.5023 | 17.79 ms | 28.8k ex/s |
| 3 | Plain Muon | `lr=0.014`, `wd=0.001`, GramNS | 86.72% | 85.40% | 0.5289 | 0.5582 | 17.08 ms | 30.0k ex/s |
| 4 | Muown | `lr=0.012`, `wd=0`, WSD | 83.36% | 83.56% | 0.4845 | 0.4902 | 20.10 ms | 25.5k ex/s |
| 5 | AdamW | `lr=0.004`, `wd=0.001`, cosine | 79.22% | 79.76% | 0.6158 | 0.6352 | 11.62 ms | 44.1k ex/s |
| 6 | EMA-Nesterov + Muown | `lr=0.012`, `wd=0.001`, `ema_beta=0.1`, `ema_gamma=0.99` | 76.54% | 75.40% | 0.6649 | 0.6954 | 21.18 ms | 24.2k ex/s |

Takeaway: Muown did not transfer to CIFAR-10 as well as it did to the
byte-level LM proxy. The current root AnchorMuon image-classification recipe
remains the best CIFAR option. EMA-Nesterov + Muon is stable and beats plain
Muon on official test in this single-seed run, but it still trails AnchorMuon.
EMA-Nesterov + Muown was unstable in no-weight-decay HPO runs and weak in the
best stable final replay.

Plots and full CSV:
`results/cifar10_muown_ema_20260601/summary.md`,
`results/cifar10_muown_ema_20260601/final_bins/val_acc.png`,
`results/cifar10_muown_ema_20260601/final_bins/val_loss.png`, and
`results/cifar10_muown_ema_20260601/final_bins/step_time_ms_bar.png`.

### ScheduleFree+ Combination Sweep

This study tested all 32 combinations of five ScheduleFree+ mechanisms wrapped
around the current PMuonEq + GramNS + NorMuon matrix direction:

- `P`: Polyak-style online LR multiplier
- `C`: `c_t` warmup
- `B`: beta annealing
- `D`: AdamC-style decay
- `M`: inner optimizer momentum

Protocol: `vit5_micro`, CIFAR-10 45k/5k train/validation split, official
10k test evaluated at the end, batch size 512, 20 epochs, seed `123`, BF16
autocast, channels-last tensors, 16 dataloader workers, and two GPUs scheduled
one trial per GPU.

Top rows from the 20-epoch run:

| Rank | Trial | Val acc | Test acc | Val loss | Test loss | Step |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93` | 84.92% | 84.41% | 0.4267 | 0.4598 | 17.26 ms |
| 2 | `sfplus_PCbDM_lr3_wd2` | 80.84% | 80.46% | 0.5665 | 0.5780 | 22.99 ms |
| 3 | `sfplus_PCBdM_lr3_wd2` | 79.94% | 79.23% | 0.5878 | 0.6098 | 23.00 ms |
| 4 | `sfplus_PCBDM_lr3_wd2` | 79.84% | 79.48% | 0.5794 | 0.6017 | 22.44 ms |
| 5 | `sfplus_PCbdM_lr3_wd2` | 79.72% | 79.49% | 0.5817 | 0.5954 | 22.92 ms |
| 6 | `adamw_cosine_lr0.004_wd0.001` | 73.44% | 72.58% | 0.7566 | 0.7796 | 11.60 ms |

Toggle marginals over SF+ rows:

| Toggle | On mean val acc | Off mean val acc | Delta |
|---|---:|---:|---:|
| Polyak LR | 74.47% | 75.20% | -0.72 |
| `c_t` warmup | 75.31% | 74.37% | +0.94 |
| beta annealing | 74.67% | 75.00% | -0.34 |
| AdamC decay | 74.97% | 74.70% | +0.27 |
| inner momentum | 78.95% | 70.73% | +8.22 |

Conclusion: SF+ did not produce a new winner in this proxy. The best SF+ row
was substantially better than AdamW but still trailed AnchorMuon WSD by 4.08
validation points and 3.95 official test points. The only large positive SF+
toggle was inner momentum; no-inner-momentum combinations are not worth more
training in this harness.

Full table and plots:
`results/sfplus_combo20_20260529/e20/summary.md`,
`results/sfplus_combo20_20260529/e20/val_acc.png`,
`results/sfplus_combo20_20260529/e20/val_loss.png`, and
`results/sfplus_combo20_20260529/e20/step_time_ms_bar.png`.

### Latest Root LR Schedule Sweep

This study used the shippable root `optimizer.py` directly. The optimizer does
not own the LR schedule; the trainer updated each parameter group's `lr`.

Protocol:

- HPO: 12 epochs, seed `123`, official training split divided into 45k train /
  5k validation.
- Schedule search: constant, cosine, linear, and WSD, plus AdamW cosine
  baseline.
- Final replay: only the best HPO config per schedule was extended to 50
  epochs; official CIFAR-10 test was evaluated only at the end.
- Shared settings: `vit5_micro`, batch size 512, BF16 autocast, channels-last,
  16 dataloader workers, two GPUs scheduled one trial per GPU.

Best 12-epoch HPO config per schedule:

| Schedule | Trial | LR | 12-epoch val loss | 12-epoch val acc | Step time |
|---|---|---:|---:|---:|---:|
| constant | `root_named_constant_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.6865 | 75.36% | 16.94 ms |
| cosine | `root_named_cosine_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5946 | 79.30% | 17.09 ms |
| linear | `root_named_linear_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5820 | 79.46% | 16.86 ms |
| WSD | `root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93` | 0.012 | 0.5632 | 80.08% | 16.06 ms |
| AdamW cosine | `adamw_cosine_lr0.004_wd0.001` | 0.004 | 0.9095 | 67.56% | 11.04 ms |

50-epoch replay of the schedule winners:

| Recipe | Schedule | Final val loss | Best val loss | Final val acc | Best val acc | Official test loss | Official test acc | Step | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnchorMuon | WSD | 0.4005 | 0.3716 | 88.36% | 88.36% | 0.4111 | 87.67% | 16.23 ms | 31.5k ex/s |
| AnchorMuon | constant | 0.4036 | 0.4036 | 86.06% | 86.54% | 0.4317 | 85.74% | 17.19 ms | 29.8k ex/s |
| AnchorMuon | cosine | 0.4511 | 0.4097 | 87.40% | 87.48% | 0.4588 | 86.90% | 16.96 ms | 30.2k ex/s |
| AnchorMuon | linear | 0.4434 | 0.4151 | 87.18% | 87.58% | 0.4597 | 86.75% | 17.02 ms | 30.1k ex/s |
| AdamW baseline | cosine | 0.6133 | 0.5998 | 79.62% | 79.64% | 0.6240 | 79.55% | 11.39 ms | 45.0k ex/s |

Conclusion: WSD is the best schedule for AnchorMuon in this single-seed study.
It improved official test accuracy by +8.12 percentage points over AdamW and
by +1.93 points over the previous three-seed mean constant-LR AnchorMuon result.
AdamW remains the fastest per step.

### Latest Component Removal Ablation

These ablations tested whether each part of the worker AnchorMuon recipe was
actually helping after short HPO and a longer replay. Each family received
12-epoch HPO, then the family winner was replayed for 50 epochs. Protocol:
`vit5_micro`, CIFAR-10 45k/5k train/validation split, official 10k test
evaluated only at the end, batch size 512, seed `123`, BF16 autocast,
channels-last tensors, 16 dataloader workers, and two GPUs scheduled one trial
per GPU.

Follow-up three-seed replay for the close variants:

| Run | Official test acc | Best val acc | Best val loss | Step | Throughput | Finding |
|---|---:|---:|---:|---:|---:|---|
| AnchorMuon full | 84.94% +/- 0.10 | 85.95% +/- 0.02 | 0.4186 +/- 0.0019 | 20.06 ms | 25.5k ex/s | Best mean quality, but the margin is tiny. |
| AnchorMuon -PMuonEq | 84.83% +/- 0.37 | 85.80% +/- 0.44 | 0.4193 +/- 0.0138 | 18.04 ms | 28.4k ex/s | Nearly tied and about 11% faster; PMuonEq is questionable on value-for-speed. |
| AnchorMuon -NorMuon | 84.54% +/- 0.63 | 85.52% +/- 0.49 | 0.4265 +/- 0.0130 | 18.92 ms | 27.1k ex/s | Worse mean quality; NorMuon still looks useful. |

The multi-seed replay does not confirm the single-seed impression that removing
PMuonEq is a quality win. It does confirm that PMuonEq costs meaningful speed
for only a very small mean quality gain in this worker harness.

Multi-seed report and plots:
`results/component_ablation_multiseed_20260529/summary.md`,
`results/component_ablation_multiseed_20260529/val_acc.png`,
`results/component_ablation_multiseed_20260529/val_loss.png`, and
`results/component_ablation_multiseed_20260529/step_time_ms_bar.png`.

Earlier single-seed 50-epoch replay, sorted by official test accuracy:

| Run | Final val loss | Best val loss | Final val acc | Best val acc | Official test loss | Official test acc | Step | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnchorMuon -PMuonEq | 0.4225 | 0.4129 | 85.98% | 86.04% | 0.4395 | 85.28% | 18.52 ms | 27.7k ex/s |
| AnchorMuon -NorMuon | 0.4237 | 0.4116 | 85.64% | 85.98% | 0.4460 | 84.99% | 18.90 ms | 27.1k ex/s |
| AnchorMuon full | 0.4385 | 0.4165 | 84.82% | 85.96% | 0.4451 | 84.88% | 20.16 ms | 25.4k ex/s |
| AnchorMuon -SODA | 0.4934 | 0.4648 | 84.66% | 84.70% | 0.5203 | 84.19% | 18.66 ms | 27.4k ex/s |
| AnchorMuon -GramNS | 0.5680 | 0.5444 | 79.66% | 80.40% | 0.5663 | 80.49% | 16.59 ms | 30.9k ex/s |
| AdamW baseline | 0.6133 | 0.5998 | 79.62% | 79.64% | 0.6240 | 79.55% | 11.56 ms | 44.3k ex/s |

Findings: SODA should not be judged from short runs alone. The no-SODA family
won the 12-epoch HPO stage, but fell behind by 50 epochs and lost 0.69 official
test points versus the full recipe. GramNS is clearly important, losing 4.39
official test points when removed. PMuonEq and NorMuon were less settled in
this single-seed worker-run ablation: removing PMuonEq gave the best official
test accuracy, while removing NorMuon gave the best validation loss. The
three-seed follow-up above reduced that uncertainty: full AnchorMuon has the
best mean quality, but PMuonEq's quality margin is small relative to its speed
cost. Validate the PMuonEq question through root `optimizer.py` before changing
the shippable default.

Full component-ablation report and plots:
`results/component_ablation_50e_20260529/summary.md`,
`results/component_ablation_50e_20260529/final_bins/val_acc.png`,
`results/component_ablation_50e_20260529/final_bins/val_loss.png`, and
`results/component_ablation_50e_20260529/final_bins/step_time_ms_bar.png`.

### Latest Proper-Split Replay

This replay holds out 5,000 examples from CIFAR-10 `train=True` for HPO and
validation, trains on the remaining 45,000 examples, and evaluates the official
CIFAR-10 `train=False` test split only once at the end of each selected final
run.

| Run | AdamW | NorMuon no aspect | NorMuon aspect |
|---|---:|---:|---:|
| 50 epoch final val loss | 0.6180 +/- 0.0043 | 0.4414 +/- 0.0228 | 0.4445 +/- 0.0130 |
| 50 epoch best val loss | 0.6043 +/- 0.0039 | 0.4310 +/- 0.0112 | 0.4231 +/- 0.0135 |
| 50 epoch final val acc | 79.69% +/- 0.22 | 84.97% +/- 0.35 | 85.25% +/- 0.32 |
| 50 epoch best val acc | 79.85% +/- 0.29 | 85.25% +/- 0.27 | 85.65% +/- 0.33 |
| Official test loss | 0.6338 +/- 0.0125 | 0.4607 +/- 0.0196 | 0.4595 +/- 0.0074 |
| Official test acc | 79.28% +/- 0.23 | 84.77% +/- 0.65 | 84.55% +/- 0.40 |
| 50 epoch step time | 11.59 ms +/- 0.09 | 19.95 ms +/- 0.08 | 19.92 ms +/- 0.66 |
| 50 epoch throughput | 44.16k ex/s +/- 0.34k | 25.67k ex/s +/- 0.10k | 25.73k ex/s +/- 0.86k |

Conclusion: the clean split preserves the large gap over AdamW. The no-aspect
NorMuon recipe is the current best by official test accuracy; the aspect recipe
has slightly better validation-checkpoint metrics but did not improve mean test
accuracy in this three-seed replay.

### Latest Aspect-Feedback Replay

This replay tested the cross-worker suggestion to add
`normuon_aspect_scale=True`, a post-NorMuon `sqrt(max(1, rows / cols))`
multiplier for tall matrices. The flag is disabled by default and the older
reported no-aspect behavior remains available.

HPO over aspect/no-aspect plus small `row_gamma`, `col_gamma`, and
`normuon_beta` changes selected the no-aspect recipe
`row_gamma=0.35`, `col_gamma=0.05`, `normuon_beta=0.95`. A 50-epoch replay used
three seeds.

| Run | AdamW | NorMuon no aspect | NorMuon aspect |
|---|---:|---:|---:|
| 50 epoch final val loss | 0.6210 +/- 0.0105 | 0.4225 +/- 0.0027 | 0.4288 +/- 0.0103 |
| 50 epoch best val loss | 0.6088 +/- 0.0063 | 0.4198 +/- 0.0043 | 0.4236 +/- 0.0083 |
| 50 epoch final val acc | 79.66% +/- 0.12 | 85.86% +/- 0.23 | 85.43% +/- 0.29 |
| 50 epoch best val acc | 79.96% +/- 0.13 | 85.98% +/- 0.09 | 85.60% +/- 0.17 |
| 50 epoch step time | 11.54 ms +/- 0.08 | 20.20 ms +/- 0.41 | 19.56 ms +/- 0.43 |
| 50 epoch throughput | 44.36k ex/s +/- 0.31k | 25.36k ex/s +/- 0.51k | 26.18k ex/s +/- 0.58k |

Conclusion: aspect scaling was close and slightly faster in this run, but did
not beat the retuned no-aspect recipe on validation loss or accuracy.

### Latest 50-Epoch Replay

This replay uses the same HPO-selected AdamW and NorMuon+base configs, but with
`AnchorMuon.sync_diagnostics=False` and `--no-sync-step-timing` so optimizer
diagnostics do not force per-step GPU synchronization.

| Run | AdamW | NorMuon+base |
|---|---:|---:|
| 50 epoch final val loss | 0.6210 +/- 0.0105 | 0.4193 +/- 0.0044 |
| 50 epoch best val loss | 0.6088 +/- 0.0063 | 0.4117 +/- 0.0018 |
| 50 epoch final val acc | 79.66% +/- 0.12 | 85.93% +/- 0.14 |
| 50 epoch best val acc | 79.96% +/- 0.13 | 86.00% +/- 0.24 |
| 50 epoch step time | 11.55 ms +/- 0.18 | 19.89 ms +/- 0.37 |
| 50 epoch throughput | 44.35k ex/s +/- 0.70k | 25.74k ex/s +/- 0.48k |

NorMuon+base is again better on validation loss and accuracy across all three
seeds. AdamW remains faster per step.

### Historical Synchronized-Diagnostics Run

See `results/cifar10_confidence_noradam_20260528/summary.md`.

| Run | AdamW | NorMuon+base |
|---|---:|---:|
| 20 epoch val loss | 0.7469 +/- 0.0110 | 0.5647 +/- 0.0086 |
| 20 epoch val acc | 73.55% +/- 0.48 | 80.09% +/- 0.20 |
| 20 epoch step time | 11.86 ms | 21.61 ms |
| 50 epoch val loss | 0.6167 +/- 0.0117 | 0.4284 +/- 0.0145 |
| 50 epoch val acc | 80.07% +/- 0.40 | 85.59% +/- 0.53 |
| 50 epoch step time | 11.78 ms | 21.69 ms |

NorMuon+base is better on validation loss and accuracy in this harness, while
AdamW is faster per step.

## Exact Reported Recipe

The latest observed best LR-schedule row uses this explicit root `AnchorMuon`
configuration:

```text
lr = 0.012
lr schedule = trainer-side 80-step warmup + WSD
lr_final_scale = 0.1
wsd_decay_frac = 0.2
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

The strongest multi-seed constant-LR rows used this explicit AnchorMuon
configuration:

```text
lr = 8e-3
lr schedule = trainer-side 80-step warmup + constant LR for the root optimizer
amuse = False
soda = "all"
soda_disables_weight_decay = True
pmuon_eq = True
pmuon_beta = 0.90
row_gamma = 0.35 for the latest proper-split winner, 0.30 for the earlier no-sync replay
col_gamma = 0.0 for the latest proper-split winner, 0.05 for the aspect-feedback replay
momentum = 0.95
normuon = True
normuon_beta = 0.93 for the latest proper-split winner, 0.95 for the aspect-feedback replay
normuon_aspect_scale = False for the current best recipe
mimuon = False
ns_steps = 5
sync_diagnostics = False for the latest run
```

The root optimizer now uses effective tensor shape only for routing and keeps
names only for summaries. A sufficiently large 2D classifier/head/embedding
matrix will therefore enter the PMuonEq/GramNS/NorMuon path unless a trainer
supplies explicit param groups.

`AnchorMuon.sync_diagnostics` now defaults to `False`. The committed historical
metrics include `mean_update_rms` and `mean_precond_matrix_rms`, but those values
required GPU-to-CPU synchronization inside `step()`. Enable
`sync_diagnostics=True` only for debugging runs where that timing perturbation is
acceptable.

## Reproduction Commands

From this folder, with the repo environment active:

```bash
python -m py_compile experiments/run_cifar10_ablation.py optim_anchormuon.py
python -m pytest -q tests/test_anchormuon_modes.py
python -m torch.distributed.run --standalone --nproc-per-node=<num_gpus> tests/ddp_smoke_anchormuon.py
```

The HPO run used:

```bash
python experiments/run_cifar10_ablation.py \
  --preset confidence \
  --output-dir results/cifar10_confidence_noradam_20260528/hpo \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 12 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 100 \
  --model vit5_micro
```

Final 20-epoch and 50-epoch runs used `--final-from-hpo` with the HPO
`best_by_family.json`, seeds as recorded in the result summaries, and all visible
GPUs scheduled one trial per GPU.

The latest no-sync final replay used:

```bash
OUT=results/cifar10_feedback_nosync_20260529/final50
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --final-from-hpo results/cifar10_confidence_noradam_20260528/hpo/best_by_family.json \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --final-epochs 50 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 200 \
  --model vit5_micro \
  --seeds 123,456,789 \
  --no-sync-step-timing
```

The latest aspect-feedback HPO and replay used:

```bash
OUT=results/cifar10_feedback_aspect_20260529/hpo
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --preset feedback \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 12 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 200 \
  --model vit5_micro \
  --no-sync-step-timing

OUT=results/cifar10_feedback_aspect_20260529/final50
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --final-from-hpo results/cifar10_feedback_aspect_20260529/hpo/best_by_family.json \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --final-epochs 50 \
  --train-subset 0 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 500 \
  --model vit5_micro \
  --seeds 123,456,789 \
  --no-sync-step-timing
```

The latest proper-split HPO and final replay used:

```bash
OUT=results/cifar10_proper_split_20260529/hpo
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --preset feedback \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --epochs 12 \
  --train-subset 0 \
  --val-source train_split --train-val-size 5000 --val-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 200 \
  --model vit5_micro \
  --no-sync-step-timing

OUT=results/cifar10_proper_split_20260529/final50
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --final-from-hpo results/cifar10_proper_split_20260529/hpo/best_by_family.json \
  --output-dir "$OUT" \
  --data-path /home/catid/screen/repos/TinyRecursiveModels/data/cifar10 \
  --final-epochs 50 \
  --train-subset 0 \
  --val-source train_split --train-val-size 5000 --val-subset 0 \
  --eval-test --test-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --warmup-steps 80 \
  --eval-bins 8 \
  --log-every 500 \
  --model vit5_micro \
  --seeds 123,456,789 \
  --no-sync-step-timing
```

The latest root LR-schedule HPO and final replay used:

```bash
OUT=results/root_lr_schedule_sweep_20260529/hpo
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --preset root_lr_schedule_sweep \
  --output-dir "$OUT" \
  --epochs 12 \
  --train-subset 0 --val-subset 0 \
  --val-source train_split --train-val-size 5000 \
  --batch-size 512 \
  --num-workers 16 \
  --seed 123 \
  --warmup-steps 80 \
  --lr-final-scale 0.1 \
  --wsd-decay-frac 0.2 \
  --eval-bins 8 \
  --log-every 200 \
  --no-sync-step-timing

OUT=results/root_lr_schedule_sweep_20260529/final50_schedule_winners
/home/catid/screen/.venv/bin/python experiments/run_cifar10_ablation.py \
  --preset root_lr_schedule_sweep \
  --only '^(adamw_cosine_lr0\.004_wd0\.001|root_named_(constant|cosine|linear|wsd)_lr0\.012_rg0\.35_pb0\.9_nb0\.93)$' \
  --output-dir "$OUT" \
  --epochs 50 \
  --train-subset 0 --val-subset 0 \
  --val-source train_split --train-val-size 5000 \
  --eval-test --test-subset 0 \
  --batch-size 512 \
  --num-workers 16 \
  --seed 123 \
  --warmup-steps 80 \
  --lr-final-scale 0.1 \
  --wsd-decay-frac 0.2 \
  --eval-bins 8 \
  --log-every 200 \
  --no-sync-step-timing
```
