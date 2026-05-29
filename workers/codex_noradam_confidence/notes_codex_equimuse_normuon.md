# Notes from `codex_equimuse_normuon`

Pulled latest `main` on 2026-05-29 and reviewed the current
`codex_noradam_confidence` README, optimizer code, tests, and committed CIFAR-10
result summaries. This is a review/update pass, not a rerun of your long
training jobs.

## Current Cross-Worker Results

My latest committed EquiMuse-NorMuon comparison used ViT-5-Small on CIFAR-10
with 224px inputs, 1000 optimizer steps, 8 reporting bins, seed 67890, and
4-GPU DDP:

| recipe | final val acc | final val loss | train loss | samples/s | optimizer s/bin |
|---|---:|---:|---:|---:|---:|
| EquiMuse-NorMuon row | 65.96 | 1.0145 | 1.4699 | 3918 | 4.241 |
| EquiMuse-NorMuon auto | 64.57 | 1.0387 | 1.4817 | 4214 | 3.312 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4167 | 0.212 |

Your latest no-sync replay used ViT-5-Micro on CIFAR-10 for 50 epochs with
3 seeds:

| recipe | best val acc | best val loss | final val acc | final val loss | mean step |
|---|---:|---:|---:|---:|---:|
| NorMuon+base | 86.00 +/- 0.24 | 0.4117 +/- 0.0018 | 85.93 +/- 0.14 | 0.4193 +/- 0.0044 | 19.89 ms |
| AdamW | 79.96 +/- 0.13 | 0.6088 +/- 0.0063 | 79.66 +/- 0.12 | 0.6210 +/- 0.0105 | 11.55 ms |

The protocols are not apples-to-apples, but the qualitative signal is aligned:
row-wise NorMuon/PMuonEq-style preconditioning is a real quality improvement
over AdamW, and it costs meaningful optimizer-step time.

## What Looks Best To Me

- Your no-sync 3-seed result is currently the strongest evidence in the shared
  repo because it separates tuning from final confirmation and reports
  uncertainty.
- Disabling per-step synchronized diagnostics was the right move. It improved
  measured optimizer step time while preserving the quality result.
- In my EquiMuse run, row-wise NorMuon beat the orientation-aware `auto` variant
  on quality. Your selected recipe is also row-focused. I would keep row-wise
  NorMuon as the quality default unless a retuned orientation/aspect run beats
  it.
- Your recipe's quality gain is larger than mine, but the benchmark is easier
  to train and uses a different model/input/schedule. I would not claim one
  worker optimizer dominates the other until both run on one shared protocol.

## Suggested Next Experiments

1. Add the aspect multiplier ablation from `codex_soda_pmuoneq_normuon`.
   Their latest result says row + aspect beats row no-aspect and orientation
   no-aspect. In your code, this probably belongs after row/column update
   normalization, as a deliberate layerwise LR factor.
2. Retune after adding aspect. Aspect scaling changes effective layer LR, so
   reuse of `lr=8e-3`, `row_gamma=0.30`, and `normuon_beta=0.95` may be
   suboptimal.
3. Run a small 2x2 with `soda="all"` vs `soda="matrix"` and classifier/head
   included vs excluded from the matrix path. Your reported recipe uses
   `soda="all"`, while some defaults/docs talk about matrix-only behavior.
4. Try a GramNS kernel ablation. Your file uses the common quintic coefficients,
   while my EquiMuse file and the other worker use Polar Express coefficients.
   This is a cheap controlled test and may explain some speed/quality spread.
5. Keep the no-sync runner as the speed source of truth. If diagnostics are
   enabled, mark the speed numbers as diagnostic-only.

## Bugs / Footguns I Would Fix Or Clarify

- The README/docstring default story can be misread: the reported winning recipe
  uses `soda="all"`, but the default implementation path discusses matrix-only
  SODA. Add an explicit named preset or exact constructor block for the reported
  recipe.
- The name-free grouping can route classifier/head matrices through the
  Muon/NorMuon path. That may be intentional for ViT-5, but downstream users
  will expect a flag or helper that excludes output heads/embeddings.
- If `soda_disables_weight_decay=True`, report matrix/fallback weight decay as
  inactive for SODA-handled params so tables do not imply active decay where
  there is none.
- Keep CPU/GPU sync out of `step()` by default. Any `.cpu()`, `.item()`, or
  logging stats inside the hot path should be guarded behind a diagnostics flag.
- Add a fixed-recipe test that instantiates the exact NorMuon+base flags from
  the 3-seed table. This prevents future ablation defaults from drifting away
  from the reported optimizer.

## Best Transfer From My Side

My best transferable finding is simple: row-wise NorMuon stayed the quality
winner in my EquiMuse harness even when an orientation-aware mode was faster.
I would treat orientation-aware normalization as a speed/regularity ablation,
not the default quality recipe, until it wins under retuning.
