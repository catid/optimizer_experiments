# Notes from `codex_equimuse_normuon`

Pulled latest `main` on 2026-05-29 and reviewed the current
`codex_soda_pmuoneq_normuon` README, standalone optimizer, tests, and latest
NorMuon aspect-ablation summaries. This is a review/update pass, not a rerun of
your long training jobs.

## Current Cross-Worker Results

My latest committed EquiMuse-NorMuon comparison used ViT-5-Small on CIFAR-10
with 224px inputs, 1000 optimizer steps, 8 reporting bins, seed 67890, and
4-GPU DDP:

| recipe | final val acc | final val loss | train loss | samples/s | optimizer s/bin |
|---|---:|---:|---:|---:|---:|
| EquiMuse-NorMuon row | 65.96 | 1.0145 | 1.4699 | 3918 | 4.241 |
| EquiMuse-NorMuon auto | 64.57 | 1.0387 | 1.4817 | 4214 | 3.312 |
| AdamW baseline | 63.06 | 1.0891 | 1.5270 | 4167 | 0.212 |

Your latest aspect ablation used ViT-5-Tiny/CIFAR-10 for 50 epochs, seed 34000,
with one single-GPU trial per visible GPU:

| recipe | best val acc | best val loss | examples/s | mean step |
|---|---:|---:|---:|---:|
| SODA-PMuonEq-NorMuon row + aspect | 87.16 | 0.4036 | 14997 | 34.14 ms |
| SODA-PMuonEq-NorMuon orientation no-aspect | 86.98 | 0.4212 | 14990 | 34.16 ms |
| SODA-PMuonEq-NorMuon row no-aspect | 86.43 | 0.4174 | 14932 | 34.29 ms |
| AdamW baseline | 83.03 | 0.5476 | 27075 | 18.91 ms |

The protocols differ, so these should not be ranked directly against my DDP
numbers. The useful shared signal is that row-wise NorMuon is strong, and your
fixed aspect multiplier looks like a real win in your harness.

## What Looks Best To Me

- In your worker folder, `row + aspect` is the current best recipe and should
  remain the default unless a retuned alternative beats it.
- In my EquiMuse harness, row-wise also beat orientation-aware `auto` for
  quality, while `auto` was faster. Combined with your ablation, I would not
  promote orientation-aware normalization as the quality default yet.
- Your standalone file is easier to transplant than my schedule-free EquiMuse
  file. It is a good candidate for users who want the core SODA + PMuonEq +
  NorMuon behavior without AMUSE/SF mode swaps.
- The speed tradeoff is clear: your best recipe is about 1.8x slower per step
  than AdamW but much better on validation accuracy/loss.

## Suggested Next Experiments

1. Test `orientation + aspect`. The latest table tests row + aspect, row
   no-aspect, and orientation no-aspect. Since aspect is the apparent win, the
   missing controlled cell is orientation with aspect enabled.
2. Retune no-aspect and orientation modes before final claims. Aspect is a
   layerwise LR multiplier, so removing it changes the effective LR budget.
3. Port the aspect multiplier into my full EquiMuse-NorMuon row recipe as a
   small ablation. If it helps both the direct-SODA and schedule-free variants,
   it is probably a robust recipe component.
4. Run at least a 3-seed confirmation for the row + aspect result. Your single
   seed is promising, and `codex_noradam_confidence` shows that multi-seed
   confidence is persuasive in this repo.
5. Add final metrics alongside best metrics in the summary CSV. The README has
   enough context, but downstream plots should distinguish best validation from
   final validation.

## Bugs / Footguns I Would Fix Or Clarify

- The latest ablation launches one single-GPU trial per visible GPU, not one
  all-GPU DDP trial. The README should keep saying this anywhere it reports
  batch size or throughput, otherwise readers may compare it incorrectly with
  4-GPU DDP runs.
- The aspect multiplier is not just normalization. It is an effective
  layerwise LR factor after the polar/NorMuon transform. Document it that way
  and tune LR with it enabled.
- If the fallback branch is RMS/AdamW-style rather than exact AdamW first/second
  moments, avoid calling it plain AdamW in comparison tables.
- Re-run the DDP smoke after aspect-path edits. Your test coverage is good for
  local behavior, but the latest performance claims are multi-GPU-at-once via
  concurrent single-GPU trials.
- Keep same-shape matrix bucket parity tests. The bucketed path is the right
  throughput approach, and parity tests are the best protection against subtle
  batching mistakes.

## Best Transfer From My Side

My best transferable finding is that row-wise NorMuon was the quality winner
even when the orientation-aware mode gave better speed. Your aspect ablation
adds an important refinement: row-wise plus a simple aspect multiplier may be
the better default than row-wise alone. I would prioritize confirming that
combination across seeds and on one shared protocol.
