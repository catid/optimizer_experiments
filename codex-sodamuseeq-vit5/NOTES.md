# Notes

## Custom Folder Ownership

This folder is the Codex worker area for the optimizer experiments monorepo.
Other workers should be able to work elsewhere in the repository without merge
conflicts.

## Current Caveats

- This package is on `codex/sodamuseeq-vit5`; the shared review notes branch is
  separate and contains notes left in the other worker folders.
- No direct peer notes addressed to `codex-sodamuseeq-vit5` were present on the
  remote branches when checked on 2026-05-28. I applied the peer feedback that
  was generally relevant to this package.
- The implementation now includes explicit `state_dict()` metadata for
  schedule-free train/eval mode and the SODA step counter. Without that, normal
  PyTorch optimizer checkpointing would restore tensor state but could restart
  the SODA schedule incorrectly after resume.

## Focused Confidence Result

The focused comparison completed in the source workspace:

- `SODA+PMuonEq+Gram` was best quality: 10k 3-seed val loss
  `0.4210 +/- 0.0097`, val acc `87.19% +/- 0.24`, test acc
  `86.90% +/- 0.35`, `42.84 +/- 0.20` steps/sec.
- `NorMuon+BaseGram` was faster but worse quality: 10k 3-seed val loss
  `0.5069 +/- 0.0100`, val acc `86.21% +/- 0.49`, test acc
  `86.05% +/- 0.36`, `47.93 +/- 0.67` steps/sec.
- `AdamW` remained fastest but substantially worse: 10k 3-seed val loss
  `0.6030 +/- 0.0167`, val acc `81.47% +/- 0.75`, test acc
  `81.27% +/- 0.74`, `58.46 +/- 0.42` steps/sec.

The same ordering held in the 20k seed-0 check.

## Feedback Applied

- Added train-mode and eval-mode checkpoint resume parity tests.
- Added same-shape matrix bucket parity coverage for batched Gram projection.
- Strengthened schedule-free `train()` / `eval()` roundtrip tests.
- Strengthened the DDP smoke test to check optimizer tensor state and eval/train
  weight transforms, not only train-mode parameters.
- Updated documentation to make SODA weight-decay replacement and checkpoint
  semantics explicit.
