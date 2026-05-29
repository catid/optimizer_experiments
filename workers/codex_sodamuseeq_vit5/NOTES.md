# Notes

## Custom Folder Ownership

This folder is the Codex worker area for the optimizer experiments monorepo.
Other workers should be able to work elsewhere in the repository without merge
conflicts.

## Current Caveats

- This package now lives under `workers/codex_sodamuseeq_vit5/` so the other
  workers' notes-discovery scripts can find it.
- Earlier no direct peer notes were addressed to this package because it lived
  at repository root as `codex-sodamuseeq-vit5/`. I renamed the notes I left in
  other folders to `notes_codex_sodamuseeq_vit5.md` to match the worker folder.
- The implementation now includes explicit `state_dict()` metadata for
  schedule-free train/eval mode and the SODA step counter. Without that, normal
  PyTorch optimizer checkpointing would restore tensor state but could restart
  the SODA schedule incorrectly after resume.

## Focused Confidence Result

## Peer Feedback Rerun

I ported the row+aspect NorMuon idea from
`workers/codex_soda_pmuoneq_normuon` into this package and reran the focused
comparison with 1k HPO, 3k top-selection, and 10k final 3-seed runs over:

- AdamW.
- NorMuon+BaseGram.
- SODA+PMuonEq+Gram.
- SODA+PMuonEq+Gram+NorMuon row+aspect.

Best final result:

| Variant | Val loss | Val acc | Test acc | Steps/sec |
| --- | ---: | ---: | ---: | ---: |
| SODA+PMuonEq+Gram+NorMuon row+aspect | 0.4079 +/- 0.0090 | 87.15% +/- 0.08 | 87.09% +/- 0.15 | 42.84 +/- 0.06 |
| SODA+PMuonEq+Gram | 0.4210 +/- 0.0097 | 87.19% +/- 0.24 | 86.90% +/- 0.35 | 43.05 +/- 0.27 |
| NorMuon+BaseGram | 0.5069 +/- 0.0100 | 86.21% +/- 0.49 | 86.05% +/- 0.36 | 47.44 +/- 0.34 |
| AdamW | 0.6030 +/- 0.0167 | 81.47% +/- 0.75 | 81.27% +/- 0.74 | 58.41 +/- 0.24 |

Best row+aspect tuning:

```text
lr = 0.012
pmuon_beta = 0.90
pmuon_row_gamma = 0.15
pmuon_col_gamma = 0.0
normuon_beta2 = 0.90
normuon_mode = row
normuon_aspect_scale = True
use_amuse = False
weight_decay = 0.0
```

Tracked artifacts are in `results/focused_peer_aspect/`.

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

Committed artifacts are in `results/focused_optimizer_confidence/`:

- `experimental_results.md`: compact written result summary.
- `val_loss_best_vs_adamw.png`: validation-loss curve for the winner versus
  AdamW.
- `train_loss_best_vs_adamw.png`: train interval-loss curve for the winner
  versus AdamW.
- `val_acc_best_vs_adamw.png`: validation-accuracy curve for the winner versus
  AdamW.
- `final_test_accuracy_bar.png`: 10k final test-accuracy comparison.
- `iteration_speed_steps_per_sec.png`: iteration-speed comparison.
- `final10k_all_runs.csv` and `long20k_all_runs.csv`: raw result tables.

## Feedback Applied

- Added train-mode and eval-mode checkpoint resume parity tests.
- Added same-shape matrix bucket parity coverage for batched Gram projection.
- Strengthened schedule-free `train()` / `eval()` roundtrip tests.
- Strengthened the DDP smoke test to check optimizer tensor state and eval/train
  weight transforms, not only train-mode parameters.
- Updated documentation to make SODA weight-decay replacement and checkpoint
  semantics explicit.
