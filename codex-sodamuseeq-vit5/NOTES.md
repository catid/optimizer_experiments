# Notes

## Custom Folder Ownership

This folder is the Codex worker area for the optimizer experiments monorepo.
Other workers should be able to work elsewhere in the repository without merge
conflicts.

## Current Caveats

- The focused confidence run was still in progress in the source workspace when
  this package was prepared.
- The GitHub CLI token in the source environment was invalid and SSH access to
  `git@github.com:catid/optimizer_experiments.git` failed with
  `Permission denied (publickey)`.
- A local commit can be produced, but pushing requires valid GitHub
  authentication.

## Intended Next Experiment

Use `vit5/experiments/focused_optimizer_confidence.py` to complete:

1. 1k-step HPO for the three requested optimizer families.
2. 3k top-config sanity pass.
3. 3-seed 10k final comparison.
4. 20k seed-0 longer-budget check.

The key ranking question is whether `NorMuon+BaseGram`, which looked strong in
early focused HPO, holds up against `SODA+PMuonEq+Gram` at 10k and 20k.
