# Golden Optimizer Validation

Updated: 2026-05-29

## Golden Target

The golden optimizer file is:

```text
workers/codex_soda_pmuoneq_normuon/golden_soda_pmuoneq_normuon.py
```

It implements only the repo-wide winning candidate for the next language-modeling round:

```text
SODA + row-only PMuonEq + GramNS + NorMuon
AMUSE: off
MiMuon: off
column PMuonEq: off
NorMuon aspect scale: off
```

The exact tuned reference row is the proper-split winner from
`workers/codex_noradam_confidence`:

```text
normuon_mlr0.008_rg0.35_cg0_mom0.95_pb0.9_nb0.93
```

That row used:

```text
matrix_lr       = 8e-3
row_gamma       = 0.35
col_gamma       = 0.0
momentum        = 0.95
pmuoneq_beta    = 0.90
normuon_beta2   = 0.93
aspect          = false
```

The older `soda_pmuoneq_normuon.py` research file can still reproduce the local
row+aspect CIFAR proxy. The golden file intentionally does not implement that
ablation, because the no-aspect row won the cleanest official-test protocol.

## Validation Performed

Command:

```bash
/home/catid/attractor/.venv/bin/python -m pytest -q \
  workers/codex_soda_pmuoneq_normuon/tests/test_golden_soda_pmuoneq_normuon.py \
  workers/codex_soda_pmuoneq_normuon/tests/test_soda_pmuoneq_normuon.py
```

Result:

```text
17 passed, 1 warning in 1.88s
```

The warning is the existing PyTorch `pynvml` deprecation warning.

## What The Tests Cover

- Named-parameter grouping keeps heads, embeddings, norms, and biases in the fallback path.
- Golden param groups do not expose removed ablation flags:
  `amuse`, `mimuon`, `col_gamma`, `normuon_mode`, `normuon_aspect_scale`.
- A golden optimizer step is finite and creates only the row PMuonEq state:
  no column EMA/factor state is created.
- Golden optimizer matches the old research optimizer exactly when the old file
  is configured as:

  ```text
  row_gamma = 0.35
  col_gamma = 0.0
  normuon_mode = "row"
  normuon_aspect_scale = False
  pmuoneq_beta = 0.90
  normuon_beta2 = 0.93
  ```

- Same-shape matrix bucketing matches split matrix groups.
- State-dict save/resume matches uninterrupted training.
- A short fixed-data training sanity check decreases loss.

## Reproduction Claim

The golden file reproduces the optimizer update rule for the selected best
no-aspect recipe. It has not rerun the full 50-epoch CIFAR job in this commit.
The full-result claim remains inherited from the previously recorded
proper-split no-aspect run; this commit validates that the stripped golden
library computes the same update path as the old implementation configured to
that recipe.
