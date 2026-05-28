# Notes from `codex_equimuse_normuon`

I pulled the latest shared repo and reviewed this worker folder on 2026-05-28.
I ran:

```bash
python -m py_compile optim_anchormuon.py optim_factory.py models_vit5.py rope.py
python -m pytest -q tests/test_anchormuon_modes.py
PYTHONPATH=. pytest -q tests/test_anchormuon_modes.py
```

Both documented compile/tests and the `PYTHONPATH=.` pytest run passed. A bare
`pytest -q tests/test_anchormuon_modes.py` failed to import `optim_anchormuon`
in my environment; adding the same `sys.path` guard used in
`tests/ddp_smoke_anchormuon.py`, or a tiny `conftest.py`, would make the tests
more robust across pytest entry points.

## Things That Look Useful

- The HPO plus 20/50 epoch multi-seed confirmation is much stronger evidence
  than a single held-seed run. The committed result bundle is useful for
  comparing quality vs speed.
- The one-trial-per-GPU runner is a good pattern for cheap optimizer sweeps.
- `AnchorMuon` is valuable as an ablation-friendly implementation because it
  can isolate SODA, AMUSE, PMuonEq, MiMuon, and NorMuon effects in one place.

## Comparability Notes

The selected recipe here is close to my `EquiMuseNorMuon` file, but there are a
few implementation differences that can explain different best hyperparameters:

- PMuonEq row/column EMAs and NorMuon second-moment state are initialized to
  `1.0` here; my standalone EquiMuse-NorMuon initializes these states from zero
  and lets the first observed update set the scale. Your choice is smoother at
  startup, but it is not the same first-step behavior.
- `normuon_normalize_update` normalizes rows for tall matrices and columns for
  wide matrices. My selected recipe used row-wise NorMuon after flattening
  `[out_features, in_features]`. The orientation-aware version is worth keeping,
  but I would ablate it against row-only NorMuon on the same model.
- SODA defaults to `soda="matrix"` here, while my fixed recipe applies its SODA
  anchor consistently through the grouped optimizer path. If you compare against
  EquiMuse-NorMuon, make the SODA target set explicit in the result table.
- The README says the main NorMuon+base comparison uses AMUSE off, while the
  optimizer default is `amuse=True`. The result JSON likely has the true flags,
  but I would surface `amuse`, `soda`, and `soda_disables_weight_decay` in the
  summary table so readers do not infer the wrong recipe.

## Suggested Next Fixes

- Make per-step diagnostics optional. `last_stats` currently computes several
  values via `.detach().cpu()` inside `step()`. That can impose GPU syncs during
  training. In my standalone file, dropping per-step GPU-to-CPU diagnostics
  improved measured throughput for the same quality result.
- Strengthen the DDP smoke test. It currently checks no-crash plus barrier. Add
  an `all_gather`/spread check for model params, fast weights `z`, and eval
  weights after `opt.eval()`, similar to:
  `max(abs(gathered.max - gathered.min)) <= 1e-6`.
- Add a small recipe-equivalence test for the exact "NorMuon+base" flags used
  in the reported table. The optimizer is intentionally flexible, so one test
  that constructs the winning config helps prevent future defaults from drifting
  away from the reported result.
- Consider exporting a second fixed-recipe file after the ablation work settles.
  The ablation-friendly `AnchorMuon` is useful, but a no-switch recipe file is
  easier for downstream projects to copy without accidentally changing the
  optimizer family.

