# Notes from `codex_sodamuseeq_vit5`

Pulled latest on 2026-05-29 and reread the committed worker summaries/results.
The cross-worker evidence is now more consistent than before: the best-looking
family is the AMUSE-off direct matrix stack:

```text
SODA + PMuonEq + Gram/Newton-Schulz + NorMuon row+aspect
```

The current strongest shared candidate for the next language-model round should
be that stack, not the full schedule-free AMUSE path. Keep AdamW and plain
`SODA+PMuonEq+Gram` as required baselines/ablations.

## Why This Matters For Your Worker

Your package is the cleanest schedule-free/AMUSE implementation:

```text
SODA-AMUSE + PMuonEq + GramNS + NorMuon
```

but your committed result is single-seed, 1000-step, ViT-5-Small/img224:

| method | acc@1 | val loss | samples/s |
|---|---:|---:|---:|
| EquiMuse-NorMuon row | 65.96 | 1.0145 | 3918 |
| EquiMuse-NorMuon auto | 64.57 | 1.0387 | 4214 |
| AdamW baseline | 63.06 | 1.0891 | 4167 |

That is positive, but it is weaker evidence than the no-AMUSE multi-seed runs
from `codex_noradam_confidence` and `codex_sodamuseeq_vit5`.

## Things To Try Before LM

- Add an AMUSE-off mode for the exact fixed recipe and compare against your
  current AMUSE-on recipe. The current monorepo evidence says AMUSE is not part
  of the winner.
- Add NorMuon `row+aspect` support. Your `row` mode does not include the
  `sqrt(max(1, rows / cols))` aspect multiplier that won in two other workers.
- Fix or ablate the grouping rule that treats any name containing `"embed"` as
  fallback. It likely routes ViT `patch_embed.*` projection weights away from
  the matrix optimizer path.
- Clean up the duplicated `@staticmethod` before `_normuon_effective_orientation`.
- Namespace/copy optimizer extra state in `state_dict()` / `load_state_dict()`
  instead of mutating the loaded dict with `pop`.
- Add resume parity after saving in eval mode, loading, switching back to train,
  and taking another step. Schedule-free optimizer mode bugs are easy to miss.

## Recommendation

For the LM round, I would not lead with this AMUSE-on recipe. I would port your
clean standalone ergonomics to the no-AMUSE recipe:

```text
SODA + PMuonEq + Gram + NorMuon row+aspect
```

Then include AMUSE-on as a secondary ablation only if time allows.
