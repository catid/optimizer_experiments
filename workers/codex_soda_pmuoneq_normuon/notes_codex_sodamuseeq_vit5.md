# Notes from `codex_sodamuseeq_vit5`

Pulled latest on 2026-05-29 and reread all worker result summaries. Your
`row+aspect` idea has become the leading candidate: it is supported by your
single-seed 50-epoch aspect ablation and by my 10k, 3-seed compact ViT-5
rerun after porting it.

## Current Cross-Worker Readout

Your latest aspect ablation:

| optimizer | best val loss | best val acc | mean step |
|---|---:|---:|---:|
| SODA-PMuonEq-NorMuon row + aspect | 0.4036 | 87.16% | 34.14 ms |
| SODA-PMuonEq-NorMuon row | 0.4174 | 86.43% | 34.29 ms |
| SODA-PMuonEq-NorMuon orientation | 0.4212 | 86.98% | 34.16 ms |
| AdamW baseline | 0.5476 | 83.03% | 18.91 ms |

My compact 10k, 3-seed rerun after porting row+aspect:

| optimizer | best val loss | test acc | steps/sec |
|---|---:|---:|---:|
| SODA+PMuonEq+Gram+NorMuon row+aspect | 0.4079 +/- 0.0090 | 87.09% +/- 0.15 | 42.84 |
| SODA+PMuonEq+Gram | 0.4210 +/- 0.0097 | 86.90% +/- 0.35 | 43.05 |
| AdamW | 0.6030 +/- 0.0167 | 81.27% +/- 0.74 | 58.41 |

This is the best evidence we have for one recipe. The quality gain over plain
`SODA+PMuonEq+Gram` is modest but repeatable in my run, and the quality gap
over AdamW is large in every worker's CIFAR result.

## Remaining Shortcomings

- Your aspect ablation is still one seed. The direction is confirmed elsewhere,
  but your own harness should get a 3-seed row+aspect vs no-aspect vs AdamW
  replay before claiming final stability.
- The optimizer is about 1.8x slower than AdamW in your small ViT-5 setting.
  For LM, report both validation loss vs step and validation loss vs wall-clock.
- The direct SODA anchor placement is not the same as AMUSE/SF fast-iterate
  SODA. That is fine, but keep calling it direct SODA, not SODA-AMUSE.
- `last_stats` only describes the last matrix group if users create multiple
  matrix groups. Aggregate stats if you expect custom grouping.
- Add or keep a `use_external_lr=True` test. LM harnesses often own their LR
  schedule externally.

## What I Would Try

- Multi-seed replay of row+aspect in this harness.
- One no-NorMuon `SODA+PMuonEq+Gram` ablation with retuned LR in this same
  harness to quantify the additive NorMuon contribution.
- LM smoke with the default row+aspect recipe on only transformer MLP/attention
  matrices, keeping embeddings, heads, norms, biases, and tiny tensors in
  fallback.

## LM Recommendation

This is the implementation style I would use for the next language-model
experiment: fixed, no AMUSE, no broad branch soup, named parameter grouping,
direct SODA, PMuonEq before Gram, NorMuon row+aspect after Gram.
