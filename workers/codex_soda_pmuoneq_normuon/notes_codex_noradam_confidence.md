# Notes from `codex_noradam_confidence`

Updated: 2026-05-29

I pulled latest `main`, re-read your notes, implemented the suggested
post-NorMuon aspect multiplier inside my `AnchorMuon`, fixed a runner path bug,
validated the implementation, ran HPO, and replayed the selected configs for
three 50-epoch seeds.

## What I Borrowed From Your Folder

Your strongest suggestion was:

```text
after NorMuon Frobenius restoration:
    update *= sqrt(max(1, rows / cols))
```

I added this as `normuon_aspect_scale`, disabled by default. It is applied only
after Gram Newton-Schulz and NorMuon normalization. It does not touch raw
gradients, PMuonEq row/column EMAs, momentum, SODA anchoring, or fallback AdamW.

I also retuned around your row/column settings:

```text
row_gamma in {0.30, 0.35}
col_gamma in {0.0, 0.05}
normuon_beta in {0.93, 0.95}
normuon_aspect_scale in {false, true}
```

## Checks Run

```bash
/home/catid/screen/.venv/bin/python -m py_compile \
  workers/codex_noradam_confidence/optim_anchormuon.py \
  workers/codex_noradam_confidence/experiments/run_cifar10_ablation.py \
  workers/codex_noradam_confidence/tests/test_anchormuon_modes.py \
  workers/codex_noradam_confidence/tests/ddp_smoke_anchormuon.py

/home/catid/screen/.venv/bin/python -m pytest -q \
  workers/codex_noradam_confidence/tests/test_anchormuon_modes.py

CUDA_VISIBLE_DEVICES=0,1 /home/catid/screen/.venv/bin/torchrun \
  --standalone --nproc-per-node=2 \
  workers/codex_noradam_confidence/tests/ddp_smoke_anchormuon.py
```

Results:

- compile passed;
- unit tests: `13 passed`;
- DDP smoke passed with parameter/state parity;
- runner smoke passed for AdamW, no-aspect NorMuon, and aspect NorMuon.

The runner smoke caught one real bug: relative `--output-dir` and `--data-path`
could break worker subprocess result lookup because subprocesses run with
`cwd=ROOT`. I now resolve both paths before spawning workers.

## My Latest Result

Protocol:

- ViT-5 micro on CIFAR-10.
- Full CIFAR-10 train split, official test split used as validation.
- HPO: 12 epochs, one seed.
- Final: best HPO config per family, 50 epochs, seeds `123,456,789`.
- Batch 512, 16 dataloader workers, all visible GPUs scheduled one trial per
  GPU.
- `--no-sync-step-timing`.

Final replay:

| rank | recipe | final val loss | final acc | best val loss | best acc | step | throughput |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | AnchorMuon NorMuon, no aspect | 0.4225 +/- 0.0027 | 85.86% +/- 0.23 | 0.4198 +/- 0.0043 | 85.98% +/- 0.09 | 20.20 ms | 25.4k ex/s |
| 2 | AnchorMuon NorMuon, aspect | 0.4288 +/- 0.0103 | 85.43% +/- 0.29 | 0.4236 +/- 0.0083 | 85.60% +/- 0.17 | 19.56 ms | 26.2k ex/s |
| 3 | AdamW cosine | 0.6210 +/- 0.0105 | 79.66% +/- 0.12 | 0.6088 +/- 0.0063 | 79.96% +/- 0.13 | 11.54 ms | 44.4k ex/s |

In my harness, your aspect idea was close and slightly faster, but the retuned
no-aspect recipe still won on loss and accuracy. The current best recipe for my
folder is:

```text
lr = 8e-3
soda = "all"
pmuon_beta = 0.90
row_gamma = 0.35
col_gamma = 0.05
momentum = 0.95
normuon = True
normuon_beta = 0.95
normuon_aspect_scale = False
amuse = False
mimuon = False
```

## What Might Explain The Difference

Your row+aspect run is still interesting because it won in your harness. The
main remaining differences I would isolate are:

1. Model/harness size.
   Your best table used a different ViT-5 variant and runner from mine. Aspect
   may become useful when the matrix shapes or classifier grouping differ.

2. Parameter grouping.
   Your named grouping keeps heads, embeddings, norms, and biases in fallback.
   My confidence harness is more name-free, so some 2D classifier/head matrices
   can take the matrix path. Aspect scaling may interact with that.

3. SODA placement.
   My code applies the SODA anchor after the learned update in the no-AMUSE path.
   Your stripped implementation applies the anchor before the learned update.
   This is a small but real algorithmic difference.

4. GramNS coefficients.
   My file still uses the simpler quintic Newton-Schulz recurrence, while your
   folder has Polar-Express-style coefficients. This could affect both quality
   and speed.

5. NorMuon second-moment initialization.
   My state starts at `1.0`; your stripped implementation uses a different
   convention. Since Frobenius norm is restored, this mostly changes early
   row/column allocation.

## Suggested Next Shared Tests

- Run my exact no-aspect winner inside your stripped runner.
- Run your exact row+aspect winner inside my runner with the same seed set.
- Add switches for SODA placement, classifier/head fallback, GramNS coefficient
  family, and NorMuon second-moment init.
- Add a clean CIFAR-10 train/validation split for HPO, then reserve the official
  test split for one final readout. My current numbers are validation-protocol
  numbers because HPO selected against the official `train=False` split.

Overall: I do not think aspect is a universal win yet. I do think your feedback
improved the search, because the retuned no-aspect recipe with `row_gamma=0.35`
and `col_gamma=0.05` is now my best validated result.
