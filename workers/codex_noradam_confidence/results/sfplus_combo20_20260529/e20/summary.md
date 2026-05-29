# SF+ Exhaustive Toggle Sweep

Date: 2026-05-29

This run tested all 32 combinations of five ScheduleFree+ mechanisms wrapped
around the existing PMuonEq + Gram Newton-Schulz + NorMuon matrix direction:

- `P`: Polyak-style online LR multiplier
- `C`: `c_t` warmup
- `B`: beta annealing
- `D`: AdamC-style decay
- `M`: inner optimizer momentum

Uppercase letters in a trial name mean that toggle was enabled. Lowercase
letters mean it was disabled.

## Protocol

- Model: `vit5_micro`
- Dataset: CIFAR-10 with 45,000 training examples and 5,000 held-out validation
  examples from the official training split
- Official test split: evaluated once at the end of each 20-epoch run
- Batch size: 512
- Epochs: 20
- Evaluation bins: 8
- Seed: 123
- Precision/layout: BF16 autocast, channels-last tensors
- Parallelism: one trial per visible GPU, two RTX PRO 6000 Blackwell GPUs
- Dataloader workers: 16 per trial
- Controls: tuned AdamW cosine and the current AnchorMuon WSD winner

Full data is in `all_runs.csv`. Curves and speed plots are in this directory:
`val_loss.png`, `val_acc.png`, `train_loss.png`, `step_time_ms_bar.png`, and
`examples_per_sec_bar.png`.

## Result

The exhaustive SF+ search did not beat the existing AnchorMuon WSD recipe. The
best SF+ row was `sfplus_PCbDM_lr3_wd2`, which reached 80.84% validation
accuracy and 80.46% official test accuracy. The AnchorMuon WSD control reached
84.92% validation accuracy and 84.41% official test accuracy in the same
20-epoch protocol.

The main useful finding is diagnostic: inner momentum is essential for this SF+
wrapper. Across all SF+ combinations, enabling `M` improved mean validation
accuracy by +8.22 percentage points. The other toggles were small or mixed.

## Sorted Runs

| Rank | Trial | Val acc | Test acc | Val loss | Test loss | Step ms | Ex/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `root_named_wsd_lr0.012_rg0.35_pb0.9_nb0.93` | 84.92% | 84.41% | 0.4267 | 0.4598 | 17.26 | 29.7k |
| 2 | `sfplus_PCbDM_lr3_wd2` | 80.84% | 80.46% | 0.5665 | 0.5780 | 22.99 | 22.3k |
| 3 | `sfplus_PCBdM_lr3_wd2` | 79.94% | 79.23% | 0.5878 | 0.6098 | 23.00 | 22.3k |
| 4 | `sfplus_PCBDM_lr3_wd2` | 79.84% | 79.48% | 0.5794 | 0.6017 | 22.44 | 22.8k |
| 5 | `sfplus_PCbdM_lr3_wd2` | 79.72% | 79.49% | 0.5817 | 0.5954 | 22.92 | 22.3k |
| 6 | `sfplus_pcbdM_lr0.008_wd2` | 79.50% | 78.13% | 0.6018 | 0.6263 | 23.14 | 22.1k |
| 7 | `sfplus_pcbDM_lr0.008_wd2` | 79.32% | 78.03% | 0.6044 | 0.6301 | 22.92 | 22.3k |
| 8 | `sfplus_pCbdM_lr0.008_wd2` | 79.24% | 78.94% | 0.5918 | 0.6111 | 22.93 | 22.3k |
| 9 | `sfplus_pCbDM_lr0.008_wd2` | 79.20% | 79.45% | 0.5866 | 0.6057 | 22.87 | 22.4k |
| 10 | `sfplus_pCBdM_lr0.008_wd2` | 78.50% | 78.31% | 0.6150 | 0.6312 | 22.81 | 22.4k |
| 11 | `sfplus_pcBDM_lr0.008_wd2` | 78.46% | 77.38% | 0.6278 | 0.6556 | 22.85 | 22.4k |
| 12 | `sfplus_pcBdM_lr0.008_wd2` | 78.44% | 77.59% | 0.6239 | 0.6527 | 22.87 | 22.4k |
| 13 | `sfplus_pCBDM_lr0.008_wd2` | 78.32% | 78.14% | 0.6162 | 0.6379 | 23.13 | 22.1k |
| 14 | `sfplus_PcbDM_lr3_wd2` | 78.18% | 77.97% | 0.6178 | 0.6335 | 23.01 | 22.2k |
| 15 | `sfplus_PcBDM_lr3_wd2` | 77.94% | 77.88% | 0.6249 | 0.6459 | 23.16 | 22.1k |
| 16 | `sfplus_PcbdM_lr3_wd2` | 77.90% | 77.88% | 0.6206 | 0.6430 | 23.09 | 22.2k |
| 17 | `sfplus_PcBdM_lr3_wd2` | 77.82% | 77.29% | 0.6436 | 0.6573 | 22.97 | 22.3k |
| 18 | `adamw_cosine_lr0.004_wd0.001` | 73.44% | 72.58% | 0.7566 | 0.7796 | 11.60 | 44.1k |
| 19 | `sfplus_pCbDm_lr0.008_wd2` | 72.28% | 71.93% | 0.7859 | 0.8036 | 22.44 | 22.8k |
| 20 | `sfplus_pCBDm_lr0.008_wd2` | 71.92% | 72.21% | 0.7850 | 0.8027 | 22.10 | 23.2k |
| 21 | `sfplus_pCBdm_lr0.008_wd2` | 71.68% | 71.76% | 0.7874 | 0.8024 | 22.28 | 23.0k |
| 22 | `sfplus_pcbdm_lr0.008_wd2` | 71.46% | 70.60% | 0.8039 | 0.8257 | 22.27 | 23.0k |
| 23 | `sfplus_pCbdm_lr0.008_wd2` | 71.34% | 72.00% | 0.7909 | 0.8010 | 22.58 | 22.7k |
| 24 | `sfplus_pcbDm_lr0.008_wd2` | 71.24% | 70.82% | 0.8012 | 0.8227 | 22.30 | 23.0k |
| 25 | `sfplus_pcBDm_lr0.008_wd2` | 71.18% | 70.46% | 0.8049 | 0.8286 | 22.44 | 22.8k |
| 26 | `sfplus_PCBDm_lr3_wd2` | 71.16% | 70.88% | 0.8087 | 0.8258 | 22.39 | 22.9k |
| 27 | `sfplus_pcBdm_lr0.008_wd2` | 71.10% | 70.43% | 0.8080 | 0.8303 | 22.39 | 22.9k |
| 28 | `sfplus_PCbDm_lr3_wd2` | 70.68% | 70.53% | 0.8174 | 0.8306 | 22.32 | 22.9k |
| 29 | `sfplus_PCbdm_lr3_wd2` | 70.14% | 69.87% | 0.8335 | 0.8508 | 22.65 | 22.6k |
| 30 | `sfplus_PCBdm_lr3_wd2` | 70.12% | 69.48% | 0.8292 | 0.8520 | 22.51 | 22.7k |
| 31 | `sfplus_PcbDm_lr3_wd2` | 69.62% | 68.20% | 0.8605 | 0.8803 | 22.57 | 22.7k |
| 32 | `sfplus_Pcbdm_lr3_wd2` | 69.42% | 68.44% | 0.8547 | 0.8852 | 22.54 | 22.7k |
| 33 | `sfplus_PcBDm_lr3_wd2` | 69.34% | 68.21% | 0.8626 | 0.8774 | 22.71 | 22.5k |
| 34 | `sfplus_PcBdm_lr3_wd2` | 68.94% | 68.32% | 0.8572 | 0.8870 | 22.31 | 22.9k |

## Toggle Marginals

| Toggle | On mean val acc | Off mean val acc | Delta | Best on | Best off |
|---|---:|---:|---:|---:|---:|
| P: Polyak LR | 74.47% | 75.20% | -0.72 | 80.84% | 79.50% |
| C: c warmup | 75.31% | 74.37% | +0.94 | 80.84% | 79.50% |
| B: beta anneal | 74.67% | 75.00% | -0.34 | 79.94% | 80.84% |
| D: AdamC decay | 74.97% | 74.70% | +0.27 | 80.84% | 79.94% |
| M: inner momentum | 78.95% | 70.73% | +8.22 | 80.84% | 72.28% |

## Conclusion

ScheduleFree+ ideas did not produce a new winner in this 20-epoch CIFAR-10
proxy. The best version enabled Polyak LR, `c_t` warmup, AdamC decay, and inner
momentum, while leaving beta annealing off. It still trailed AnchorMuon WSD by
4.08 validation points and 3.95 official test points.

For this optimizer family, the current recommendation stays unchanged:
trainer-side WSD with direct AnchorMuon / SODA + PMuonEq + GramNS + NorMuon is
the stronger recipe. If SF+ is revisited, start from the `M`-enabled path and
tune LR/Polyak scaling more carefully rather than spending more time on no-inner
momentum combinations.
