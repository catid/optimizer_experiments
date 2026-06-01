# Paused CIFAR-10 EMA-Muown HPO

This is a partial fixed-scheduler-ratio rerun of the EMA-Nesterov + Muown
CIFAR-10 HPO. It was paused on 2026-06-01 so the GPUs could be reassigned to
the newer request to run the LLM benchmark with a real tokenizer.

Status:

- Completed HPO summaries: `96`.
- Planned HPO candidates: `268`.
- Final replay was not reached.
- These rows should not be treated as final optimizer conclusions.

Top partial HPO rows by validation accuracy at pause time:

| Rank | Family | Trial | Best val acc | Best val loss | Step time |
|---:|---|---|---:|---:|---:|
| 1 | root AnchorMuon | `anchor_wsd_lr0.016_flr0.5_atan2_rg0.35_pb0.9_nb0.93` | 78.46% | 0.6058 | 17.79 ms |
| 2 | EMA-Muown | `ema_muown_wsd_lr0.01_wd0_flr1_mag1_b0.06_g0.995` | 72.80% | 0.7740 | 21.72 ms |
| 3 | Muown | `muown_wsd_lr0.014_wd0_flr0.5_mag1` | 72.70% | 0.7735 | 20.67 ms |

Operational note: this run includes the corrected scheduler behavior that
preserves per-group learning-rate ratios while applying the trainer-side
schedule.
