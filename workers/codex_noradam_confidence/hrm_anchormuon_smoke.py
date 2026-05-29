"""Tiny HRM + AnchorMuon smoke test.

This validates that a patched clone of https://github.com/sapientinc/HRM can
construct its model optimizer with the root ``optimizer.py`` AnchorMuon class
and complete one CUDA forward/backward/step.

Usage from repo root:

    DISABLE_COMPILE=1 HRM_ALLOW_SDPA_FALLBACK=1 \
      .venv/bin/python workers/codex_noradam_confidence/hrm_anchormuon_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    hrm = repo / "workers/codex_noradam_confidence/external/HRM"
    if not (hrm / "pretrain.py").exists():
        raise SystemExit(
            f"Missing HRM clone at {hrm}. Clone sapientinc/HRM there and apply "
            "workers/codex_noradam_confidence/hrm_anchormuon.patch first."
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the HRM smoke test.")

    os.environ.setdefault("DISABLE_COMPILE", "1")
    os.environ.setdefault("HRM_ALLOW_SDPA_FALLBACK", "1")
    sys.path.insert(0, str(hrm))

    from dataset.common import PuzzleDatasetMetadata
    from pretrain import ArchConfig, LossConfig, PretrainConfig, init_train_state, train_batch

    torch.manual_seed(123)
    config = PretrainConfig(
        arch=ArchConfig(
            name="hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
            loss=LossConfig(name="losses@ACTLossHead", loss_type="softmax_cross_entropy"),
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            hidden_size=64,
            expansion=4.0,
            num_heads=4,
            halt_max_steps=1,
            halt_exploration_prob=0.0,
            puzzle_emb_ndim=64,
            pos_encodings="rope",
        ),
        data_path="unused",
        global_batch_size=4,
        epochs=1,
        lr=1e-4,
        lr_min_ratio=1.0,
        lr_warmup_steps=0,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        puzzle_emb_lr=1e-4,
        puzzle_emb_weight_decay=0.1,
        optimizer="anchormuon",
        anchormuon_row_gamma=0.35,
        anchormuon_pmuoneq_beta=0.90,
        anchormuon_normuon_beta2=0.93,
        eval_interval=1,
    )
    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=11,
        seq_len=9,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=4.0,
        sets=["all"],
    )
    state = init_train_state(config, metadata, world_size=1)
    batch = {
        "inputs": torch.randint(1, metadata.vocab_size, (4, metadata.seq_len), dtype=torch.int32),
        "labels": torch.randint(1, metadata.vocab_size, (4, metadata.seq_len), dtype=torch.int32),
        "puzzle_identifiers": torch.zeros((4,), dtype=torch.int32),
    }
    metrics = train_batch(config, state, batch, global_batch_size=4, rank=0, world_size=1)
    model_optimizer = state.optimizers[1]
    stats = model_optimizer.last_stats
    assert state.step == 1
    assert stats["matrix_count"] > 0
    assert metrics is not None
    assert torch.isfinite(torch.tensor(float(metrics["train/lm_loss"])))
    print("HRM AnchorMuon smoke passed")
    print("optimizers", [type(o).__name__ for o in state.optimizers])
    print("group_summary", model_optimizer.group_summary())
    print("anchormuon_stats", stats)


if __name__ == "__main__":
    main()
