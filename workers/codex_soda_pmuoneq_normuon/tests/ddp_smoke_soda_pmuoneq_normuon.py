"""Tiny DDP consistency smoke test for SodaPmuonEqNorMuon.

Run from the repository root:

    torchrun --standalone --nproc_per_node=2 \
      workers/codex_soda_pmuoneq_normuon/tests/ddp_smoke_soda_pmuoneq_normuon.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soda_pmuoneq_normuon import SodaPmuonEqNorMuon, build_soda_pmuoneq_normuon_param_groups


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(16)
        self.classifier_head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(self.norm(F.gelu(self.fc1(x))))


def _setup() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world, device


def _max_rank_spread(tensor: torch.Tensor) -> float:
    if not dist.is_initialized():
        return 0.0
    work = tensor.detach().float()
    max_tensor = work.clone()
    min_tensor = work.clone()
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
    return float((max_tensor - min_tensor).abs().max().detach().cpu())


def main() -> None:
    rank, world, device = _setup()
    torch.manual_seed(1234)
    model = TinyClassifier().to(device)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = SodaPmuonEqNorMuon(
        build_soda_pmuoneq_normuon_param_groups(ddp.module.named_parameters(), matrix_lr=1e-3, adam_lr=1e-4),
        warmup_steps=2,
    )

    for step in range(4):
        torch.manual_seed(9000 + step + rank)
        x = torch.randn(16, 8, device=device)
        y = torch.randint(0, 4, (16,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp(x), y)
        loss.backward()
        opt.step()

    spreads: dict[str, float] = {}
    for name, param in ddp.module.named_parameters():
        spreads[f"param:{name}"] = _max_rank_spread(param)
    for name, param in ddp.module.named_parameters():
        state = opt.state[param]
        for key, value in state.items():
            if torch.is_tensor(value):
                spreads[f"state:{name}:{key}"] = _max_rank_spread(value)

    max_spread = max(spreads.values(), default=0.0)
    if rank == 0:
        print(json.dumps({"world_size": world, "max_rank_spread": max_spread, "num_checked": len(spreads)}, sort_keys=True))
    if max_spread > 1e-6:
        raise SystemExit(f"DDP rank spread too high: {max_spread}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

