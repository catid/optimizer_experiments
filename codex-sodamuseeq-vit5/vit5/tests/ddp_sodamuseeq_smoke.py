import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim_sodamuseeq import SodaMuseEq, make_sodamuseeq_param_groups


class TinyViTLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(12, 16)
        self.norm = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.head = nn.Linear(16, 5)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.norm(x + self.mlp(self.norm(x)))
        return self.head(x)


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)
    model = TinyViTLike().to(device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    opt = SodaMuseEq(
        make_sodamuseeq_param_groups(ddp.module, lr=1e-2, weight_decay=0.01),
        warmup_steps=2,
        soda_warmup_steps=1,
        stats_interval=1,
    )
    x = torch.randn(32, 12, device=device)
    y = torch.randint(0, 5, (32,), device=device)
    for _ in range(3):
        opt.train()
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(ddp(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        max_diff = torch.zeros((), device=device)
        for p in ddp.module.parameters():
            ref = p.detach().clone()
            dist.broadcast(ref, src=0)
            max_diff = torch.maximum(max_diff, (p.detach() - ref).abs().max())
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f"ddp_sodamuseeq_smoke max_param_diff={float(max_diff.item()):.6g}")
    assert float(max_diff.item()) < 1e-5
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
