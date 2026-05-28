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
        train_param_diff = _max_rank0_diff([p.detach() for p in ddp.module.parameters()], device)
        state_tensors = []
        for p in ddp.module.parameters():
            for value in opt.state[p].values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    state_tensors.append(value.detach())
        state_diff = _max_rank0_diff(state_tensors, device)

        opt.eval()
        eval_param_diff = _max_rank0_diff([p.detach() for p in ddp.module.parameters()], device)
        opt.train()
        restored_param_diff = _max_rank0_diff([p.detach() for p in ddp.module.parameters()], device)
    if dist.get_rank() == 0:
        print(
            "ddp_sodamuseeq_smoke "
            f"train_param_diff={float(train_param_diff.item()):.6g} "
            f"state_diff={float(state_diff.item()):.6g} "
            f"eval_param_diff={float(eval_param_diff.item()):.6g} "
            f"restored_param_diff={float(restored_param_diff.item()):.6g}"
        )
    assert float(train_param_diff.item()) < 1e-5
    assert float(state_diff.item()) < 1e-5
    assert float(eval_param_diff.item()) < 1e-5
    assert float(restored_param_diff.item()) < 1e-5
    dist.destroy_process_group()


def _max_rank0_diff(tensors, device):
    max_diff = torch.zeros((), device=device)
    for tensor in tensors:
        ref = tensor.detach().clone()
        dist.broadcast(ref, src=0)
        max_diff = torch.maximum(max_diff, (tensor.detach() - ref).abs().max())
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    return max_diff


if __name__ == "__main__":
    main()
