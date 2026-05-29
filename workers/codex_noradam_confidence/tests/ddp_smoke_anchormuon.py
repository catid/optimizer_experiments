from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optim_anchormuon import AnchorMuon


class TinyDDP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(32, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def assert_replicated(name: str, tensor: torch.Tensor, *, atol: float = 1e-6) -> None:
    flat = tensor.detach().float().reshape(-1)
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    stacked = torch.stack(gathered, dim=0)
    spread = (stacked.max(dim=0).values - stacked.min(dim=0).values).max()
    if spread.item() > atol:
        raise AssertionError(f"{name} differs across DDP ranks: max spread {spread.item():.3e}")


def assert_optimizer_replicated(model: TinyDDP, opt: AnchorMuon) -> None:
    for name, param in model.named_parameters():
        assert_replicated(f"param:{name}", param)
        for state_name, value in opt.state[param].items():
            if torch.is_tensor(value):
                assert_replicated(f"state:{name}:{state_name}", value)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    torch.manual_seed(100 + dist.get_rank())
    model = TinyDDP().cuda()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    opt = AnchorMuon(
        ddp.parameters(),
        lr=1e-3,
        warmup_steps=2,
        soda="all",
        pmuon_eq=True,
        mimuon=True,
        normuon=True,
        normuon_aspect_scale=True,
    )
    for _ in range(3):
        opt.train()
        x = torch.randn(32, 16, device="cuda")
        y = torch.randint(0, 8, (32,), device="cuda")
        loss = torch.nn.functional.cross_entropy(ddp(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)
        assert_optimizer_replicated(model, opt)
    opt.eval()
    assert_optimizer_replicated(model, opt)
    with torch.no_grad():
        _ = ddp(torch.randn(8, 16, device="cuda"))
    opt.train()
    assert_optimizer_replicated(model, opt)
    dist.barrier()
    if dist.get_rank() == 0:
        print("ddp smoke passed with parameter/state parity")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
