# Validation

Source workstation:

- 4x NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs
- Python 3.12
- PyTorch nightly `2.13.0.dev20260513+cu130`

Checks run before export:

```text
python -m py_compile equimuse_normuon.py third_party/ViT-5-equimuse/main.py third_party/ViT-5-equimuse/run_equimuse_ablation.py
PYTHONPATH=. pytest -q tests/test_equimuse_normuon_standalone.py tests/test_equimuse_standalone.py
torchrun --standalone --nproc_per_node=4 check_equimuse_ddp.py --steps 3 --output reports/equimuse_normuon_standalone_ddp_check.json
```

Results:

- Unit/parity tests: 18 passed.
- DDP consistency: zero rank spread for model params, fast weights, and eval weights.
- Fixed ViT-5-Small/CIFAR-10 img224 validation, all 4 GPUs, 1000 optimizer steps:

| implementation | acc@1 | val loss | train loss | samples/s |
| --- | ---: | ---: | ---: | ---: |
| prior ablation path | 65.96 | 1.0145 | 1.4699 | 3918 |
| standalone `equimuse_normuon.py` | 65.96 | 1.0144 | 1.4699 | 4335 |

The standalone file reproduced the previous best accuracy and train loss while
improving measured throughput by avoiding per-step GPU-to-CPU diagnostic syncs.

