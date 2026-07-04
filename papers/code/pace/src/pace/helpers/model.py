"""Model loading for PACE training."""

import torch
from transformers import AutoModelForCausalLM


def load_model(name: str, device: str, dtype=torch.bfloat16):
    """Load a causal-LM by HuggingFace name in the given dtype, on ``device``."""
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    return model
