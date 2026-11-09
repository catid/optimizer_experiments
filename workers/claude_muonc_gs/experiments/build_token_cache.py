#!/usr/bin/env python3
"""Build the FineWeb-Edu GPT-2 token caches used by ``run_muonc_gs_lm.py``.

This calls the repo runner's own ``prepare_hf_token_cache`` so the caches are
the ones the README benchmark uses: one shuffled stream (buffer 10000, seed
123) whose first 524,288 tokens are the validation set and whose next
``--train-tokens`` tokens are the training set.  A larger training cache
therefore starts with the smaller one.

    python build_token_cache.py --train-tokens 8000000      # "std" protocol
    python build_token_cache.py --train-tokens 512000000    # "sp" protocol (about 5 minutes)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CODEX_EXPERIMENTS = REPO_ROOT / "workers" / "codex_noradam_confidence" / "experiments"
for import_root in (REPO_ROOT, CODEX_EXPERIMENTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPO_ROOT / "workers" / "codex_noradam_confidence" / "data" / "fineweb_gpt2_tokens",
    )
    cli = parser.parse_args()

    import run_wikitext_llm50m as codex_runner

    args = argparse.Namespace(
        dataset_source="hf_text", tokenizer_mode="hf", tokenizer_name="gpt2", tokenizer_add_eos=True,
        hf_dataset="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT", hf_split="train", hf_val_split="",
        hf_text_field="text", hf_shuffle_buffer=10_000, cache_dir=cli.cache_dir,
        max_train_tokens=cli.train_tokens, max_val_tokens=524_288, seed=123,
    )
    started = time.time()
    train_path, val_path = codex_runner.prepare_hf_token_cache(args)
    print(f"{train_path.name}\n{val_path.name}\n{time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
