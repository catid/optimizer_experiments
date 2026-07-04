"""Prepare and load tokenized SFT data. ``prepare_sft`` builds a dataset from a chat
dataset; ``load_prepared`` loads one. See the README for the data recipe.
"""

import os
from typing import Optional, Tuple

import torch
from datasets import load_from_disk

SMOLTALK_REPO = "HuggingFaceTB/smol-smoltalk"


def collate_fn(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_prepared(path: str, max_samples: Optional[int] = None) -> Tuple:
    """Load a prepared dataset saved under ``<path>/train`` and ``<path>/val``."""
    train_ds = load_from_disk(os.path.join(path, "train"))
    val_ds = load_from_disk(os.path.join(path, "val"))
    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(max_samples, len(val_ds))))
    return train_ds, val_ds


def tokenize_conversation(messages, tokenizer, max_seq_len):
    """Tokenize a conversation as ChatML, supervising only the assistant responses."""
    input_ids, labels = [], []
    for msg in messages:
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value")
        # ShareGPT role aliases map onto ChatML role names.
        if role == "gpt":
            role = "assistant"
        elif role == "human":
            role = "user"
        is_assistant = role in ("assistant", "gpt")

        header_ids = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        body_ids = tokenizer.encode(f"{content}<|im_end|>\n", add_special_tokens=False)

        input_ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))

        input_ids.extend(body_ids)
        labels.extend(body_ids if is_assistant else [-100] * len(body_ids))

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels


def _build_split(rows, tokenizer, max_seq_len, pad_id, name):
    """Tokenize, filter, and pad a list of conversation rows into a Dataset."""
    from datasets import Dataset

    out_ids, out_attn, out_labels, skipped = [], [], [], 0
    for ex in rows:
        messages = ex.get("messages") or ex.get("conversations")
        # Skip rows with no assistant content to supervise
        if not messages or not any(
            (m.get("role") or m.get("from")) in ("assistant", "gpt") for m in messages
        ):
            skipped += 1
            continue
        ids, attn, labels = tokenize_conversation(messages, tokenizer, max_seq_len)
        if len(ids) < 10 or all(label == -100 for label in labels):
            skipped += 1
            continue
        out_ids.append(ids)
        out_attn.append(attn)
        out_labels.append(labels)

    # Pad to a fixed length for efficient batching
    for i in range(len(out_ids)):
        pad = max_seq_len - len(out_ids[i])
        out_ids[i] = out_ids[i] + [pad_id] * pad
        out_attn[i] = out_attn[i] + [0] * pad
        out_labels[i] = out_labels[i] + [-100] * pad

    print(f"  {name}: kept {len(out_ids)}, skipped {skipped}")
    return Dataset.from_dict(
        {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_labels}
    )


def prepare_sft(
    out_dir: str,
    tokenizer_name: str,
    dataset_name: str = SMOLTALK_REPO,
    max_seq_len: int = 1280,
    subset_ratio: float = 1.0,
    seed: int = 42,
    revision: Optional[str] = None,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
) -> str:
    """Tokenize an SFT dataset and save it under ``out_dir/{train,val}``.

    Full dataset by default; ``subset_ratio < 1`` keeps a seeded random subset.
    ``max_train``/``max_val`` instead stream a tiny first-N slice for a quick test.
    """
    import random
    from datasets import load_dataset
    from transformers import AutoTokenizer

    rev = revision  # optionally pin a dataset version; None tracks the current one

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if max_train:
        # Streamed first-N slice (no full download); used for the self-contained test
        n_val = max_val or max(1, max_train // 4)
        train_stream = load_dataset(dataset_name, split="train", streaming=True, revision=rev)
        train_rows = [r for _, r in zip(range(max_train), train_stream)]
        try:
            val_stream = load_dataset(dataset_name, split="test", streaming=True, revision=rev)
        except Exception:
            val_stream = load_dataset(dataset_name, split="train", streaming=True, revision=rev)
        val_rows = [r for _, r in zip(range(n_val), val_stream)]
    else:
        ds = load_dataset(dataset_name, revision=rev)
        train_split = ds["train"]
        val_split = ds.get("test") or ds.get("validation")

        def subset(split, ratio):
            if ratio >= 1.0:
                return split
            n = int(len(split) * ratio)
            idx = sorted(random.Random(seed).sample(range(len(split)), n))
            return split.select(idx)

        if val_split is not None:
            train_rows = subset(train_split, subset_ratio)
            val_rows = subset(val_split, subset_ratio)
        else:
            # No native eval split: hold out the last 5% of train for validation
            n_val = max(1, int(len(train_split) * 0.05))
            cut = len(train_split) - n_val
            train_rows = subset(train_split.select(range(cut)), subset_ratio)
            val_rows = train_split.select(range(cut, len(train_split)))

    sample = train_rows[0] if len(train_rows) else {}
    if not (sample.get("messages") or sample.get("conversations")):
        raise ValueError(
            "prepare_sft expects a chat/SFT dataset with a 'messages' or "
            "'conversations' field; raw-text / pretraining corpora (e.g. FineWeb) "
            "are not supported by the shipped SFT trainer."
        )

    pad_id = tok.pad_token_id
    train_ds = _build_split(train_rows, tok, max_seq_len, pad_id, "train")
    val_ds = _build_split(val_rows, tok, max_seq_len, pad_id, "val")
    train_ds.save_to_disk(os.path.join(out_dir, "train"))
    val_ds.save_to_disk(os.path.join(out_dir, "val"))
    return out_dir


def main():
    """CLI: ``pace-data --out data/smoltalk --tokenizer HuggingFaceTB/SmolLM2-1.7B``."""
    import argparse

    p = argparse.ArgumentParser(description="Prepare a tokenized SFT dataset for pace-train.")
    p.add_argument("--out", required=True, help="output directory (writes <out>/train and <out>/val)")
    p.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer/model name")
    p.add_argument("--dataset", default=SMOLTALK_REPO, help="HuggingFace chat dataset")
    p.add_argument("--max-seq-len", type=int, default=1280)
    p.add_argument("--subset-ratio", type=float, default=1.0)
    p.add_argument("--max-train", type=int, default=None,
                   help="stream only the first N train rows (a quick slice, no full download)")
    p.add_argument("--max-val", type=int, default=None, help="first N val rows for the slice")
    p.add_argument("--revision", default=None, help="pin a dataset revision")
    args = p.parse_args()
    out = prepare_sft(args.out, tokenizer_name=args.tokenizer, dataset_name=args.dataset,
                      max_seq_len=args.max_seq_len, subset_ratio=args.subset_ratio,
                      max_train=args.max_train, max_val=args.max_val, revision=args.revision)
    print(f"Saved prepared dataset to {out}/")


if __name__ == "__main__":
    main()
