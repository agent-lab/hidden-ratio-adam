#!/usr/bin/env python
"""Build KellerJordan-style FineWeb .bin shards with a 32k LLaMA tokenizer.

The output format matches the existing fineweb_*.bin shards used by plainLM:
- 256 x int32 header
- uint16 token payload

To preserve the repo's existing style, we prepend a single document delimiter
token per document and do not append EOS. For LLaMA-style tokenizers we use
the BOS token when available, falling back to EOS only if BOS is missing.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


MAGIC = 20240520
VERSION = 1

_TOKENIZER = None
_DOC_START_ID = None


def write_datafile(filename: str, toks: np.ndarray) -> None:
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(toks)
    if not isinstance(toks, np.ndarray) or toks.dtype != np.uint16:
        toks_np = np.asarray(toks, dtype=np.uint16)
    else:
        toks_np = toks
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def init_worker(tokenizer_name: str) -> None:
    global _TOKENIZER, _DOC_START_ID
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if _TOKENIZER.vocab_size >= 2**16:
        raise ValueError(f"Tokenizer vocab too large for uint16 payload: {_TOKENIZER.vocab_size}")
    _DOC_START_ID = _TOKENIZER.bos_token_id
    if _DOC_START_ID is None:
        _DOC_START_ID = _TOKENIZER.eos_token_id
    if _DOC_START_ID is None:
        raise ValueError("Tokenizer must expose bos_token_id or eos_token_id")


def tokenize_doc(doc: dict) -> np.ndarray:
    tokens = [_DOC_START_ID]
    tokens.extend(_TOKENIZER.encode(doc["text"], add_special_tokens=False))
    toks_np = np.asarray(tokens, dtype=np.int64)
    if not ((0 <= toks_np).all() and (toks_np < 2**16).all()):
        raise ValueError("Tokenizer produced ids outside uint16 range")
    return toks_np.astype(np.uint16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FineWeb uint16 .bin shards with a LLaMA-style tokenizer")
    parser.add_argument(
        "--preset",
        choices=["fineweb10b", "fineweb100b", "custom"],
        default="fineweb10b",
        help="Convenience preset for FineWeb token budgets and remote config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for fineweb_train_*.bin / fineweb_val_*.bin",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-step-50K-105b",
        help="HF tokenizer to use for re-tokenization",
    )
    parser.add_argument(
        "--remote-name",
        type=str,
        default=None,
        help="FineWeb subset name. Defaults to sample-10BT or sample-100BT based on --preset.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=os.environ.get("HF_DATASETS_CACHE"),
        help="Optional datasets cache dir",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=os.environ.get("HF_HOME"),
        help="Optional HF_HOME override; mostly for logging metadata",
    )
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--shard-size", type=int, default=10**8)
    parser.add_argument(
        "--val-token-budget",
        type=int,
        default=None,
        help="How many tokens to keep for the val split (defaults to one KJ-style shard)",
    )
    parser.add_argument(
        "--train-token-budget",
        type=int,
        default=None,
        help="How many train-split tokens to write before stopping",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    presets = {
        "fineweb10b": {
            "remote_name": "sample-10BT",
            "output_dir": Path("./data/fineweb10B_llama20m_tinyllama32k_bin"),
            "train_token_budget": 10_000_000_000,
            "val_token_budget": 100_000_000,
        },
        "fineweb100b": {
            "remote_name": "sample-100BT",
            "output_dir": Path("./data/fineweb100B_llama20m_tinyllama32k_bin"),
            "train_token_budget": 100_000_000_000,
            "val_token_budget": 100_000_000,
        },
        "custom": {
            "remote_name": "sample-10BT",
            "output_dir": Path("./data/fineweb_custom_llama20m_tinyllama32k_bin"),
            "train_token_budget": 1_500_793_344,
            "val_token_budget": 100_000_000,
        },
    }
    preset = presets[args.preset]
    if args.remote_name is None:
        args.remote_name = preset["remote_name"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]
    if args.train_token_budget is None:
        args.train_token_budget = preset["train_token_budget"]
    if args.val_token_budget is None:
        args.val_token_budget = preset["val_token_budget"]
    return args


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    ready_path = out_dir / "_READY"
    meta_path = out_dir / "_METADATA.json"
    tmp_path = out_dir / "_BUILDING"

    if out_dir.exists():
        if ready_path.exists() and not args.overwrite:
            print(f"[skip] {ready_path} already exists; dataset is ready at {out_dir}")
            return
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(
                f"Output dir {out_dir} already exists without _READY. "
                "Re-run with --overwrite to rebuild from scratch."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(f"building started at {time.ctime()}\n", encoding="utf-8")

    meta = {
        "dataset": "HuggingFaceFW/fineweb",
        "remote_name": args.remote_name,
        "tokenizer": args.tokenizer,
        "shard_size": args.shard_size,
        "val_token_budget": args.val_token_budget,
        "train_token_budget": args.train_token_budget,
        "format": "KellerJordan fineweb bin (256-int32 header + uint16 payload)",
        "doc_delimiter": "prepend BOS per document; no EOS append",
        "hf_home": args.hf_home,
        "dataset_cache_dir": args.dataset_cache_dir,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"[load] fineweb/{args.remote_name}")
    fw = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.remote_name,
        split="train",
        cache_dir=args.dataset_cache_dir,
    )

    nprocs = max(1, args.num_workers)
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    total_tokens = 0
    num_docs = 0
    current_split = "val"
    split_budgets = {"val": int(args.val_token_budget), "train": int(args.train_token_budget)}
    split_written = {"val": 0, "train": 0}

    with mp.Pool(nprocs, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        for tokens in pool.imap(tokenize_doc, fw, chunksize=16):
            num_docs += 1
            total_tokens += int(len(tokens))
            offset = 0
            while offset < len(tokens):
                if split_written[current_split] + token_count >= split_budgets[current_split]:
                    if token_count:
                        filename = out_dir / f"fineweb_{current_split}_{shard_index:06d}.bin"
                        write_datafile(str(filename), all_tokens_np[:token_count])
                        split_written[current_split] += token_count
                        shard_index += 1
                        token_count = 0
                        if progress_bar is not None:
                            progress_bar.close()
                            progress_bar = None
                    if current_split == "val":
                        current_split = "train"
                        continue
                    meta["stopped_early_after_budget"] = True
                    offset = len(tokens)
                    break

                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

                remaining_shard = args.shard_size - token_count
                remaining_budget = split_budgets[current_split] - (split_written[current_split] + token_count)
                take = min(len(tokens) - offset, remaining_shard, remaining_budget)
                if take <= 0:
                    break

                all_tokens_np[token_count : token_count + take] = tokens[offset : offset + take]
                token_count += take
                offset += take
                progress_bar.update(take)

                shard_budget_hit = split_written[current_split] + token_count >= split_budgets[current_split]
                shard_full = token_count >= args.shard_size
                if shard_full or shard_budget_hit:
                    filename = out_dir / f"fineweb_{current_split}_{shard_index:06d}.bin"
                    write_datafile(str(filename), all_tokens_np[:token_count])
                    split_written[current_split] += token_count
                    shard_index += 1
                    token_count = 0
                    if progress_bar is not None:
                        progress_bar.close()
                        progress_bar = None
                    if shard_budget_hit and current_split == "val":
                        current_split = "train"
                    elif shard_budget_hit and current_split == "train":
                        meta["stopped_early_after_budget"] = True
                        break

            if split_written["train"] >= split_budgets["train"]:
                break

    if token_count and split_written[current_split] < split_budgets[current_split]:
        filename = out_dir / f"fineweb_{current_split}_{shard_index:06d}.bin"
        write_datafile(str(filename), all_tokens_np[:token_count])
        split_written[current_split] += token_count
        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None

    meta.update(
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_docs": num_docs,
            "approx_total_tokens_with_bos": total_tokens,
            "split_written_tokens": split_written,
            "ready_marker": str(ready_path),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ready_path.write_text(json.dumps({"status": "ready", "completed_at": meta["completed_at"]}) + "\n", encoding="utf-8")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[done] wrote bins to {out_dir}")


if __name__ == "__main__":
    main()
