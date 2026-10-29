import argparse
import os

import datasets


def to_messages(example):
    msgs = example.get("messages") or example.get("prompt") or example.get("conversation")
    if msgs is None:
        raise ValueError(f"No messages-like field found in example keys={list(example.keys())}")
    return {"messages": msgs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="examples/sft/ultrachat/data")
    args = parser.parse_args()

    ds = datasets.load_dataset("HuggingFaceH4/ultrachat_200k")
    train = ds["train_sft"].map(to_messages, remove_columns=ds["train_sft"].column_names)
    test_split = "test_sft" if "test_sft" in ds else "train_sft"
    test = ds[test_split].map(to_messages, remove_columns=ds[test_split].column_names)

    out = os.path.expanduser(args.local_save_dir)
    os.makedirs(out, exist_ok=True)
    train.to_parquet(os.path.join(out, "train.parquet"))
    test.to_parquet(os.path.join(out, "test.parquet"))
    print(out)
