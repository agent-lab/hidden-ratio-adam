import argparse
import os

import datasets


def _normalize_prompt_messages(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict) and "role" in item and "content" in item:
                role = item["role"]
                if role == "assistant":
                    break
                if role in {"system", "user"}:
                    messages.append({"role": role, "content": item["content"]})
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
            else:
                return []
        return messages

    return []


def to_rl_example(example, data_source, split, idx):
    prompt = []
    for key in ["prompt", "messages", "conversation"]:
        prompt = _normalize_prompt_messages(example.get(key))
        if prompt:
            break

    if not prompt:
        raise ValueError(f"Cannot extract prompt from keys={list(example.keys())}")

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "alignment",
        "reward_model": {
            "style": "model",
            "ground_truth": "",
        },
        "extra_info": {"split": split, "index": idx},
    }


def convert_split(dataset, split, data_source):
    return dataset.map(
        lambda example, idx: to_rl_example(example, data_source, split, idx),
        with_indices=True,
        remove_columns=dataset.column_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="argilla/ultrafeedback-binarized-preferences-cleaned")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--test_size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_save_dir", default="data/ultrafeedback_rl")
    args = parser.parse_args()

    load_kwargs = {"path": args.dataset_name}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    ds = datasets.load_dataset(**load_kwargs)

    train = ds[args.train_split]
    if args.test_split in ds:
        test = ds[args.test_split]
    else:
        split = train.train_test_split(test_size=args.test_size, seed=args.seed)
        train = split["train"]
        test = split["test"]

    data_source = args.dataset_name
    train = convert_split(train, "train", data_source)
    test = convert_split(test, "test", data_source)

    out = os.path.expanduser(args.local_save_dir)
    os.makedirs(out, exist_ok=True)
    train.to_parquet(os.path.join(out, "train.parquet"))
    test.to_parquet(os.path.join(out, "test.parquet"))
    print(out)
