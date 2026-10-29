import io
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


SOURCE = Path(os.environ.get("CIFAR100_PARQUET_DIR", "./data/cifar100_hf"))
OUTPUT = Path(os.environ.get("CIFAR100_TENSOR_FILE", "./data/cifar100_uint8.pt"))


def decode(path):
    table = pq.read_table(path)
    encoded = table.column("img").combine_chunks().field("bytes").to_pylist()
    images = np.empty((len(encoded), 3, 32, 32), dtype=np.uint8)
    for index, raw in enumerate(encoded):
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        images[index] = image.transpose(2, 0, 1)
    targets = table.column("fine_label").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return torch.from_numpy(images), torch.from_numpy(targets.copy())


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    train_images, train_targets = decode(SOURCE / "train.parquet")
    test_images, test_targets = decode(SOURCE / "test.parquet")
    if train_images.shape != (50000, 3, 32, 32) or test_images.shape != (10000, 3, 32, 32):
        raise RuntimeError(f"Unexpected CIFAR-100 shapes: {train_images.shape}, {test_images.shape}")
    if train_targets.min().item() != 0 or train_targets.max().item() != 99:
        raise RuntimeError("Unexpected CIFAR-100 fine-label range")
    torch.save(
        {
            "train_images": train_images,
            "train_targets": train_targets,
            "test_images": test_images,
            "test_targets": test_targets,
        },
        OUTPUT,
    )
    print({
        "output": str(OUTPUT),
        "train_images": tuple(train_images.shape),
        "test_images": tuple(test_images.shape),
        "label_range": (int(train_targets.min()), int(train_targets.max())),
        "bytes": OUTPUT.stat().st_size,
    })


if __name__ == "__main__":
    main()
