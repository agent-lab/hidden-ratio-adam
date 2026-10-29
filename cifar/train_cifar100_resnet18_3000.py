import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from adam_nsr import AdamNSR


SEED = int(os.environ.get("CIFAR100_SEED", "1"))
BATCH_SIZE = int(os.environ.get("CIFAR100_BATCH_SIZE", "512"))
LR = float(os.environ.get("CIFAR100_LR", "2e-4"))
MAX_STEPS = int(os.environ.get("CIFAR100_MAX_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("CIFAR100_EVAL_EVERY", "98"))
NUM_WORKERS = int(os.environ.get("CIFAR100_NUM_WORKERS", "4"))
LOG_HIDDEN_RATIO = os.environ.get("CIFAR100_LOG_HIDDEN_RATIO", "0") == "1"
BETA = float(os.environ.get("ADAM_BETA", "0.90"))
OPTIMIZER_NAME = os.environ.get("CIFAR100_OPTIMIZER", "adam").lower()
DATA_FILE = Path(os.environ.get("CIFAR100_DATA_FILE", "./data/cifar100_uint8.pt"))
RUN_DIR = Path(os.environ.get("CIFAR100_RUN_DIR", "./outputs/cifar100"))
ADAM_NSR_SOURCE_SHA256 = "1371e1a0500795b7a4d949968ae71c9f31b32e2a778e643160dffc07a0600fe5"


class TensorCIFAR100(Dataset):
    def __init__(self, images, targets, train):
        self.images = images
        self.targets = targets
        self.train = train
        self.mean = torch.tensor((0.5071, 0.4867, 0.4408), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.2675, 0.2565, 0.2761), dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return self.targets.numel()

    def __getitem__(self, index):
        image = self.images[index].float().div_(255.0)
        if self.train:
            image = F.pad(image, (4, 4, 4, 4), mode="reflect")
            top = int(torch.randint(0, 9, ()).item())
            left = int(torch.randint(0, 9, ()).item())
            image = image[:, top:top + 32, left:left + 32]
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
        return (image - self.mean) / self.std, int(self.targets[index])


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(self.in_channels, out_channels, 1) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class HiddenRatioAccumulator:
    def __init__(self, device):
        self.edges = np.linspace(0.0, 20.0, 81, dtype=np.float64)
        self.counts = {stage: torch.zeros(80, dtype=torch.int64, device=device) for stage in ("early", "middle", "late")}
        self.overflow = {stage: torch.zeros((), dtype=torch.int64, device=device) for stage in self.counts}
        self.total = {stage: 0 for stage in self.counts}
        self.snapshots = {stage: 0 for stage in self.counts}

    @staticmethod
    def stage_for_step(step):
        if step <= 1000:
            return "early"
        if step <= 2000:
            return "middle"
        return "late"

    @torch.no_grad()
    def add(self, optimizer, step):
        stage = self.stage_for_step(step)
        for group in optimizer.param_groups:
            beta = group["betas"][0]
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if "exp_avg" not in state or "exp_avg_sq" not in state:
                    continue
                m = state["exp_avg"].detach().float()
                v = state["exp_avg_sq"].detach().float()
                m2 = m.square()
                y = ((1.0 - beta) / beta) * (v - m2).clamp_min_(0.0) / m2.clamp_min(1e-30)
                y = torch.nan_to_num(y, nan=float("inf"), posinf=float("inf"), neginf=0.0)
                flat = y.flatten()
                in_range = flat <= 20.0
                self.counts[stage].add_(torch.histc(flat[in_range], bins=80, min=0.0, max=20.0).to(torch.int64))
                self.overflow[stage].add_((~in_range).sum())
                self.total[stage] += flat.numel()
        self.snapshots[stage] += 1

    def save(self, run_dir, metadata):
        cpu_counts = {stage: value.cpu().numpy() for stage, value in self.counts.items()}
        cpu_overflow = {stage: int(value.cpu().item()) for stage, value in self.overflow.items()}
        with (run_dir / "hidden_ratio_stages.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "y_left", "y_right", "early_mass", "middle_mass", "late_mass",
                "early_cdf", "middle_cdf", "late_cdf", "theory_mass", "theory_cdf",
            ])
            cumulative = {stage: 0.0 for stage in self.counts}
            for index in range(80):
                left, right = self.edges[index], self.edges[index + 1]
                masses = {}
                cdfs = {}
                for stage in self.counts:
                    masses[stage] = cpu_counts[stage][index] / max(1, self.total[stage])
                    cumulative[stage] += masses[stage]
                    cdfs[stage] = cumulative[stage]
                reference_cdf = lambda value: 0.0 if value <= 0 else math.erfc(math.sqrt(1.0 / value))
                writer.writerow([
                    left, right, masses["early"], masses["middle"], masses["late"],
                    cdfs["early"], cdfs["middle"], cdfs["late"],
                    reference_cdf(right) - reference_cdf(left), reference_cdf(right),
                ])
        metadata["hidden_ratio"] = {
            "definition": "((1-beta)/beta)*(exp_avg_sq-exp_avg^2)/exp_avg^2",
            "range": [0.0, 20.0],
            "num_bins": 80,
            "bin_width": 0.25,
            "averaging_mode": "every optimizer step, all parameter entries",
            "stage_windows": {
                "early": "optimizer steps 1-1000",
                "middle": "optimizer steps 1001-2000",
                "late": "optimizer steps 2001-3000",
            },
            "snapshots": self.snapshots,
            "sampled_parameter_entries": self.total,
            "overflow_fraction": {stage: cpu_overflow[stage] / max(1, self.total[stage]) for stage in self.counts},
        }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, targets).item()
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.numel()
    return loss_sum / total, correct / total


def main():
    if OPTIMIZER_NAME not in ("adam", "tr_adam"):
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")
    if LOG_HIDDEN_RATIO and OPTIMIZER_NAME != "adam":
        raise ValueError("Hidden-ratio logging is enabled only for Adam in this sweep")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(DATA_FILE, map_location="cpu")
    train_set = TensorCIFAR100(data["train_images"], data["train_targets"], train=True)
    test_set = TensorCIFAR100(data["test_images"], data["test_targets"], train=False)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=1024, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )

    model = CIFARResNet18().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if OPTIMIZER_NAME == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(BETA, BETA), eps=1e-8)
        optimizer_metadata = {"name": "Adam"}
    else:
        optimizer = AdamNSR(
            model.parameters(), lr=LR, betas=(BETA, BETA), eps=1e-8, weight_decay=0.0,
            y_max=64.0, moment1_dtype="fp32", moment1_quant="per_column",
            moment2_dtype="fp32", pseudo_quantize_state_dtype=None, compute_dtype="fp32",
            skip_update_on_bad_y=False,
        )
        optimizer_metadata = {
            "name": "TR-Adam",
            "implementation": "AdamNSR",
            "y_table": list(optimizer.param_groups[0]["y_table"]),
            "y_max": 64.0,
            "moment1_dtype": "fp32",
            "moment1_quant": "per_column (no effect while moment1_dtype is fp32)",
            "source_sha256": ADAM_NSR_SOURCE_SHA256,
        }
    optimizer_metadata.update({
        "lr": LR, "betas": [BETA, BETA], "eps": 1e-8, "weight_decay": 0.0, "scheduler": None,
    })

    criterion = nn.CrossEntropyLoss()
    accumulator = HiddenRatioAccumulator(device) if LOG_HIDDEN_RATIO else None
    history = []
    optimizer_step = 0
    epoch = 0
    started = time.time()
    print(json.dumps({
        "event": "start", "device": str(device), "parameters": parameter_count,
        "optimizer": optimizer_metadata["name"], "beta": BETA, "batch_size": BATCH_SIZE,
    }), flush=True)

    while optimizer_step < MAX_STEPS:
        epoch += 1
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer_step += 1
            if accumulator is not None:
                accumulator.add(optimizer, optimizer_step)

            if optimizer_step % EVAL_EVERY == 0 or optimizer_step == MAX_STEPS:
                test_loss, test_accuracy = evaluate(model, test_loader, device)
                record = {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "train_batch_loss": float(loss.item()),
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "elapsed_seconds": time.time() - started,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
                model.train()
            if optimizer_step >= MAX_STEPS:
                break

    metadata = {
        "dataset": "CIFAR-100 fine labels, decoded losslessly from Hugging Face parquet",
        "seed": SEED,
        "optimizer_steps": optimizer_step,
        "evaluation_every_steps": EVAL_EVERY,
        "batch_size": BATCH_SIZE,
        "augmentation": ["RandomCrop(32,padding=4,padding_mode=reflect)", "RandomHorizontalFlip"],
        "normalization": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
        "network": {
            "type": "CIFARResNet18",
            "blocks": [2, 2, 2, 2],
            "stem": "3x3 stride 1, no max-pool",
            "parameters": parameter_count,
            "classes": 100,
        },
        "optimizer": optimizer_metadata,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "best_test_loss": min(row["test_loss"] for row in history),
        "best_test_loss_step": min(history, key=lambda row: row["test_loss"])["optimizer_step"],
        "final_test_loss": history[-1]["test_loss"],
        "final_test_accuracy": history[-1]["test_accuracy"],
    }
    if accumulator is not None:
        accumulator.save(RUN_DIR, metadata)
    else:
        metadata["hidden_ratio"] = {"logging": "disabled"}
    with (RUN_DIR / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (RUN_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"event": "done", **metadata}), flush=True)


if __name__ == "__main__":
    main()
