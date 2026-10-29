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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


SEED = int(os.environ.get("CIFAR10_SEED", "1"))
BATCH_SIZE = 512
LR = float(os.environ.get("CIFAR10_LR", "2e-4"))
MAX_STEPS = 3000
EVAL_EVERY = 98
NUM_WORKERS = int(os.environ.get("CIFAR10_NUM_WORKERS", "8"))
LOG_HIDDEN_RATIO = os.environ.get("CIFAR10_LOG_HIDDEN_RATIO", "1") == "1"
BETA = float(os.environ.get("ADAM_BETA", "0.90"))
HIDDEN = [int(value) for value in os.environ.get("CIFAR10_HIDDEN", "1024,1024,512").split(",")]
if len(HIDDEN) != 3:
    raise ValueError("CIFAR10_HIDDEN must contain exactly three comma-separated widths")
DATA_FILE = Path(os.environ.get("CIFAR10_DATA_FILE", "./data/cifar10_uint8.pt"))
RUN_DIR = Path(os.environ.get("CIFAR10_RUN_DIR", "./outputs/cifar10_adam"))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN[1], HIDDEN[2]),
            nn.ReLU(),
            nn.Linear(HIDDEN[2], 10),
        )

    def forward(self, x):
        return self.net(x)


class TensorCIFAR10(Dataset):
    def __init__(self, images, targets, train):
        self.images = images
        self.targets = targets
        self.train = train
        self.mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).view(3, 1, 1)

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
        image = (image - self.mean) / self.std
        return image, int(self.targets[index])


class HiddenRatioAccumulator:
    def __init__(self, device):
        self.device = device
        self.edges = np.linspace(0.0, 20.0, 81, dtype=np.float64)
        self.counts = {
            stage: torch.zeros(80, dtype=torch.int64, device=device)
            for stage in ("early", "middle", "late")
        }
        self.overflow = {
            stage: torch.zeros((), dtype=torch.int64, device=device)
            for stage in self.counts
        }
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
                if "exp_avg" not in state:
                    continue
                m = state["exp_avg"].detach().float()
                v = state["exp_avg_sq"].detach().float()
                m2 = m.square()
                y = ((1.0 - beta) / beta) * (v - m2).clamp_min_(0.0) / m2.clamp_min(1e-30)
                y = torch.nan_to_num(y, nan=float("inf"), posinf=float("inf"), neginf=0.0)
                flat = y.flatten()
                in_range = flat <= 20.0
                self.counts[stage].add_(
                    torch.histc(flat[in_range], bins=80, min=0.0, max=20.0).to(torch.int64)
                )
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

                def reference_cdf(value):
                    return 0.0 if value <= 0 else math.erfc(math.sqrt(1.0 / value))

                writer.writerow([
                    left, right,
                    masses["early"], masses["middle"], masses["late"],
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
            "overflow_fraction": {
                stage: cpu_overflow[stage] / max(1, self.total[stage])
                for stage in self.counts
            },
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
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    return loss_sum / total, correct / total


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    data = torch.load(DATA_FILE, map_location="cpu")
    train_set = TensorCIFAR10(data["train_images"], data["train_targets"], train=True)
    test_set = TensorCIFAR10(data["test_images"], data["test_targets"], train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = MLP().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(BETA, BETA), eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    accumulator = HiddenRatioAccumulator(device) if LOG_HIDDEN_RATIO else None
    history = []
    optimizer_step = 0
    epoch = 0
    started = time.time()
    print(json.dumps({"event": "start", "device": str(device), "parameters": parameter_count, "beta": BETA}), flush=True)

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
        "dataset": "CIFAR-10, decoded losslessly from the same Hugging Face parquet files as the prior runs",
        "seed": SEED,
        "optimizer_steps": optimizer_step,
        "evaluation_every_steps": EVAL_EVERY,
        "batch_size": BATCH_SIZE,
        "augmentation": ["RandomCrop(32,padding=4,padding_mode=reflect)", "RandomHorizontalFlip"],
        "normalization": {"mean": mean, "std": std},
        "network": {
            "type": "MLP",
            "hidden_layers": HIDDEN,
            "activation": "ReLU",
            "parameters": parameter_count,
        },
        "optimizer": {
            "name": "Adam",
            "lr": LR,
            "betas": [BETA, BETA],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "scheduler": None,
        },
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
        metadata["hidden_ratio"] = {"logging": "disabled for learning-rate sweep"}
    with (RUN_DIR / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (RUN_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"event": "done", **metadata}), flush=True)


if __name__ == "__main__":
    main()
