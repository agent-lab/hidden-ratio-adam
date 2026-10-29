# The Hidden Ratio in Adam

Compact code release for **The Hidden Ratio in Adam: Stable Structure, Compression, and Sign Dynamics**.
TR-Adam is named `AdamNSR` in the implementation.

| Directory | Contents |
| --- | --- |
| `optimizer/` | Standalone TR-Adam optimizer (PyTorch only) |
| `pretrain/` | LLaMA-style 20M and Pythia pretraining, data preparation, example YAMLs |
| `posttrain/` | Bundled verl runtime and the ReMax / UltraChat SFT launch examples |
| `cifar/` | CIFAR-10 MLP and CIFAR-100 ResNet-18 experiments |

Historical logs, checkpoints, datasets, tracking files, cluster launchers, upstream docs/tests and unrelated examples are excluded. The bundled verl runtime is retained to preserve its internal imports and configuration dependencies.

## Optimizer

From this directory, with PyTorch installed:

```python
from optimizer import AdamNSR

optimizer = AdamNSR(
    model.parameters(), lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
    weight_decay=0.0,
    y_max=64.0, moment1_dtype="fp32", moment2_dtype="fp32",
    compute_dtype="fp32", skip_update_on_bad_y=False,
)
```

Use tied betas. The paper's 16-value codebook is selected internally; no table argument is needed.
`moment1_dtype="fp8_e4m3"` enables the FP8 momentum variant. State quantization is simulated with FP32 storage tensors; this release does not implement packed 4-bit memory. The current ratio is quantized before the parameter update. FP8 momentum is quantized for state writeback; the current update uses the fresh momentum working value.

See each directory's README for installation and launch commands. Use separate environments for pretraining and post-training because their inherited NumPy requirements differ. The launch configs are starting examples, not an exhaustive manifest of every paper row; match hyperparameters and budgets to the paper.

A small CPU check (no training) is available:

```bash
python tests/check_optimizer.py
```
