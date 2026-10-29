# CIFAR appendix experiments

```bash
cd cifar
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Prepare CIFAR Hugging Face parquet files in `data/cifar10_hf/` or `data/cifar100_hf/`, each containing `train.parquet` and `test.parquet`. The converters expect `img.bytes` plus `label` (CIFAR-10) or `fine_label` (CIFAR-100), and preserve image bytes and example order. Run:

```bash
python convert_cifar10_parquet_to_tensor.py
python convert_cifar100_parquet_to_tensor.py
```

Override `CIFAR10_PARQUET_DIR` / `CIFAR100_PARQUET_DIR` and `CIFAR10_TENSOR_FILE` / `CIFAR100_TENSOR_FILE` for other paths. The training inputs are `data/cifar10_uint8.pt` and `data/cifar100_uint8.pt` by default.

Example runs (3000 optimizer steps, batch512, LR2e-4):

```bash
ADAM_BETA=0.90 CIFAR10_SEED=1 python train_cifar10_473m_3000.py
ADAM_BETA=0.90 CIFAR10_SEED=1 python train_cifar10_473m_3000_adam_nsr.py
CIFAR100_OPTIMIZER=adam CIFAR100_LOG_HIDDEN_RATIO=1 CIFAR100_RUN_DIR=outputs/cifar100_adam python train_cifar100_resnet18_3000.py
CIFAR100_OPTIMIZER=tr_adam CIFAR100_RUN_DIR=outputs/cifar100_tr_adam python train_cifar100_resnet18_3000.py
```

CIFAR-10 defaults to the 4.73M-parameter MLP. Set `CIFAR10_HIDDEN=128,128,128` for the 427.7K model. For the paper comparison use betas .90/.92/.95 and seeds1/2, with a distinct `CIFAR10_RUN_DIR` or `CIFAR100_RUN_DIR` for each run. Average each seed's test loss at steps2842/2940/3000, then report mean and sample standard deviation across seeds.

Adam ratio diagnostics use seed1 and pool all parameter entries at every step in the three 1000-step windows. CIFAR-10 TR-Adam diagnostics, when enabled, instead describe the quantized recursive state. Logging can be disabled with `CIFAR10_LOG_HIDDEN_RATIO=0` or `CIFAR100_LOG_HIDDEN_RATIO=0`. These switches do not change the optimizer.
