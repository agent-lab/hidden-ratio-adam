# Pretraining

Use a separate Python 3.12 environment with a CUDA-compatible PyTorch installation:

```bash
cd pretrain
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation
```

The example YAMLs request FlashAttention 2. Alternatively, set `attn_implementation: sdpa` for a supported PyTorch/Transformers backend; that changes the execution backend from the provided example. Dependencies are inherited lower bounds, not a recovered historical lockfile.

Prepare FineWeb shards for the 32k-tokenizer LLaMA-style 20M model:

```bash
python scripts/build_fineweb_bins.py --preset fineweb10b --output-dir ./data/fineweb10B_llama20m_tinyllama32k_bin
python train.py --config=config/llama20m_adam_nsr.yaml
```

For Pythia, use the GPT-2-tokenized cached shards:

```bash
python scripts/download_cached_fineweb_bins.py --preset fineweb10b --output-dir ./data/fineweb10B
# Set trainset_bin_dir / validset_bin_dir in the model YAML to this directory.
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --config=config/pythia160m_adam_nsr.yaml
# Pythia-1B example: config/pythia1b_adam_nsr.yaml (adjust GPU count / accumulation).
```

Copy a YAML and set `optim: adamw` for the Adam baseline, or use `optim: adam_nsr` for TR-Adam. Set `beta1`, `beta2`, learning rate and `seed` explicitly. `moment1_dtype: fp8_e4m3` selects FP8 momentum. The token budget per step is sequence length × micro-batch size × gradient accumulation × GPU count; preserve it when changing GPU count.

For epsilon ablations, change `eps` in a copied YAML. The isolated zero-epsilon implementation uses algebraic cancellation of |m| and sign(0)=0:

```bash
python train.py --config=config/llama20m_tr_adam_eps0.yaml
```

The zero-epsilon example uses beta=.90, LR=.018, seed100. Other example YAML learning rates are not a complete paper sweep. Positive-epsilon experiments use the standard `adam_nsr` implementation. Training outputs are written under `outputs/`; W&B is off by default.
