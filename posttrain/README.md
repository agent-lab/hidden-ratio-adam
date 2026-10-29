# ReMax and SFT

This directory retains the archived verl runtime with AdamNSR integration. Use Linux with CUDA GPUs, a separate Python environment, access to the requested Hugging Face model weights, and compatible PyTorch / vLLM / FlashAttention versions.

```bash
cd posttrain
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[vllm]'
uv pip install flash-attn --no-build-isolation
```

The package dependency constraints are inherited from the bundled runtime, not a recovered exact training lockfile. GPU installation and end-to-end training have not been revalidated during packaging.

## ReMax

```bash
python examples/data_preprocess/ultrafeedback_rl.py --local_save_dir data/ultrafeedback_rl
bash examples/RL_HF/run_llama32_1b_ultrafeedback_skywork_rm_remax_adamnsr.sh
```

The default actor is `meta-llama/Llama-3.2-1B-Instruct`, with `Skywork/Skywork-Reward-V2-Llama-3.2-3B` as the reward model. Override `MODEL_PATH`, `REWARD_MODEL_PATH`, `NGPUS_PER_NODE`, `TRAIN_FILES`, `TEST_FILES`, `BETA1`, `BETA2` and `ACTOR_LR` as needed. Pass extra Hydra arguments at the end, for example `trainer.total_training_steps=600`. This is a launch example, not the complete later rebuttal run configuration.

## SFT

```bash
python examples/data_preprocess/ultrachat_200k_sft.py --local_save_dir examples/sft/ultrachat/data
bash examples/sft/ultrachat/run_llama32_1b_ultrachat_adamnsr.sh 2 outputs/sft_adamnsr
```

Override `MODEL_PATH`, `TRAIN_FILES`, `VAL_FILES`, `GLOBAL_BSZ`, `MICRO_BSZ`, `LR`, `BETA1`, `BETA2` and `TOTAL_TRAINING_STEPS`. The SFT example retains the original base-model default and text template.

Both examples use FP32 momentum plus the paper's 16-value ratio codebook. For FP8, append `actor_rollout_ref.actor.optim.override_optimizer_config.moment1_dtype=fp8_e4m3` (ReMax) or `optim.override_optimizer_config.moment1_dtype=fp8_e4m3` (SFT). Console logging is enabled by default; W&B is optional via `trainer.logger='["console","wandb"]'`.
