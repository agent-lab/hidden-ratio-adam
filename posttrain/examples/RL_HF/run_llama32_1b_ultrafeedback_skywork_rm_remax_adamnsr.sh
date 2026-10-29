#!/usr/bin/env bash
set -euo pipefail

unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
CACHE_BASE="${CACHE_BASE:-${ROOT_DIR}/.cache}"
mkdir -p "${CACHE_BASE}"
CACHE_BASE="$(cd "${CACHE_BASE}" && pwd)"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_BASE}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export RAY_TMPDIR="${RAY_TMPDIR:-${CACHE_BASE}/ray}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${RAY_TMPDIR}"

nnodes="${NNODES:-1}"

# This script expects prompt-only RL parquet files. To prepare the default data:
#   python examples/data_preprocess/ultrafeedback_rl.py --local_save_dir data/ultrafeedback_rl
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
ULTRAFEEDBACK_TRAIN_PATH="${ULTRAFEEDBACK_TRAIN_PATH:-${DATA_ROOT}/ultrafeedback_rl/train.parquet}"
ULTRAFEEDBACK_TEST_PATH="${ULTRAFEEDBACK_TEST_PATH:-${DATA_ROOT}/ultrafeedback_rl/test.parquet}"
train_files="${TRAIN_FILES:-$ULTRAFEEDBACK_TRAIN_PATH}"
test_files="${TEST_FILES:-$ULTRAFEEDBACK_TEST_PATH}"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-Skywork/Skywork-Reward-V2-Llama-3.2-3B}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-${SCRIPT_DIR}/llama3_simple_chat_template.jinja}"
export CHAT_TEMPLATE_FILE

max_prompt_length="${MAX_PROMPT_LENGTH:-512}"
max_response_length="${MAX_RESPONSE_LENGTH:-512}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-128}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-32}"
micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-2}"
critic_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-2}"
ppo_epochs="${PPO_EPOCHS:-1}"
project_name="${PROJECT_NAME:-verl_rlhf_ultrafeedback}"
exp_name="${EXP_NAME:-llama32_1b_skyworkrm3b_remax_adamnsr}"

adv_estimator="${ADV_ESTIMATOR:-remax}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-1}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"

use_kl_in_reward="${USE_KL_IN_REWARD:-True}"
kl_coef="${KL_COEF:-0.001}"
use_kl_loss="${USE_KL_LOSS:-False}"
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}"
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.2}"
clip_ratio_c="${CLIP_RATIO_C:-3.0}"
loss_agg_mode="${LOSS_AGG_MODE:-token-mean}"
entropy_coeff="${ENTROPY_COEFF:-0.0}"

rollout_name="${ROLLOUT_NAME:-vllm}"
rollout_logprob_mb="${ROLLOUT_LOGPROB_MB_PER_GPU:-2}"
ref_logprob_mb="${REF_LOGPROB_MB_PER_GPU:-2}"
gen_tp="${GEN_TP:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
gpu_mem_util="${GPU_MEMORY_UTILIZATION:-0.35}"

rm_num_workers="${RM_NUM_WORKERS:-4}"
rm_rollout_name="${RM_ROLLOUT_NAME:-vllm}"
rm_rollout_gpu_mem_util="${RM_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
rm_rollout_tp="${RM_ROLLOUT_TP:-1}"
rm_prompt_length="${RM_PROMPT_LENGTH:-1536}"
rm_response_length="${RM_RESPONSE_LENGTH:-512}"

beta1="${BETA1:-0.90}"
beta2="${BETA2:-0.90}"
actor_lr="${ACTOR_LR:-1e-6}"
weight_decay="${WEIGHT_DECAY:-0.0}"
y_max="${Y_MAX:-64}"

out_dir="${OUT_DIR:-${ROOT_DIR}/outputs/${project_name}/${exp_name}}"
mkdir -p "${out_dir}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${adv_estimator}" \
  data.train_files="${train_files}" \
  data.val_files="${test_files}" \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  data.train_batch_size="${train_prompt_bsz}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.custom_chat_template_file="${CHAT_TEMPLATE_FILE}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.optimizer=AdamNSR \
  actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.adam_nsr \
  actor_rollout_ref.actor.optim.lr="${actor_lr}" \
  actor_rollout_ref.actor.optim.weight_decay="${weight_decay}" \
  actor_rollout_ref.actor.optim.betas="[${beta1},${beta2}]" \
  +actor_rollout_ref.actor.optim.override_optimizer_config.y_max="${y_max}" \
  +actor_rollout_ref.actor.optim.override_optimizer_config.moment1_dtype=fp32 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.moment1_quant=per_column \
  +actor_rollout_ref.actor.optim.override_optimizer_config.moment2_dtype=fp32 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.skip_update_on_bad_y=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batch_size_per_gpu}" \
  actor_rollout_ref.actor.ppo_epochs="${ppo_epochs}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
  actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
  actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
  actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
  actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
  actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
  actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
  actor_rollout_ref.actor.entropy_coeff="${entropy_coeff}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_logprob_mb}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
  actor_rollout_ref.rollout.name="${rollout_name}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_mem_util}" \
  actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
  actor_rollout_ref.rollout.temperature="${temperature}" \
  actor_rollout_ref.rollout.top_p="${top_p}" \
  actor_rollout_ref.rollout.top_k="${top_k}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_logprob_mb}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-True}" \
  critic.enable=False \
  critic.model.path="${MODEL_PATH}" \
  critic.optim.lr="${CRITIC_LR:-1e-5}" \
  critic.ppo_micro_batch_size_per_gpu="${critic_micro_batch_size_per_gpu}" \
  reward.reward_model.enable=True \
  reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
  reward.reward_model.rollout.name="${rm_rollout_name}" \
  reward.reward_model.rollout.gpu_memory_utilization="${rm_rollout_gpu_mem_util}" \
  reward.reward_model.rollout.tensor_model_parallel_size="${rm_rollout_tp}" \
  reward.reward_model.rollout.prompt_length="${rm_prompt_length}" \
  reward.reward_model.rollout.response_length="${rm_response_length}" \
  reward.num_workers="${rm_num_workers}" \
  algorithm.use_kl_in_reward="${use_kl_in_reward}" \
  algorithm.kl_penalty="${KL_PENALTY:-kl}" \
  algorithm.kl_ctrl.kl_coef="${kl_coef}" \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-2}" \
  trainer.nnodes="${nnodes}" \
  trainer.save_freq="${SAVE_FREQ:-20}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-3}" \
  trainer.default_local_dir="${out_dir}" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  "$@" 2>&1 | tee "${out_dir}/${project_name}_${exp_name}_remax_adamnsr.log"
