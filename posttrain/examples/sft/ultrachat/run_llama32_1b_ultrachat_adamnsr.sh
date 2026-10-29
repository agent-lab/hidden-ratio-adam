#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: run_llama32_1b_ultrachat_adamnsr.sh <nproc_per_node> <save_path> [other_configs...]"
  exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B}"
TRAIN_FILES="${TRAIN_FILES:-${SCRIPT_DIR}/data/train.parquet}"
VAL_FILES="${VAL_FILES:-${SCRIPT_DIR}/data/test.parquet}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-${SCRIPT_DIR}/pythia_simple_chat_template.jinja}"
PROJECT_NAME="${PROJECT_NAME:-verl_sft}"
EXP_NAME="${EXP_NAME:-llama32_1b_ultrachat_adamnsr}"
GLOBAL_BSZ="${GLOBAL_BSZ:-128}"
MICRO_BSZ="${MICRO_BSZ:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
BETA1="${BETA1:-0.90}"
BETA2="${BETA2:-0.90}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2000}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:--1}"
Y_MAX="${Y_MAX:-64}"

cd "${ROOT_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.messages_key=messages \
  data.ignore_input_ids_mismatch=True \
  data.train_batch_size="${GLOBAL_BSZ}" \
  data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=right \
  model.path="${MODEL_PATH}" \
  +model.custom_chat_template_file="${CHAT_TEMPLATE_FILE}" \
  model.use_remove_padding=true \
  optim.optimizer=AdamNSR \
  optim.optimizer_impl=verl.utils.adam_nsr \
  optim.lr="${LR}" \
  optim.weight_decay="${WEIGHT_DECAY}" \
  optim.betas="[${BETA1},${BETA2}]" \
  optim.lr_scheduler_type=constant \
  optim.lr_warmup_steps="${WARMUP_STEPS:-200}" \
  optim.lr_warmup_steps_ratio=0.0 \
  +optim.override_optimizer_config.y_max="${Y_MAX}" \
  +optim.override_optimizer_config.moment1_dtype=fp32 \
  +optim.override_optimizer_config.moment1_quant=per_column \
  +optim.override_optimizer_config.moment2_dtype=fp32 \
  +optim.override_optimizer_config.skip_update_on_bad_y=False \
  trainer.default_local_dir="${save_path}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.logger='["console"]' \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  "$@"
