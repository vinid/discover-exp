#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-kb_gptoss120b_baseline}"
CONFIG="src/kernelbench_tinker/config/rl_kernelbench.yaml"

python -m kernelbench_tinker.scripts.train_kernel_rl \
  --config "${CONFIG}" \
  model_name="openai/gpt-oss-120b" \
  lora_rank=32 \
  learning_rate=0.00004 \
  max_tokens=32768 \
  temperature=1.0 \
  kl_penalty_coef=0.1 \
  save_every=2 \
  dataset_builder.renderer_name="gpt_oss_high_reasoning" \
  dataset_builder.batch_size=8 \
  dataset_builder.group_size=64 \
  log_path="./runs/${RUN_NAME}"
