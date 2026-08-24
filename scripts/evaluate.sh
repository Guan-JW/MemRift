#!/usr/bin/env bash
set -euo pipefail
: "${DATASET_REVISION:?DATASET_REVISION must be the exact 40-hex revision already present in the mounted dataset cache}"

run_name="evaluation-$(date -u +%Y%m%dT%H%M%SZ)"
exec /opt/venvs/training/bin/python /workspace/scripts/run_result_driver.py \
  --run-name "$run_name" --results-dir /results --timeout-seconds "${TIMEOUT_SECONDS:-2400}" \
  --min-available-gib "${MIN_AVAILABLE_GIB:-4}" \
  --model-id "${MODEL_LOGICAL_ID:-reviewer-model}" \
  --checkpoint-id "${CHECKPOINT_LOGICAL_ID:-reviewer-checkpoint}" \
  --dataset-revision "$DATASET_REVISION" -- \
  /opt/venvs/training/bin/python /workspace/src/train_memrift.py \
  --model /models/model --checkpoint /checkpoints/model --results-dir "/results/$run_name" \
  --save-model-dir "/results/$run_name/models" \
  --dataset-cache /cache/huggingface --device cuda:0 --wandb-mode disabled \
  --finetune_type lora --hook --weight --weight_async --activation --act_async \
  --gradient_checkpointing --gc_keep_recompute_weights --gc_no_recompute_prefetch \
  --weight_async_concurrency 1 --act_compact_concurrency 1 \
  --max_length "${CONTEXT_TOKENS:-2048}" --batch_size "${BATCH_SIZE:-1}" \
  --round "${ROUNDS:-7}" --warmup_rounds "${WARMUP_ROUNDS:-1}" "$@"
