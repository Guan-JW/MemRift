#!/usr/bin/env bash
set -euo pipefail

run_name="smoke-$(date -u +%Y%m%dT%H%M%SZ)"
exec /opt/venvs/training/bin/python /workspace/scripts/run_result_driver.py \
  --run-name "$run_name" --results-dir /results --timeout-seconds "${TIMEOUT_SECONDS:-900}" \
  --min-available-gib "${MIN_AVAILABLE_GIB:-4}" \
  --model-id "${MODEL_LOGICAL_ID:-reviewer-model}" \
  --checkpoint-id "${CHECKPOINT_LOGICAL_ID:-reviewer-checkpoint}" -- \
  /opt/venvs/training/bin/python /workspace/src/train_memrift.py \
  --model /models/model --checkpoint /checkpoints/model --results-dir "/results/$run_name" \
  --save-model-dir "/results/$run_name/models" \
  --dataset-cache /cache/huggingface --device cuda:0 --wandb-mode disabled \
  --tegra-csv tegrastats.csv \
  --synthetic-data \
  --finetune_type lora --hook --weight --weight_async --activation --act_async \
  --gradient_checkpointing --gc_keep_recompute_weights --gc_no_recompute_prefetch \
  --weight_async_concurrency 1 --act_compact_concurrency 1 \
  --max_length 2048 --batch_size 1 --round 2 --warmup_rounds 1 "$@"
