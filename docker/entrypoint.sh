#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "${1:-validate}" != "validate" ]]; then
  /opt/venvs/training/bin/python /workspace/scripts/validate_environment.py
fi

case "${1:-validate}" in
  validate)
    shift || true
    exec /opt/venvs/training/bin/python /workspace/scripts/validate_environment.py "$@"
    ;;
  smoke|evaluate)
    action="$1"
    shift
    validation_args=(--require-model /models/model --require-checkpoint /checkpoints/model)
    if [[ "$action" == "evaluate" ]]; then
      validation_args+=(--require-dataset-cache /cache/huggingface --dataset-revision "${DATASET_REVISION:-}")
    fi
    /opt/venvs/training/bin/python /workspace/scripts/validate_environment.py "${validation_args[@]}"
    exec "/workspace/scripts/${action}.sh" "$@"
    ;;
  loading)
    shift
    exec /opt/venvs/loading/bin/python "$@"
    ;;
  training)
    shift
    exec /opt/venvs/training/bin/python "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
