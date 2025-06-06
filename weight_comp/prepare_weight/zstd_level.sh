#!/usr/bin/env bash
set -euo pipefail                     # 出错即退出，捕获未定义变量
PY_SCRIPT="prepare_weight.py"
MODEL_PATH="/opt/models/hf/Mistral-7B-v0.1"

for level in {21..22}; do
    OUTDIR="./zstd_comped_weights_level${level}"   # 输出目录按 level 命名
    mkdir -p "${OUTDIR}"                           # 若目录不存在则创建
    echo "=== Running level ${level} → ${OUTDIR} ==="
    python3 -u "${PY_SCRIPT}" \
        --model "${MODEL_PATH}" \
        --outdir "${OUTDIR}" \
        --level "${level}" 
done

# PY_SCRIPT="restore_from_compressed.py"
# for level in {-1..0}; do
#     OUTDIR="./zstd_comped_weights_level${level}"
#     echo "=== Running level ${level} → ${OUTDIR} ==="
#     python3 -u "${PY_SCRIPT}" \
#         --compdir "${OUTDIR}" \
#         --check_diff
# done