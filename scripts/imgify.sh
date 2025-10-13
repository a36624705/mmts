#!/usr/bin/env bash
# 图像化（imgify）：读取配置文件 -> 后台运行 -> 写日志
# 用法：
#   bash scripts/imgify.sh configs/defaults.yaml
# 或：
#   bash scripts/imgify.sh    # 默认 configs/defaults.yaml

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CONFIG="${1:-configs/defaults.yaml}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="imgify_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[IMGIFY] GPU=${CUDA_VISIBLE_DEVICES}"
echo "[IMGIFY] CONFIG=${CONFIG}"
echo "[IMGIFY] LOG=${LOG_FILE}"

nohup python -u -m mmts.cli.imgify --config "${CONFIG}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "[IMGIFY] Started PID=${PID} (nohup)"
echo "[IMGIFY] Use 'tail -f ${LOG_FILE}' to check progress."
