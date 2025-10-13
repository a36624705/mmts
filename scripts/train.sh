#!/usr/bin/env bash
# 训练（train）：读取配置文件 -> 后台运行 -> 写日志
# 用法：
#   bash scripts/train.sh configs/defaults.yaml
# 或：
#   bash scripts/train.sh

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CONFIG="${1:-configs/defaults.yaml}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="train_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[TRAIN] GPU=${CUDA_VISIBLE_DEVICES}"
echo "[TRAIN] CONFIG=${CONFIG}"
echo "[TRAIN] LOG=${LOG_FILE}"

nohup python -u -m mmts.cli.train --config "${CONFIG}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "[TRAIN] Started PID=${PID} (nohup)"
echo "[TRAIN] Use 'tail -f ${LOG_FILE}' to monitor training progress."
