#!/usr/bin/env bash
# 图像化脚本（生成时间序列对应图像）

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="imgify_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[图像化] 使用 GPU=${CUDA_VISIBLE_DEVICES}"
echo "[图像化] 日志文件=${LOG_FILE}"


nohup python -u -m mmts.cli.imgify "$@" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "[图像化] 启动进程 PID=${PID}"
echo "查看日志：tail -f ${LOG_FILE}"
