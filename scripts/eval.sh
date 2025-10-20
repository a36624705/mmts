#!/usr/bin/env bash
# 评估脚本（依赖配置文件和 eval.py 自动选择最新 LoRA/Processor）

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="eval_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[评估] 使用 GPU=${CUDA_VISIBLE_DEVICES}"
echo "[评估] 日志文件=${LOG_FILE}"

nohup python -u -m mmts.cli.eval "$@" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[评估] 启动进程 PID=${PID}"
echo "查看日志：tail -f ${LOG_FILE}"
