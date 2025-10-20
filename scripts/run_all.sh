#!/usr/bin/env bash
# 一键：图像化 → 训练 → 评估（顺序执行，整体后台）

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="run_all_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[全流程] 使用 GPU=${CUDA_VISIBLE_DEVICES}"
echo "[全流程] 日志文件=${LOG_FILE}"

nohup bash -c '
  set -euo pipefail

  echo "[1/3] 图像化..."
  python -u -m mmts.cli.imgify "$@" || exit 1
  echo "[1/3] 完成"

  echo "[2/3] 训练..."
  python -u -m mmts.cli.train "$@" || exit 1
  echo "[2/3] 完成"

  echo "[3/3] 评估..."
  python -u -m mmts.cli.eval "$@" || exit 1
  echo "[3/3] 完成"

  echo "[全流程] 完成"
' _ "$@" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[全流程] 启动进程 PID=${PID}"
echo "查看日志：tail -f ${LOG_FILE}"
echo "停止进程：kill ${PID}"
