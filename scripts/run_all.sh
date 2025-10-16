#!/usr/bin/env bash
# 一键：图像化 → 训练 → 评估（顺序执行，整体后台）

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
export OUTPUTS_ROOT   # 传入子 shell
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="run_all_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[全流程] GPU=${CUDA_VISIBLE_DEVICES}"
echo "[全流程] 日志=${LOG_FILE}"

nohup bash -c '
  set -euo pipefail
  echo "[1/3] 图像化..."
  python -u -m mmts.cli.imgify "$@" || exit 1
  echo "[1/3] 完成"

  echo "[2/3] 训练..."
  python -u -m mmts.cli.train "$@" || exit 1
  echo "[2/3] 完成"

  echo "[3/3] 评估准备：查找最新 LoRA/Processor..."
  LORA_DIR=$(ls -dt "${OUTPUTS_ROOT}/lora"/*/ 2>/dev/null | head -n 1 || true)
  PROC_DIR=$(ls -dt "${OUTPUTS_ROOT}/processor"/*/ 2>/dev/null | head -n 1 || true)
  echo "[3/3] LORA=${LORA_DIR:-<未找到>}"
  echo "[3/3] PROC=${PROC_DIR:-<未找到>}"

  if [[ -z "${LORA_DIR}" || -z "${PROC_DIR}" ]]; then
    echo "[错误] 未找到 LoRA 或 Processor 目录" >&2
    exit 1
  fi

  # 去掉末尾斜杠，避免 Hydra 解析路径时带 //
  LORA_DIR=${LORA_DIR%/}
  PROC_DIR=${PROC_DIR%/}

  echo "[3/3] 评估..."
  python -u -m mmts.cli.eval \
    "eval.lora_dir=${LORA_DIR}" \
    "eval.processor_dir=${PROC_DIR}" \
    "$@" || exit 1
  echo "[3/3] 完成"

  echo "[全流程] 完成"
' _ "$@" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[全流程] PID=${PID}"
echo "查看日志：tail -f ${LOG_FILE}"
echo "停止进程：kill ${PID}"
