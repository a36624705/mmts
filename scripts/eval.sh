#!/usr/bin/env bash
# 评估脚本（自动发现最新 LoRA 和 Processor）

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="eval_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[评估] 使用 GPU=${CUDA_VISIBLE_DEVICES}"

# 自动查找最新 LoRA 与 Processor 目录
latest_dir() {
  local parent="$1"
  if compgen -G "${parent}/*/" > /dev/null; then
    ls -dt "${parent}"/*/ | head -n 1
  else
    echo ""
  fi
}

LORA_DIR="$(latest_dir "${OUTPUTS_ROOT}/lora")"
PROC_DIR="$(latest_dir "${OUTPUTS_ROOT}/processor")"

echo "[评估] LoRA目录=${LORA_DIR:-<未找到>}"
echo "[评估] Processor目录=${PROC_DIR:-<未找到>}"
echo "[评估] 日志文件=${LOG_FILE}"


nohup python -u -m mmts.cli.eval \
  "eval.lora_dir=${LORA_DIR}" \
  "eval.processor_dir=${PROC_DIR}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[评估] 启动进程 PID=${PID}"
echo "使用命令查看日志：tail -f ${LOG_FILE}"
