#!/usr/bin/env bash
# 评估（eval）：读取配置文件 -> 自动发现最新的 LoRA/Processor -> 后台运行 -> 写日志
# 用法：
#   bash scripts/eval.sh configs/defaults.yaml
# 或：
#   bash scripts/eval.sh

set -euo pipefail

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CONFIG="${1:-configs/defaults.yaml}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="eval_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[EVAL] GPU=${CUDA_VISIBLE_DEVICES}"
echo "[EVAL] CONFIG=${CONFIG}"

# 自动发现最新 LoRA / Processor
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

echo "[EVAL] LORA_DIR=${LORA_DIR:-<not found>}"
echo "[EVAL] PROCESSOR_DIR=${PROC_DIR:-<not found>}"
echo "[EVAL] LOG=${LOG_FILE}"

nohup python -u -m mmts.cli.eval \
  --config "${CONFIG}" \
  --lora-dir "${LORA_DIR}" \
  --processor-dir "${PROC_DIR}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[EVAL] Started PID=${PID} (nohup)"
echo "[EVAL] Use 'tail -f ${LOG_FILE}' to check evaluation progress."
