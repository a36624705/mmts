#!/usr/bin/env bash
# 一键全流程：imgify -> train -> eval（同步执行，但整体后台运行）
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CONFIG="${1:-configs/defaults.yaml}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="run_all_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[RUN-ALL] CONFIG=${CONFIG} | GPU=${CUDA_VISIBLE_DEVICES}"
echo "[RUN-ALL] LOG=${LOG_FILE}"

# 后台启动整个同步流程（每一步都是阻塞执行）
nohup bash -c "
  echo '[1/3] IMGIFY starting...'
  python -u -m mmts.cli.imgify --config '${CONFIG}' || exit 1
  echo '[1/3] IMGIFY done.'

  echo '[2/3] TRAIN starting...'
  python -u -m mmts.cli.train --config '${CONFIG}' || exit 1
  echo '[2/3] TRAIN done.'

  echo '[3/3] EVAL starting...'
  python -u -m mmts.cli.eval --config '${CONFIG}' || exit 1
  echo '[3/3] EVAL done.'

  echo '[RUN-ALL] ✅ 全流程完成！'
" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[RUN-ALL] Started PID=${PID} (nohup)"
echo "[RUN-ALL] 查看进度: tail -f ${LOG_FILE}"
echo "[RUN-ALL] 停止进程: kill ${PID}"
