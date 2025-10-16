#!/usr/bin/env bash
# 训练脚本

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }
RUN_ID="train_$(ts)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "[训练] 使用 GPU=${CUDA_VISIBLE_DEVICES}"
echo "[训练] 日志文件=${LOG_FILE}"

# 说明：
# - 不再传 --config，Hydra 会自动读取装饰器默认配置。
# - 可在命令行传递覆盖项，例如：
#   bash scripts/train.sh train.epochs=5 model.base_id=Qwen/Qwen2.5-VL-3B-Instruct

nohup python -u -m mmts.cli.train "$@" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "[训练] 启动进程 PID=${PID}"
echo "查看日志：tail -f ${LOG_FILE}"
