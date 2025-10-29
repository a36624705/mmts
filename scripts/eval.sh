#!/usr/bin/env bash
# 评估脚本（优化版本）

set -euo pipefail

GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[评估] 使用 GPU=${CUDA_VISIBLE_DEVICES}"
echo "[评估] 将使用最新实验进行评估"

python -u -m mmts.cli.eval "$@"

echo "[评估] 评估完成！"
echo "查看结果: ls experiments/*/figures/"
