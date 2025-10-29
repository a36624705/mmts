#!/usr/bin/env bash
# 训练脚本（优化版本）

set -euo pipefail

GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 使用实验管理器，不再需要单独的日志目录
echo "[训练] 使用 GPU=${CUDA_VISIBLE_DEVICES}"
echo "[训练] 实验将保存在 experiments/ 目录下"

# 直接运行训练，Hydra已禁用自动输出目录
python -u -m mmts.cli.train "$@"

echo "[训练] 训练完成！"
echo "查看实验: ls experiments/"
