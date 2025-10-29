#!/usr/bin/env bash
# 图像化脚本（生成时间序列对应图像）

set -euo pipefail

GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[图像化] 使用 GPU=${CUDA_VISIBLE_DEVICES}"
echo "[图像化] 开始图像化处理..."

# 直接运行图像化，Hydra已禁用自动输出目录
python -u -m mmts.cli.imgify "$@"

echo "[图像化] 图像化完成！"
