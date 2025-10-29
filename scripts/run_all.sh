#!/usr/bin/env bash
# 一键：图像化 → 训练 → 评估（顺序执行）

set -euo pipefail

GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 生成统一的实验ID
EXPERIMENT_ID="run_all_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "MMTS 完整流程启动"
echo "=========================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "实验ID: $EXPERIMENT_ID"
echo "=========================================="

echo ""
echo "------------------------------------------"
echo "[1/3] 图像化阶段"
echo "------------------------------------------"
python -u -m mmts.cli.imgify "$@" || {
    echo "[ERROR] 图像化失败"
    exit 1
}
echo "[1/3] 图像化完成"

echo ""
echo "------------------------------------------"
echo "[2/3] 训练阶段"
echo "------------------------------------------"
python -u -m mmts.cli.train +experiment_id="$EXPERIMENT_ID" "$@" || {
    echo "[ERROR] 训练失败"
    exit 1
}
echo "[2/3] 训练完成"

echo ""
echo "------------------------------------------"
echo "[3/3] 评估阶段"
echo "------------------------------------------"
python -u -m mmts.cli.eval eval.experiment_id="$EXPERIMENT_ID" eval.use_latest=false "$@" || {
    echo "[ERROR] 评估失败"
    exit 1
}
echo "[3/3] 评估完成"

echo ""
echo "=========================================="
echo "完整流程成功完成！"
echo "=========================================="
echo "实验ID: $EXPERIMENT_ID"
echo "结果目录: experiments/$EXPERIMENT_ID/"
echo "=========================================="
