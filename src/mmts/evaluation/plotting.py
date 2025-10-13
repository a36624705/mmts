# src/mmts/evaluation/plotting.py
# -*- coding: utf-8 -*-
"""
回归可视化（Plotting for Regression）

本模块提供三类典型图表：
1) 序列对比图（前 N 样本的 y_true vs. y_pred 折线）；
2) Parity（散点）图：y_true 为 x 轴，y_pred 为 y 轴，并绘制 y=x 参考线；
3) 误差分布图：e = y_pred - y_true 的直方图（可叠加均值±标准差标注）。

特点与约定
---------
- 仅依赖 matplotlib（不使用 seaborn），减少环境依赖；
- 不设置特定配色方案，交给 matplotlib 默认；
- 所有保存操作委托给 `mmts.utils.io.save_matplotlib_figure`，自动创建父目录；
- 函数既可返回 matplotlib Figure，也可在传入 out_path 时直接落盘并返回路径。

注意
----
- 输入将被转换为 1D numpy 数组（float32）；
- 自动屏蔽 NaN/±Inf；对于 Parity/Histogram，会仅在有效样本上绘制；
- 出图函数不负责指标计算，若需在标题中显示 RMSE/MAE/R²，请在外层提前计算并传入 title。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from ..utils.io import save_matplotlib_figure


# ============================================================================
# 一、内部工具
# ============================================================================

def _to_float_np1d(x) -> np.ndarray:
    """将输入转为 1D float32 numpy 数组。"""
    arr = np.asarray(list(x), dtype=np.float32).reshape(-1)
    return arr


def _valid_mask(a: np.ndarray) -> np.ndarray:
    """返回有限值（非 NaN 且非 Inf）的布尔掩码。"""
    return np.isfinite(a)


def _mask_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回二者均为有限值的位置布尔掩码。"""
    return _valid_mask(a) & _valid_mask(b)


# ============================================================================
# 二、序列对比图
# ============================================================================

def plot_series_comparison(
    y_true,
    y_pred,
    first_n: Optional[int] = 200,
    title: str = "Prediction vs. Ground Truth",
    xlabel: str = "Index",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (10, 4),
    out_path: Optional[Union[str, Path]] = None,
):
    """
    绘制前 N 个样本的折线对比图（y_true 与 y_pred）。
    - 若 first_n 为 None 或大于长度，则绘制全部；
    - 自动屏蔽 NaN/Inf：不可用位置的 y_pred 会显示为缺失（仅在图形层面体现）。

    返回：
    - 若 out_path 为 None：返回 matplotlib Figure；
    - 否则将图片保存到 out_path，并返回 Path。
    """
    yt = _to_float_np1d(y_true)
    yp = _to_float_np1d(y_pred)

    n = yt.shape[0]
    if yp.shape[0] != n:
        raise ValueError(f"长度不一致：y_true={n}, y_pred={yp.shape[0]}")

    if first_n is None or first_n <= 0 or first_n > n:
        first_n = n

    # 为了视觉整洁，仅将不可用的 y_pred 用 NaN 替代（折线将断开）
    mask = _valid_mask(yp)
    yp_plot = yp.copy()
    yp_plot[~mask] = np.nan

    fig = plt.figure(figsize=figsize)
    plt.plot(yt[:first_n], label="True")
    plt.plot(yp_plot[:first_n], label="Pred")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (first {first_n})")
    plt.tight_layout()

    if out_path is None:
        return fig
    out_path = save_matplotlib_figure(fig, out_path)
    return out_path


# ============================================================================
# 三、Parity（散点）图
# ============================================================================

def plot_parity(
    y_true,
    y_pred,
    title: str = "Parity Plot",
    xlabel: str = "True",
    ylabel: str = "Pred",
    figsize: Tuple[int, int] = (5, 5),
    out_path: Optional[Union[str, Path]] = None,
    draw_identity: bool = True,
):
    """
    绘制 Parity 散点图（y_true 为 x，y_pred 为 y）。
    - 会自动仅使用同时有限的样本；
    - 可选绘制 y=x 的参考线。

    返回：
    - 若 out_path 为 None：返回 matplotlib Figure；
    - 否则保存并返回 Path。
    """
    yt = _to_float_np1d(y_true)
    yp = _to_float_np1d(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"长度不一致：y_true={len(yt)}, y_pred={len(yp)}")

    m = _mask_pair(yt, yp)
    yt = yt[m]; yp = yp[m]
    if yt.size == 0:
        # 空数据也输出一个空图，避免上层逻辑崩溃
        yt = np.array([0.0], dtype=np.float32)
        yp = np.array([0.0], dtype=np.float32)

    # 轴范围：按数据范围稍微留白
    x_min, x_max = float(np.min(yt)), float(np.max(yt))
    y_min, y_max = float(np.min(yp)), float(np.max(yp))
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo -= pad; hi += pad

    fig = plt.figure(figsize=figsize)
    plt.scatter(yt, yp, s=12, alpha=0.7)
    if draw_identity:
        xs = np.linspace(lo, hi, 100)
        plt.plot(xs, xs, linestyle="--", linewidth=1.0)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if out_path is None:
        return fig
    out_path = save_matplotlib_figure(fig, out_path)
    return out_path


# ============================================================================
# 四、误差直方图
# ============================================================================

def plot_error_hist(
    y_true,
    y_pred,
    bins: int = 30,
    title: str = "Error Distribution (pred - true)",
    xlabel: str = "Error",
    ylabel: str = "Count",
    figsize: Tuple[int, int] = (6, 4),
    out_path: Optional[Union[str, Path]] = None,
    show_stats: bool = True,
):
    """
    绘制误差 e = y_pred - y_true 的直方图。

    参数：
    - bins:      直方图箱数
    - show_stats:是否叠加均值与 ±1σ 的竖线标注

    返回：
    - 若 out_path 为 None：返回 matplotlib Figure；
    - 否则保存并返回 Path。
    """
    yt = _to_float_np1d(y_true)
    yp = _to_float_np1d(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"长度不一致：y_true={len(yt)}, y_pred={len(yp)}")

    m = _mask_pair(yt, yp)
    err = (yp - yt)[m]
    if err.size == 0:
        err = np.array([0.0], dtype=np.float32)

    mu = float(np.mean(err))
    sigma = float(np.std(err))

    fig = plt.figure(figsize=figsize)
    plt.hist(err, bins=bins)
    if show_stats:
        plt.axvline(mu, linestyle="--", linewidth=1.0)
        plt.axvline(mu - sigma, linestyle=":", linewidth=1.0)
        plt.axvline(mu + sigma, linestyle=":", linewidth=1.0)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if show_stats:
        plt.title(f"{title}\nmean={mu:.3f}, std={sigma:.3f}")
    else:
        plt.title(title)
    plt.tight_layout()

    if out_path is None:
        return fig
    out_path = save_matplotlib_figure(fig, out_path)
    return out_path
