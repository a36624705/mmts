# src/mmts/evaluation/plotting.py
# -*- coding: utf-8 -*-
"""回归结果可视化：序列对比 / Parity 散点 / 误差直方图。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from ..utils.io import save_matplotlib_figure


# -------------------------
# 内部工具
# -------------------------

def _to_float_np1d(x) -> np.ndarray:
    """转为 1D float32 向量。"""
    return np.asarray(list(x), dtype=np.float32).reshape(-1)


def _valid_mask(a: np.ndarray) -> np.ndarray:
    """有限值（非 NaN/Inf）掩码。"""
    return np.isfinite(a)


def _mask_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a、b 同时有限的掩码。"""
    return _valid_mask(a) & _valid_mask(b)


# -------------------------
# 序列对比图
# -------------------------

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
    绘制前 N 个样本的折线对比图。
    - NaN/Inf 的预测值在图上显示为“断线”。
    - out_path 为 None 返回 Figure；否则保存并返回 Path。
    """
    yt = _to_float_np1d(y_true)
    yp = _to_float_np1d(y_pred)
    n = yt.shape[0]
    if yp.shape[0] != n:
        raise ValueError(f"长度不一致：y_true={n}, y_pred={yp.shape[0]}")

    if first_n is None or first_n <= 0 or first_n > n:
        first_n = n

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
    return save_matplotlib_figure(fig, out_path)


# -------------------------
# Parity 散点图
# -------------------------

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
    绘制 Parity（y_true 作 x，y_pred 作 y）。
    - 仅使用同时为有限值的样本。
    - 可选绘制 y=x 参考线。
    """
    yt = _to_float_np1d(y_true)
    yp = _to_float_np1d(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"长度不一致：y_true={len(yt)}, y_pred={len(yp)}")

    m = _mask_pair(yt, yp)
    yt = yt[m]
    yp = yp[m]
    if yt.size == 0:
        yt = np.array([0.0], dtype=np.float32)
        yp = np.array([0.0], dtype=np.float32)

    x_min, x_max = float(np.min(yt)), float(np.max(yt))
    y_min, y_max = float(np.min(yp)), float(np.max(yp))
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo -= pad
    hi += pad

    fig = plt.figure(figsize=figsize)
    plt.scatter(yt, yp, s=12, alpha=0.7)
    if draw_identity:
        xs = np.linspace(lo, hi, 100)
        plt.plot(xs, xs, linestyle="--", linewidth=1.0)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if out_path is None:
        return fig
    return save_matplotlib_figure(fig, out_path)


# -------------------------
# 误差直方图
# -------------------------

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
    - show_stats=True 会标注均值与 ±1σ。
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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nmean={mu:.3f}, std={sigma:.3f}" if show_stats else title)
    plt.tight_layout()

    if out_path is None:
        return fig
    return save_matplotlib_figure(fig, out_path)
