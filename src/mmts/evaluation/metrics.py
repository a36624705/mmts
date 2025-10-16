# src/mmts/evaluation/metrics.py
# -*- coding: utf-8 -*-
"""回归指标：RMSE / MAE / R²（含 NaN/Inf 掩蔽与有效率统计）。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np


# -------------------------
# 结果数据结构
# -------------------------

@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    valid_ratio: float
    n_valid: int
    n_total: int

    def as_dict(self) -> dict:
        """转为字典（便于日志/存盘）。"""
        return asdict(self)


# -------------------------
# 内部工具
# -------------------------

def _to_float_np(x: Iterable) -> np.ndarray:
    """转为 1D float32 向量。"""
    arr = np.asarray(list(x), dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)


def _safe_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    additional_mask: Optional[np.ndarray],
) -> np.ndarray:
    """构造有效样本掩码：两边有限值，且满足额外掩码。"""
    base = np.isfinite(y_true) & np.isfinite(y_pred)
    if additional_mask is not None:
        if additional_mask.shape != y_true.shape:
            raise ValueError(f"additional_mask 形状不匹配：{additional_mask.shape} vs {y_true.shape}")
        base &= additional_mask.astype(bool)
    return base


# -------------------------
# 指标计算
# -------------------------

def compute_regression_metrics(
    y_true_in: Iterable,
    y_pred_in: Iterable,
    additional_mask: Optional[Iterable] = None,
) -> RegressionMetrics:
    """计算 RMSE / MAE / R² 与有效率。"""
    y_true = _to_float_np(y_true_in)
    y_pred = _to_float_np(y_pred_in)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true 与 y_pred 长度不一致：{len(y_true)} vs {len(y_pred)}")

    mask_extra = None
    if additional_mask is not None:
        mask_extra = np.asarray(list(additional_mask), dtype=bool).reshape(-1)
        if mask_extra.shape[0] != y_true.shape[0]:
            raise ValueError(f"additional_mask 长度不一致：{len(mask_extra)} vs {len(y_true)}")

    mask = _safe_mask(y_true, y_pred, mask_extra)

    n_total = int(y_true.shape[0])
    n_valid = int(mask.sum())
    valid_ratio = float(n_valid / n_total) if n_total > 0 else 0.0

    if n_valid == 0:
        return RegressionMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            r2=float("nan"),
            valid_ratio=valid_ratio,
            n_valid=n_valid,
            n_total=n_total,
        )

    yt = y_true[mask]
    yp = y_pred[mask]

    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yt - yp)))

    if n_valid < 2:
        r2 = float("nan")
    else:
        ss_res = float(np.sum((yt - yp) ** 2))
        mu = float(np.mean(yt))
        ss_tot = float(np.sum((yt - mu) ** 2))
        r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        valid_ratio=valid_ratio,
        n_valid=n_valid,
        n_total=n_total,
    )


def compute_regression_metrics_dict(
    y_true_in: Iterable,
    y_pred_in: Iterable,
    additional_mask: Optional[Iterable] = None,
) -> dict:
    """同上，返回字典。"""
    return compute_regression_metrics(y_true_in, y_pred_in, additional_mask).as_dict()
