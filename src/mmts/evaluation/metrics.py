# src/mmts/evaluation/metrics.py
# -*- coding: utf-8 -*-
"""
回归指标计算（Evaluation Metrics for Regression）

功能与设计
--------
- 面向时序/图像回归等场景，提供常见的 RMSE / MAE / R² 指标；
- 对预测/标签中的 NaN、±Inf 自动做掩蔽（mask），并计算“有效率”（被纳入指标计算的样本占比）；
- 允许传入可选的布尔掩码 additional_mask，用于进一步筛选样本（例如只评估某区间）；
- 返回结构化结果（数据类 + 字典），便于打印、记录与序列化。

注意
----
- 当有效样本数 < 2 时，R² 不定义（返回 NaN）；
- 当没有有效样本时，所有指标返回 NaN，有效率为 0；
- 输入可以是 list/tuple/np.ndarray，内部统一转换为 float32 的 numpy 数组。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


# ============================================================================
# 一、数据类：回归指标
# ============================================================================

@dataclass
class RegressionMetrics:
    """
    回归指标结果数据类。

    字段：
    - rmse:  Root Mean Squared Error
    - mae:   Mean Absolute Error
    - r2:    R-squared（决定系数）
    - valid_ratio: 被纳入计算的有效样本占比（0~1）
    - n_valid:     有效样本个数（整数）
    - n_total:     总样本个数（整数）
    """
    rmse: float
    mae: float
    r2: float
    valid_ratio: float
    n_valid: int
    n_total: int

    def as_dict(self) -> Dict[str, float]:
        """将结果转为字典，适合日志与保存 JSON。"""
        return asdict(self)  # type: ignore[return-value]


# ============================================================================
# 二、内部工具：统一清洗输入
# ============================================================================

def _to_float_np(x: Iterable) -> np.ndarray:
    """
    将输入转换为 float32 的 numpy 向量。
    - 支持 list/tuple/ndarray；
    - 不尝试从字符串解析数字（上游应已处理）。
    """
    arr = np.asarray(list(x), dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1).astype(np.float32)
    return arr


def _safe_mask(y_true: np.ndarray,
               y_pred: np.ndarray,
               additional_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    生成“有效样本”掩码：
    - y_true/y_pred 都是有限值（非 NaN 且非 ±Inf）；
    - 若 additional_mask 非空，则与其做逻辑与（&）。
    """
    base = np.isfinite(y_true) & np.isfinite(y_pred)
    if additional_mask is not None:
        if additional_mask.shape != y_true.shape:
            raise ValueError(f"additional_mask 形状不匹配：{additional_mask.shape} vs {y_true.shape}")
        base &= additional_mask.astype(bool)
    return base


# ============================================================================
# 三、指标计算主函数
# ============================================================================

def compute_regression_metrics(
    y_true_in: Iterable,
    y_pred_in: Iterable,
    additional_mask: Optional[Iterable] = None,
) -> RegressionMetrics:
    """
    计算 RMSE / MAE / R² 与有效率。

    参数：
    - y_true_in:         真实值序列（可为 list/tuple/ndarray）
    - y_pred_in:         预测值序列（与 y_true 等长）
    - additional_mask:   可选的布尔掩码（True 表示保留该样本），长度需与 y_true 相同

    返回：
    - RegressionMetrics 数据类
    """
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
        # 无有效样本——返回 NaN 指标
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

    # RMSE
    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))

    # MAE
    mae = float(np.mean(np.abs(yt - yp)))

    # R²（决定系数）
    # 若有效样本数 < 2，R² 不定义，返回 NaN
    if n_valid < 2:
        r2 = float("nan")
    else:
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
        if ss_tot == 0.0:
            # 所有 y_true 相同 → R² 退化为 1.0（若预测也完全一致），否则为 -inf；
            # 为稳健与便于阅读，这里将 ss_tot==0 的情况设为 NaN。
            r2 = float("nan")
        else:
            r2 = float(1.0 - ss_res / ss_tot)

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        valid_ratio=valid_ratio,
        n_valid=n_valid,
        n_total=n_total,
    )


# ============================================================================
# 四、便捷包装：同时返回字典
# ============================================================================

def compute_regression_metrics_dict(
    y_true_in: Iterable,
    y_pred_in: Iterable,
    additional_mask: Optional[Iterable] = None,
) -> Dict[str, float]:
    """
    与 compute_regression_metrics 相同，但直接返回字典格式，适合日志与 JSON 存盘。
    """
    return compute_regression_metrics(y_true_in, y_pred_in, additional_mask).as_dict()
