# src/mmts/imaging/base.py
# -*- coding: utf-8 -*-
"""将数值时序窗口（2D 矩阵）渲染为图像的抽象、缩放规范与注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import numpy as np
import inspect


# -------------------------
# 缩放规范
# -------------------------

@dataclass
class ScaleSpec:
    """
    将原始矩阵缩放到 [0,1] 的规范：
    - mode: "global" | "percentile" | "sample" | "none"
    - global/percentile 需先 fit_global() 得到 (vmin, vmax)
    - none 模式表示不进行缩放，直接使用原始值
    """
    mode: str = "percentile"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    pmin: float = 1.0
    pmax: float = 99.0

    def fit_global(self, array: np.ndarray) -> "ScaleSpec":
        """基于整体样本估计 (vmin, vmax)；percentile 模式用 pmin/pmax。"""
        flat = np.asarray(array, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            self.vmin, self.vmax = 0.0, 1.0
            return self

        if self.mode == "global":
            self.vmin = float(np.nanmin(flat))
            self.vmax = float(np.nanmax(flat))
        elif self.mode == "percentile":
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                self.vmin, self.vmax = 0.0, 1.0
            else:
                self.vmin = float(np.percentile(flat, self.pmin))
                self.vmax = float(np.percentile(flat, self.pmax))
        elif self.mode == "sample":
            self.vmin, self.vmax = None, None
        elif self.mode == "none":
            self.vmin, self.vmax = None, None
        else:
            raise ValueError(f"Unknown scale mode: {self.mode}")
        return self

    def transform(self, sample: np.ndarray) -> np.ndarray:
        """把单个样本 (T,F) 归一化到 [0,1]。"""
        x = np.asarray(sample, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expect 2D sample (T,F), got shape {x.shape}")

        if self.mode == "sample":
            vmin = float(np.nanmin(x))
            vmax = float(np.nanmax(x))
        elif self.mode == "none":
            # none 模式：不进行缩放，直接返回原始值（假设已经在 [0,1] 范围内）
            return np.clip(x, 0.0, 1.0)
        else:
            if self.vmin is None or self.vmax is None:
                raise RuntimeError("ScaleSpec not fitted for global/percentile mode.")
            vmin, vmax = self.vmin, self.vmax

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros_like(x, dtype=np.float32)

        y = (x - vmin) / (vmax - vmin)
        return np.clip(y, 0.0, 1.0)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    """将 [0,1] 浮点矩阵转换为 uint8 [0,255]。"""
    y = np.clip(x01, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype("uint8")


# -------------------------
# 渲染器抽象与注册表
# -------------------------

class ImageRenderer:
    """抽象基类：sample(2D) -> PIL.Image。子类需实现 render()."""
    name: str = "base"

    def render(self, sample: np.ndarray, *, scaler: ScaleSpec):
        """子类实现：返回 PIL.Image.Image。"""
        raise NotImplementedError


_RENDERERS: Dict[str, Type[ImageRenderer]] = {}


def register_renderer(cls: Type[ImageRenderer]) -> None:
    """注册渲染器类到全局表。"""
    name = getattr(cls, "name", None)
    if not name or not isinstance(name, str):
        raise ValueError("Renderer class must define a string attribute `name`.")
    key = name.strip().lower()
    if key in _RENDERERS:
        raise ValueError(f"Renderer name already registered: {key}")
    _RENDERERS[key] = cls


def get_renderer(name: str) -> Type[ImageRenderer]:
    """按名称获取渲染器类（未实例化）。"""
    key = name.strip().lower()
    if key not in _RENDERERS:
        available = ", ".join(sorted(_RENDERERS.keys())) or "(empty)"
        raise KeyError(f"Unknown renderer '{name}'. Available: {available}")
    return _RENDERERS[key]


def create_renderer(name: str, **kwargs) -> ImageRenderer:
    """便捷工厂：按名称实例化渲染器（自动过滤掉构造函数不接受的参数）。"""
    cls = get_renderer(name)
    sig = inspect.signature(cls.__init__)
    accept = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in accept}
    return cls(**filtered)


# 预留：可用于策略警告/弃用提示
_TRENDING_WARNINGS: Dict[str, str] = {}
