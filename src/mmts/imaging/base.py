# src/mmts/imaging/base.py
# -*- coding: utf-8 -*-
"""
Imaging 基础设施：把数值时序窗口（2D 矩阵）渲染为图像的抽象、缩放规范与注册表。

使用方式（概览）
---------------
# 1) 在具体渲染器中继承 ImageRenderer 并实现 render()
from .base import ImageRenderer, register_renderer, ScaleSpec, to_uint8

class GrayRenderer(ImageRenderer):
    name = "grayscale"
    def render(self, sample: "np.ndarray", *, scaler: ScaleSpec) -> "PIL.Image.Image":
        # sample: (T, F) 2D
        arr = scaler.transform(sample)          # 归一化到 [0, 1]
        img8 = to_uint8(arr)                    # 转为 uint8 [0,255]
        from PIL import Image
        return Image.fromarray(img8, mode="L")  # 灰度

register_renderer(GrayRenderer)

# 2) CLI 中通过名字选择：
#    renderer = get_renderer("grayscale")()

设计要点
--------
- 可插拔：不同渲染策略 = 不同子类文件；通过注册表按名字选择，便于实验对比。
- 可复现：统一的缩放规范 ScaleSpec 支持 global / percentile / sample 三种模式。
- 轻依赖：默认只依赖 numpy / PIL；matplotlib 仅由具体 renderer 自行选择是否使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import numpy as np


# =============================================================================
# 一、缩放规范（Scaling / Normalization）
# =============================================================================

@dataclass
class ScaleSpec:
    """
    定义将原始矩阵缩放到 [0,1] 的规范（与 renderer 解耦，便于统一控制）。

    三种模式：
    - mode="global"      ：使用 fit() 得到的全局 vmin/vmax，对所有样本一致；
    - mode="percentile"  ：使用全局百分位 pmin/pmax 计算得到的 vmin/vmax；
    - mode="sample"      ：对每个样本单独 min/max（无需预先 fit）。

    约定：
    - 对于 "global" 与 "percentile"，应当先调用 fit_global(array) 得到 vmin/vmax；
    - transform(sample) 返回浮点数组，范围在 [0,1]（若 vmax<=vmin，退化为全零）。
    """
    mode: str = "percentile"            # "global" | "percentile" | "sample"
    vmin: Optional[float] = None        # 仅 global/percentile 模式使用
    vmax: Optional[float] = None
    pmin: float = 1.0                   # 仅 percentile 模式使用
    pmax: float = 99.0

    def fit_global(self, array: np.ndarray) -> "ScaleSpec":
        """
        基于一个大的样本集合（例如 X_train 的所有窗口合在一起的 2D/3D 数组），
        估计全局的 (vmin, vmax)。percentile 模式下用 pmin/pmax 计算。
        - 支持输入维度：(N, T, F) 或 (T, F) 或已经拉平的任意形状。
        - 返回 self（便于链式调用）。
        """
        flat = np.asarray(array, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            # 空数组兜底：给一个安全区间
            self.vmin, self.vmax = 0.0, 1.0
            return self

        if self.mode == "global":
            self.vmin = float(np.nanmin(flat))
            self.vmax = float(np.nanmax(flat))
        elif self.mode == "percentile":
            # 注意：np.percentile 忽略 NaN 需手动处理；这里简单剔除 NaN
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                self.vmin, self.vmax = 0.0, 1.0
            else:
                self.vmin = float(np.percentile(flat, self.pmin))
                self.vmax = float(np.percentile(flat, self.pmax))
        elif self.mode == "sample":
            # sample 模式不需要全局拟合，但为保持接口一致，给出默认区间
            self.vmin, self.vmax = None, None
        else:
            raise ValueError(f"Unknown scale mode: {self.mode}")
        return self

    def transform(self, sample: np.ndarray) -> np.ndarray:
        """
        将单个样本（2D 矩阵，如形状 (window_size, n_features)）归一化到 [0,1]。
        - 对于 sample 模式：使用该样本的 min/max。
        - 对于 global/percentile 模式：使用事先 fit_global() 得到的 vmin/vmax。
        """
        x = np.asarray(sample, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expect 2D sample (T,F), got shape {x.shape}")

        if self.mode == "sample":
            vmin = float(np.nanmin(x))
            vmax = float(np.nanmax(x))
        else:
            if self.vmin is None or self.vmax is None:
                raise RuntimeError("ScaleSpec not fitted: call fit_global() first for global/percentile mode.")
            vmin, vmax = self.vmin, self.vmax

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            # 异常或退化区间：返回全零，避免 NaN 扩散
            return np.zeros_like(x, dtype=np.float32)

        y = (x - vmin) / (vmax - vmin)
        # clip 到 [0,1]，应对极端值
        return np.clip(y, 0.0, 1.0)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    """
    将 [0,1] 浮点矩阵转换为 uint8 [0,255]。
    - 输入会被 clip 到 [0,1]。
    - 返回值 shape 与输入相同，dtype=uint8。
    """
    y = np.clip(x01, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype("uint8")


# =============================================================================
# 二、渲染器抽象与注册表
# =============================================================================

class ImageRenderer:
    """
    抽象基类：将一个 2D 时序窗口 sample -> PIL.Image。
    - 子类需实现：render(sample, scaler=ScaleSpec) -> PIL.Image.Image
    - name: 用于注册/选择的唯一字符串（例如 "grayscale"、"heatmap"）
    """
    #: 注册名（子类覆盖）
    name: str = "base"

    def render(self, sample: np.ndarray, *, scaler: ScaleSpec):
        """
        子类实现：把 sample 渲染为 PIL.Image。
        约定：
        - sample: 形状 (T, F) 的 2D numpy 数组，float/uint8 皆可（内部自行处理）。
        - scaler: 统一的缩放规范（通常先 transform 到 [0,1]，再转 uint8）。
        """
        raise NotImplementedError


# 注册表：字符串 name -> 渲染器类
_RENDERERS: Dict[str, Type[ImageRenderer]] = {}


def register_renderer(cls: Type[ImageRenderer]) -> None:
    """
    将渲染器类注册到全局注册表中。
    用法：
        @register_renderer
        class MyRenderer(ImageRenderer):
            name = "my_renderer"
            ...
    或：
        register_renderer(MyRenderer)
    """
    name = getattr(cls, "name", None)
    if not name or not isinstance(name, str):
        raise ValueError("Renderer class must define a string attribute `name`.")
    key = name.strip().lower()
    if key in _TRENDING_WARNINGS:
        # 保留位：目前不做特殊处理
        pass
    if key in _RENDERERS:
        raise ValueError(f"Renderer name already registered: {key}")
    _RENDERERS[key] = cls


def get_renderer(name: str) -> Type[ImageRenderer]:
    """
    根据名字获取渲染器类（未实例化）。大小写不敏感。
    """
    key = name.strip().lower()
    if key not in _RENDERERS:
        available = ", ".join(sorted(_RENDERERS.keys())) or "(empty)"
        raise KeyError(f"Unknown renderer '{name}'. Available: {available}")
    return _RENDERERS[key]


def create_renderer(name: str, **kwargs) -> ImageRenderer:
    """
    便捷工厂：根据名字实例化渲染器（传入 kwargs 作为 __init__ 参数）。
    """
    cls = get_renderer(name)
    return cls(**kwargs)


# 预留：可在未来用于发出某些策略的警告/弃用提示
_TRENDING_WARNINGS: Dict[str, str] = {}
