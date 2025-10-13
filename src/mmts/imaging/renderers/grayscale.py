# src/mmts/imaging/renderers/grayscale.py
# -*- coding: utf-8 -*-
"""
灰度矩阵渲染器（Grayscale Matrix Renderer）

功能
----
把形如 (window_size, n_features) 的 2D 数组渲染为**灰度图**（PIL.Image）。
- 归一化到 [0,1] 的逻辑统一由 ScaleSpec 处理（支持 global / percentile / sample）。
- 渲染阶段将 [0,1] 浮点矩阵转换为 uint8 [0,255] 并生成 "L" 模式灰度图。
- 提供基础的方向/翻转控制，方便与你现有脚本输出保持一致。

与原脚本差异
------------
- 原脚本用 matplotlib.imsave，这里直接返回 PIL.Image（更轻依赖、速度也更快）。
- 批量导出/保存由 CLI (mmts.cli.imgify) 承担；本类只负责“单样本 -> Image”。

典型用法
--------
from mmts.imaging.base import ScaleSpec
from mmts.imaging.renderers.grayscale import GrayRenderer

renderer = GrayRenderer(orientation="row-major")  # 或 "col-major"
scaler = ScaleSpec(mode="percentile", pmin=1, pmax=99).fit_global(X_train)  # 若用 global/percentile
img = renderer.render(X_sample, scaler=scaler)  # PIL.Image
img.save("sample_00000.png")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from ..base import ImageRenderer, ScaleSpec, to_uint8, register_renderer


@register_renderer  # 注册到全局渲染器表，名字见类属性 name
class GrayRenderer(ImageRenderer):
    """
    基于灰度的简单矩阵渲染器。

    参数
    ----
    orientation : {"row-major", "col-major"}
        - "row-major"：按原始 shape (T, F) 直接渲染，T 作为行（垂直方向），F 作为列（水平方向）。
        - "col-major"：先转置为 (F, T) 再渲染（有些场景希望时间轴在横向显示）。
    flip_vertical : bool
        是否对图像做上下翻转（等价 np.flipud，默认 False）。
    flip_horizontal : bool
        是否对图像做左右翻转（等价 np.fliplr，默认 False）。

    说明
    ----
    - 缩放：由 ScaleSpec 控制。如果使用 "global"/"percentile" 模式，务必先对训练集或选定集合调用
      `ScaleSpec.fit_global()`；如果使用 "sample"，每张图会按各自 min/max 缩放（无需预拟合）。
    - 输出：PIL.Image（mode="L"），像素范围 [0,255]。
    """

    name: str = "grayscale"

    def __init__(
        self,
        orientation: str = "row-major",
        *,
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
    ) -> None:
        orientation = orientation.strip().lower()
        if orientation not in {"row-major", "col-major"}:
            raise ValueError("orientation 必须是 'row-major' 或 'col-major'")
        self.orientation = orientation
        self.flip_vertical = bool(flip_vertical)
        self.flip_horizontal = bool(flip_horizontal)

    # 核心接口：将单个 2D 矩阵渲染为 PIL.Image
    def render(self, sample: np.ndarray, *, scaler: ScaleSpec) -> Image.Image:
        """
        渲染单样本矩阵。

        参数
        ----
        sample : np.ndarray
            形状 (T, F) 的二维数组（float / int 均可）。
        scaler : ScaleSpec
            缩放规范；负责把矩阵转换到 [0,1]（见 ScaleSpec.transform）。

        返回
        ----
        PIL.Image.Image
            模式为 "L" 的灰度图。
        """
        # 1) 归一化到 [0,1]
        x01 = scaler.transform(sample)  # (T, F) in [0,1]

        # 2) 可选的方向调整
        if self.orientation == "col-major":
            x01 = x01.T  # (F, T)

        # 3) 可选翻转（视觉取向因人而异，给到最小控制）
        if self.flip_vertical:
            x01 = np.flipud(x01)
        if self.flip_horizontal:
            x01 = np.fliplr(x01)

        # 4) 转为 uint8 并构造灰度图
        img8 = to_uint8(x01)  # (H, W) uint8
        return Image.fromarray(img8, mode="L")
