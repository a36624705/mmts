# src/mmts/imaging/renderers/grayscale.py
# -*- coding: utf-8 -*-
"""将 (T, F) 矩阵渲染为灰度图（PIL.Image），缩放由 ScaleSpec 统一处理。"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from ..base import ImageRenderer, ScaleSpec, to_uint8, register_renderer


@register_renderer
class GrayRenderer(ImageRenderer):
    """
    简单灰度渲染器。
    - orientation: "row-major" 直接渲染 (T,F)；"col-major" 渲染转置 (F,T)
    - flip_vertical / flip_horizontal: 可选翻转
    """
    name: str = "grayscale"

    def __init__(
        self,
        orientation: str = "row-major",
        *,
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
    ) -> None:
        o = orientation.strip().lower()
        if o not in {"row-major", "col-major"}:
            raise ValueError("orientation 必须为 'row-major' 或 'col-major'")
        self.orientation = o
        self.flip_vertical = bool(flip_vertical)
        self.flip_horizontal = bool(flip_horizontal)

    def render(self, sample: np.ndarray, *, scaler: ScaleSpec) -> Image.Image:
        """渲染单个样本为灰度图（mode='L'）。"""
        x01 = scaler.transform(sample)  # (T, F) in [0,1]
        if self.orientation == "col-major":
            x01 = x01.T
        if self.flip_vertical:
            x01 = np.flipud(x01)
        if self.flip_horizontal:
            x01 = np.fliplr(x01)
        return Image.fromarray(to_uint8(x01), mode="L")
