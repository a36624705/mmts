# src/mmts/imaging/__init__.py
# -*- coding: utf-8 -*-
"""Imaging 模块初始化：导出基础接口并自动注册内置渲染器。"""

from __future__ import annotations

from .base import (
    ScaleSpec,
    ImageRenderer,
    register_renderer,
    get_renderer,
    create_renderer,
)

# 自动加载内置渲染器（触发 @register_renderer）
try:
    from .renderers import grayscale  # noqa: F401
    from .renderers import gasf
except Exception:
    pass
