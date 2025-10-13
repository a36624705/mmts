# src/mmts/imaging/__init__.py
# -*- coding: utf-8 -*-

# 基础导出
from .base import ScaleSpec, ImageRenderer, register_renderer, get_renderer, create_renderer

# 自动加载内置渲染器（触发 @register_renderer）
try:
    from .renderers import grayscale  # noqa: F401
except Exception:
    pass
