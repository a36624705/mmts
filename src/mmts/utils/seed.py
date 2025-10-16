# src/mmts/utils/seed.py
# -*- coding: utf-8 -*-
"""
随机性与后端配置工具：
- set_global_seed：统一设置 Python / NumPy / PyTorch 种子
- set_torch_backends：配置 TF32、cuDNN 行为
- get_device：返回首选可用设备字符串
- set_env_seed_from_var：从环境变量读取并应用种子
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # 允许在无 torch 环境下导入


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """设置随机种子；可选严格确定性（可能降低性能）。"""
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def set_torch_backends(
    allow_tf32: bool = True,
    cudnn_benchmark: Optional[bool] = None,
    cudnn_deterministic: Optional[bool] = None,
) -> None:
    """配置 TF32 与 cuDNN 选项；仅在传入参数时修改对应项。"""
    if torch is None:
        return

    # TF32（无 CUDA 或老版本时静默跳过）
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass

    if cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if cudnn_deterministic is not None:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)


def get_device(prefer: str = "cuda") -> str:
    """返回 'cuda' / 'mps' / 'cpu' 中可用的设备（按 prefer 优先）。"""
    p = (prefer or "cpu").lower()
    if p == "cpu":
        return "cpu"

    if p == "cuda":
        return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    if p == "mps":
        has_mps = (
            torch is not None
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        return "mps" if has_mps else "cpu"

    return "cpu"


def set_env_seed_from_var(env_key: str = "GLOBAL_SEED", default: int = 42) -> int:
    """从环境变量读取种子并应用；返回最终使用的种子。"""
    raw = os.environ.get(env_key, "")
    try:
        seed = int(raw)
    except Exception:
        seed = int(default)

    set_global_seed(seed=seed, deterministic=False)
    return seed
