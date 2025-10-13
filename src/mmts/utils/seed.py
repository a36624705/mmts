# src/mmts/utils/seed.py
# -*- coding: utf-8 -*-
"""
提供与随机性和数值加速相关的工具函数：
1) 统一设置 Python / NumPy / PyTorch 的随机种子，保证实验可复现；
2) 配置 PyTorch 的 TF32 选项，在兼容硬件上加速矩阵运算；
3) 配置 cuDNN 的确定性/benchmark 行为；
4) 简单的设备探测工具。

使用建议：
- 在任何训练/评估脚本的最开头调用 set_global_seed() 和 set_torch_backends()。
- 不要在训练中途多次修改这些全局开关，避免产生不可预期的行为。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # 允许在无 torch 环境下导入该模块（例如仅解析配置时）


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    统一设置随机种子，尽量保证可复现。

    参数说明：
    - seed: 随机种子整数（建议使用固定常量，例如 42/3407 等）。
    - deterministic: 是否启用严格确定性模式（True 时一些算子会退化为确定性实现，
                     可能降低性能；False 时追求速度，允许微小非确定性）。

    作用范围：
    - Python 内置 random 模块
    - NumPy
    - PyTorch（CPU/GPU），若当前环境安装了 torch

    注意：
    - 完全意义上的可复现依赖于：
        * 固定数据加载的随机性（例如 DataLoader 的 worker_init_fn、shuffle 等）；
        * 固定算子/库版本、驱动与硬件；
        * 禁用某些非确定性算子或为其选择确定性替代实现。
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 时确保每张卡也被设置
        # cudnn 的确定性与 benchmark 逻辑通常放在 set_torch_backends() 中控制
        if deterministic:
            # 严格确定性：牺牲速度换取可复现
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            # 允许使用非确定性算法，追求更高吞吐
            torch.use_deterministic_algorithms(False)


def set_torch_backends(
    allow_tf32: bool = True,
    cudnn_benchmark: Optional[bool] = None,
    cudnn_deterministic: Optional[bool] = None,
) -> None:
    """
    配置 PyTorch 的底层加速与 cuDNN 行为。

    参数说明：
    - allow_tf32: 是否在支持的硬件上启用 TF32（Ada/Ampere 架构等）。
                  TF32 对 FP32 GEMM/卷积有明显加速，通常对训练/推理精度影响极小。
    - cudnn_benchmark: 是否开启 cuDNN benchmark。
                       * 当输入尺寸固定时，设为 True 可加速；尺寸变化多时不建议开启。
                       * 若为 None，则根据环境变量或不做修改（保持 PyTorch 默认值）。
    - cudnn_deterministic: 是否开启 cuDNN 的确定性模式。
                           * True 可提升可复现性，但会禁用某些高性能算法；
                           * 若为 None，则根据环境变量或不做修改（保持默认值）。

    提示：
    - 典型搭配：
        * 追求速度（默认）：allow_tf32=True, cudnn_benchmark=True, cudnn_deterministic=False
        * 追求复现：allow_tf32 可 True/False，cudnn_benchmark=False, cudnn_deterministic=True
    """
    if torch is None:
        return

    # 1) TF32：对 matmul 与 cudnn 中的 conv/gemm 生效
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        # 在无 CUDA 或老版本 torch 环境下静默跳过
        pass

    # 2) cuDNN 行为配置（仅在用户明确传入时修改）
    if cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if cudnn_deterministic is not None:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)


def get_device(prefer: str = "cuda") -> str:
    """
    简单的设备字符串探测与选择。

    参数：
    - prefer: 首选设备类型字符串，常见取值：
        * "cuda"：若可用则返回 "cuda"；否则回退到 "cpu"
        * "mps"：适配 Apple Metal（PyTorch MPS），若可用则返回 "mps"，否则回退 "cpu"
        * "cpu"：强制使用 CPU

    返回：
    - "cuda" / "mps" / "cpu" 三者之一

    说明：
    - 对于多卡环境，更细粒度的选择（如 "cuda:0"）建议由外层脚本通过
      CUDA_VISIBLE_DEVICES 或显式 device map 处理。
    """
    p = prefer.lower()
    if p == "cpu":
        return "cpu"

    if p == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if p == "mps":
        # PyTorch 对 MPS 的支持取决于版本与系统
        if torch is not None and getattr(torch.backends, "mps", None):
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    # 未知输入时兜底为 CPU
    return "cpu"


def set_env_seed_from_var(env_key: str = "GLOBAL_SEED", default: int = 42) -> int:
    """
    从环境变量读取种子并应用，便于通过 shell/Makefile 控制实验种子。

    参数：
    - env_key: 环境变量名（例如 "GLOBAL_SEED"）
    - default: 若环境变量不存在或非法时使用的默认种子

    返回：
    - 实际最终应用的整数种子值
    """
    raw = os.environ.get(env_key, "")
    try:
        seed = int(raw)
    except Exception:
        seed = int(default)

    # 采用较宽松的确定性设定：避免过度牺牲性能
    set_global_seed(seed=seed, deterministic=False)
    return seed
