# src/mmts/models/loader.py
# -*- coding: utf-8 -*-
"""
模型与处理器加载器（Model & Processor Loader）

职责：
- 统一加载 HuggingFace 的 AutoProcessor 与 多模态基础模型（VL 模型）；
- 在不同环境（CPU / CUDA）下，自动选择合适的 torch.dtype 与 device_map；
- 提供推理阶段常用的开关（关闭梯度检查点、打开 KV Cache）；
- 与上层 LoRA 装配解耦：本文件只返回“可被 LoRA 注入的基础模型”。

设计取舍：
- 不引入训练细节，只处理“加载与简单配置”；
- 对不同 transformers 版本做轻量兼容（AutoModelForImageTextToText / AutoModelForVision2Seq）；
- 参数保持中性，具体数值（如 model_id）由外层配置或 CLI 传入。
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # 允许仅做配置解析时缺少 torch

from transformers import AutoProcessor

# 兼容不同版本 transformers 中视觉-语言模型的类名
try:  # 新一些的命名
    from transformers import AutoModelForImageTextToText as AutoVLModel
except Exception:  # 老一些的命名
    from transformers import AutoModelForVision2Seq as AutoVLModel


# ============================================================================
# 公有 API
# ============================================================================

def load_model_and_processor(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional["torch.dtype"] = None,
    trust_remote_code: bool = True,
    use_fast_tokenizer: bool = False,
    device_map: Optional[str] = None,
) -> Tuple["torch.nn.Module", "AutoProcessor"]:
    """
    加载多模态基础模型与处理器。

    参数：
    - model_id: HuggingFace Hub 模型标识，例如 "Qwen/Qwen2.5-VL-3B-Instruct"
    - device: 设备字符串
        * None：自动选择（若 CUDA 可用 → "cuda"，否则 "cpu"）
        * "cuda" / "cpu" / "mps"
    - dtype: torch.dtype
        * None：自动选择（CUDA 上默认 bfloat16，否则 float32）
        * 也可显式传入 torch.float16 / torch.bfloat16 / torch.float32 等
    - trust_remote_code: 是否信任远程代码（对于带自定义逻辑的仓库通常需要 True）
    - use_fast_tokenizer: 是否使用 fast tokenizer（部分模型在 fast 模式下可能存在兼容性差异）
    - device_map: 设备映射
        * None：自动选择（CUDA 上为 "auto"，CPU/MPS 为 None）
        * 也可显式指定 "auto" 或字典映射（本函数不校验字典细节）

    返回：
    - (model, processor) 二元组

    说明：
    - 本函数**不做**任何 LoRA 注入；如需 LoRA，请在上层调用 attach_lora(model, cfg)。
    - 对于推理优化（KV Cache）与训练优化（grad checkpoint），请使用下方辅助函数进行开关设置。
    """
    if torch is None:
        raise ImportError("需要安装 PyTorch 才能加载模型。")

    # 1) 设备与 dtype 的自动推断
    device = _auto_pick_device(device)
    dtype = _auto_pick_dtype(dtype, device)

    # 2) 处理器加载：不同模型的 AutoProcessor 负责文本模板/图像预处理等
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
    )

    # 3) 模型加载：尽可能利用 bfloat16 + device_map="auto" 来节省显存与加速
    #    - CPU/MPS 环境下不设置 device_map，交由上层 to(device) 或 PEFT 来处理
    actual_device_map = device_map
    if actual_device_map is None:
        actual_device_map = "auto" if device == "cuda" else None

    model = AutoVLModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=actual_device_map,
        trust_remote_code=trust_remote_code,
    )

    # 4) 如果没有使用 device_map（例如在 CPU/MPS），确保模型放到指定设备
    #    CUDA + "auto" 会分片到 GPU 上；这里不主动覆盖
    if actual_device_map is None and hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            # 某些远程实现可能不支持标准 .to() 行为，静默跳过
            pass

    return model, processor


def configure_for_inference(
    model: "torch.nn.Module",
    enable_kv_cache: bool = True,
    disable_gradient_checkpointing: bool = True,
) -> None:
    """
    推理阶段的便捷配置：
    - enable_kv_cache=True：打开 KV Cache，提升生成速度；
    - disable_gradient_checkpointing=True：关闭梯度检查点（防止生成阶段不必要的开销）。

    注意：
    - 这些开关的可用性取决于具体模型实现（不同 repo 可能字段名略有差异）。
    - 函数为“尽力而为”，遇到不存在的属性会安全跳过。
    """
    # 1) 关闭梯度检查点（若模型支持）
    if disable_gradient_checkpointing:
        try:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
        except Exception:
            pass

    # 2) 开启 KV Cache（尽力设置多个可能位置）
    if enable_kv_cache:
        try:
            if hasattr(model, "config"):
                setattr(model.config, "use_cache", True)
        except Exception:
            pass
        try:
            if hasattr(model, "generation_config"):
                setattr(model.generation_config, "use_cache", True)
        except Exception:
            pass


# ============================================================================
# 内部工具
# ============================================================================

def _auto_pick_device(device: Optional[str]) -> str:
    """
    根据可用性自动选择设备：
    - 显式传入则直接返回；
    - 未传入：CUDA 可用 → "cuda"，否则 "cpu"；
    - 简化处理：如需 "cuda:0" 之类的细粒度选择，建议用 CUDA_VISIBLE_DEVICES 控制。
    """
    if device:
        return device

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    # 简单兜底，不做 MPS 的特殊处理（如需 MPS 请显式传入）
    return "cpu"


def _auto_pick_dtype(dtype: Optional["torch.dtype"], device: str) -> "torch.dtype":
    """
    根据设备自动选择默认 dtype：
    - CUDA：默认 bfloat16（更稳、更兼容 AMPere/Ada）；
    - 非 CUDA：默认 float32；
    - 显式传入 dtype 时直接返回。
    """
    if dtype is not None:
        return dtype

    if device == "cuda":
        # bfloat16 在 A100/A800/3090/4090 等上普遍支持，稳定性优于 float16
        return torch.bfloat16
    return torch.float32
