# src/mmts/models/loader.py
# -*- coding: utf-8 -*-
"""模型与处理器加载：统一加载 AutoProcessor/模型，自动选择 device/dtype，并提供推理配置。"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

# 仅供类型检查使用（避免 Pylance 报 torch 为 Any）
if TYPE_CHECKING:  # noqa: SIM108
    import torch as T

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # 运行期兜底

from transformers import AutoProcessor

# 兼容不同 transformers 版本
try:
    from transformers import AutoModelForImageTextToText as AutoVLModel
except Exception:
    from transformers import AutoModelForVision2Seq as AutoVLModel


def load_model_and_processor(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional["T.dtype"] = None,   # 仅在类型检查期解析
    trust_remote_code: bool = True,
    use_fast_tokenizer: bool = False,
    device_map: Optional[str] = None,
) -> Tuple[object, object]:
    """加载基础模型与处理器。"""
    if torch is None:
        raise ImportError("需要安装 PyTorch 才能加载模型。")

    device = _auto_pick_device(device)
    dtype = _auto_pick_dtype(dtype, device)

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
    )

    actual_device_map = device_map or ("auto" if device == "cuda" else None)
    model = AutoVLModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=actual_device_map,
        trust_remote_code=trust_remote_code,
    )

    if actual_device_map is None and hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            pass

    return model, processor


def configure_for_inference(
    model: object,
    enable_kv_cache: bool = True,
    disable_gradient_checkpointing: bool = True,
) -> None:
    """推理期便捷配置（KV Cache / 关闭梯度检查点）。"""
    if disable_gradient_checkpointing:
        try:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()  # type: ignore[attr-defined]
        except Exception:
            pass

    if enable_kv_cache:
        for attr in ("config", "generation_config"):
            try:
                if hasattr(model, attr):
                    setattr(getattr(model, attr), "use_cache", True)
            except Exception:
                pass


# ---- 内部工具 ----

def _auto_pick_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _auto_pick_dtype(dtype: Optional["T.dtype"], device: str):  # -> "T.dtype"
    if dtype is not None:
        return dtype
    if torch is None:
        return None
    return torch.bfloat16 if device == "cuda" else torch.float32
