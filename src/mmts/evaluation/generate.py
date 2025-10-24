# src/mmts/evaluation/generate.py
# -*- coding: utf-8 -*-
"""
推理与生成（Inference & Text-to-JSON 解析）

- 单样本：generate_for_image
- 批处理：generate_for_dataset
- 稳健抽取与解析首个 JSON（可配合 RUL 场景的 validate_rul_object）
- 推理前可选启用 KV Cache、关闭梯度检查点（configure_for_inference）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from ..core.loader import configure_for_inference
from ..utils.json_parse import (
    extract_first_json_block,
    find_and_parse_first_json,
    validate_rul_object,
)


# =========================
# 配置与结果结构
# =========================

@dataclass
class GenConfig:
    image_size: int = 448
    max_new_tokens: int = 64
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50


@dataclass
class GenResult:
    raw_text: str
    json_text: Optional[str]
    json_obj: Optional[dict]


# =========================
# 内部：消息构造与编码
# =========================

def _build_infer_messages_single_image(img: Image.Image, rules: str) -> List[Dict[str, Any]]:
    """仅包含 user（image + rules）的推理消息。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": rules},
            ],
        }
    ]


def _encode_for_generate(
    processor: object,
    img: Image.Image,
    rules: str,
    image_size: int,
    device: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    """
    将推理消息编码为模型输入张量，并移动到指定 device。
    依赖：processor.apply_chat_template 与 processor(...) 编码接口（duck typing）。
    """
    prompt_text = processor.apply_chat_template(  # type: ignore[attr-defined]
        _build_infer_messages_single_image(img, rules),
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = processor(  # type: ignore[call-arg]
        text=prompt_text,
        images=img,
        return_tensors="pt",
        max_pixels=image_size * image_size,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _decode_new_tokens(processor: object, out_ids: torch.Tensor, input_len: int) -> str:
    """
    解码新生成部分。优先使用 processor.tokenizer.decode。
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        # 兜底：尝试 processor.decode；否则抛错提示处理器不兼容
        if hasattr(processor, "decode"):
            return processor.decode(out_ids[0][input_len:])  # type: ignore[attr-defined]
        raise AttributeError("processor 不包含 tokenizer.decode 或 decode，无法解码生成结果。")
    return tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)


# =========================
# 单样本生成
# =========================

def generate_for_image(
    model: "torch.nn.Module",
    processor: object,
    image: Union[str, Path, Image.Image],
    rules: str,
    gen: Optional[GenConfig] = None,
    setup_infer: bool = True,
) -> GenResult:
    """
    对单张图像进行生成推理并解析 JSON，返回 GenResult(raw_text/json_text/json_obj)。
    """
    if setup_infer:
        configure_for_inference(model, enable_kv_cache=True, disable_gradient_checkpointing=True)

    gen = gen or GenConfig()

    img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

    try:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
    except Exception:
        device = torch.device("cpu")

    enc = _encode_for_generate(processor, img, rules, gen.image_size, device)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=int(gen.max_new_tokens),
        do_sample=bool(gen.do_sample),
        num_beams=int(gen.num_beams),
    )
    if gen.do_sample:
        gen_kwargs.update(
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            top_k=int(gen.top_k),
        )

    with torch.inference_mode():
        out_ids = model.generate(**enc, **gen_kwargs)

    input_len = enc["input_ids"].shape[-1]
    raw = _decode_new_tokens(processor, out_ids, input_len)

    js = extract_first_json_block(raw)
    obj = find_and_parse_first_json(js) if js is not None else None

    return GenResult(raw_text=raw, json_text=js, json_obj=obj)


# =========================
# 批量（HFDataset）生成
# =========================

def generate_for_dataset(
    model: "torch.nn.Module",
    processor: object,
    dataset: object,
    rules: str,
    gen: Optional[GenConfig] = None,
    setup_infer: bool = True,
) -> Tuple[np.ndarray, List[Optional[str]], List[Optional[dict]]]:
    """
    串行遍历 HFDataset，依次生成并解析 JSON。
    返回：raw_texts(np.ndarray[object])，json_texts(List[str|None])，json_objs(List[dict|None])
    """
    if setup_infer:
        configure_for_inference(model, enable_kv_cache=True, disable_gradient_checkpointing=True)

    gen = gen or GenConfig()
    try:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
    except Exception:
        device = torch.device("cpu")

    raw_texts: List[str] = []
    json_texts: List[Optional[str]] = []
    json_objs: List[Optional[dict]] = []

    for i in tqdm(range(len(dataset)), desc="Eval-Generate", ncols=100):
        ex = dataset[i]
        img_path = ex.get("path", None)
        if img_path is None:
            raise KeyError("数据集中缺少 'path' 键，无法读取图像进行推理。")

        img = Image.open(img_path).convert("RGB")
        enc = _encode_for_generate(processor, img, rules, gen.image_size, device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=bool(gen.do_sample),
            num_beams=int(gen.num_beams),
        )
        if gen.do_sample:
            gen_kwargs.update(
                temperature=float(gen.temperature),
                top_p=float(gen.top_p),
                top_k=int(gen.top_k),
            )

        with torch.inference_mode():
            out_ids = model.generate(**enc, **gen_kwargs)

        input_len = enc["input_ids"].shape[-1]
        raw = _decode_new_tokens(processor, out_ids, input_len)

        js = extract_first_json_block(raw)
        obj = find_and_parse_first_json(js) if js is not None else None

        raw_texts.append(raw)
        json_texts.append(js)
        json_objs.append(obj)

    return np.array(raw_texts, dtype=object), json_texts, json_objs


# =========================
# RUL 便捷数值提取
# =========================

def extract_rul_values(
    json_objs: Sequence[Optional[dict]],
    low: float,
    high: float,
    reason_key: str = "reason",
) -> Tuple[np.ndarray, List[str]]:
    """
    将一组 JSON 对象解析为（数值, reason）：
    - 对每个元素调用 validate_rul_object，得到 (value_or_None, reason)
    - value 为 None → 以 NaN 占位
    """
    vals: List[float] = []
    reasons: List[str] = []
    for obj in json_objs:
        if obj is None:
            vals.append(np.nan)
            reasons.append("")
            continue
        v, r = validate_rul_object(obj, low=low, high=high, reason_key=reason_key)
        vals.append(v if v is not None else np.nan)
        reasons.append(r or "")
    return np.asarray(vals, dtype=np.float32), reasons
