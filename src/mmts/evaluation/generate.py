# src/mmts/evaluation/generate.py
# -*- coding: utf-8 -*-
"""
推理与生成（Inference & Text-to-JSON 解析）

目标
----
1) 提供对单样本（单张图像）的便捷推理接口 `generate_for_image`；
2) 提供对数据集（HFDataset）的批量推理接口 `generate_for_dataset`；
3) 稳健抽取、解析首个 JSON 对象，并可选进行数值字段裁剪（RUL 等任务复用）；
4) 推理前尽力配置模型为“生成友好”（开启 KV Cache、关闭 grad ckpt）。

与其他模块的关系
---------------
- `mmts.models.loader.configure_for_inference`：设置 KV Cache、关闭梯度检查点；
- `mmts.utils.json_parse`：`extract_first_json_block / find_and_parse_first_json / validate_rul_object`；
- 本模块不关心可视化与指标，图与分数由 `plotting.py`、`metrics.py` 处理。

注意
----
- 这里默认**单图像输入**；如需扩展“多图像/跨模态”，可在 `_build_infer_messages_single_image` 处拓展；
- 我们提供通用的 JSON 解析方法，同时提供 RUL 场景的薄封装（validate_rul_object）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from transformers import AutoProcessor

from ..models.loader import configure_for_inference
from ..utils.json_parse import (
    extract_first_json_block,
    find_and_parse_first_json,
    validate_rul_object,
)


# ============================================================================
# 一、数据结构：生成配置与结果
# ============================================================================

@dataclass
class GenConfig:
    """
    生成（generate）时的关键超参数。

    字段：
    - image_size:      图像编码像素上限（max_pixels ≈ image_size^2）
    - max_new_tokens:  生成的最大新 token 数
    - do_sample:       是否随机采样（False=贪心/束搜索）
    - num_beams:       束搜索条数（do_sample=False 且 num_beams>1 时生效）
    - temperature:     采样温度（do_sample=True 时有意义）
    - top_p:           nucleus 采样阈值
    - top_k:           top-k 采样阈值
    """
    image_size: int = 448
    max_new_tokens: int = 64
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50


@dataclass
class GenResult:
    """
    单样本的生成结果。
    - raw_text:   从模型 decode 出的**完整**文本（可能含前后噪声）
    - json_text:  稳健抽取出的首个 JSON 子串（若失败为 None）
    - json_obj:   解析后的 dict（若失败为 None）
    """
    raw_text: str
    json_text: Optional[str]
    json_obj: Optional[dict]


# ============================================================================
# 二、内部：消息构造与编码
# ============================================================================

def _build_infer_messages_single_image(img: Image.Image, rules: str) -> List[Dict[str, Any]]:
    """
    构造“推理阶段”的 user 消息：
      - 仅包含 user 角色：[image + rules]；
      - 不含 assistant（因为要生成）。

    注意：
    - 训练阶段的消息形态与此不同（训练时会附带 assistant 目标文本用于监督）。
    """
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
    processor: AutoProcessor,
    img: Image.Image,
    rules: str,
    image_size: int,
    device: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    """
    将“推理消息 + 图像”编码为模型输入张量字典，并放到指定 device 上。

    步骤：
    1) 使用 apply_chat_template(add_generation_prompt=True) 生成“待续写”的文本；
    2) 将文本与图像一并编码（VL 处理器会产生 input_ids/pixel_values/image_grid_thw 等）；
    3) 把张量移动到模型所在设备；
    """
    prompt_text = processor.apply_chat_template(
        _build_infer_messages_single_image(img, rules),
        add_generation_prompt=True,
        tokenize=False,
    )

    enc = processor(
        text=prompt_text,
        images=img,
        return_tensors="pt",
        max_pixels=image_size * image_size,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc


# ============================================================================
# 三、单样本生成
# ============================================================================

def generate_for_image(
    model: "torch.nn.Module",
    processor: AutoProcessor,
    image: Union[str, Path, Image.Image],
    rules: str,
    gen: Optional[GenConfig] = None,
    setup_infer: bool = True,
) -> GenResult:
    """
    对单个样本（单张图像）进行生成推理，并解析 JSON。

    参数：
    - model:    已加载的 VL 模型（可已注入 LoRA）
    - processor:与模型配套的处理器
    - image:    图像路径或 PIL.Image
    - rules:    任务规则文本（会拼接到 user 消息中）
    - gen:      生成超参数（若 None，使用 GenConfig() 默认）
    - setup_infer: 调用前是否自动配置模型以适配生成（KV Cache/关闭 ckpt）

    返回：
    - GenResult(raw_text, json_text, json_obj)

    说明：
    - 生成时默认关闭采样（do_sample=False）以获得稳定输出；
      如需更具多样性的解释语句，可在外部传入 do_sample=True。
    """
    if setup_infer:
        configure_for_inference(model, enable_kv_cache=True, disable_gradient_checkpointing=True)

    gen = gen or GenConfig()

    # 1) 读图像
    img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

    # 2) 编码并移动到模型设备（若模型分片或 device_map=auto，HF 会处理）
    device = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
    enc = _encode_for_generate(processor, img, rules, gen.image_size, device)

    # 3) 生成
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=int(gen.max_new_tokens),
        do_sample=bool(gen.do_sample),
        num_beams=int(gen.num_beams),
    )
    # 仅在采样场景下注入采样相关参数，避免对部分实现造成冲突
    if gen.do_sample:
        gen_kwargs.update(
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            top_k=int(gen.top_k),
        )

    with torch.inference_mode():
        out_ids = model.generate(**enc, **gen_kwargs)

    # 4) 解码（仅取新生成部分）
    input_len = enc["input_ids"].shape[-1]
    raw = processor.decode(out_ids[0][input_len:])

    # 5) 解析 JSON
    js = extract_first_json_block(raw)
    obj = None
    if js is not None:
        obj = find_and_parse_first_json(js)  # 第二次校验：获得 dict

    return GenResult(raw_text=raw, json_text=js, json_obj=obj)


# ============================================================================
# 四、针对数据集的批量生成（串行，重视稳健性）
# ============================================================================

def generate_for_dataset(
    model: "torch.nn.Module",
    processor: AutoProcessor,
    dataset: "datasets.Dataset",
    rules: str,
    gen: Optional[GenConfig] = None,
    setup_infer: bool = True,
) -> Tuple[np.ndarray, List[Optional[str]], List[Optional[dict]]]:
    """
    对 HFDataset 的每个样本**逐个**生成（串行），返回解析好的结果列表。
    """
    if setup_infer:
        configure_for_inference(model, enable_kv_cache=True, disable_gradient_checkpointing=True)

    gen = gen or GenConfig()
    device = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))

    raw_texts: List[str] = []
    json_texts: List[Optional[str]] = []
    json_objs: List[Optional[dict]] = []

    # ✅ 加 tqdm 进度条
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
        raw = processor.decode(out_ids[0][input_len:])

        js = extract_first_json_block(raw)
        obj = None
        if js is not None:
            obj = find_and_parse_first_json(js)

        raw_texts.append(raw)
        json_texts.append(js)
        json_objs.append(obj)

    return np.array(raw_texts, dtype=object), json_texts, json_objs


# ============================================================================
# 五、便捷：针对 RUL 场景的数值提取（可复制到其他数值任务）
# ============================================================================

def extract_rul_values(
    json_objs: Sequence[Optional[dict]],
    low: float,
    high: float,
    reason_key: str = "reason",
) -> Tuple[np.ndarray, List[str]]:
    """
    将一组 JSON 对象解析为（数值, reason）：
    - 对每个元素调用 `validate_rul_object`，获取 (value_or_None, reason)；
    - 若 value 为 None，则以 NaN 占位；
    - 返回：
        * values:  np.ndarray[float]，长度与输入一致
        * reasons: List[str]          ，长度与输入一致
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
