# src/mmts/data/datasets.py
# -*- coding: utf-8 -*-
"""SFT 数据集：图像 → JSON（仅监督数值 token）。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# -------------------------
# 仅监督数值 token 的配置
# -------------------------

@dataclass
class NumberSupervisionSpec:
    """仅监督数字 token 的配置。"""
    value_key: str = "rul"
    reason_key: str = "reason"
    reason_placeholder: str = ""
    fallback_full_assistant: bool = True  # 定位失败时是否监督整段 assistant


# -------------------------
# 工具函数
# -------------------------

def _find_subsequence(hay: Sequence[int], needle: Sequence[int]) -> List[int]:
    """在 hay 中寻找 needle 的所有起始下标。"""
    hits: List[int] = []
    n, m = len(hay), len(needle)
    if m == 0 or n < m:
        return hits
    for i in range(n - m + 1):
        if list(hay[i : i + m]) == list(needle):
            hits.append(i)
    return hits


def _build_training_messages_single_image(
    img: Image.Image,
    rules: str,
    target_json: str,
) -> List[Dict[str, Any]]:
    """构造 user(image+rules) → assistant(target_json) 的 messages。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": rules},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_json}],
        },
    ]


def _make_target_json_number(value: float, spec: NumberSupervisionSpec) -> str:
    """构造训练用 JSON：仅对数值取整，其余为占位。"""
    v = int(round(float(value)))
    return f'{{ "{spec.value_key}": {v}, "{spec.reason_key}": "{spec.reason_placeholder}" }}'


# -------------------------
# 数据集
# -------------------------

class ImageJsonNumberSFTDataset(Dataset):
    """
    图像 → JSON(含数值字段) 的 SFT 数据集：
    - 使用 chat template 生成文本；
    - 仅监督数字 token，其它位置为 -100；
    - 当前实现支持单图像输入。
    """

    def __init__(
        self,
        root: Path | str,
        processor: object,  # 需具备 .apply_chat_template / __call__ / .tokenizer
        rules: str,
        split: Optional[str] = None,
        image_glob: str = "sample_*.png",
        label_filename: str = "y.npy",
        supervision: Optional[NumberSupervisionSpec] = None,
        image_size: int = 448,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.processor = processor
        self.rules = str(rules)
        self.split = split or ""
        self.image_glob = image_glob
        self.label_filename = label_filename
        self.spec = supervision or NumberSupervisionSpec()
        self.image_size = int(image_size)

        # 收集样本与标签
        self.img_paths = sorted(self.root.glob(self.image_glob))
        if not self.img_paths:
            raise FileNotFoundError(f"[{self.split}] 未找到图像: {self.root} ({self.image_glob})")

        y_path = self.root / self.label_filename
        if not y_path.exists():
            raise FileNotFoundError(f"[{self.split}] 未找到标签文件: {y_path}")

        self.y = np.load(y_path).astype(np.float32)
        if len(self.img_paths) != len(self.y):
            raise AssertionError(f"[{self.split}] 样本数不一致: images={len(self.img_paths)} vs labels={len(self.y)}")

        # 轻量固定随机性
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回一个样本的编码字典。"""
        path = self.img_paths[idx]
        y_val = float(self.y[idx])

        # 1) 读图
        img = Image.open(path).convert("RGB")

        # 2) 目标 JSON（reason 使用占位符；仅 value 参与监督）
        target_json = _make_target_json_number(y_val, self.spec)

        # 3) messages 与模板化文本
        messages = _build_training_messages_single_image(img, self.rules, target_json)
        full_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        # 4) 同时编码文本与图像
        enc = self.processor(
            text=full_text,
            images=img,
            return_tensors="pt",
            max_pixels=self.image_size * self.image_size,
        )

        # 5) 拆张量
        input_ids_full = enc["input_ids"].squeeze(0)      # (L,)
        attn_mask_full = enc["attention_mask"].squeeze(0) # (L,)
        pixel_values = enc["pixel_values"].squeeze(0)     # (C,H,W)
        image_grid_thw = enc.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)    # (3,)

        # 6) 计算 prompt_len：完整序列长度 - assistant 文本长度
        tokenizer = self.processor.tokenizer  # 不做类型限定，duck typing
        asst_ids: List[int] = list(tokenizer(target_json, add_special_tokens=False)["input_ids"])
        asst_len = len(asst_ids)
        full_len = int(input_ids_full.shape[0])
        prompt_len = full_len - asst_len
        if prompt_len <= 0:
            raise ValueError(
                f"[{self.split}] prompt_len 异常：full_len={full_len}, asst_len={asst_len}, path={path}"
            )

        # 7) 初始化 labels：-100
        labels = torch.full_like(input_ids_full, fill_value=-100)

        # 8) 仅监督数字 token
        num_txt = str(int(round(float(y_val))))
        num_ids: List[int] = list(tokenizer(num_txt, add_special_tokens=False)["input_ids"])
        num_hits = _find_subsequence(asst_ids, num_ids)

        if not num_hits:
            if self.spec.fallback_full_assistant:
                labels[prompt_len:] = input_ids_full[prompt_len:]
        else:
            for h in num_hits:
                start = prompt_len + h
                end = start + len(num_ids)
                labels[start:end] = input_ids_full[start:end]

        # 9) 输出
        out: Dict[str, Any] = {
            "input_ids": input_ids_full,
            "attention_mask": attn_mask_full,
            "labels": labels,
            "pixel_values": pixel_values,
            "y": y_val,
            "path": str(path),
        }
        if image_grid_thw is not None:
            out["image_grid_thw"] = image_grid_thw
        return out
