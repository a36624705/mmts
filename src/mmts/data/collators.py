# src/mmts/data/collators.py
# -*- coding: utf-8 -*-
"""VL-SFT 批处理组装器：pad 文本序列、拼接图像张量，保留可选元信息。"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class VLSFTDataCollator:
    """
    视觉-语言 SFT 的数据整理器：
    - pad: input_ids / attention_mask
    - labels: 对齐到同长度，pad 位置填 -100
    - pixel_values: 按 batch 维堆叠
    - image_grid_thw: 若存在则堆叠
    - 可选附加 _aux_y / _aux_path
    """

    def __init__(self, tokenizer: object, return_aux_in_batch: bool = False) -> None:
        """
        参数
        ----
        tokenizer : object
            仅要求实现 .pad(...) 和 .pad_token_id
        return_aux_in_batch : bool
            是否把 y/path 等辅助信息一并返回（默认 False，仅供调试）。
        """
        self.tokenizer = tokenizer
        self.return_aux_in_batch = bool(return_aux_in_batch)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("VLSFTDataCollator 收到空特征列表。")

        # ---- 1) pad 文本序列（交给 tokenizer 处理 input_ids/attention_mask）----
        text_batches = {
            "input_ids": [torch.as_tensor(f["input_ids"]) for f in features],
            "attention_mask": [torch.as_tensor(f["attention_mask"]) for f in features],
        }
        padded = self.tokenizer.pad(text_batches, padding=True, return_tensors="pt")
        max_len = int(padded["input_ids"].shape[1])

        # ---- 2) labels 手动 pad 到 max_len，pad 值为 -100 ----
        labels_list = [torch.as_tensor(f["labels"]) for f in features]
        padded_labels = torch.full((len(labels_list), max_len), fill_value=-100, dtype=labels_list[0].dtype)
        for i, lab in enumerate(labels_list):
            L = int(lab.shape[0])
            if L > max_len:
                padded_labels[i, :] = lab[:max_len]  # 极端保护（通常不会发生）
            else:
                padded_labels[i, :L] = lab

        # ---- 3) 图像张量堆叠 ----
        try:
            pixel_values = torch.stack([torch.as_tensor(f["pixel_values"]) for f in features], dim=0)  # (B,C,H,W)
        except KeyError as e:
            raise KeyError(f"样本缺少必要键 'pixel_values'：{e}")

        batch: Dict[str, Any] = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": padded_labels,
            "pixel_values": pixel_values,
        }

        # ---- 4) 可选键：image_grid_thw ----
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.stack(
                [torch.as_tensor(f["image_grid_thw"]) for f in features], dim=0
            )

        # ---- 5) 可选调试信息（不参与训练）----
        if self.return_aux_in_batch:
            batch["_aux_y"] = [float(f.get("y", float("nan"))) for f in features]
            batch["_aux_path"] = [str(f.get("path", "")) for f in features]

        return batch
