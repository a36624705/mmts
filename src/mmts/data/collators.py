# src/mmts/data/collators.py
# -*- coding: utf-8 -*-
"""
批处理组装（Data Collator）

本文件实现用于 VL（视觉-语言）SFT 的数据整理器：
- 对文本相关张量（input_ids / attention_mask / labels）做 pad；
- 将图像张量（pixel_values）按 batch 维拼接；
- 对可选键（image_grid_thw）进行拼接；
- 保留必要的元信息（如 y、path）以便调试（默认不返回给 Trainer）。

与 HuggingFace Trainer 的兼容性：
- `__call__(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]`
- 返回字典至少包含：input_ids、attention_mask、labels、pixel_values
- 若存在 image_grid_thw，则也应返回该键，形状 (B, 3)

注意：
- labels 中 pad 位置会被替换为 -100（loss 忽略标记）
- 本 Collator 假设每个样本已由处理器产生统一键位：
    * input_ids: (L,)
    * attention_mask: (L,)
    * labels: (L,)
    * pixel_values: (C, H, W)
    * (可选) image_grid_thw: (3,)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class VLSFTDataCollator:
    """
    视觉-语言 SFT 的数据整理器（适配单图像输入）。

    参数：
    - tokenizer:            用于 pad 文本序列的 tokenizer（必须支持 pad）
    - return_aux_in_batch:  是否把 "y" 与 "path" 等辅助字段一并返回（默认 False）。
                            当设为 True 时会以列表形式返回，便于自定义调试/可视化；
                            注意：Trainer 不会使用这些字段，它们不会影响梯度。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, return_aux_in_batch: bool = False) -> None:
        self.tokenizer = tokenizer
        self.return_aux_in_batch = bool(return_aux_in_batch)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将若干样本（features）合并为一个批次。

        处理步骤：
        1) 文本三件套（input_ids / attention_mask / labels）统一 pad 到批内最大长度；
           - 使用 tokenizer.pad 完成对 input_ids/attention_mask 的填充；
           - 对 labels 中等于 pad_token_id 的位置替换为 -100；
        2) 图像张量 pixel_values 直接按 batch 维拼接（要求形状一致，前处理器保证）；
        3) 若样本包含 image_grid_thw，则一并拼接；
        4) 视需要附加辅助字段（y/path），仅用于调试观察。

        返回：
        - dict，包含供模型前向所需的张量，以及可选辅助信息。
        """
        if len(features) == 0:
            raise ValueError("VLSFTDataCollator 收到空特征列表。")

        # ---- 1) 文本相关键收集并 pad ----
        # 将单样本 (L,) 张量转为 list[Tensor]，交给 tokenizer 统一 pad
        text_keys = ["input_ids", "attention_mask", "labels"]
        text_batches: Dict[str, List[torch.Tensor]] = {k: [] for k in text_keys}

        for f in features:
            for k in text_keys:
                if k not in f:
                    raise KeyError(f"样本缺少必要键 '{k}'。实际键：{list(f.keys())}")
                # 转成 1D Tensor（保险起见，避免是 numpy）
                text_batches[k].append(torch.as_tensor(f[k]))

        # tokenizer.pad 期望 {key: List[Tensor]}，并会按 input_ids 的 pad_token_id 填充
        padded = self.tokenizer.pad(
            text_batches,
            padding=True,            # pad 到批内最大长度
            return_tensors="pt",     # 返回 PyTorch 张量
        )

        # labels 中 pad 的位置替换为 -100，以便 CrossEntropyLoss 忽略
        labels: torch.Tensor = padded["labels"]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # 部分 tokenizer 可能没有 pad_token_id（少见），这里使用一个不会命中的 id 兜底
            # 但通常应在构建 tokenizer 时设置 pad_token_id
            pad_id = -1
        labels[labels == pad_id] = -100

        # ---- 2) 图像张量拼接 ----
        # 要求上游处理器输出统一 (C, H, W)
        try:
            pixel_values = torch.stack(
                [torch.as_tensor(f["pixel_values"]) for f in features],
                dim=0,  # (B, C, H, W)
            )
        except KeyError as e:
            raise KeyError(f"样本缺少必要键 'pixel_values'：{e}")

        # ---- 3) 可选键：image_grid_thw ----
        batch: Dict[str, Any] = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
            "pixel_values": pixel_values,
        }

        if "image_grid_thw" in features[0]:
            # 若存在，则假设每个样本都有该键；若个别样本缺失会抛错提醒
            image_grid_thw = torch.stack(
                [torch.as_tensor(f["image_grid_thw"]) for f in features],
                dim=0,  # (B, 3)
            )
            batch["image_grid_thw"] = image_grid_thw

        # ---- 4) 可选：附加辅助字段（非张量，不参与训练，仅便于调试）----
        if self.return_aux_in_batch:
            batch["_aux_y"] = [float(f.get("y", float("nan"))) for f in features]
            batch["_aux_path"] = [str(f.get("path", "")) for f in features]

        return batch
