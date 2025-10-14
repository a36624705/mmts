# src/mmts/data/datasets.py
# -*- coding: utf-8 -*-
"""
数据集定义（Supervised Fine-tuning for JSON 数值输出）

本文件提供一个通用的数据集类 `ImageJsonNumberSFTDataset`：
- 面向“图像（或多图像）→ 文本(JSON)”的 SFT 训练；
- 目标 JSON 至少包含一个 **数值字段**（默认键名为 "rul"），模型训练时**仅监督该数字的 token**，
  其余 token（如 JSON 结构、reason 文本）全部 mask（-100），以减少模型对格式与无关文本的过拟合；
- 与 HuggingFace 的 `AutoProcessor` / VL 模型配合：使用 `apply_chat_template` 构建对话风格输入。

当前实现聚焦**单图像输入**（与 Qwen-VL 等模型兼容）。如需多图像/跨模态（图像+文本），可在后续扩展。

典型目录结构（与原脚本一致）：
  <root>/
    sample_000001.png
    sample_000002.png
    ...
    y.npy                      # 长度与图像数一致的数值标签数组（float32）

使用方式：
  ds = ImageJsonNumberSFTDataset(
          root=Path("images/FD001/train"),
          processor=processor,
          rules=...,
          value_key="rul",
          image_glob="sample_*.png",
          label_filename="y.npy",
          reason_placeholder="",
          image_size=448,
       )

返回样本的字段：
  - input_ids:         (L,)
  - attention_mask:    (L,)
  - labels:            (L,)  仅数字 token 处为标签，其它位置为 -100
  - pixel_values:      (C,H,W)
  - image_grid_thw:    (3,)  若处理器返回该键，则一并输出
  - y:                 (float) 原始标量标签
  - path:              (str) 图像路径
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# transformers 仅作类型与接口使用（无需导入具体模型）
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# ============================================================================
# 一、辅助数据结构
# ============================================================================

@dataclass
class NumberSupervisionSpec:
    """
    控制“仅监督数字 token”的行为配置。

    字段说明：
    - value_key:           JSON 中数值字段的键名（默认 "rul"）
    - reason_key:          JSON 中解释字段的键名（默认 "reason"），仅用于构造 target_json
    - reason_placeholder:  训练阶段可将 reason 置为空串/占位符，避免模型学会输出 "N/A"
    - fallback_full_assistant: 当无法精准定位数字子序列时，是否回退为“监督整段 assistant”
                               （默认 True，与原脚本保持一致，保证训练可继续）
    """
    value_key: str = "rul"
    reason_key: str = "reason"
    reason_placeholder: str = ""
    fallback_full_assistant: bool = True


# ============================================================================
# 二、核心工具函数
# ============================================================================

def _find_subsequence(hay: Sequence[int], needle: Sequence[int]) -> List[int]:
    """
    在整型序列 hay 中查找子序列 needle 的所有起始下标。
    - 由于 tokenizer 的编码可能将数字拆分为多个 token，此处采用“精确子序列匹配”。

    返回：
    - 命中的起始位置列表（可能为空）
    """
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
    """
    构造“训练阶段”的多轮对话消息（messages），采用常见的 user→assistant 形式：
      - user:   [image + rules 文本]
      - assistant: 目标 JSON 文本

    注意：
    - 之所以在训练时带上 assistant 的完整 JSON，是为了构建监督序列；
    - 真正的监督掩码在后续通过 labels 实现，仅数字 token 处会被标注。
    """
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


def _make_target_json_number(
    value: float,
    spec: NumberSupervisionSpec,
) -> str:
    """
    构造训练阶段的目标 JSON 文本串。
    - 当前约定的输出形如：{ "<value_key>": <int>, "<reason_key>": "<占位>" }
    - 仅将数值取整（round→int），保持与原脚本一致；
      * 标签本身不裁剪范围，裁剪应在“推理后评估”阶段进行。
    """
    v = int(round(float(value)))
    return f'{{ "{spec.value_key}": {v}, "{spec.reason_key}": "{spec.reason_placeholder}" }}'


# ============================================================================
# 三、数据集定义
# ============================================================================

class ImageJsonNumberSFTDataset(Dataset):
    """
    针对“图像→JSON(含数值字段)”的监督微调数据集。

    功能要点：
    1) 读取统一文件命名的图像（默认 "sample_*.png"）与与之对齐的标签数组 y.npy；
    2) 使用 AutoProcessor 的 `apply_chat_template` 生成对话风格输入文本；
    3) 将文本与图像一并编码为张量（input_ids/attention_mask/pixel_values 等）；
    4) 构造 labels：**仅对数值字段对应的数字 token 位置赋监督标签，其余置 -100**；
       - 若 tokenizer 无法精确定位（极少），可按配置回退到“监督整段 assistant”。

    约束：
    - 当前仅支持“单图像输入”，适配主流 VL 模型（Qwen-VL / InternVL / LLaVA 系等）。
    """

    def __init__(
        self,
        root: Path | str,
        processor: AutoProcessor,
        rules: str,
        split: Optional[str] = None,
        image_glob: str = "sample_*.png",
        label_filename: str = "y.npy",
        supervision: Optional[NumberSupervisionSpec] = None,
        image_size: int = 448,
        seed: int = 42,
    ) -> None:
        """
        参数说明：
        - root:            数据根目录（包含若干 sample_*.png 与 y.npy）
        - processor:       HF AutoProcessor（需支持 chat template 与图像预处理）
        - rules:           任务规则/提示词（会作为 user 的文本内容）
        - split:           可选的分割标签（仅用于记录/打印，无逻辑分支）
        - image_glob:     匹配图像的 glob 模式（默认 "sample_*.png"）
        - label_filename: 标签文件名（默认 "y.npy"）
        - supervision:     数字监督配置（NumberSupervisionSpec），可为 None 使用默认
        - image_size:      编码时 max_pixels ≈ image_size^2，用于限制显存占用
        - seed:            随机种子（用于 index 打乱等微小随机性）
        """
        super().__init__()
        self.root = Path(root)
        self.processor = processor
        self.rules = str(rules)
        self.split = split or ""
        self.image_glob = image_glob
        self.label_filename = label_filename
        self.spec = supervision or NumberSupervisionSpec()
        self.image_size = int(image_size)

        # 1) 收集图像路径并做一致性检查
        self.img_paths = sorted(self.root.glob(self.image_glob))
        if not self.img_paths:
            raise FileNotFoundError(f"[{self.split}] 未在目录中找到图像: {self.root} ({self.image_glob})")

        # 2) 读取标签数组 y.npy（float32）
        y_path = self.root / self.label_filename
        if not y_path.exists():
            raise FileNotFoundError(f"[{self.split}] 未找到标签文件: {y_path}")
        self.y = np.load(y_path).astype(np.float32)

        if len(self.img_paths) != len(self.y):
            raise AssertionError(f"[{self.split}] 样本数不一致: images={len(self.img_paths)} vs labels={len(self.y)}")

        # 3) 固定随机性（这里主要防止某些潜在顺序依赖；不改变样本顺序）
        random.seed(seed)
        np.random.seed(seed)

    # -----------------------
    # 基本魔法方法
    # -----------------------
    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回单个样本的编码张量与元数据。

        步骤：
        1) 读取图像与对应标量标签；
        2) 构造目标 JSON（只在训练阶段使用 reason 占位）；
        3) 拼装训练消息 → 通过 processor.apply_chat_template 生成文本；
        4) 用 processor 同时编码文本与图像，获得 token 与像素张量；
        5) 基于“完整长度 - assistant 字符串长度”定位 assistant 起点（prompt_len）；
        6) 定位 **数字 token 的子序列**，将 labels 中对应位置设置为 token id，其余置 -100；
           - 若定位失败且允许回退，则监督整个 assistant 片段。

        返回字典键：
        - input_ids, attention_mask, labels, pixel_values, (可选)image_grid_thw, y, path
        """
        path = self.img_paths[idx]
        y_val = float(self.y[idx])

        # 1) 读图像
        img = Image.open(path).convert("RGB")

        # 2) 目标 JSON（仅 value_key 的值会用于监督；reason 仅占位）
        target_json = _make_target_json_number(y_val, self.spec)

        # 3) 构造 messages（user: image+rules；assistant: target_json）
        messages = _build_training_messages_single_image(img, self.rules, target_json)

        # 4) 生成可供 tokenizer 使用的“对话模板化文本”
        #    - add_generation_prompt=False：训练时包含 assistant 目标文本
        full_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        # 5) 统一编码文本与图像（VL 处理器会返回 pixel_values / image_grid_thw 等）
        enc = self.processor(
            text=full_text,
            images=img,
            return_tensors="pt",
            max_pixels=self.image_size * self.image_size,
        )

        # 6) 拆出张量（去 batch 维）
        input_ids_full = enc["input_ids"].squeeze(0)         # (L,)
        attn_mask_full = enc["attention_mask"].squeeze(0)    # (L,)
        pixel_values = enc["pixel_values"].squeeze(0)        # (C,H,W)
        image_grid_thw = enc.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)       # (3,)

        # 7) 计算 prompt_len = full_len - assistant_len
        #    方法：对 target_json 单独 tokenizer，再与完整序列长度作差
        tokenizer: PreTrainedTokenizerBase = self.processor.tokenizer
        tok_asst = tokenizer(target_json, add_special_tokens=False)
        asst_ids: List[int] = list(tok_asst["input_ids"])
        asst_len = len(asst_ids)
        full_len = int(input_ids_full.shape[0])
        prompt_len = full_len - asst_len
        if prompt_len <= 0:
            raise ValueError(
                f"[{self.split}] prompt_len 异常：full_len={full_len}, asst_len={asst_len}. "
                f"样本路径：{path}"
            )

        # 8) 初始化 labels，全置为 -100（忽略监督）
        labels = torch.full_like(input_ids_full, fill_value=-100)

        # 9) 仅监督“数字 token”：
        #    - 将标量 y_val → 四舍五入后转为字符串，再通过 tokenizer 得到其 token 序列；
        #    - 在 asst_ids 中查找该 token 子序列的所有命中位置；
        #    - 将这些位置映射回 labels（需要加上 prompt_len 偏移）。
        num_txt = str(int(round(float(y_val))))
        tok_num = tokenizer(num_txt, add_special_tokens=False)
        num_ids: List[int] = list(tok_num["input_ids"])
        num_hits = _find_subsequence(asst_ids, num_ids)

        if not num_hits:
            # 兜底方案：若精确定位失败，可以选择监督整段 assistant，以避免样本被完全忽略
            if self.spec.fallback_full_assistant:
                labels[prompt_len:] = input_ids_full[prompt_len:]
            # 否则就保持“全 -100”，相当于跳过该样本的损失贡献
        else:
            for h in num_hits:
                for j in range(len(num_ids)):
                    pos = prompt_len + h + j
                    labels[pos] = input_ids_full[pos]

        # 10) 组装输出
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
