# src/mmts/data/builders.py
# -*- coding: utf-8 -*-
"""
数据集构建器（Builders）

职责
----
1) 基于本项目的底层 torch Dataset（ImageJsonNumberSFTDataset），
   构建 HuggingFace 的 Dataset（datasets.Dataset，简称 HFDataset）：
   - 支持明确的 train_dir/test_dir；
   - 或仅提供一个数据目录时，按比例切分（train_ratio）。
2) 提供少量辅助工具（如快速统计“监督到的数字 token 数量”），
   用于 sanity check 与数据管线自检。

设计理念
--------
- 上层训练入口（cli/train.py）应尽量简单，只负责读取配置并调用这里的构建函数；
- 该模块只关注“数据如何组织到 HFDataset”，不介入训练细节与模型逻辑；
- 任何与模型相关的参数（如 rules、image_size）都通过函数参数传入，避免隐式依赖。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .datasets import (
    ImageJsonNumberSFTDataset,
    NumberSupervisionSpec,
)


# ============================================================================
# 一、组装 HFDataset 的主函数
# ============================================================================

@dataclass
class DatasetBuildConfig:
    """
    数据集构建的核心参数集合。

    字段：
    - rules:                任务规则文本（会拼到 user 输入中）
    - image_glob:          匹配图像文件的 glob（默认 "sample_*.png"）
    - label_filename:      标签文件名（默认 "y.npy"）
    - image_size:          处理器编码时的像素上限（max_pixels≈image_size^2）
    - seed:                随机种子（用于划分/随机性）
    - supervision:         “仅监督数字 token”的规则（如 value_key、是否 fallback）
    """
    rules: str
    image_glob: str = "sample_*.png"
    label_filename: str = "y.npy"
    image_size: int = 448
    seed: int = 42
    supervision: Optional[NumberSupervisionSpec] = None


def _torchds_to_hfds(torch_ds: ImageJsonNumberSFTDataset) -> HFDataset:
    """
    将底层 torch Dataset 转为 HuggingFace Dataset。
    实现：逐个 __getitem__ 收集为 list，再用 HFDataset.from_list 创建。

    说明：
    - 这样做的好处是简单直观，且便于 Trainer 与 map 操作；
    - 对样本数中小规模的数据足够；如需极大规模数据，可考虑 Streaming 等方案。
    """
    records: List[Dict] = [torch_ds[i] for i in range(len(torch_ds))]
    return HFDataset.from_list(records)


def build_hfds_from_explicit(
    processor: AutoProcessor,
    train_root: Path | str,
    test_root: Path | str,
    cfg: DatasetBuildConfig,
) -> Tuple[HFDataset, HFDataset]:
    """
    基于**显式的 train/test 目录**构建两个 HFDataset。

    参数：
    - processor:   HF AutoProcessor
    - train_root:  训练数据目录（含若干图片与 y.npy）
    - test_root:   测试数据目录
    - cfg:         数据构建相关参数（规则、glob、image_size 等）

    返回：
    - (train_hf, test_hf) 两个 HFDataset
    """
    spec = cfg.supervision or NumberSupervisionSpec()

    train_ds = ImageJsonNumberSFTDataset(
        root=train_root,
        processor=processor,
        rules=cfg.rules,
        split="train",
        image_glob=cfg.image_glob,
        label_filename=cfg.label_filename,
        supervision=spec,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )
    test_ds = ImageJsonNumberSFTDataset(
        root=test_root,
        processor=processor,
        rules=cfg.rules,
        split="test",
        image_glob=cfg.image_glob,
        label_filename=cfg.label_filename,
        supervision=spec,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )

    return _torchds_to_hfds(train_ds), _torchds_to_hfds(test_ds)


def build_hfds_from_single_root(
    processor: AutoProcessor,
    root: Path | str,
    cfg: DatasetBuildConfig,
    train_ratio: float = 0.8,
) -> Tuple[HFDataset, HFDataset]:
    """
    只有一个数据目录时，按比例切分为 train/test 两份 HFDataset。

    步骤：
    1) 先构建完整的 torch Dataset；
    2) 用固定随机种子打乱索引；
    3) 按 train_ratio 切分出索引；
    4) 将两个子集逐条 materialize 成 list，再转为 HFDataset。

    参数：
    - processor:    HF AutoProcessor
    - root:         单一数据目录（包含图片与 y.npy）
    - cfg:          数据构建参数
    - train_ratio:  训练集占比（0~1），默认 0.8

    返回：
    - (train_hf, test_hf)
    """
    spec = cfg.supervision or NumberSupervisionSpec()
    base = ImageJsonNumberSFTDataset(
        root=root,
        processor=processor,
        rules=cfg.rules,
        split="full",
        image_glob=cfg.image_glob,
        label_filename=cfg.label_filename,
        supervision=spec,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )

    n = len(base)
    idx = np.arange(n)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx)

    n_tr = int(round(n * float(train_ratio)))
    n_tr = max(1, min(n - 1, n_tr))  # 既不为空，又不全占

    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr:]

    def _subset_to_list(ds: ImageJsonNumberSFTDataset, indices: np.ndarray) -> List[Dict]:
        return [ds[int(i)] for i in indices]

    train_list = _subset_to_list(base, tr_idx)
    test_list = _subset_to_list(base, te_idx)

    return HFDataset.from_list(train_list), HFDataset.from_list(test_list)


# ============================================================================
# 二、数据质量快速检查（可选）
# ============================================================================

def preview_supervised_token_counts(
    hf_train: HFDataset,
    tokenizer: PreTrainedTokenizerBase,
    k: int = 3,
) -> List[int]:
    """
    取训练集前 k 个样本，统计其 labels 中 **有效监督 token** 的数量。

    用途：
    - 快速判断“仅监督数字 token”的逻辑是否生效；
    - 若出现连续 0，可能意味着 tokenizer 对数字分词与目标 JSON 不匹配，
      需检查：rules、target_json 格式或 fallback 策略。

    参数：
    - hf_train:   训练集 HFDataset
    - tokenizer:  用于 pad 的 tokenizer（这里仅用于获取 pad_token_id）
    - k:          检查样本个数（默认 3）

    返回：
    - 长度为 k 的整数列表，每个元素是该样本的“被监督 token 数量”
    """
    k = int(max(1, min(k, len(hf_train))))
    counts: List[int] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    for i in range(k):
        ex = hf_train[i]
        labels = torch.as_tensor(ex["labels"])
        # 被监督的定义：labels != -100 且 != pad_id（稳妥起见）
        n_valid = int(((labels != -100) & (labels != pad_id)).sum().item())
        counts.append(n_valid)
    return counts
