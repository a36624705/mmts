# src/mmts/data/builders.py
# -*- coding: utf-8 -*-
"""数据集构建器：将底层 Torch Dataset 组装为 HuggingFace Dataset。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset

from .datasets import ImageJsonNumberSFTDataset, NumberSupervisionSpec


@dataclass
class DatasetBuildConfig:
    """构建参数。"""
    rules: str
    image_glob: str = "sample_*.png"
    label_filename: str = "y.npy"
    image_size: int = 448
    seed: int = 42
    supervision: Optional[NumberSupervisionSpec] = None


def _torchds_to_hfds(torch_ds: ImageJsonNumberSFTDataset) -> HFDataset:
    """逐条 materialize 到 list，再构建 HFDataset。"""
    records: List[Dict] = [torch_ds[i] for i in range(len(torch_ds))]
    return HFDataset.from_list(records)


def build_hfds_from_explicit(
    processor: object,  # 需具备 .apply_chat_template / __call__ / .tokenizer
    train_root: Path | str,
    test_root: Path | str,
    cfg: DatasetBuildConfig,
) -> Tuple[HFDataset, HFDataset]:
    """基于显式 train/test 目录构建 HFDataset。"""
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
    processor: object,
    root: Path | str,
    cfg: DatasetBuildConfig,
    train_ratio: float = 0.8,
) -> Tuple[HFDataset, HFDataset]:
    """单目录按比例切分为 train/test 的 HFDataset。"""
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
    n_tr = max(1, min(n - 1, n_tr))  # 非空且不全占

    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr:]

    def _subset_to_list(ds: ImageJsonNumberSFTDataset, indices: np.ndarray) -> List[Dict]:
        return [ds[int(i)] for i in indices]

    train_list = _subset_to_list(base, tr_idx)
    test_list = _subset_to_list(base, te_idx)

    return HFDataset.from_list(train_list), HFDataset.from_list(test_list)


def preview_supervised_token_counts(
    hf_train: HFDataset,
    tokenizer: object,  # 宽松类型：需具备 pad_token_id 属性
    k: int = 3,
) -> List[int]:
    """统计前 k 个样本中被监督的 token 数（labels != -100 且 != pad_id）。"""
    k = int(max(1, min(k, len(hf_train))))
    counts: List[int] = []
    pad_id = getattr(tokenizer, "pad_token_id", -1)

    for i in range(k):
        ex = hf_train[i]
        labels = torch.as_tensor(ex["labels"])
        n_valid = int(((labels != -100) & (labels != pad_id)).sum().item())
        counts.append(n_valid)
    return counts
