# src/mmts/training/arguments.py
# -*- coding: utf-8 -*-
"""TrainingArguments 构建器：从项目参数生成 HF TrainingArguments。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import TrainingArguments


@dataclass
class TrainConfig:
    """项目统一训练超参数配置。"""
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"  # 不再用 Literal，避免类型表达式警告

    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: Optional[int] = None

    do_eval: bool = False
    eval_strategy: str = "no"
    eval_steps: int = 1000

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    dataloader_num_workers: int = 0
    max_grad_norm: float = 1.0
    report_to: str = "none"

    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None


def build_training_arguments(output_dir: str | Path, cfg: TrainConfig) -> TrainingArguments:
    """
    将项目内部的 TrainConfig 转换为 transformers.TrainingArguments。
    """
    output_dir = str(Path(output_dir))

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,

        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps if cfg.save_strategy == "steps" else None,
        save_total_limit=cfg.save_total_limit,

        do_eval=cfg.do_eval,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps if cfg.eval_strategy == "steps" else None,

        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,

        dataloader_num_workers=cfg.dataloader_num_workers,
        max_grad_norm=cfg.max_grad_norm,
        report_to=cfg.report_to,
    )

    if cfg.load_best_model_at_end:
        args.load_best_model_at_end = True
        if cfg.metric_for_best_model is not None:
            args.metric_for_best_model = cfg.metric_for_best_model
        if cfg.greater_is_better is not None:
            args.greater_is_better = cfg.greater_is_better

    return args
