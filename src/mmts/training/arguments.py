# src/mmts/training/arguments.py
# -*- coding: utf-8 -*-
"""
TrainingArguments 构建器

职责
----
将训练中常见的可调超参数集中到一个数据类里（便于从配置文件或 CLI 映射），
并提供一个函数把这些参数安全地转换为 HuggingFace 的
`transformers.TrainingArguments`。

设计取舍
--------
- 只暴露当前项目确实要用到的字段，避免过度配置；
- 提供合理的默认值，做到“零参数也能跑”；
- 与上层 Trainer 构建解耦：这里不涉及数据集/模型，只产出 TrainingArguments。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from transformers import TrainingArguments


# ============================================================================
# 一、数据类：训练参数（项目内通用）
# ============================================================================

SchedulerType = Literal["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
SaveStrategy = Literal["no", "epoch", "steps"]
EvalStrategy = Literal["no", "epoch", "steps"]  # 虽然当前流程训练中不做评估，但保留选项以备扩展


@dataclass
class TrainConfig:
    """
    与本项目契合的一组训练超参数。

    字段说明（常用项）：
    - num_train_epochs:            训练轮数
    - per_device_train_batch_size: 每设备 batch（总有效 batch=该值×梯度累积×设备数）
    - gradient_accumulation_steps: 梯度累积步数（显存不足时可增大，以换取更大有效 batch）
    - learning_rate:               初始学习率（LoRA 场景常在 1e-4 ~ 2e-4）
    - weight_decay:                权重衰减
    - warmup_ratio:                预热比例（相对总步数）
    - lr_scheduler_type:           学习率调度器
    - logging_steps:               训练日志打印间隔（step）
    - save_strategy:               保存策略（按 epoch 或按 step）
    - save_steps:                  当 save_strategy="steps" 时有效；多少步保存一次
    - save_total_limit:            最多保留的 checkpoint 个数（旧的会被自动删除），None 表示不限制
    - do_eval:                     训练过程中是否做评估（当前项目默认 False）
    - bf16:                        是否启用 bfloat16 训练（CUDA 建议 True）
    - fp16:                        是否启用 float16 训练（与 bf16 互斥，优先建议用 bf16）
    - gradient_checkpointing:      是否开启梯度检查点（适合显存吃紧，代价是前向更慢）
    - dataloader_num_workers:      DataLoader 的工作进程数（I/O 与预处理开销较小可设 0）
    - max_grad_norm:               梯度裁剪阈值
    - report_to:                   日志上报目标（"none"|"tensorboard"|"wandb"|...）

    其它：
    - eval_strategy/ eval_steps:   若 do_eval=True 可配合使用（这里保留字段以备扩展）
    """
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: SchedulerType = "cosine"

    logging_steps: int = 10
    save_strategy: SaveStrategy = "epoch"
    save_steps: int = 500
    save_total_limit: Optional[int] = None

    do_eval: bool = False
    eval_strategy: EvalStrategy = "no"
    eval_steps: int = 1000  # 仅当 eval_strategy="steps" 时使用

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    dataloader_num_workers: int = 0
    max_grad_norm: float = 1.0
    report_to: str = "none"

    # 可选：是否在训练结束时加载最佳模型（需要 eval 才有意义，这里留钩子）
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None


# ============================================================================
# 二、构建函数：TrainingArguments
# ============================================================================

def build_training_arguments(
    output_dir: str | Path,
    cfg: TrainConfig,
) -> TrainingArguments:
    """
    将项目内的 `TrainConfig` 转换为 `transformers.TrainingArguments`。

    参数：
    - output_dir: checkpoint 等训练产物输出目录（建议使用统一的 outputs/checkpoints/...）
    - cfg:        训练超参数数据类

    返回：
    - TrainingArguments 实例，可直接传入 `transformers.Trainer`
    """
    output_dir = str(Path(output_dir))

    # 注意：eval_strategy 若为 "no"，则在 TrainingArguments 中对应为 "no"
    # transformers 会将 "no" 视为不进行评估（等价于不设置/None）
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

        # 与我们当前流程一致：默认不使用 TensorBoard 以外的报告
        # disable_tqdm 可在 CLI 决定是否关闭进度条，这里不强制设置
    )

    # 当需要在训练结束自动加载最佳模型时（通常配合 eval）：
    if cfg.load_best_model_at_end:
        # TrainingArguments 中这些字段不存在“条件设置”的强制要求，
        # 但为了清晰，我们仅当用户显式启用该选项时才设置。
        args.load_best_model_at_end = True
        if cfg.metric_for_best_model is not None:
            args.metric_for_best_model = cfg.metric_for_best_model
        if cfg.greater_is_better is not None:
            args.greater_is_better = cfg.greater_is_better

    return args
