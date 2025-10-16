# src/mmts/training/trainer.py
# -*- coding: utf-8 -*-
"""基于模型/数据/Collator/TrainingArguments 构建 HuggingFace Trainer，并提供保存工具。"""

from __future__ import annotations

from typing import Any, Callable, Optional

from transformers import Trainer, TrainingArguments

# 可选：打印参数统计（若未装 LoRA 也不报错）
try:
    from ..models.lora import print_trainable_parameters  # 仅用于统计输出
except Exception:  # pragma: no cover
    def print_trainable_parameters(model):  # type: ignore
        return {}


# ---------------------------------------------------------------------
# 构建 Trainer
# ---------------------------------------------------------------------
def build_trainer(
    model: object,
    args: TrainingArguments,
    train_dataset: object,
    eval_dataset: Optional[object] = None,
    data_collator: Optional[Callable[[list], dict]] = None,
    compute_metrics: Optional[Callable[[dict], dict]] = None,
    print_param_stats: bool = True,
) -> Trainer:
    """
    返回一个 `transformers.Trainer` 实例。
    - model/args/train_dataset 为必需；
    - eval_dataset/data_collator/compute_metrics 按需提供；
    - print_param_stats=True 时打印一次可训练参数占比（便于确认 LoRA 是否生效）。
    """
    if print_param_stats:
        _ = print_trainable_parameters(model)

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


# ---------------------------------------------------------------------
# 训练产物保存：LoRA 适配器 + Processor
# ---------------------------------------------------------------------
def save_lora_and_processor(
    trainer: Trainer,
    lora_output_dir: str,
    processor: object,
    processor_output_dir: str,
) -> None:
    """
    将 `trainer.model` 的 LoRA 适配器权重与配套 Processor 分别保存到目录。
    - 仅负责“适配器”保存；基础模型权重不在此复制。
    """
    model = getattr(trainer, "model", None)
    if model is None:
        raise RuntimeError("trainer.model 为空，无法保存 LoRA 适配器。")

    if not hasattr(model, "save_pretrained"):
        raise TypeError("trainer.model 不支持 save_pretrained 方法。")
    model.save_pretrained(lora_output_dir)

    if processor is None or not hasattr(processor, "save_pretrained"):
        raise TypeError("传入的 processor 不支持 save_pretrained 方法。")
    processor.save_pretrained(processor_output_dir)
