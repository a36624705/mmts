# src/mmts/training/trainer.py
# -*- coding: utf-8 -*-
"""
Trainer 构建与训练产物保存

职责
----
1) 基于模型、数据集、collator 与 TrainingArguments 构建 HuggingFace `Trainer`；
2) 提供轻量的保存工具：将 LoRA 适配器与 Processor 保存到指定目录；
3) （可选）打印一次可训练参数统计，便于确认 LoRA 装配是否生效。

注意
----
- 本模块**不**包含任何与配置或 I/O 目录组织的逻辑（这部分在 utils/io.py 中）；
- 也**不**包含训练入口或评估流程（这部分放在 src/mmts/cli/ 下的 train.py / eval.py）；
- 这里仅专注于把依赖对象“就位”，并提供最常用的薄封装。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from transformers import Trainer, TrainingArguments

# 可选：打印参数统计（若用户未装配 LoRA，此函数也能工作）
try:
    from ..models.lora import print_trainable_parameters  # 仅用于打印统计信息
except Exception:  # pragma: no cover
    def print_trainable_parameters(model):  # type: ignore
        return {}


# ============================================================================
# 一、Trainer 构建
# ============================================================================

def build_trainer(
    model: Any,
    args: TrainingArguments,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    data_collator: Optional[Callable[[list[dict]], dict]] = None,
    compute_metrics: Optional[Callable[[dict], dict]] = None,
    print_param_stats: bool = True,
) -> Trainer:
    """
    构建并返回一个 HuggingFace `Trainer` 实例。

    参数：
    - model:           已加载（且可能注入 LoRA）的 transformers 模型实例
    - args:            由 `mmts.training.arguments.build_training_arguments` 构造的 TrainingArguments
    - train_dataset:   训练集（可为 torch Dataset 或 HuggingFace Dataset）
    - eval_dataset:    可选的评估集（若 args.do_eval=False，则 Trainer 不会在训练中调用评估）
    - data_collator:   批处理组装函数/对象（通常为 `mmts.data.collators.VLSFTDataCollator`）
    - compute_metrics: 可选的指标回调（仅在 eval 时使用）
    - print_param_stats: 构建时是否打印一次可训练参数统计（LoRA 场景下很有用）

    返回：
    - `transformers.Trainer` 实例

    说明：
    - 本函数不额外改写 Trainer 的默认行为；如需自定义日志/回调，可在外层注入 callbacks。
    """
    if print_param_stats:
        _ = print_trainable_parameters(model)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer


# ============================================================================
# 二、训练产物保存（LoRA + Processor）
# ============================================================================

def save_lora_and_processor(
    trainer: Trainer,
    lora_output_dir: str,
    processor: Any,
    processor_output_dir: str,
) -> None:
    """
    将当前 `trainer.model` 的 **LoRA 适配器** 与 **Processor** 分别保存到指定目录。

    使用场景：
    - 在训练结束后于 CLI 中调用：
        save_lora_and_processor(trainer, "outputs/lora/my_run", processor, "outputs/processor/my_run")

    参数：
    - trainer:              已完成训练/微调的 Trainer
    - lora_output_dir:      LoRA 适配器保存目录（会调用 model.save_pretrained）
    - processor:            与模型配套的处理器（AutoProcessor 或兼容对象）
    - processor_output_dir: 处理器保存目录（会调用 processor.save_pretrained）

    注意：
    - 这里只负责“适配器”保存；基础大模型权重通常不在此复制；
    - 目录创建与有效性检查交由各对象的 save_pretrained 自身处理；
    - 对于异常（例如对象不具备 save_pretrained），将抛出 TypeError/AttributeError。
    """
    model = getattr(trainer, "model", None)
    if model is None:
        raise RuntimeError("trainer.model 为空，无法保存 LoRA 适配器。")

    # 保存 LoRA 适配器（PEFT 模型的 save_pretrained 会仅写出适配器权重）
    if not hasattr(model, "save_pretrained"):
        raise TypeError("trainer.model 不支持 save_pretrained 方法，无法保存 LoRA 适配器。")
    model.save_pretrained(lora_output_dir)

    # 保存处理器
    if processor is None or not hasattr(processor, "save_pretrained"):
        raise TypeError("传入的 processor 不支持 save_pretrained 方法。")
    processor.save_pretrained(processor_output_dir)
