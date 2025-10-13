# src/mmts/prompts/templates.py
# -*- coding: utf-8 -*-
"""
消息模板构造（Prompt Templates for Multi-Modal JSON Regression）

功能概述
--------
本模块统一生成用于训练/推理的多模态对话模板（messages）。
通过这些函数，可以轻松构造：
- 含图像输入和文本规则的 user 消息；
- 含目标 JSON（或空字符串）的 assistant 消息。

这样在不同任务（RUL、温度预测、信号回归等）中，只需替换规则内容和字段名，
就能快速扩展新任务，而不用改底层数据集逻辑。

核心接口：
----------
- `build_train_messages(img, rules, target_json)`：训练阶段，包含 user+assistant；
- `build_infer_messages(img, rules)`：推理阶段，仅包含 user；
- `apply_chat_template(processor, messages, add_generation_prompt, tokenize)`：
    将 messages 转换成符合模型输入的文本串（内部封装 AutoProcessor.apply_chat_template）。
"""

from __future__ import annotations
from typing import Any, Dict, List
from PIL import Image
from transformers import AutoProcessor


# ============================================================================
# 一、基础消息模板构造
# ============================================================================

def build_train_messages(
    img: Image.Image,
    rules: str,
    target_json: str,
) -> List[Dict[str, Any]]:
    """
    构造训练阶段的对话消息：
    - user：包含图像 + 规则文本；
    - assistant：包含目标 JSON。

    返回：
    [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": rules}]},
        {"role": "assistant", "content": [{"type": "text", "text": target_json}]}
    ]
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
            "content": [
                {"type": "text", "text": target_json},
            ],
        },
    ]


def build_infer_messages(
    img: Image.Image,
    rules: str,
) -> List[Dict[str, Any]]:
    """
    构造推理阶段的对话消息：
    - 仅包含 user；
    - assistant 由模型生成。

    返回：
    [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": rules}]}
    ]
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": rules},
            ],
        }
    ]


# ============================================================================
# 二、统一封装：调用 processor.apply_chat_template
# ============================================================================

def apply_chat_template(
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = False,
    tokenize: bool = False,
) -> str:
    """
    将 messages 结构转成模型可接受的 prompt 字符串。

    参数：
    - processor: AutoProcessor 实例（已加载模板逻辑）；
    - messages: 由 build_train_messages 或 build_infer_messages 生成的结构；
    - add_generation_prompt: 是否在最后添加生成提示（推理时应为 True）；
    - tokenize: 是否直接返回已分词结果（通常 False，只返回字符串）。

    返回：
    - prompt_text: 处理后的文本模板串
    """
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
    )
