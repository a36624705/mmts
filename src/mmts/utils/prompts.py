# src/mmts/utils/prompts.py
# -*- coding: utf-8 -*-
"""多模态 JSON 回归任务的消息模板构造与封装。"""

from __future__ import annotations
from typing import Any, Dict, List
from PIL import Image


# -------------------------
# 训练与推理消息构造
# -------------------------

def build_train_messages(
    img: Image.Image,
    rules: str,
    target_json: str,
) -> List[Dict[str, Any]]:
    """训练阶段消息：user(图片+规则) + assistant(目标JSON)。"""
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


def build_infer_messages(
    img: Image.Image,
    rules: str,
) -> List[Dict[str, Any]]:
    """推理阶段消息：仅 user(图片+规则)。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": rules},
            ],
        }
    ]


# -------------------------
# 统一封装：processor.apply_chat_template
# -------------------------

def apply_chat_template(
    processor: object,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = False,
    tokenize: bool = False,
) -> str:
    """
    将 messages 转为模型可接受的 prompt 字符串。
    依赖 processor.apply_chat_template（duck typing，避免强依赖 transformers 类型）。
    """
    return processor.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
    )
