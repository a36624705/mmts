# src/mmts/models/lora.py
# -*- coding: utf-8 -*-
"""
LoRA（Low-Rank Adaptation）装配与配置工具。

职责
----
1) 以统一方式构建 `LoraConfig`；
2) 将 LoRA 注入到已加载的基础模型（get_peft_model）；
3) 打印/返回可训练参数计数，便于确认“只训练 LoRA”；
4) 提供一份对 VL/LLM 常见模块名的默认 target_modules 列表。

设计原则
--------
- 不绑定具体模型结构，默认的 target_modules 基于通用命名（q/k/v/o_proj + MLP 三线性）；
- 允许调用方显式传入 target_modules / modules_to_save 覆盖默认；
- 不在此处处理“训练细节”（如优化器、梯度检查点），只负责注入与统计。

依赖
----
需要安装 `peft`（HuggingFace）。若未安装，会在调用时给出清晰错误信息。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Tuple, Union


# -------------------------
# 兼容性导入（延迟检查）
# -------------------------
try:
    from peft import LoraConfig, get_peft_model
except Exception as _e:  # pragma: no cover
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    _PEFT_IMPORT_ERROR = _e
else:
    _PEFT_IMPORT_ERROR = None


# ============================================================================
# 一、数据类：LoRA 参数
# ============================================================================

@dataclass
class LoRAParams:
    """
    LoRA 超参数与目标模块集合。

    字段说明：
    - r:          秩（低秩分解的维度，典型取 4/8/16/32）
    - alpha:      LoRA 缩放系数（一般等于或略大于 r）
    - dropout:    LoRA 层的 dropout（训练稳健性）
    - bias:       LoRA 偏置策略（"none"|"all"|"lora_only"），通常 "none"
    - task_type:  任务类型（"CAUSAL_LM" 常见于自回归生成）
    - target_modules:  需要注入 LoRA 的子模块名列表（字符串匹配，见下默认）
    - modules_to_save:  非 LoRA 权重但需要与适配器一起保存的模块名（如 "lm_head"）
    """
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[Sequence[str]] = None
    modules_to_save: Optional[Sequence[str]] = None


# ============================================================================
# 二、默认的 target_modules 建议
# ============================================================================

_DEFAULT_TARGETS: Tuple[str, ...] = (
    # 注意：为兼容大部分 LLM/VL 模型，使用子字符串匹配策略；
    # 下列名称通常覆盖注意力的 q/k/v/o 以及 MLP 三线性层。
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",

    # 兼容某些实现可能使用的简写
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "ffn.gate_proj", "ffn.up_proj", "ffn.down_proj",
)


def default_target_modules() -> List[str]:
    """
    返回一份默认的 target_modules 列表。
    - 采取“尽量覆盖常见命名”的思路；
    - 最终是否命中由 PEFT 内部的名字匹配决定（基于子字符串）。
    """
    return list(_DEFAULT_TARGETS)


# ============================================================================
# 三、构建 LoRA 配置 & 注入
# ============================================================================

def build_lora_config(params: LoRAParams) -> "LoraConfig":
    """
    根据 LoRAParams 构建 `LoraConfig`。

    - 若 `params.target_modules` 为空，则使用默认列表；
    - 若 `params.modules_to_save` 为空，默认保存 "lm_head"（常见情况下需要同时序列化输出层）。

    返回：
    - peft.LoraConfig 实例
    """
    _ensure_peft_available()

    targets = list(params.target_modules) if params.target_modules else default_target_modules()
    modules_to_save = list(params.modules_to_save) if params.modules_to_save else ["lm_head"]

    cfg = LoraConfig(
        r=params.r,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        bias=params.bias,
        task_type=params.task_type,
        target_modules=targets,
        modules_to_save=modules_to_save,
    )
    return cfg


def attach_lora(
    model,
    params_or_cfg: Union[LoRAParams, "LoraConfig"],
    verbose: bool = True,
):
    """
    将 LoRA 注入到基础模型上，并返回带适配器的模型（PEFT 模型）。

    参数：
    - model: 已加载的基础模型（transformers 模型实例）
    - params_or_cfg: 可以传入 LoRAParams（本函数内部转换为 LoraConfig），
                     或直接传入已有的 LoraConfig。
    - verbose: 是否打印可训练参数统计信息。

    返回：
    - 带 LoRA 适配器的模型实例（可直接用于 Trainer 训练）

    典型用法：
    >>> model, processor = load_model_and_processor(...)
    >>> lora = LoRAParams(r=16, alpha=32, dropout=0.05)
    >>> model = attach_lora(model, lora)
    """
    _ensure_peft_available()

    cfg = params_or_cfg if _is_lora_config(params_or_cfg) else build_lora_config(params_or_cfg)  # type: ignore
    peft_model = get_peft_model(model, cfg)

    if verbose:
        _ = print_trainable_parameters(peft_model)

    return peft_model


# ============================================================================
# 四、统计与打印：可训练参数
# ============================================================================

def count_parameters(model) -> dict:
    """
    统计模型参数数量。

    返回字典：
    {
        "total": 总参数数目,
        "trainable": 可训练参数数目,
        "frozen": 冻结参数数目,
        "trainable_ratio": 可训练/总参数比例（0~1 浮点）
    }
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    frozen = total - trainable
    ratio = (trainable / total) if total > 0 else 0.0
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
        "trainable_ratio": float(ratio),
    }


def print_trainable_parameters(model) -> dict:
    """
    打印并返回可训练参数统计信息。
    - 便于在日志中快速确认“只训练 LoRA 层 + 少量保存模块”。

    返回：
    - 同 `count_parameters` 的字典
    """
    stats = count_parameters(model)
    pct = stats["trainable_ratio"] * 100.0
    print(
        f"[LoRA] Parameters: total={stats['total']:,d} | "
        f"trainable={stats['trainable']:,d} ({pct:.2f}%) | "
        f"frozen={stats['frozen']:,d}"
    )
    return stats


# ============================================================================
# 内部辅助
# ============================================================================

def _ensure_peft_available() -> None:
    """若 peft 未正确导入，则抛出更友好的错误信息。"""
    if (LoraConfig is None) or (get_peft_model is None):
        raise ImportError(
            "未检测到 `peft` 库，请先安装：\n"
            "  pip install -U peft\n\n"
            f"原始导入错误：{_PEFT_IMPORT_ERROR!r}"
        )


def _is_lora_config(x) -> bool:
    """运行时判断对象是否为 LoraConfig（避免直接依赖类型检查工具）。"""
    if LoraConfig is None:
        return False
    return isinstance(x, LoraConfig)
