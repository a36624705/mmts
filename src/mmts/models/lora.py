# src/mmts/models/lora.py
# -*- coding: utf-8 -*-
"""LoRA 装配与配置：构建 LoraConfig、注入 LoRA、参数统计。"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

# 仅供类型检查使用，避免在注解中引用运行期的 Any
if TYPE_CHECKING:  # noqa: SIM108
    from peft import LoraConfig as _LoraConfig

# 运行期延迟导入，便于给出友好错误
try:
    from peft import LoraConfig, get_peft_model  # type: ignore
except Exception as _e:  # pragma: no cover
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    _PEFT_IMPORT_ERROR = _e
else:
    _PEFT_IMPORT_ERROR = None


@dataclass
class LoRAParams:
    """LoRA 超参数与目标模块集合。"""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[Sequence[str]] = None
    modules_to_save: Optional[Sequence[str]] = None


# 常见注意力/MLP 线性层命名
_DEFAULT_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "ffn.gate_proj", "ffn.up_proj", "ffn.down_proj",
)

def default_target_modules() -> List[str]:
    return list(_DEFAULT_TARGETS)


def build_lora_config(params: LoRAParams) -> "_LoraConfig":
    """由 LoRAParams 构建 LoraConfig；缺省时使用默认 targets 与 ['lm_head']。"""
    _ensure_peft_available()
    targets = list(params.target_modules) if params.target_modules else default_target_modules()
    modules_to_save = list(params.modules_to_save) if params.modules_to_save else ["lm_head"]
    return LoraConfig(  # type: ignore[call-arg]
        r=params.r,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        bias=params.bias,
        task_type=params.task_type,
        target_modules=targets,
        modules_to_save=modules_to_save,
    )


def attach_lora(
    model: object,
    params_or_cfg: Union[LoRAParams, "_LoraConfig"],
    verbose: bool = True,
) -> object:
    """将 LoRA 注入基础模型并返回带适配器的模型。"""
    _ensure_peft_available()
    cfg = params_or_cfg if _is_lora_config(params_or_cfg) else build_lora_config(params_or_cfg)  # type: ignore[arg-type]
    peft_model = get_peft_model(model, cfg)  # type: ignore[misc]
    if verbose:
        _ = print_trainable_parameters(peft_model)
    return peft_model


def count_parameters(model: object) -> dict:
    """返回 total/trainable/frozen/ratio 的参数统计。"""
    total = 0
    trainable = 0
    for p in model.parameters():  # type: ignore[attr-defined]
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


def print_trainable_parameters(model: object) -> dict:
    """打印并返回参数统计。"""
    stats = count_parameters(model)
    pct = stats["trainable_ratio"] * 100.0
    print(
        f"[LoRA] Parameters: total={stats['total']:,d} | "
        f"trainable={stats['trainable']:,d} ({pct:.2f}%) | "
        f"frozen={stats['frozen']:,d}"
    )
    return stats


def _ensure_peft_available() -> None:
    if (LoraConfig is None) or (get_peft_model is None):
        raise ImportError(
            "未检测到 `peft`，请安装：pip install -U peft\n"
            f"原始导入错误：{_PEFT_IMPORT_ERROR!r}"
        )


def _is_lora_config(x: object) -> bool:
    if LoraConfig is None:
        return False
    return isinstance(x, LoraConfig)
