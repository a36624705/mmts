# src/mmts/utils/io.py
# -*- coding: utf-8 -*-
"""
I/O 与路径工具：
- 统一管理 outputs 目录结构（checkpoints/、figures/、logs/、lora/）
- 生成 run_name（基于时间戳或自定义前缀）
- 保存/加载 HuggingFace 的 Processor
- 保存 LoRA 适配器（PEFT 模型）
- 保存 Matplotlib 图像到文件（自动创建父目录）
- 简易的文本/JSON 文件保存工具

设计目标：
- 最小依赖、与上层训练逻辑解耦；
- 任何函数仅做路径与文件层面的职责，不引入训练/评估细节；
- 尽可能避免硬编码：默认使用 outputs/，也允许调用方传入自定义基路径。
"""

from __future__ import annotations

import json
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    # transformers 不是强依赖；仅在需要保存/加载 Processor 时才用到
    from transformers import AutoProcessor, ProcessorMixin
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore
    ProcessorMixin = object  # 兜底，避免类型检查报错


# =========================
# 目录结构与常量
# =========================

DEFAULT_OUTPUT_ROOT = Path("outputs")  # 顶层输出目录（与 .gitignore 配套忽略）
SUBDIR_CHECKPOINTS = "checkpoints"     # 训练权重与 Trainer 状态
SUBDIR_FIGURES = "figures"             # 评估/可视化图像
SUBDIR_LOGS = "logs"                   # 文本日志
SUBDIR_LORA = "lora"                   # LoRA 适配器权重
SUBDIR_PROCESSOR = "processor"         # Processor 序列化目录（可与 lora 并列）


@dataclass(frozen=True)
class OutputPaths:
    """
    一个简单的数据类，用于集中管理当前 run 使用到的输出路径。
    说明：
    - root: 顶层 outputs 目录
    - checkpoints/figures/logs/lora/processor: 各子目录的具体 Path
    - run_dir: 针对某次运行的“会话目录”，常用于聚合该次训练的产物
               （如果不需要分 run 目录，可不使用该字段）
    """
    root: Path
    checkpoints: Path
    figures: Path
    logs: Path
    lora: Path
    processor: Path
    run_dir: Optional[Path] = None


# =========================
# 基础工具
# =========================

def ensure_dir(p: Path) -> Path:
    """
    确保目录存在；若不存在则递归创建。
    返回创建/存在的目录 Path，便于链式调用。
    """
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_output_paths(root: Path | str = DEFAULT_OUTPUT_ROOT,
                       with_run: bool = False,
                       run_name: Optional[str] = None) -> OutputPaths:
    """
    构建并（可选）创建标准输出目录结构。

    参数：
    - root: 顶层输出根目录（默认 'outputs'）
    - with_run: 是否额外创建一个 run 专属子目录（常用于一次训练会话）
    - run_name: run 子目录的名称；当 with_run=True 且未提供时，会自动生成时间戳名称

    返回：
    - OutputPaths 数据类实例，其中包含各子目录 Path。
    """
    root = Path(root)
    ensure_dir(root)

    ckpt = ensure_dir(root / SUBDIR_CHECKPOINTS)
    figs = ensure_dir(root / SUBDIR_FIGURES)
    logs = ensure_dir(root / SUBDIR_LOGS)
    lora = ensure_dir(root / SUBDIR_LORA)
    proc = ensure_dir(root / SUBDIR_PROCESSOR)

    run_dir: Optional[Path] = None
    if with_run:
        run_dir = ensure_dir(root / _auto_run_name(run_name))

    return OutputPaths(
        root=root,
        checkpoints=ckpt,
        figures=figs,
        logs=logs,
        lora=lora,
        processor=proc,
        run_dir=run_dir,
    )


def _auto_run_name(hint: Optional[str] = None) -> str:
    """
    生成一个 run 名称：
    - 当提供 hint 时，以 hint 为前缀；
    - 统一追加时间戳（到秒），避免名称冲突。

    示例：
    - hint=None  -> 'run_2025-10-13_21-30-05'
    - hint='fd001' -> 'fd001_2025-10-13_21-30-05'
    """
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "run" if not hint else str(hint).strip()
    return f"{prefix}_{ts}"


def gen_run_name(prefix: str = "run") -> str:
    """
    对外暴露的 run 名称生成函数（语义化包装）。
    训练脚本可在记录日志/保存配置时使用该名称作为会话标识。
    """
    return _auto_run_name(prefix)


# =========================
# 序列化工具：Processor / LoRA
# =========================

def save_processor(processor: "ProcessorMixin",
                   out_dir: Path | str) -> Path:
    """
    保存 HuggingFace Processor 到指定目录。
    - processor: 任何继承自 ProcessorMixin 的处理器（AutoProcessor 等）
    - out_dir: 保存目录路径（会自动创建）

    返回：
    - 实际保存的目录 Path
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    if not hasattr(processor, "save_pretrained"):
        raise TypeError("传入的 processor 不支持 save_pretrained 方法。")
    processor.save_pretrained(str(out_dir))
    return out_dir


def load_processor(from_dir: Path | str) -> "ProcessorMixin":
    """
    从目录加载 Processor。
    要求：transformers 已安装，并且目录内存在 config/processor 相关文件。
    """
    if AutoProcessor is None:
        raise ImportError("需要 transformers 才能加载 AutoProcessor。")
    from_dir = Path(from_dir)
    if not from_dir.exists():
        raise FileNotFoundError(f"Processor 目录不存在：{from_dir}")
    return AutoProcessor.from_pretrained(str(from_dir), trust_remote_code=True)


def save_lora_adapter(model: Any, out_dir: Path | str) -> Path:
    """
    保存 LoRA 适配器权重（PEFT 模型）。
    要求：model 对象拥有 save_pretrained(out_dir) 方法。
    注意：这里只负责保存“适配器”本身，而非完整基础模型权重。
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    if not hasattr(model, "save_pretrained"):
        raise TypeError("传入的 model 不支持 save_pretrained 方法。")
    model.save_pretrained(str(out_dir))
    return out_dir


# =========================
# Matplotlib 图像保存
# =========================

def save_matplotlib_figure(fig,
                           out_path: Path | str,
                           dpi: int = 300,
                           bbox_inches: str = "tight",
                           close: bool = True) -> Path:
    """
    将 Matplotlib Figure 保存到文件。
    - fig: 一个 Matplotlib 的 Figure 对象（例如 plt.gcf()）
    - out_path: 输出路径（包含文件名与扩展名，如 'outputs/figures/curve.png'）
    - dpi/bbox_inches: 常用保存参数
    - close: 保存后是否调用 fig.clf()/plt.close(fig) 释放内存

    返回：
    - 实际保存的文件 Path
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches=bbox_inches)
    if close:
        try:
            import matplotlib.pyplot as plt  # 延迟导入，避免无图环境报错
            plt.close(fig)
        except Exception:
            pass
    return out_path


# =========================
# 文本/JSON 简易保存
# =========================

def save_text(content: str, out_path: Path | str, encoding: str = "utf-8") -> Path:
    """
    将字符串写入到指定路径（自动创建父目录，覆盖写）。
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(content, encoding=encoding)
    return out_path


def save_json(obj: Any,
              out_path: Path | str,
              ensure_ascii: bool = False,
              indent: int = 2) -> Path:
    """
    将 Python 对象以 JSON 格式写入文件。
    - ensure_ascii=False：允许写入可读的非 ASCII 字符（中文等）
    - indent=2：默认美化缩进，便于人工查看
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    return out_path


# -------------------------------
# JSON / 文本 读写便捷函数
# -------------------------------
from pathlib import Path
import json

def load_json(path: str | Path):
    """
    读取 JSON 文件并返回对象（dict/list/...）
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_text(path: str | Path) -> str:
    """
    读取纯文本文件并返回字符串
    """
    return Path(path).read_text(encoding="utf-8")
