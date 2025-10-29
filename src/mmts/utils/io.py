# src/mmts/utils/io.py
# -*- coding: utf-8 -*-
"""
I/O 与路径工具（仅做文件与目录层面的职责）：
- 输出目录结构与 run 名生成
- Processor/LoRA 适配器的保存与加载
- Matplotlib 图像与文本/JSON 的读写
"""

from __future__ import annotations

import json
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

# transformers 为可选依赖，仅在需要时使用
try:
    from transformers import AutoProcessor  # type: ignore
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore


# -------------------------
# 目录结构与常量
# -------------------------

DEFAULT_OUTPUT_ROOT = Path("outputs")
SUBDIR_CHECKPOINTS = "checkpoints"
SUBDIR_FIGURES = "figures"
SUBDIR_LOGS = "logs"
SUBDIR_LORA = "lora"
SUBDIR_PROCESSOR = "processor"


@dataclass(frozen=True)
class OutputPaths:
    """集中管理一次运行使用到的输出路径。"""
    root: Path
    checkpoints: Path
    figures: Path
    logs: Path
    lora: Path
    processor: Path
    run_dir: Optional[Path] = None


# -------------------------
# 基础工具
# -------------------------

def ensure_dir(p: Path) -> Path:
    """确保目录存在并返回该路径。"""
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_output_paths(
    root: Path | str = DEFAULT_OUTPUT_ROOT,
    with_run: bool = False,
    run_name: Optional[str] = None,
) -> OutputPaths:
    """构建标准输出目录结构，可选创建 run 子目录。"""
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
    """生成带时间戳的 run 名称；可选使用 hint 作为前缀。"""
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "run" if not hint else str(hint).strip()
    return f"{prefix}_{ts}"


def gen_run_name(prefix: str = "run") -> str:
    """对外暴露的 run 名生成函数。"""
    return _auto_run_name(prefix)


# -------------------------
# 序列化：Processor / LoRA
# -------------------------

@runtime_checkable
class _SavePretrainedLike(Protocol):
    def save_pretrained(self, *args, **kwargs) -> Any: ...


def save_processor(processor: _SavePretrainedLike, out_dir: Path | str) -> Path:
    """保存 HuggingFace Processor 到目录。"""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    if not hasattr(processor, "save_pretrained"):
        raise TypeError("传入对象不支持 save_pretrained。")
    processor.save_pretrained(str(out_dir))
    return out_dir


def load_processor(from_dir: Path | str):
    """从目录加载 Processor（需要 transformers）。"""
    if AutoProcessor is None:
        raise ImportError("需要安装 transformers 才能加载 AutoProcessor。")
    from_dir = Path(from_dir)
    if not from_dir.exists():
        raise FileNotFoundError(f"Processor 目录不存在：{from_dir}")
    return AutoProcessor.from_pretrained(str(from_dir), trust_remote_code=True)


def save_lora_adapter(model: _SavePretrainedLike, out_dir: Path | str) -> Path:
    """仅保存 LoRA 适配器权重（要求对象实现 save_pretrained）。"""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    if not hasattr(model, "save_pretrained"):
        raise TypeError("传入对象不支持 save_pretrained。")
    model.save_pretrained(str(out_dir))
    return out_dir


# -------------------------
# 统一模型保存/加载（新优化版本）
# -------------------------

def save_unified_model(
    trainer,
    processor: _SavePretrainedLike,
    output_dir: Path | str,
    save_checkpoint: bool = True
) -> Path:
    """
    统一保存模型（LoRA + Processor）到同一目录。
    
    Args:
        trainer: 训练器对象，包含LoRA模型
        processor: 处理器对象
        output_dir: 输出目录
        save_checkpoint: 是否保存训练checkpoint信息
    
    Returns:
        输出目录路径
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    # 保存LoRA适配器
    lora_dir = output_dir / "lora"
    ensure_dir(lora_dir)
    model = getattr(trainer, "model", None)
    if model is None:
        raise RuntimeError("trainer.model 为空，无法保存 LoRA 适配器。")
    if not hasattr(model, "save_pretrained"):
        raise TypeError("trainer.model 不支持 save_pretrained 方法。")
    model.save_pretrained(str(lora_dir))
    
    # 保存Processor
    processor_dir = output_dir / "processor"
    ensure_dir(processor_dir)
    if processor is None or not hasattr(processor, "save_pretrained"):
        raise TypeError("传入的 processor 不支持 save_pretrained 方法。")
    processor.save_pretrained(str(processor_dir))
    
    # 可选：保存训练checkpoint信息
    if save_checkpoint:
        # 安全地序列化训练参数和状态
        def safe_serialize(obj):
            """安全地序列化对象，跳过不可序列化的属性"""
            if isinstance(obj, dict):
                return {k: safe_serialize(v) for k, v in obj.items() 
                       if not k.startswith('_') and not callable(v)}
            elif isinstance(obj, (list, tuple)):
                return [safe_serialize(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)  # 将其他对象转换为字符串
        
        checkpoint_info = {
            "training_args": safe_serialize(getattr(trainer.args, "__dict__", {})),
            "trainer_state": safe_serialize(getattr(trainer.state, "__dict__", {})),
            "saved_at": _dt.datetime.now().isoformat()
        }
        save_json(checkpoint_info, output_dir / "checkpoint_info.json")
    
    return output_dir


def load_unified_model(
    model_dir: Path | str,
    base_model_id: str,
    device: Optional[str] = None,
    dtype: Optional[Any] = None,
    trust_remote_code: bool = True
) -> tuple[Any, Any]:
    """
    从统一目录加载模型（LoRA + Processor）。
    
    Args:
        model_dir: 模型目录（包含lora和processor子目录）
        base_model_id: 基础模型ID
        device: 设备
        dtype: 数据类型
        trust_remote_code: 是否信任远程代码
    
    Returns:
        (model, processor) 元组
    """
    model_dir = Path(model_dir)
    
    # 加载基础模型和处理器
    from mmts.core.loader import load_model_and_processor
    base_model, base_processor = load_model_and_processor(
        model_id=base_model_id,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code
    )
    
    # 尝试加载保存的processor
    processor_dir = model_dir / "processor"
    if processor_dir.exists():
        try:
            processor = load_processor(processor_dir)
        except Exception as e:
            print(f"[Warning] 无法加载保存的processor ({e})，使用基础模型processor")
            processor = base_processor
    else:
        processor = base_processor
    
    # 尝试加载LoRA适配器
    lora_dir = model_dir / "lora"
    if lora_dir.exists():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, str(lora_dir))
        except Exception as e:
            print(f"[Warning] 无法加载LoRA适配器 ({e})，使用基础模型")
            model = base_model
    else:
        model = base_model
    
    return model, processor


def find_latest_model(experiments_dir: Path | str = "experiments") -> Optional[Path]:
    """
    查找最新的模型目录。
    
    Args:
        experiments_dir: 实验目录
    
    Returns:
        最新模型目录路径，如果没找到则返回None
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        return None
    
    latest_exp = None
    latest_time = None
    
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
            
        models_dir = exp_dir / "models"
        if not models_dir.exists():
            continue
            
        # 检查是否有完整的模型文件
        lora_dir = models_dir / "lora"
        processor_dir = models_dir / "processor"
        
        if lora_dir.exists() and processor_dir.exists():
            # 使用目录修改时间作为排序依据
            mtime = exp_dir.stat().st_mtime
            if latest_time is None or mtime > latest_time:
                latest_time = mtime
                latest_exp = models_dir
    
    return latest_exp


# -------------------------
# Matplotlib 图像保存
# -------------------------

def save_matplotlib_figure(
    fig,
    out_path: Path | str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    close: bool = True,
) -> Path:
    """保存 Matplotlib Figure 到文件。"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches=bbox_inches)
    if close:
        try:
            import matplotlib.pyplot as plt  # 延迟导入以降低依赖
            plt.close(fig)
        except Exception:
            pass
    return out_path


# -------------------------
# 文本/JSON 读写
# -------------------------

def save_text(content: str, out_path: Path | str, encoding: str = "utf-8") -> Path:
    """写入文本（覆盖）。"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(content, encoding=encoding)
    return out_path


def save_json(
    obj: Any,
    out_path: Path | str,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> Path:
    """写入 JSON（默认美化缩进，保留非 ASCII 字符）。"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    return out_path


def load_json(path: str | Path):
    """读取 JSON 并返回对象。"""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str | Path) -> str:
    """读取纯文本并返回字符串。"""
    return Path(path).read_text(encoding="utf-8")
