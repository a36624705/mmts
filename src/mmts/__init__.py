# src/mmts/__init__.py
# -*- coding: utf-8 -*-
"""
MMTS: Multi-Modal Time-series (and numeric JSON) fine-tuning toolkit.

模块定位
--------
本包提供“基于多模态大模型进行数值回归（以 JSON 输出承载）”的最小工程化框架，
包含模型加载、LoRA 注入、数据集/Collator、训练封装、推理与评估可视化等组件。

你可以：
    >>> import mmts as M
    >>> model, processor = M.load_model_and_processor("Qwen/Qwen2.5-VL-3B-Instruct")
    >>> model = M.attach_lora(model, M.LoRAParams())
    >>> # 构建数据、训练、评估等参见子模块文档

对外暴露
--------
- models：加载模型/处理器、推理配置、LoRA 构建与注入
- data：   基础数据集、collator、HF Dataset 构建
- training：TrainingArguments 构建、Trainer 组装与保存
- evaluation：生成（generate）、指标与绘图
- utils：  随机数种子与 I/O 辅助

注意：
- 这是轻量入口，未对所有内部符号进行强行导出；如需深入使用，请直接 from 子模块导入。
"""

from __future__ import annotations

# -------------------------
# 版本号（手动维护或后续接入 setuptools_scm）
# -------------------------
__version__ = "0.1.0"

# -------------------------
# models 子模块：模型加载与 LoRA
# -------------------------
from .models.loader import (
    load_model_and_processor,   # 加载 AutoProcessor + VL 模型
    configure_for_inference,    # 推理阶段的便捷开关（KV cache / 关闭梯度检查点）
)
from .models.lora import (
    LoRAParams,                 # LoRA 超参数数据类
    build_lora_config,          # 将 LoRAParams 构建为 peft.LoraConfig
    attach_lora,                # 注入 LoRA 到基础模型
    print_trainable_parameters, # 打印/返回可训练参数统计
)

# -------------------------
# data 子模块：数据集/构建器/Collator
# -------------------------
from .data.datasets import (
    ImageJsonNumberSFTDataset,  # 单图像 → JSON(含数值字段) 的训练数据集
    NumberSupervisionSpec,      # 控制“仅监督数字 token”的策略
)
from .data.collators import (
    VLSFTDataCollator,          # 视觉-语言 SFT 的批处理整理器
)
from .data.builders import (
    DatasetBuildConfig,         # 数据集构建参数
    build_hfds_from_explicit,   # 显式 train/test 目录 → HFDataset
    build_hfds_from_single_root,# 单目录按比例切分 → HFDataset
    preview_supervised_token_counts,  # 快速查看被监督 token 数
)

# -------------------------
# training 子模块：TrainingArguments 与 Trainer
# -------------------------
from .training.arguments import (
    TrainConfig,                # 项目内训练超参数
    build_training_arguments,   # 生成 transformers.TrainingArguments
)
from .training.trainer import (
    build_trainer,              # 构建 transformers.Trainer
    save_lora_and_processor,    # 保存 LoRA 适配器与 Processor
)

# -------------------------
# evaluation 子模块：生成/指标/绘图
# -------------------------
from .evaluation.generate import (
    GenConfig,                  # 推理生成的超参数
    GenResult,                  # 单样本生成结果
    generate_for_image,         # 单图像生成
    generate_for_dataset,       # 数据集批量生成
    extract_rul_values,         # 从 JSON 对象中抽取 RUL 数值与 reason
)
from .evaluation.metrics import (
    RegressionMetrics,          # 回归指标数据类
    compute_regression_metrics, # 计算 RMSE/MAE/R²/valid_ratio
    compute_regression_metrics_dict,
)
from .evaluation.plotting import (
    plot_series_comparison,     # 序列对比图
    plot_parity,                # Parity 散点图
    plot_error_hist,            # 误差直方图
)

# -------------------------
# utils 子模块：随机数与 I/O 辅助
# -------------------------
from .utils.seed import (
    set_global_seed,            # 统一设置随机种子
    set_torch_backends,         # 配置 torch TF32/CUDNN 等后端
)
from .utils.io import (
    build_output_paths,         # 统一的输出目录结构
    gen_run_name,               # 运行命名（前缀+时间戳）
    save_json, load_json,       # JSON 读写
    save_text, load_text,       # 文本读写
    save_matplotlib_figure,     # matplotlib 图像保存
    load_processor,             # AutoProcessor 读取（与 save_pretrained 对应）
)

# 便于 IDE/工具自动补全
__all__ = [
    "__version__",
    # models
    "load_model_and_processor", "configure_for_inference",
    "LoRAParams", "build_lora_config", "attach_lora", "print_trainable_parameters",
    # data
    "ImageJsonNumberSFTDataset", "NumberSupervisionSpec",
    "VLSFTDataCollator",
    "DatasetBuildConfig", "build_hfds_from_explicit", "build_hfds_from_single_root",
    "preview_supervised_token_counts",
    # training
    "TrainConfig", "build_training_arguments",
    "build_trainer", "save_lora_and_processor",
    # evaluation
    "GenConfig", "GenResult", "generate_for_image", "generate_for_dataset", "extract_rul_values",
    "RegressionMetrics", "compute_regression_metrics", "compute_regression_metrics_dict",
    "plot_series_comparison", "plot_parity", "plot_error_hist",
    # utils
    "set_global_seed", "set_torch_backends",
    "build_output_paths", "gen_run_name",
    "save_json", "load_json", "save_text", "load_text", "save_matplotlib_figure",
    "load_processor",
]
