# src/mmts/__init__.py
# -*- coding: utf-8 -*-
"""
MMTS: Multi-Modal Time-series (and numeric JSON) fine-tuning toolkit.

定位
----
基于多模态大模型进行“图像/时序 -> JSON(数值)”的最小工程化框架：模型加载、LoRA、数据/Collator、
训练封装、推理与评估可视化。

快速上手
--------
>>> import mmts as M
>>> model, processor = M.load_model_and_processor("Qwen/Qwen2.5-VL-3B-Instruct")
>>> model = M.attach_lora(model, M.LoRAParams())
"""

from __future__ import annotations

__version__ = "0.1.0"

# -------------------------
# core：加载与 LoRA
# -------------------------
from .core.loader import (
    load_model_and_processor,
    configure_for_inference,
)
from .core.lora import (
    LoRAParams,
    build_lora_config,
    attach_lora,
    print_trainable_parameters,
)

# -------------------------
# data：Dataset / Collator / 构建器
# -------------------------
from .data.datasets import (
    ImageJsonNumberSFTDataset,
    NumberSupervisionSpec,
)
from .data.collators import VLSFTDataCollator
from .data.builders import (
    DatasetBuildConfig,
    build_hfds_from_explicit,
    build_hfds_from_single_root,
    preview_supervised_token_counts,
)

# -------------------------
# training：TrainingArguments / Trainer
# -------------------------
from .training.arguments import (
    TrainConfig,
    build_training_arguments,
)
from .training.trainer import (
    build_trainer,
    save_lora_and_processor,
)

# -------------------------
# evaluation：生成/指标/绘图
# -------------------------
from .evaluation.generate import (
    GenConfig,
    GenResult,
    generate_for_image,
    generate_for_dataset,
    extract_rul_values,
)
from .evaluation.metrics import (
    RegressionMetrics,
    compute_regression_metrics,
    compute_regression_metrics_dict,
)
from .evaluation.plotting import (
    plot_series_comparison,
    plot_parity,
    plot_error_hist,
)

# -------------------------
# utils：随机性与 I/O
# -------------------------
from .utils.seed import (
    set_global_seed,
    set_torch_backends,
)
from .utils.io import (
    build_output_paths,
    gen_run_name,
    save_json,
    load_json,
    save_text,
    load_text,
    save_matplotlib_figure,
    load_processor,
)
from .utils.prompts import (
    build_train_messages,
    build_infer_messages,
    apply_chat_template,
)

__all__ = [
    "__version__",
    # models
    "load_model_and_processor", "configure_for_inference",
    "LoRAParams", "build_lora_config", "attach_lora", "print_trainable_parameters",
    # data
    "ImageJsonNumberSFTDataset", "NumberSupervisionSpec", "VLSFTDataCollator",
    "DatasetBuildConfig", "build_hfds_from_explicit", "build_hfds_from_single_root",
    "preview_supervised_token_counts",
    # training
    "TrainConfig", "build_training_arguments", "build_trainer", "save_lora_and_processor",
    # evaluation
    "GenConfig", "GenResult", "generate_for_image", "generate_for_dataset", "extract_rul_values",
    "RegressionMetrics", "compute_regression_metrics", "compute_regression_metrics_dict",
    "plot_series_comparison", "plot_parity", "plot_error_hist",
    # utils
    "set_global_seed", "set_torch_backends",
    "build_output_paths", "gen_run_name",
    "save_json", "load_json", "save_text", "load_text", "save_matplotlib_figure",
    "load_processor",
    "build_train_messages", "build_infer_messages", "apply_chat_template",
]
