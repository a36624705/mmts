# src/mmts/cli/eval.py
# -*- coding: utf-8 -*-
"""
评估入口（优化版本）
- 使用实验管理器统一管理模型加载
- 简化模型保存和加载流程
- 支持指定实验ID或使用最新实验
- 统一的输出目录管理
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig

import numpy as np

# ==== 项目内模块 ====
from ..utils.io import save_json, save_matplotlib_figure
from ..data.builders import (
    DatasetBuildConfig,
    build_hfds_from_explicit,
    build_hfds_from_single_root,
)
from ..evaluation.generate import GenConfig, generate_for_dataset, extract_rul_values
from ..evaluation.metrics import compute_regression_metrics_dict
from ..evaluation.plotting import plot_series_comparison, plot_parity, plot_error_hist


def _cfg_get(cfg: DictConfig, dotted: str, default: Any = None) -> Any:
    """从嵌套配置中安全获取值。"""
    keys = dotted.split(".")
    cur = cfg
    for k in keys:
        if not hasattr(cur, k):
            return default
        cur = getattr(cur, k)
    return cur


def _resolve_with_renderer(path: Optional[str], renderer: Optional[str]) -> Optional[str]:
    """若给了 renderer，则返回 path/renderer；否则返回 path。"""
    if path is None:
        return None
    p = Path(path)
    return str((p / renderer).as_posix()) if renderer else str(p.as_posix())


def _load_rules_text(rules_file: Optional[str]) -> str:
    """优先读配置指定的规则文件；否则读取项目根目录的 configs/base_rules.txt。"""
    if rules_file:
        return Path(rules_file).read_text(encoding="utf-8")
    # 项目根目录默认
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        default_rules = project_root / "configs" / "base_rules.txt"
        return default_rules.read_text(encoding="utf-8")
    except Exception:
        raise FileNotFoundError(
            "未能读取规则文本。请在配置中设置 prompts.rules_file，或确保项目根目录存在 configs/base_rules.txt。"
        )


# ---------------- 主流程（Hydra） ----------------
@hydra.main(config_path='../../../configs', config_name='defaults', version_base='1.3')
def main(cfg: DictConfig) -> None:
    # ---- 读取关键配置 ----
    model_id = _cfg_get(cfg, "model.model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    experiments_root = _cfg_get(cfg, "paths.experiments_root", "experiments")
    
    # 支持指定实验ID或使用最新实验
    experiment_id = _cfg_get(cfg, "eval.experiment_id", None)
    use_latest = _cfg_get(cfg, "eval.use_latest", True)

    test_dir_raw   = _cfg_get(cfg, "data.test_dir", None)
    data_root      = _cfg_get(cfg, "data.data_root", None)
    train_ratio    = float(_cfg_get(cfg, "data.train_ratio", 0.8))
    renderer       = _cfg_get(cfg, "data.renderer", None)

    rules_file     = _cfg_get(cfg, "prompts.rules_file", None)
    image_glob     = _cfg_get(cfg, "data.image_glob", "sample_*.png")
    label_filename = _cfg_get(cfg, "data.label_filename", "y.npy")
    image_size     = int(_cfg_get(cfg, "data.image_size", 448))

    max_new_tokens = int(_cfg_get(cfg, "eval.max_new_tokens", 32))
    do_sample      = bool(_cfg_get(cfg, "eval.do_sample", False))
    num_beams      = int(_cfg_get(cfg, "eval.num_beams", 1))
    rul_low        = float(_cfg_get(cfg, "eval.rul_low", 0.0))
    rul_high       = float(_cfg_get(cfg, "eval.rul_high", 150.0))
    
    # 样本数量限制（从图像化配置中读取）
    test_max_samples = _cfg_get(cfg, "data.imgify.test.max_samples", None)

    # ---- 创建实验管理器并选择实验 ----
    from mmts.utils.experiment import get_experiment_manager
    exp_manager = get_experiment_manager(experiments_root)
    
    if experiment_id:
        try:
            exp_info, exp_paths = exp_manager.load_experiment(experiment_id)
            print(f"[Experiment] 使用指定实验: {experiment_id}")
        except FileNotFoundError:
            print(f"[Error] 实验不存在: {experiment_id}")
            return
    elif use_latest:
        latest = exp_manager.get_latest_experiment()
        if latest is None:
            print(f"[Error] 没有找到任何实验，请先训练模型")
            return
        exp_info, exp_paths = latest
        print(f"[Experiment] 使用最新实验: {exp_info.experiment_id}")
    else:
        print(f"[Error] 请指定 experiment_id 或设置 use_latest=true")
        return
    
    print(f"[Experiment] 实验目录: {exp_paths.root}")

    # ---- 规则文本 ----
    rules = _load_rules_text(rules_file)

    # ---- 加载统一模型（LoRA + Processor） ----
    from mmts.utils.io import load_unified_model
    print(f"[Info] Loading unified model from: {exp_paths.models}")
    model, processor = load_unified_model(
        model_dir=exp_paths.models,
        base_model_id=model_id,
        trust_remote_code=True
    )

    # ---- 构建测试集 ----
    ds_cfg = DatasetBuildConfig(
        rules=rules,
        image_glob=image_glob,
        label_filename=label_filename,
        image_size=image_size,
        seed=42,
        supervision=None,
    )

    test_dir = _resolve_with_renderer(test_dir_raw, renderer) if test_dir_raw else None

    if test_dir:
        # 复用构建函数：我们只要 test_hf
        _, test_hf = build_hfds_from_explicit(
            processor=processor,
            train_root=test_dir,
            test_root=test_dir,
            cfg=ds_cfg,
        )
    elif data_root:
        if renderer:
            print("[Warn] 单目录切分模式不会自动附加 renderer 子目录；若数据在 test/<renderer>，请使用 data.test_dir。")
        _, test_hf = build_hfds_from_single_root(
            processor=processor,
            root=data_root,
            cfg=ds_cfg,
            train_ratio=float(train_ratio),
        )
    else:
        raise ValueError("请在配置中提供 data.test_dir 或 data.data_root。")

    # ---- 应用测试样本数量限制 ----
    if test_max_samples is not None and len(test_hf) > test_max_samples:
        print(f"[Info] 限制测试样本数量: {len(test_hf)} -> {test_max_samples}")
        test_hf = test_hf.select(range(test_max_samples))

    # ---- 批量推理 ----
    gen_cfg = GenConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        temperature=1.0,
        top_p=1.0,
    )

    print(f"[Generate] 开始批量推理，样本数: {len(test_hf)}")
    raw_texts, json_texts, json_objs = generate_for_dataset(model, processor, test_hf, rules, gen_cfg)
    print(f"[Generate] 推理完成")

    # ---- 解析 RUL 值 ----
    print("[Parse] 解析 RUL 值...")
    rul_pred, reasons = extract_rul_values(json_objs, low=rul_low, high=rul_high)
    rul_true = np.array([item["y"] for item in test_hf])

    # ---- 计算指标 ----
    print("[Metrics] 计算评估指标...")
    metrics = compute_regression_metrics_dict(rul_true, rul_pred)
    
    # 打印指标
    print("\n=== 评估结果 ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # ---- 保存结果 ----
    # 保存指标
    metrics_file = exp_paths.logs / f"metrics_eval_{exp_info.experiment_id}.json"
    save_json(metrics, metrics_file)
    print(f"[Save] 指标已保存到: {metrics_file}")

    # ---- 生成图表 ----
    print("[Plot] 生成评估图表...")
    
    # 时间序列对比图
    fig1 = plot_series_comparison(rul_true, rul_pred, title="RUL Prediction Comparison")
    fig1_path = exp_paths.figures / f"predictions_eval_{exp_info.experiment_id}.png"
    save_matplotlib_figure(fig1, fig1_path)
    print(f"[Plot] 时间序列对比图: {fig1_path}")

    # 散点图
    fig2 = plot_parity(rul_true, rul_pred, title="RUL Prediction Scatter")
    fig2_path = exp_paths.figures / f"parity_eval_{exp_info.experiment_id}.png"
    save_matplotlib_figure(fig2, fig2_path)
    print(f"[Plot] 散点图: {fig2_path}")

    # 误差分布图
    fig3 = plot_error_hist(rul_true, rul_pred, title="RUL Prediction Error Distribution")
    fig3_path = exp_paths.figures / f"error_hist_eval_{exp_info.experiment_id}.png"
    save_matplotlib_figure(fig3, fig3_path)
    print(f"[Plot] 误差分布图: {fig3_path}")

    print(f"\n[Complete] 评估完成！结果保存在: {exp_paths.root}")


if __name__ == "__main__":
    main()