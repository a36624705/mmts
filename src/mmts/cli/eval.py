# src/mmts/cli/eval.py
# -*- coding: utf-8 -*-
"""
评估入口（Hydra 版）
- 读取根目录 configs/ 下的配置（可被 CLI 覆盖）
- 加载基础 VL 模型与 LoRA 适配器
- 加载 Processor（优先使用保存目录，其次使用模型自带）
- 构建测试集，批量生成，解析 JSON，计算指标并出图落盘
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import hydra
from omegaconf import DictConfig

import numpy as np
import torch

# ==== 项目内模块 ====
from ..utils.io import build_output_paths, gen_run_name, save_json
from ..models.loader import load_model_and_processor
from ..data.builders import (
    DatasetBuildConfig,
    build_hfds_from_explicit,
    build_hfds_from_single_root,
)
from ..evaluation.generate import GenConfig, generate_for_dataset, extract_rul_values
from ..evaluation.metrics import compute_regression_metrics_dict
from ..evaluation.plotting import plot_series_comparison, plot_parity, plot_error_hist

# 可选：Processor 加载封装
try:
    from ..utils.io import load_processor  # AutoProcessor 的封装
except Exception:  # pragma: no cover
    load_processor = None  # type: ignore

# LoRA（PEFT）
try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover
    PeftModel = None  # type: ignore
    _PEFT_IMPORT_ERROR = e
else:
    _PEFT_IMPORT_ERROR = None


def _cfg_get(cfg: DictConfig, dotted: str, default: Any = None) -> Any:
    """从 DictConfig 里用点路径取值；不存在则返回 default。"""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, (dict, DictConfig)) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _resolve_with_renderer(path: Optional[str], renderer: Optional[str]) -> Optional[str]:
    """若给了 renderer，则返回 path/renderer；否则返回 path。"""
    if path is None:
        return None
    p = Path(path)
    return str((p / renderer).as_posix()) if renderer else str(p.as_posix())


def _load_rules_text(rules_file: Optional[str]) -> str:
    """优先读配置指定的规则文件；否则读取包内 mmts/prompts/base_rules.txt。"""
    if rules_file:
        return Path(rules_file).read_text(encoding="utf-8")
    try:
        from importlib import resources
        return resources.files("mmts.prompts").joinpath("base_rules.txt").read_text(encoding="utf-8")
    except Exception:
        raise FileNotFoundError(
            "未能读取规则文本。请在配置中设置 prompts.rules_file，或确保包内存在 mmts/prompts/base_rules.txt。"
        )


# ===== 新增：latest_dir() 的 Python 等价实现 =====
def _latest_subdir(parent: Path) -> Optional[str]:
    """
    返回 parent 目录下一层“最近修改时间”的子目录路径（字符串），若不存在返回 None。
    """
    if not parent.exists() or not parent.is_dir():
        return None
    # 仅考虑一级子目录
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    return str(latest.as_posix())


def _maybe_auto_select_dirs(
    lora_dir: Optional[str],
    processor_dir: Optional[str],
    outputs_root: str,
) -> Tuple[str, Optional[str], dict]:
    """
    若 lora_dir / processor_dir 未显式提供或传 "latest"，
    则从 outputs_root/{lora,processor} 下自动选择最近修改的子目录。
    返回 (lora_dir, processor_dir, info)。
    """
    info = {"auto_lora": False, "auto_processor": False}

    # LoRA：必须能解析出目录，否则报错
    if lora_dir is None or str(lora_dir).strip().lower() == "latest":
        auto = _latest_subdir(Path(outputs_root) / "lora")
        if not auto:
            raise ValueError(
                f"未找到可用的 LoRA 目录。请显式设置 eval.lora_dir，或确保 {outputs_root}/lora/ 下存在子目录。"
            )
        lora_dir = auto
        info["auto_lora"] = True

    # Processor：可自动，也可回退到 base model 的 processor
    if processor_dir is None or str(processor_dir).strip().lower() == "latest":
        auto = _latest_subdir(Path(outputs_root) / "processor")
        if auto:
            processor_dir = auto
            info["auto_processor"] = True
        else:
            # 不报错，后续将回退到 base model processor
            processor_dir = None

    # 规范化去尾斜杠
    lora_dir = str(Path(lora_dir).as_posix())
    if processor_dir is not None:
        processor_dir = str(Path(processor_dir).as_posix())

    return lora_dir, processor_dir, info


@hydra.main(config_path="../../../configs", config_name="defaults", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- 读取关键配置 ----
    model_id       = _cfg_get(cfg, "model.model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    lora_dir       = _cfg_get(cfg, "eval.lora_dir", None)
    processor_dir  = _cfg_get(cfg, "eval.processor_dir", None)
    outputs_root   = _cfg_get(cfg, "paths.outputs_root", "outputs")

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

    # ==== 新增：默认开启“latest”行为（显式传入优先） ====
    lora_dir, processor_dir, _auto_info = _maybe_auto_select_dirs(lora_dir, processor_dir, outputs_root)
    if _auto_info.get("auto_lora"):
        print(f"[Auto] Using latest LoRA under '{outputs_root}/lora': {lora_dir}")
    else:
        print(f"[Info] Using provided LoRA dir: {lora_dir}")
    if _auto_info.get("auto_processor"):
        print(f"[Auto] Using latest Processor under '{outputs_root}/processor': {processor_dir}")
    else:
        print(f"[Info] Processor dir: {processor_dir or '<none - will use base model processor>'}")

    # ---- 规则文本 ----
    rules = _load_rules_text(rules_file)

    # ---- 输出目录（本次评估专属 run）---
    run_name = gen_run_name("eval")
    paths = build_output_paths(root=outputs_root, with_run=False)

    # ---- 加载基础模型 + Processor ----
    print(f"[Info] Loading base model: {model_id}")
    base_model, proc_from_model = load_model_and_processor(
        model_id=model_id,
        trust_remote_code=True,
        use_fast_tokenizer=False,
    )

    if processor_dir:
        if load_processor is None:
            raise ImportError("缺少 Processor 加载工具或 transformers 未安装。")
        print(f"[Info] Loading processor from: {processor_dir}")
        processor = load_processor(processor_dir)
    else:
        processor = proc_from_model
        print("[Info] Using processor from base model.")

    # ---- 注入 LoRA ----
    if PeftModel is None:
        raise ImportError(
            "未检测到 `peft` 库，请先安装：\n  pip install -U peft\n\n"
            f"原始导入错误：{_PEFT_IMPORT_ERROR!r}"
        )
    print(f"[Info] Loading LoRA adapter from: {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)

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

    # ---- 批量推理 ----
    gen_cfg = GenConfig(
        image_size=image_size,
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
        num_beams=int(num_beams),
    )
    print("[Eval] Generating predictions on test set...")
    raw_texts, json_texts, json_objs = generate_for_dataset(
        model=model,
        processor=processor,
        dataset=test_hf,
        rules=rules,
        gen=gen_cfg,
        setup_infer=True,
    )

    # ---- 抽取数值并裁剪 ----
    y_true = np.asarray([float(test_hf[i]["y"]) for i in range(len(test_hf))], dtype=np.float32)
    y_pred, reasons = extract_rul_values(
        json_objs=json_objs,
        low=float(rul_low),
        high=float(rul_high),
        reason_key="reason",
    )

    # ---- 指标 ----
    metrics = compute_regression_metrics_dict(y_true, y_pred)
    print(f"[Eval] RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f} | "
          f"R2={metrics['r2']:.3f} | valid={metrics['valid_ratio']*100:.1f}% "
          f"(n={int(metrics['n_valid'])}/{int(metrics['n_total'])})")

    # ---- 落盘 ----
    metrics_path = paths.logs / f"metrics_{run_name}.json"
    save_json(metrics, metrics_path)
    print(f"[Save] Metrics saved to: {metrics_path}")

    print("[Eval] Saving figures...")
    fig_series_path = paths.figures / f"series_{run_name}.png"
    fig_parity_path = paths.figures / f"parity_{run_name}.png"
    fig_hist_path   = paths.figures / f"errors_{run_name}.png"

    plot_series_comparison(
        y_true, y_pred,
        first_n=min(200, len(y_true)),
        title=f"Test RUL Prediction | RMSE={metrics['rmse']:.2f}",
        out_path=fig_series_path,
    )
    plot_parity(
        y_true, y_pred,
        title="Parity Plot (Test)",
        out_path=fig_parity_path,
    )
    plot_error_hist(
        y_true, y_pred,
        title="Error Distribution (pred - true)",
        out_path=fig_hist_path,
    )
    print(f"[Save] Figures ->\n  {fig_series_path}\n  {fig_parity_path}\n  {fig_hist_path}")
    print("[Eval] Done.")


if __name__ == "__main__":
    main()
