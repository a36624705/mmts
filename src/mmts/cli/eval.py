# src/mmts/cli/eval.py
# -*- coding: utf-8 -*-
"""
评估入口（Command Line Interface - eval）

功能概述
--------
- （可选）从 YAML 配置读取参数，并与 CLI 合并（优先级：CLI > YAML > 代码默认）；
- 加载基础多模态模型（HF Hub）与 LoRA 适配器（本地目录）；
- 加载保存好的 Processor（与训练时一致）或使用模型自带 Processor（若未提供）；
- 构建测试集（显式 test-dir，或从单目录切分）；
- 批量推理（串行）→ 解析首个 JSON → 抽出 RUL → 指标与图表落盘。

使用示例
--------
1) 显式 test 目录：
   python -m mmts.cli.eval \
     --model-id Qwen/Qwen2.5-VL-3B-Instruct \
     --lora-dir outputs/lora/train_2025-10-13_12-00-00 \
     --processor-dir outputs/processor/train_2025-10-13_12-00-00 \
     --test-dir data/images/FD001/test/grayscale \
     --outputs-root outputs

2) 用 YAML（并可被 CLI 覆盖）：
   python -m mmts.cli.eval \
     --config configs/defaults.yaml \
     --lora-dir outputs/lora/train_... \
     --processor-dir outputs/processor/train_...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

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
from ..configs.loader import load_config_tree

# 可选：使用我们封装的 Processor 加载函数（若未提供 processor-dir，可跳过）
try:
    from ..utils.io import load_processor  # AutoProcessor 的封装
except Exception:  # pragma: no cover
    load_processor = None  # type: ignore

# LoRA 加载（PEFT）
try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover
    PeftModel = None  # type: ignore
    _PEFT_IMPORT_ERROR = e
else:
    _PEFT_IMPORT_ERROR = None


# =========================
# 与训练一致的默认规则（可被 --rules-file 覆盖）
# =========================
_DEFAULT_RULES = (
    "You are an assistant for numerical regression from images.\n"
    "Task: Predict Remaining Useful Life (RUL) in engine cycles.\n"
    "Valid range: [0, 150].\n"
    "Output JSON with two fields only:\n"
    '{ "rul": <number>, "reason": "<short explanation in English>" }\n'
    "Rules:\n"
    "- Return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
    "- `rul` must be a number inside the valid range.\n"
    "- `reason` must be concise (<= 20 words) and MUST NOT be 'N/A'.\n"
)

# =========================
# 小工具：cfg 读取与路径解析
# =========================
def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """从 dict 中按点路径取值：cfg[\"a\"][\"b\"][\"c\"] ← \"a.b.c\""""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def pick_from_cli_or_cfg(args, defaults, cfg: Dict[str, Any], arg_name: str, cfg_key: str, fallback: Any):
    """
    选择参数值，优先级：CLI(显式传入) > YAML(cfg) > fallback(通常是 parser 默认)。
    """
    cli_val = getattr(args, arg_name)
    def_val = getattr(defaults, arg_name)
    if cli_val != def_val:
        return cli_val
    cfg_val = cfg_get(cfg, cfg_key, None)
    if cfg_val is not None:
        return cfg_val
    return fallback

def resolve_with_renderer(path: Optional[str], renderer: Optional[str]) -> Optional[str]:
    """
    强制拼接 renderer 子目录。
    无论上层目录是否存在，都会返回 path/renderer。
    这样不同 renderer 各自独立，避免歧义。
    """
    if path is None:
        return None
    p = Path(path)
    if renderer:
        return str((p / renderer).as_posix())
    return str(p.as_posix())



# =========================
# CLI 解析
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a VL+LoRA model on a numeric JSON task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 配置入口 ----
    p.add_argument("--config", type=str, default=None, help="YAML 配置入口（例如 configs/defaults.yaml）")

    # ---- 模型与产物 ----
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HuggingFace 模型标识。可在 YAML: model.base_id")
    p.add_argument("--lora-dir", type=str, default=None,
                   help="训练后保存的 LoRA 适配器目录。可在 YAML: eval.lora_dir")
    p.add_argument("--processor-dir", type=str, default=None,
                   help="保存的 Processor 目录；未提供则使用模型自带。可在 YAML: eval.processor_dir")
    p.add_argument("--outputs-root", type=str, default="outputs",
                   help="评估产物输出根目录。可在 YAML: paths.outputs_root")

    # ---- 数据来源（两种方式二选一）----
    p.add_argument("--test-dir", type=str, default=None,
                   help="显式测试集目录（包含 sample_*.png 与 y.npy）。可在 YAML: data.test_dir")
    p.add_argument("--data-root", type=str, default=None,
                   help="只提供一个目录时按比例划分，并仅评估 test 部分。可在 YAML: data.data_root")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="单目录划分时训练集占比（仅 --data-root 生效）。可在 YAML: data.train_ratio")

    # ---- 渲染器（用于自动拼接 test_dir/<renderer>）----
    p.add_argument("--renderer", type=str, default=None,
                   help="图像渲染方法名；若 test_dir 指向上层目录，将尝试自动拼接 /<renderer>。可在 YAML: data.renderer")

    # ---- 规则与数据细节 ----
    p.add_argument("--rules-file", type=str, default=None,
                   help="规则文本文件路径；若不提供，使用内置默认规则。可在 YAML: prompts.rules_file")
    p.add_argument("--image-glob", type=str, default="sample_*.png",
                   help="匹配图像文件的 glob。可在 YAML: data.image_glob")
    p.add_argument("--label-filename", type=str, default="y.npy",
                   help="标签文件名。可在 YAML: data.label_filename")
    p.add_argument("--image-size", type=int, default=448,
                   help="编码时的图像像素上限。可在 YAML: data.image_size")

    # ---- 生成与裁剪区间 ----
    p.add_argument("--max-new-tokens", type=int, default=32,
                   help="生成的最大新 token 数。可在 YAML: eval.max_new_tokens")
    p.add_argument("--do-sample", action="store_true", default=False,
                   help="是否启用采样（默认关闭，保证稳定）。可在 YAML: eval.do_sample")
    p.add_argument("--num-beams", type=int, default=1,
                   help="束搜索条数（do-sample=False 时可>1）。可在 YAML: eval.num_beams")
    p.add_argument("--rul-low", type=float, default=0.0,
                   help="RUL 合法下界（用于裁剪与指标计算）。可在 YAML: eval.rul_low")
    p.add_argument("--rul-high", type=float, default=150.0,
                   help="RUL 合法上界（用于裁剪与指标计算）。可在 YAML: eval.rul_high")

    return p


# =========================
# 主流程
# =========================
def main():
    parser = build_argparser()
    args = parser.parse_args()
    defaults = parser.parse_args([])

    # --- YAML 配置（可选） ---
    cfg = load_config_tree(args.config) if args.config else {}

    # --- 合并参数：CLI > YAML > 默认 ---
    model_id       = pick_from_cli_or_cfg(args, defaults, cfg, "model_id",       "model.base_id",        defaults.model_id)
    lora_dir       = pick_from_cli_or_cfg(args, defaults, cfg, "lora_dir",       "eval.lora_dir",        None)
    processor_dir  = pick_from_cli_or_cfg(args, defaults, cfg, "processor_dir",  "eval.processor_dir",   None)
    outputs_root   = pick_from_cli_or_cfg(args, defaults, cfg, "outputs_root",   "paths.outputs_root",   defaults.outputs_root)

    test_dir_raw   = pick_from_cli_or_cfg(args, defaults, cfg, "test_dir",       "data.test_dir",        None)
    data_root      = pick_from_cli_or_cfg(args, defaults, cfg, "data_root",      "data.data_root",       None)
    train_ratio    = float(pick_from_cli_or_cfg(args, defaults, cfg, "train_ratio","data.train_ratio",   defaults.train_ratio))
    renderer       = pick_from_cli_or_cfg(args, defaults, cfg, "renderer",       "data.renderer",        None)

    rules_file     = pick_from_cli_or_cfg(args, defaults, cfg, "rules_file",     "prompts.rules_file",   None)
    image_glob     = pick_from_cli_or_cfg(args, defaults, cfg, "image_glob",     "data.image_glob",      defaults.image_glob)
    label_filename = pick_from_cli_or_cfg(args, defaults, cfg, "label_filename", "data.label_filename",  defaults.label_filename)
    image_size     = int(pick_from_cli_or_cfg(args, defaults, cfg, "image_size", "data.image_size",      defaults.image_size))

    max_new_tokens = int(pick_from_cli_or_cfg(args, defaults, cfg, "max_new_tokens","eval.max_new_tokens", defaults.max_new_tokens))
    do_sample      = bool(pick_from_cli_or_cfg(args, defaults, cfg, "do_sample",    "eval.do_sample",      defaults.do_sample))
    num_beams      = int(pick_from_cli_or_cfg(args, defaults, cfg, "num_beams",     "eval.num_beams",       defaults.num_beams))
    rul_low        = float(pick_from_cli_or_cfg(args, defaults, cfg, "rul_low",     "eval.rul_low",         defaults.rul_low))
    rul_high       = float(pick_from_cli_or_cfg(args, defaults, cfg, "rul_high",    "eval.rul_high",        defaults.rul_high))

    if lora_dir is None:
        raise ValueError("必须提供 LoRA 目录：--lora-dir 或在 YAML: eval.lora_dir")

    # --- 读取规则 ---
    rules = Path(rules_file).read_text(encoding="utf-8") if rules_file else _DEFAULT_RULES

    # --- 输出目录（本次评估专属 run）---
    run_name = gen_run_name("eval")
    paths = build_output_paths(root=outputs_root, with_run=False)

    # --- 加载基础模型 + Processor ---
    print(f"[Info] Loading base model: {model_id}")
    base_model, proc_from_model = load_model_and_processor(
        model_id=model_id,
        trust_remote_code=True,
        use_fast_tokenizer=False,
    )

    if processor_dir:
        if load_processor is None:
            raise ImportError("缺少 Processor 加载工具，或 transformers 未安装。")
        print(f"[Info] Loading processor from: {processor_dir}")
        processor = load_processor(processor_dir)
    else:
        processor = proc_from_model  # 使用模型自带 Processor
        print("[Info] Using processor from base model.")

    # --- 注入 LoRA 适配器 ---
    if PeftModel is None:
        raise ImportError(
            "未检测到 `peft` 库，请先安装：\n  pip install -U peft\n\n"
            f"原始导入错误：{_PEFT_IMPORT_ERROR!r}"
        )
    print(f"[Info] Loading LoRA adapter from: {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)

    # --- 构建测试集（两种路径） ---
    ds_cfg = DatasetBuildConfig(
        rules=rules,
        image_glob=image_glob,
        label_filename=label_filename,
        image_size=image_size,
        seed=42,               # 评估不敏感，这里固定
        supervision=None,      # 构建 HF 数据集时会包含 labels 等键，但评估只用 path/y
    )

    # 处理 renderer 子目录：若用户给的是 test 上层目录，则自动追加 /<renderer>
    test_dir = resolve_with_renderer(test_dir_raw, renderer) if test_dir_raw else None

    if test_dir:
        _, test_hf = build_hfds_from_explicit(
            processor=processor,
            train_root=test_dir,  # 这里传同一路径占位，不使用返回的 train_hf
            test_root=test_dir,
            cfg=ds_cfg,
        )
    elif data_root:
        if renderer:
            print("[Warn] 单目录切分模式不会自动附加 renderer 子目录；若你的数据在 test/<renderer>，"
                  "请使用 --test-dir 显式指定。")
        _, test_hf = build_hfds_from_single_root(
            processor=processor,
            root=data_root,
            cfg=ds_cfg,
            train_ratio=float(train_ratio),
        )
    else:
        raise ValueError("请提供 --test-dir 或 --data-root（与 --train-ratio 搭配）。")

    # --- 批量推理 ---
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

    # --- 抽取数值并裁剪到区间 ---
    y_true = np.asarray([float(test_hf[i]["y"]) for i in range(len(test_hf))], dtype=np.float32)
    y_pred, reasons = extract_rul_values(
        json_objs=json_objs,
        low=float(rul_low),
        high=float(rul_high),
        reason_key="reason",
    )

    # --- 计算指标 ---
    metrics = compute_regression_metrics_dict(y_true, y_pred)
    print(f"[Eval] RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f} | "
          f"R2={metrics['r2']:.3f} | valid={metrics['valid_ratio']*100:.1f}% "
          f"(n={int(metrics['n_valid'])}/{int(metrics['n_total'])})")

    # --- 保存指标 JSON ---
    metrics_path = paths.logs / f"metrics_{run_name}.json"
    save_json(metrics, metrics_path)
    print(f"[Save] Metrics saved to: {metrics_path}")

    # --- 画图并保存 ---
    print("[Eval] Saving figures...")
    fig_series_path = paths.figures / f"series_{run_name}.png"
    fig_parity_path = paths.figures / f"parity_{run_name}.png"
    fig_hist_path = paths.figures / f"errors_{run_name}.png"

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
