# src/mmts/cli/train.py
# -*- coding: utf-8 -*-
"""
训练入口（Command Line Interface - train）

功能概述
--------
- （可选）读取 YAML 配置并与 CLI 参数合并（优先级：CLI > YAML > 代码默认）；
- 设置随机性与加速后端（TF32）；
- 加载多模态基础模型与处理器（AutoProcessor + AutoVLModel）；
- 注入 LoRA 适配器（仅语言侧线性层，默认 target_modules 见 models.lora）；
- 构建数据集（显式 train/test 目录，或单目录按比例划分）与 Data Collator；
- 组装 TrainingArguments 与 Trainer，启动训练；
- 训练结束后保存 LoRA 适配器与 Processor 到 outputs/ 对应子目录。

注意事项
--------
- 本文件不做评估/画图（这些在 eval.py 实现），仅负责训练与保存产物；
- 你可以通过外部 shell 设置 CUDA_VISIBLE_DEVICES / nohup 等运行方式；
- 若需要自定义规则文本（rules），可通过 --rules-file 指定，否则使用内置简短默认规则。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Any, Dict

import torch

# ==== 项目内模块 ====
from ..utils.seed import set_global_seed, set_torch_backends
from ..utils.io import build_output_paths, gen_run_name
from ..models.loader import load_model_and_processor
from ..models.lora import LoRAParams, attach_lora
from ..data.builders import (
    DatasetBuildConfig,
    build_hfds_from_explicit,
    build_hfds_from_single_root,
    preview_supervised_token_counts,
)
from ..data.collators import VLSFTDataCollator
from ..training.arguments import TrainConfig, build_training_arguments
from ..training.trainer import build_trainer, save_lora_and_processor
from ..configs.loader import load_config_tree

# =========================
# 内置一个精简默认规则（可被 --rules-file 覆盖）
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
# 工具：从 cfg 读取嵌套键
# =========================
def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """
    从 dict 中按点路径取值：cfg["a"]["b"]["c"] ← "a.b.c"
    若任一层不存在，返回 default。
    """
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def pick_from_cli_or_cfg(args, defaults, cfg: Dict[str, Any], arg_name: str, cfg_key: str, fallback: Any):
    """
    选择一个参数值，优先级：CLI(显式传入) > YAML(cfg) > fallback(通常是 parser 默认)
    - args: 解析后的命令行命名空间
    - defaults: p.parse_args([]) 得到的默认命名空间
    - cfg: 已加载的 YAML 配置 dict
    - arg_name: argparse 的参数名（如 "model_id"）
    - cfg_key: YAML 中的点路径（如 "model.base_id"）
    - fallback: 兜底值（通常取 defaults.<arg_name>）
    """
    cli_val = getattr(args, arg_name)
    def_val = getattr(defaults, arg_name)
    if cli_val != def_val:          # 用户在 CLI 显式传了值
        return cli_val
    cfg_val = cfg_get(cfg, cfg_key, None)
    if cfg_val is not None:         # YAML 给了值
        return cfg_val
    return fallback                  # 兜底

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
        description="Train LoRA on a VL model to predict a numeric field in JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="YAML 配置入口（例如 configs/defaults.yaml）")

    # ---- 基础路径与模型 ----
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HuggingFace 模型标识。可在 YAML: model.base_id")
    p.add_argument("--outputs-root", type=str, default="outputs",
                   help="训练产物输出根目录。可在 YAML: paths.outputs_root")

    # ---- 数据来源（两种方式二选一）----
    p.add_argument("--train-dir", type=str, default=None,
                   help="显式训练集目录（包含 sample_*.png 与 y.npy）。可在 YAML: data.train_dir")
    p.add_argument("--test-dir", type=str, default=None,
                   help="显式测试集目录。可在 YAML: data.test_dir")
    p.add_argument("--data-root", type=str, default=None,
                   help="若只提供一个目录，将按 --train-ratio 划分 train/test。可在 YAML: data.data_root")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="单目录划分时的训练集占比（仅 --data-root 生效）。可在 YAML: data.train_ratio")

    # ---- 渲染器（控制图像化子目录，如 grayscale/heatmap/...）----
    p.add_argument("--renderer", type=str, default=None,
                   help="图像渲染方法名；若 train/test 目录未包含该子目录，将自动拼接。可在 YAML: data.renderer")

    # ---- 规则与数据细节 ----
    p.add_argument("--rules-file", type=str, default=None,
                   help="规则文本文件路径；若不提供，使用内置默认规则。可在 YAML: prompts.rules_file")
    p.add_argument("--image-glob", type=str, default="sample_*.png",
                   help="匹配图像文件的 glob。可在 YAML: data.image_glob")
    p.add_argument("--label-filename", type=str, default="y.npy",
                   help="标签文件名。可在 YAML: data.label_filename")
    p.add_argument("--image-size", type=int, default=448,
                   help="编码时的图像像素上限（max_pixels≈image_size^2）。可在 YAML: data.image_size")

    # ---- 随机性与后端 ----
    p.add_argument("--seed", type=int, default=42, help="随机种子。可在 YAML: train.seed")
    p.add_argument("--allow-tf32", action="store_true", default=True, help="允许 TF32 加速。可在 YAML: train.allow_tf32")
    p.add_argument("--no-allow-tf32", action="store_false", dest="allow_tf32")

    # ---- LoRA 超参数 ----
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank。可在 YAML: lora.r")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA 缩放系数。可在 YAML: lora.alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout。可在 YAML: lora.dropout")

    # ---- 训练超参数 ----
    p.add_argument("--epochs", type=int, default=10, help="YAML: train.epochs")
    p.add_argument("--batch-size", type=int, default=1, help="YAML: train.batch_size")
    p.add_argument("--grad-accum", type=int, default=8, help="YAML: train.grad_accum")
    p.add_argument("--lr", type=float, default=2e-4, help="YAML: train.lr")
    p.add_argument("--weight-decay", type=float, default=0.01, help="YAML: train.weight_decay")
    p.add_argument("--warmup-ratio", type=float, default=0.05, help="YAML: train.warmup_ratio")
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                   help="YAML: train.scheduler")
    p.add_argument("--logging-steps", type=int, default=10, help="YAML: train.logging_steps")
    p.add_argument("--save-strategy", type=str, default="epoch", choices=["no", "epoch", "steps"], help="YAML: train.save_strategy")
    p.add_argument("--save-steps", type=int, default=500, help="YAML: train.save_steps")
    p.add_argument("--save-total-limit", type=int, default=None, help="YAML: train.save_total_limit")
    p.add_argument("--no-eval-during-train", action="store_true", default=True, help="YAML: train.no_eval_during_train")

    # ---- 设备与精度 ----
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available(), help="YAML: train.bf16")
    p.add_argument("--fp16", action="store_true", default=False, help="YAML: train.fp16（与 bf16 互斥，建议优先 bf16）")
    p.add_argument("--grad-ckpt", action="store_true", default=True, help="YAML: train.grad_ckpt")

    return p


# =========================
# 训练主逻辑
# =========================
def main():
    parser = build_argparser()
    args = parser.parse_args()
    defaults = parser.parse_args([])  # 获取 parser 的默认值，用于判断 CLI 是否显式覆盖

    # --- YAML 配置（可选） ---
    cfg = load_config_tree(args.config) if args.config else {}

    # --- 合并参数：CLI > YAML > 默认 ---
    model_id       = pick_from_cli_or_cfg(args, defaults, cfg, "model_id",       "model.base_id",          defaults.model_id)
    outputs_root   = pick_from_cli_or_cfg(args, defaults, cfg, "outputs_root",   "paths.outputs_root",     defaults.outputs_root)

    renderer       = pick_from_cli_or_cfg(args, defaults, cfg, "renderer",       "data.renderer",          None)

    train_dir_raw  = pick_from_cli_or_cfg(args, defaults, cfg, "train_dir",      "data.train_dir",         None)
    test_dir_raw   = pick_from_cli_or_cfg(args, defaults, cfg, "test_dir",       "data.test_dir",          None)
    data_root      = pick_from_cli_or_cfg(args, defaults, cfg, "data_root",      "data.data_root",         None)
    train_ratio    = float(pick_from_cli_or_cfg(args, defaults, cfg, "train_ratio","data.train_ratio",     defaults.train_ratio))

    rules_file     = pick_from_cli_or_cfg(args, defaults, cfg, "rules_file",     "prompts.rules_file",     None)
    image_glob     = pick_from_cli_or_cfg(args, defaults, cfg, "image_glob",     "data.image_glob",        defaults.image_glob)
    label_filename = pick_from_cli_or_cfg(args, defaults, cfg, "label_filename", "data.label_filename",    defaults.label_filename)
    image_size     = int(pick_from_cli_or_cfg(args, defaults, cfg, "image_size", "data.image_size",        defaults.image_size))

    seed           = int(pick_from_cli_or_cfg(args, defaults, cfg, "seed",       "train.seed",             defaults.seed))
    allow_tf32     = bool(pick_from_cli_or_cfg(args, defaults, cfg, "allow_tf32","train.allow_tf32",       defaults.allow_tf32))

    lora_r         = int(pick_from_cli_or_cfg(args, defaults, cfg, "lora_r",     "lora.r",                 defaults.lora_r))
    lora_alpha     = int(pick_from_cli_or_cfg(args, defaults, cfg, "lora_alpha", "lora.alpha",             defaults.lora_alpha))
    lora_dropout   = float(pick_from_cli_or_cfg(args, defaults, cfg, "lora_dropout","lora.dropout",        defaults.lora_dropout))

    epochs         = int(pick_from_cli_or_cfg(args, defaults, cfg, "epochs",     "train.epochs",           defaults.epochs))
    batch_size     = int(pick_from_cli_or_cfg(args, defaults, cfg, "batch_size", "train.batch_size",       defaults.batch_size))
    grad_accum     = int(pick_from_cli_or_cfg(args, defaults, cfg, "grad_accum", "train.grad_accum",       defaults.grad_accum))
    lr             = float(pick_from_cli_or_cfg(args, defaults, cfg, "lr",       "train.lr",               defaults.lr))
    weight_decay   = float(pick_from_cli_or_cfg(args, defaults, cfg, "weight_decay","train.weight_decay",  defaults.weight_decay))
    warmup_ratio   = float(pick_from_cli_or_cfg(args, defaults, cfg, "warmup_ratio","train.warmup_ratio",  defaults.warmup_ratio))
    scheduler      = pick_from_cli_or_cfg(args, defaults, cfg, "scheduler",      "train.scheduler",        defaults.scheduler)
    logging_steps  = int(pick_from_cli_or_cfg(args, defaults, cfg, "logging_steps","train.logging_steps",  defaults.logging_steps))
    save_strategy  = pick_from_cli_or_cfg(args, defaults, cfg, "save_strategy",  "train.save_strategy",    defaults.save_strategy)
    save_steps     = int(pick_from_cli_or_cfg(args, defaults, cfg, "save_steps", "train.save_steps",       defaults.save_steps))
    save_total_limit = pick_from_cli_or_cfg(args, defaults, cfg, "save_total_limit","train.save_total_limit", defaults.save_total_limit)
    no_eval_during_train = bool(pick_from_cli_or_cfg(args, defaults, cfg, "no_eval_during_train","train.no_eval_during_train", defaults.no_eval_during_train))

    bf16          = bool(pick_from_cli_or_cfg(args, defaults, cfg, "bf16",       "train.bf16",             defaults.bf16))
    fp16          = bool(pick_from_cli_or_cfg(args, defaults, cfg, "fp16",       "train.fp16",             defaults.fp16))
    grad_ckpt     = bool(pick_from_cli_or_cfg(args, defaults, cfg, "grad_ckpt",  "train.grad_ckpt",        defaults.grad_ckpt))

    # --- 读取/准备规则文本 ---
    rules = Path(rules_file).read_text(encoding="utf-8") if rules_file else _DEFAULT_RULES

    # --- 运行命名与输出目录 ---
    run_name = gen_run_name("train")
    paths = build_output_paths(root=outputs_root, with_run=False)

    # --- 随机性与后端 ---
    set_global_seed(seed, deterministic=False)
    set_torch_backends(allow_tf32=allow_tf32, cudnn_benchmark=True, cudnn_deterministic=False)

    # --- 加载模型与处理器 ---
    print(f"[Info] Loading model: {model_id}")
    model, processor = load_model_and_processor(
        model_id=model_id,
        trust_remote_code=True,
        use_fast_tokenizer=False,
    )

    # --- 注入 LoRA ---
    lora_params = LoRAParams(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None,
        modules_to_save=None,
    )
    model = attach_lora(model, lora_params, verbose=True)

    # --- 构建数据集（显式 or 单目录切分） ---
    ds_cfg = DatasetBuildConfig(
        rules=rules,
        image_glob=image_glob,
        label_filename=label_filename,
        image_size=image_size,
        seed=seed,
        supervision=None,
    )

    # 处理 renderer 子目录：若用户给的是 train/test 上层目录，则自动追加 /<renderer>
    # （若原目录已存在，则不追加；若不存在而 <dir>/<renderer> 存在，则用后者）
    train_dir = resolve_with_renderer(train_dir_raw, renderer) if train_dir_raw else None
    test_dir  = resolve_with_renderer(test_dir_raw,  renderer) if test_dir_raw  else None

    if train_dir and test_dir:
        if (train_dir_raw and train_dir_raw != train_dir) or (test_dir_raw and test_dir_raw != test_dir):
            print(f"[Info] Using renderer subdir -> train: {train_dir} ; test: {test_dir}")
        train_hf, test_hf = build_hfds_from_explicit(
            processor=processor,
            train_root=train_dir,
            test_root=test_dir,
            cfg=ds_cfg,
        )
    elif data_root:
        # 单根目录模式：若 data_root 是上层（如 data/images/FD001），你应当确保其下直接有 sample_*.png/y.npy；
        # 如果你的目录结构是 data/images/FD001/train/<renderer> 和 test/<renderer>，
        # 建议显式使用 --train-dir/--test-dir；单根目录模式不做递归搜寻以避免歧义。
        if renderer:
            print("[Warn] 单目录切分模式不会自动附加 renderer 子目录；若你的数据在 train/<renderer>/test/<renderer>，"
                  "请使用 --train-dir / --test-dir 显式指定。")
        train_hf, test_hf = build_hfds_from_single_root(
            processor=processor,
            root=data_root,
            cfg=ds_cfg,
            train_ratio=float(train_ratio),
        )
    else:
        raise ValueError("请提供 (--train-dir 与 --test-dir) 或 (--data-root) 之一。")

    # --- 简短 sanity check：统计前 3 个样本的“被监督 token 数量” ---
    try:
        cnts = preview_supervised_token_counts(train_hf, processor.tokenizer, k=3)
        for i, c in enumerate(cnts, 1):
            print(f"[Debug] sample {i}: supervised_token_count = {c}")
    except Exception as e:
        print(f"[Warn] 统计被监督 token 数量失败（可忽略）：{e}")

    # --- Data Collator ---
    collator = VLSFTDataCollator(processor.tokenizer)

    # --- TrainingArguments ---
    tr_cfg = TrainConfig(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        do_eval=(not no_eval_during_train),
        eval_strategy="no" if no_eval_during_train else "epoch",
        bf16=bf16,
        fp16=(False if bf16 else fp16),
        gradient_checkpointing=grad_ckpt,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        report_to="none",
    )
    ckpt_dir = paths.checkpoints / f"vl_lora_{run_name}"
    args_hf = build_training_arguments(output_dir=ckpt_dir, cfg=tr_cfg)

    # --- Trainer ---
    trainer = build_trainer(
        model=model,
        args=args_hf,
        train_dataset=train_hf,
        eval_dataset=(test_hf if tr_cfg.do_eval else None),
        data_collator=collator,
        compute_metrics=None,  # 训练阶段不计算指标
        print_param_stats=True,
    )

    # --- 开始训练 ---
    print("[Train] Start SFT...")
    trainer.train()
    print("[Train] Finished.")

    # --- 保存 LoRA 与 Processor ---
    lora_out = (paths.lora / run_name).as_posix()
    proc_out = (paths.processor / run_name).as_posix()
    save_lora_and_processor(trainer, lora_output_dir=lora_out,
                            processor=processor, processor_output_dir=proc_out)
    print(f"[Save] LoRA adapter saved to: {lora_out}")
    print(f"[Save] Processor saved to:   {proc_out}")


if __name__ == "__main__":
    main()
