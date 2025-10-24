# src/mmts/cli/train.py
# -*- coding: utf-8 -*-
"""
训练入口（Hydra 版）

用法（示例）
-----------
python -m mmts.cli.train \
  model.model_id=Qwen/Qwen2.5-VL-3B-Instruct \
  data.train_dir=/path/to/train \
  data.test_dir=/path/to/test \
  prompts.rules_file=configs/prompts/base_rules.txt

说明
----
- 配置从项目根目录的 `configs/` 读取（见 @hydra.main 的 config_path）。
- 一切键都可用 CLI 覆盖，例如：train.epochs=5 data.image_size=448
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

# ==== 项目内模块 ====
from ..utils.seed import set_global_seed, set_torch_backends
from ..utils.io import build_output_paths, gen_run_name
from ..core.loader import load_model_and_processor
from ..core.lora import LoRAParams, attach_lora
from ..data.builders import (
    DatasetBuildConfig,
    build_hfds_from_explicit,
    build_hfds_from_single_root,
    preview_supervised_token_counts,
)
from ..data.collators import VLSFTDataCollator
from ..training.arguments import TrainConfig, build_training_arguments
from ..training.trainer import build_trainer, save_lora_and_processor


# -------------------------
# 小工具
# -------------------------
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
    """优先读配置指定的规则文件；否则读取项目根目录的 configs/base_rules.txt。"""
    if rules_file:
        return Path(rules_file).read_text(encoding="utf-8")
    # 项目根目录默认
    try:
        # 获取项目根目录（当前工作目录的父目录）
        project_root = Path(__file__).parent.parent.parent.parent
        default_rules = project_root / "configs" / "base_rules.txt"
        return default_rules.read_text(encoding="utf-8")
    except Exception:
        raise FileNotFoundError(
            "未能读取规则文本。请在配置中设置 prompts.rules_file，或确保项目根目录存在 configs/base_rules.txt。"
        )


# -------------------------
# 训练主逻辑（Hydra）
# -------------------------
@hydra.main(config_path="../../../configs", config_name="defaults", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 可打印最终配置（调试用）：print(OmegaConf.to_yaml(cfg))
    # ---- 读取关键配置 ----
    model_id       = _cfg_get(cfg, "model.model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    outputs_root   = _cfg_get(cfg, "paths.outputs_root", "outputs")

    renderer       = _cfg_get(cfg, "data.renderer", None)
    train_dir_raw  = _cfg_get(cfg, "data.train_dir", None)
    test_dir_raw   = _cfg_get(cfg, "data.test_dir", None)
    data_root      = _cfg_get(cfg, "data.data_root", None)
    train_ratio    = float(_cfg_get(cfg, "data.train_ratio", 0.8))

    rules_file     = _cfg_get(cfg, "prompts.rules_file", None)
    image_glob     = _cfg_get(cfg, "data.image_glob", "sample_*.png")
    label_filename = _cfg_get(cfg, "data.label_filename", "y.npy")
    image_size     = int(_cfg_get(cfg, "data.image_size", 448))

    seed           = int(_cfg_get(cfg, "train.seed", 42))
    allow_tf32     = bool(_cfg_get(cfg, "train.allow_tf32", True))

    lora_r         = int(_cfg_get(cfg, "lora.r", 16))
    lora_alpha     = int(_cfg_get(cfg, "lora.alpha", 32))
    lora_dropout   = float(_cfg_get(cfg, "lora.dropout", 0.05))

    # 训练超参
    epochs         = int(_cfg_get(cfg, "train.epochs", 10))
    batch_size     = int(_cfg_get(cfg, "train.batch_size", 1))
    grad_accum     = int(_cfg_get(cfg, "train.grad_accum", 8))
    lr             = float(_cfg_get(cfg, "train.lr", 2e-4))
    weight_decay   = float(_cfg_get(cfg, "train.weight_decay", 0.01))
    warmup_ratio   = float(_cfg_get(cfg, "train.warmup_ratio", 0.05))
    scheduler      = _cfg_get(cfg, "train.scheduler", "cosine")
    logging_steps  = int(_cfg_get(cfg, "train.logging_steps", 10))
    save_strategy  = _cfg_get(cfg, "train.save_strategy", "epoch")
    save_steps     = int(_cfg_get(cfg, "train.save_steps", 500))
    save_total_limit = _cfg_get(cfg, "train.save_total_limit", None)
    no_eval_during_train = bool(_cfg_get(cfg, "train.no_eval_during_train", True))

    bf16          = bool(_cfg_get(cfg, "train.bf16", torch.cuda.is_available()))
    fp16          = bool(_cfg_get(cfg, "train.fp16", False))
    grad_ckpt     = bool(_cfg_get(cfg, "train.grad_ckpt", True))

    # ---- 规则文本 ----
    rules = _load_rules_text(rules_file)

    # ---- 输出目录与随机性/后端 ----
    run_name = gen_run_name("train")
    paths = build_output_paths(root=outputs_root, with_run=False)

    set_global_seed(seed, deterministic=False)
    set_torch_backends(allow_tf32=allow_tf32, cudnn_benchmark=True, cudnn_deterministic=False)

    # ---- 加载模型与处理器 ----
    print(f"[Info] Loading model: {model_id}")
    model, processor = load_model_and_processor(
        model_id=model_id,
        trust_remote_code=True,
        use_fast_tokenizer=False,
    )

    # ---- 注入 LoRA ----
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

    # ---- 构建数据集 ----
    ds_cfg = DatasetBuildConfig(
        rules=rules,
        image_glob=image_glob,
        label_filename=label_filename,
        image_size=image_size,
        seed=seed,
        supervision=None,
    )

    train_dir = _resolve_with_renderer(train_dir_raw, renderer) if train_dir_raw else None
    test_dir  = _resolve_with_renderer(test_dir_raw,  renderer) if test_dir_raw  else None

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
        if renderer:
            print("[Warn] 单目录模式不会自动附加 renderer 子目录；若数据在 train/<renderer>/test/<renderer>，"
                  "请改用 data.train_dir / data.test_dir。")
        train_hf, test_hf = build_hfds_from_single_root(
            processor=processor,
            root=data_root,
            cfg=ds_cfg,
            train_ratio=float(train_ratio),
        )
    else:
        raise ValueError("请在配置中提供 (data.train_dir 与 data.test_dir) 或 (data.data_root) 之一。")

    # ---- 简短 sanity check ----
    try:
        cnts = preview_supervised_token_counts(train_hf, processor.tokenizer, k=3)
        for i, c in enumerate(cnts, 1):
            print(f"[Debug] sample {i}: supervised_token_count = {c}")
    except Exception as e:
        print(f"[Warn] 统计被监督 token 数量失败（可忽略）：{e}")

    # ---- Collator / Args / Trainer ----
    collator = VLSFTDataCollator(processor.tokenizer)

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

    trainer = build_trainer(
        model=model,
        args=args_hf,
        train_dataset=train_hf,
        eval_dataset=(test_hf if tr_cfg.do_eval else None),
        data_collator=collator,
        compute_metrics=None,
        print_param_stats=True,
    )

    # ---- 训练 ----
    print("[Train] Start SFT...")
    trainer.train()
    print("[Train] Finished.")

    # ---- 保存 LoRA 与 Processor ----
    lora_out = (paths.lora / run_name).as_posix()
    proc_out = (paths.processor / run_name).as_posix()
    save_lora_and_processor(
        trainer,
        lora_output_dir=lora_out,
        processor=processor,
        processor_output_dir=proc_out,
    )
    print(f"[Save] LoRA adapter saved to: {lora_out}")
    print(f"[Save] Processor saved to:   {proc_out}")


if __name__ == "__main__":
    main()
