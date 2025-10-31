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
    experiments_root = _cfg_get(cfg, "paths.experiments_root", "experiments")

    renderer       = _cfg_get(cfg, "data.renderer", None)
    train_dir_raw  = _cfg_get(cfg, "data.train_dir", None)
    test_dir_raw   = _cfg_get(cfg, "data.test_dir", None)
    data_root      = _cfg_get(cfg, "data.data_root", None)
    train_ratio    = float(_cfg_get(cfg, "data.train_ratio", 0.8))
    
    # 样本数量限制（从图像化配置中读取）
    train_max_samples = _cfg_get(cfg, "data.imgify.train.max_samples", None)
    test_max_samples = _cfg_get(cfg, "data.imgify.test.max_samples", None)

    rules_file     = _cfg_get(cfg, "prompts.rules_file", None)
    image_glob     = _cfg_get(cfg, "data.image_glob", "sample_*.png")
    label_filename = _cfg_get(cfg, "data.label_filename", "y.npy")
    image_size     = int(_cfg_get(cfg, "data.image_size", 448))

    seed           = int(_cfg_get(cfg, "train.seed", 42))
    allow_tf32     = bool(_cfg_get(cfg, "train.allow_tf32", True))

    lora_r         = int(_cfg_get(cfg, "lora.r", 16))
    lora_alpha     = int(_cfg_get(cfg, "lora.alpha", 32))
    lora_dropout   = float(_cfg_get(cfg, "lora.dropout", 0.05))
    lora_target_modules = _cfg_get(cfg, "lora.target_modules", None)
    lora_modules_to_save = _cfg_get(cfg, "lora.modules_to_save", None)

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

    # ---- 创建实验管理器 ----
    from mmts.utils.experiment import get_experiment_manager
    exp_manager = get_experiment_manager(experiments_root)
    
    # 检查是否指定了实验ID
    experiment_id = _cfg_get(cfg, "experiment_id", None)
    
    if experiment_id:
        # 使用指定的实验ID创建实验
        exp_name = f"vl_lora_train"
        exp_description = f"训练LoRA模型 - epochs={epochs}, lr={lr}, batch_size={batch_size}"
        exp_info, exp_paths = exp_manager.create_experiment(
            name=exp_name,
            config=cfg,
            description=exp_description,
            experiment_id=experiment_id
        )
    else:
        # 自动生成实验ID
        exp_name = f"vl_lora_train"
        exp_description = f"训练LoRA模型 - epochs={epochs}, lr={lr}, batch_size={batch_size}"
        exp_info, exp_paths = exp_manager.create_experiment(
            name=exp_name,
            config=cfg,
            description=exp_description
        )
    
    print(f"[Experiment] 创建实验: {exp_info.experiment_id}")
    print(f"[Experiment] 实验目录: {exp_paths.root}")

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
    # PEFT会自动处理参数冻结：
    # - target_modules中的模块：会用LoRA包装，LoRA参数可训练，原始参数自动冻结
    # - modules_to_save中的模块：完整保存和训练（不使用LoRA），参数可训练
    # - 其他所有参数：自动冻结
    lora_params = LoRAParams(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
        modules_to_save=lora_modules_to_save,
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

    # ---- 应用样本数量限制 ----
    if train_max_samples is not None and len(train_hf) > train_max_samples:
        print(f"[Info] 限制训练样本数量: {len(train_hf)} -> {train_max_samples}")
        train_hf = train_hf.select(range(train_max_samples))
    
    if test_max_samples is not None and len(test_hf) > test_max_samples:
        print(f"[Info] 限制测试样本数量: {len(test_hf)} -> {test_max_samples}")
        test_hf = test_hf.select(range(test_max_samples))

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
    # 使用实验目录作为checkpoint目录
    ckpt_dir = exp_paths.checkpoints
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
    try:
        trainer.train()
        print("[Train] Finished.")
        
        # 更新实验状态为完成
        exp_manager.update_experiment_status(exp_info.experiment_id, "completed")
        
    except Exception as e:
        print(f"[Train] Failed: {e}")
        exp_manager.update_experiment_status(exp_info.experiment_id, "failed")
        raise

    # ---- 保存统一模型（LoRA + Processor） ----
    from mmts.utils.io import save_unified_model
    model_output_dir = save_unified_model(
        trainer=trainer,
        processor=processor,
        output_dir=exp_paths.models,
        save_checkpoint=True
    )
    print(f"[Save] 统一模型已保存到: {model_output_dir}")


if __name__ == "__main__":
    main()
