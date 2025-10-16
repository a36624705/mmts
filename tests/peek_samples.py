# tests/peek_samples.py
# 使用 MMTS API 进行样本推理的最小测试脚本，仅在终端输出结果，不生成任何文件。

from __future__ import annotations
from pathlib import Path
import os
import random
import sys
import argparse

# === 默认仅使用第 0 张 GPU（可用环境变量覆盖） ===
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[Device] 默认使用 GPU:0")

import mmts as M

# ===== 默认参数（可以改这里；也可用环境变量覆盖） =====
MODEL_ID     = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
OUTPUTS_ROOT = Path(os.getenv("OUTPUTS_ROOT", "outputs"))
TEST_DIR     = Path(os.getenv("TEST_DIR", "data/images/FD001/test/grayscale"))
IMAGE_GLOB   = os.getenv("IMAGE_GLOB", "sample_*.png")
IMAGE_SIZE   = int(os.getenv("IMAGE_SIZE", "448"))
RANDOM_K     = int(os.getenv("RANDOM_K", "3"))  # 随机挑几张；可改

def latest_subdir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    dirs = [p for p in parent.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]

def parse_args():
    ap = argparse.ArgumentParser(
        description="Peek a few samples with raw/json output (no files)."
    )
    ap.add_argument(
        "--no-train",
        action="store_true",
        default=os.getenv("PEEK_NO_TRAIN", "0") == "1",
        help="不加载 LoRA，仅使用底座预训练模型进行推理（等价于只看未微调效果）",
    )
    ap.add_argument(
        "--random-k", type=int, default=RANDOM_K, help="随机挑选样本数（默认来自 RANDOM_K 或 3）"
    )
    ap.add_argument(
        "--image-size", type=int, default=IMAGE_SIZE, help="推理图像边长像素限制（默认来自 IMAGE_SIZE 或 448）"
    )
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) 发现最新 LoRA / Processor（若 --no-train 则跳过 LoRA）
    lora_dir = None if args.no_train else latest_subdir(OUTPUTS_ROOT / "lora")
    proc_dir = latest_subdir(OUTPUTS_ROOT / "processor")

    if args.no_train:
        print("[Info] --no-train：跳过加载 LoRA，使用底座模型推理。")
    else:
        if lora_dir is None:
            print(f"[Error] 没找到 LoRA 目录：{(OUTPUTS_ROOT / 'lora').as_posix()}", file=sys.stderr)
            print("        请先训练/导出 LoRA，或使用 --no-train 只看底座模型。", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"[Load] LoRA (latest): {lora_dir.as_posix()}")

    # 2) 加载基础模型与 Processor；若有保存的 Processor 就用之
    print(f"[Load] Base model: {MODEL_ID}")
    base_model, proc_from_model = M.load_model_and_processor(
        model_id=MODEL_ID,
        trust_remote_code=True,
        use_fast_tokenizer=False,
    )
    if proc_dir is not None:
        try:
            from mmts.utils.io import load_processor
        except Exception as e:
            print(f"[Warn] 无法加载已保存的 Processor（{e!r}），改用模型自带 Processor。")
            processor = proc_from_model
        else:
            print(f"[Load] Processor: {proc_dir.as_posix()}")
            processor = load_processor(proc_dir.as_posix())
    else:
        print("[Load] Using processor from base model.")
        processor = proc_from_model

    # 3) 如需 LoRA 则注入；否则直接用底座模型
    if lora_dir is not None:
        try:
            from peft import PeftModel
        except Exception as e:
            print("[Error] 需要安装 peft：pip install -U peft", file=sys.stderr)
            print(f"        原始错误：{e!r}", file=sys.stderr)
            sys.exit(1)
        model = PeftModel.from_pretrained(base_model, lora_dir.as_posix())
    else:
        model = base_model

    # 4) 选样本
    if not TEST_DIR.exists():
        print(f"[Error] 测试目录不存在：{TEST_DIR.as_posix()}", file=sys.stderr)
        sys.exit(1)
    cands = sorted(TEST_DIR.glob(IMAGE_GLOB))
    if not cands:
        print(f"[Error] 未找到任何样本（目录：{TEST_DIR}；glob：{IMAGE_GLOB}）", file=sys.stderr)
        sys.exit(1)

    k = min(max(args.random_k, 1), len(cands))
    random.seed(42)
    picks = random.sample(cands, k=k)

    # 5) 推理并打印（纯命令行输出）
    gen = M.GenConfig(image_size=args.image_size, max_new_tokens=32, do_sample=False, num_beams=1)

    print(f"[Peek] Samples: {k} / {len(cands)} | image_size={args.image_size} | no_train={args.no_train}")
    for i, p in enumerate(picks):
        res = M.generate_for_image(
            model=model,
            processor=processor,
            image=p,
            rules=_load_rules(),
            gen=gen,
            setup_infer=True,
        )
        raw = (res.raw_text or "").strip().replace("\n", " ")[:180]
        js  = (res.json_text or "").strip().replace("\n", " ")[:180]
        print(f"[#{i:02d}] path={p.as_posix()}")
        print("  raw :", raw)
        print("  json:", js)

def _load_rules() -> str:
    # 读取包内默认规则（与 eval 一致）
    from importlib import resources
    return resources.files("mmts.prompts").joinpath("base_rules.txt").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
