# src/mmts/cli/imgify.py
# -*- coding: utf-8 -*-
"""
图像化入口（Command Line Interface - imgify）

- 支持从 YAML 读取 data.imgify.* 配置；
- 当未显式指定 (--x-npy/--y-npy/--out-dir) 时，会尝试在配置中读取
  data.imgify.train 与 data.imgify.test，并分别生成；
- 单次生成的输出路径为 out_dir/<renderer>/...，避免不同渲染器互相覆盖。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..imaging.base import ScaleSpec, create_renderer
from ..configs.loader import load_config_tree


# ---------------- 工具：配置读取与优先级合并 ----------------
def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def pick_from_cli_or_cfg(args, defaults, cfg: Dict[str, Any], arg_name: str, cfg_key: str, fallback: Any):
    cli_val = getattr(args, arg_name)
    def_val = getattr(defaults, arg_name)
    if cli_val != def_val:
        return cli_val
    cfg_val = cfg_get(cfg, cfg_key, None)
    if cfg_val is not None:
        return cfg_val
    return fallback


# ---------------- CLI ----------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render time-series windows (X.npy) into images for VL fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="YAML 配置入口（如 configs/defaults.yaml）")

    # 单次生成（若三者都不传，则自动读取 YAML 的 train/test 两份）
    p.add_argument("--x-npy", type=str, default=None, help="输入窗口数组 X.npy（N,T,F）")
    p.add_argument("--y-npy", type=str, default=None, help="输入标签数组 y.npy（N,）")
    p.add_argument("--out-dir", type=str, default=None, help="输出上层目录（内部会再建 <renderer>/）")
    p.add_argument("--max-samples", type=int, default=None, help="导出样本上限；缺省=全量")

    # 渲染公共参数
    p.add_argument("--renderer", type=str, default="grayscale")
    p.add_argument("--orientation", type=str, default="row-major", choices=["row-major", "col-major"])
    p.add_argument("--flip-vertical", action="store_true", default=False)
    p.add_argument("--flip-horizontal", action="store_true", default=False)

    # 缩放
    p.add_argument("--scale-mode", type=str, default="percentile", choices=["global", "percentile", "sample"])
    p.add_argument("--pmin", type=float, default=1.0)
    p.add_argument("--pmax", type=float, default=99.0)
    return p


# ---------------- 生成一个拆分（train 或 test）的核心函数 ----------------
def _run_one_split(
    x_path: Path,
    y_path: Path,
    out_dir_root: Path,
    renderer: str,
    orientation: str,
    flip_v: bool,
    flip_h: bool,
    scale_mode: str,
    pmin: float,
    pmax: float,
    max_samples: Optional[int],
):
    out_dir = out_dir_root / renderer.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Imgify] Renderer={renderer}  -> Out={out_dir}")
    print(f"[Imgify] Loading: X={x_path} ; y={y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 3:
        raise ValueError(f"X 应为 (N,T,F)，当前 {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y 应为 (N,)，当前 {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y 数量不一致：{X.shape[0]} vs {y.shape[0]}")

    N = X.shape[0]
    n_export = min(max_samples if max_samples is not None else N, N)
    print(f"[Imgify] Will export {n_export}/{N} samples")

    scaler = ScaleSpec(mode=scale_mode, pmin=float(pmin), pmax=float(pmax))
    if scaler.mode in ("global", "percentile"):
        print(f"[Imgify] Fitting global scale ({scaler.mode}) ...")
        scaler.fit_global(X)

    renderer_obj = create_renderer(
        renderer,
        orientation=orientation,
        flip_vertical=bool(flip_v),
        flip_horizontal=bool(flip_h),
    )

    for i in tqdm(range(n_export), desc="Imgify", ncols=100):
        img = renderer_obj.render(X[i], scaler=scaler)
        img.save(out_dir / f"sample_{i:05d}.png")

    np.save(out_dir / "y.npy", y[:n_export])
    print(f"[Imgify] Saved y.npy -> {out_dir/'y.npy'}")


# ---------------- 主流程 ----------------
def main():
    parser = build_argparser()
    args = parser.parse_args()
    defaults = parser.parse_args([])

    cfg = load_config_tree(args.config) if args.config else {}

    # 合并公共渲染与缩放参数
    renderer    = pick_from_cli_or_cfg(args, defaults, cfg, "renderer",       "data.renderer",       defaults.renderer)
    orientation = pick_from_cli_or_cfg(args, defaults, cfg, "orientation",    "data.imgify.orientation",    defaults.orientation)
    flip_v      = bool(pick_from_cli_or_cfg(args, defaults, cfg, "flip_vertical",   "data.imgify.flip_vertical",   defaults.flip_vertical))
    flip_h      = bool(pick_from_cli_or_cfg(args, defaults, cfg, "flip_horizontal", "data.imgify.flip_horizontal", defaults.flip_horizontal))
    scale_mode  = pick_from_cli_or_cfg(args, defaults, cfg, "scale_mode",     "data.imgify.scale_mode",     defaults.scale_mode)
    pmin        = float(pick_from_cli_or_cfg(args, defaults, cfg, "pmin",     "data.imgify.pmin",           defaults.pmin))
    pmax        = float(pick_from_cli_or_cfg(args, defaults, cfg, "pmax",     "data.imgify.pmax",           defaults.pmax))

    # 如果显式给了 x/y/out_dir，就只生成一份
    if args.x_npy and args.y_npy and args.out_dir:
        x_path = Path(args.x_npy)
        y_path = Path(args.y_npy)
        out_dir_root = Path(args.out_dir)
        _run_one_split(
            x_path, y_path, out_dir_root,
            renderer, orientation, args.flip_vertical, args.flip_horizontal,
            scale_mode, pmin, pmax, args.max_samples,
        )
        print("[Imgify] Done (single split).")
        return

    # 否则：从配置读取 train/test 的路径；能读到哪个就生成哪个
    splits = []
    for split_name in ("train", "test"):
        base = f"data.imgify.{split_name}"
        x_npy = cfg_get(cfg, f"{base}.x_npy")
        y_npy = cfg_get(cfg, f"{base}.y_npy")
        out_dir = cfg_get(cfg, f"{base}.out_dir")
        max_samples = cfg_get(cfg, f"{base}.max_samples", None)
        if x_npy and y_npy and out_dir:
            splits.append((
                Path(str(x_npy)),
                Path(str(y_npy)),
                Path(str(out_dir)),
                max_samples,
                split_name,
            ))

    if not splits:
        raise ValueError(
            "未提供 (--x-npy/--y-npy/--out-dir)，同时配置文件中也没有 data.imgify.train/test 的三元组。"
        )

    for x_path, y_path, out_dir_root, max_samples, split_name in splits:
        print(f"\n[Imgify] === Split: {split_name} ===")
        _run_one_split(
            x_path, y_path, out_dir_root,
            renderer, orientation, args.flip_vertical, args.flip_horizontal,
            scale_mode, pmin, pmax, max_samples,
        )

    print("\n[Imgify] Done (multi-split).")


if __name__ == "__main__":
    main()
