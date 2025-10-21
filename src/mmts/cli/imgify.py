# src/mmts/cli/imgify.py
# -*- coding: utf-8 -*-
"""
图像化入口（Hydra 版）

功能
----
- 从根目录 configs/ 读取 data.imgify.* 配置（可用 Hydra CLI 覆盖）；
- 若提供 data.imgify.single.{x_npy,y_npy,out_dir}，则执行单次导出；
- 否则尝试 data.imgify.{train,test} 两套配置，分别导出；
- 输出目录会追加 /<renderer>/ 以避免不同渲染器互相覆盖。

用法示例
--------
# 单次导出
python -m mmts.cli.imgify data.imgify.single.x_npy=data/X.npy \
                          data.imgify.single.y_npy=data/y.npy \
                          data.imgify.single.out_dir=images/FD001 \
                          data.imgify.renderer=grayscale

# 读 configs/imgify.yaml 中的 train/test 两套并导出
python -m mmts.cli.imgify

# 临时覆盖缩放与方向
python -m mmts.cli.imgify data.imgify.scale_mode=percentile data.imgify.pmin=2 data.imgify.pmax=98 \
                          data.imgify.orientation=col-major
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
from hydra import utils as hyutils
from omegaconf import DictConfig

import numpy as np
from tqdm import tqdm

from ..imaging.base import ScaleSpec, create_renderer


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
    patch_window: int,
    patch_stride: int,
    cmap: str
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
        patch_window=patch_window,
        patch_stride=patch_stride,
        cmap=cmap
    )

    for i in tqdm(range(n_export), desc="Imgify", ncols=100):
        img = renderer_obj.render(X[i], scaler=scaler)
        img.save(out_dir / f"sample_{i:05d}.png")

    np.save(out_dir / "y.npy", y[:n_export])
    print(f"[Imgify] Saved y.npy -> {out_dir/'y.npy'}")


def _to_abs(p: Optional[str]) -> Optional[Path]:
    """将（可能是相对的）配置路径，转换为 *原始工作目录* 下的绝对路径。"""
    if p is None:
        return None
    pth = Path(p)
    if pth.is_absolute():
        return pth
    return Path(hyutils.get_original_cwd()) / pth


# ---------------- 主流程（Hydra） ----------------
@hydra.main(config_path='../../../configs', config_name='defaults', version_base='1.3')
def main(cfg: DictConfig) -> None:
    # 读取公共渲染/缩放参数
    dcfg = cfg.get("data", {})
    icfg = dcfg.get("imgify", {})

    renderer     = str(icfg.get("renderer", "grayscale"))
    orientation  = str(icfg.get("orientation", "row-major"))
    flip_v       = bool(icfg.get("flip_vertical", False))
    flip_h       = bool(icfg.get("flip_horizontal", False))
    scale_mode   = str(icfg.get("scale_mode", "percentile"))
    pmin         = float(icfg.get("pmin", 1.0))
    pmax         = float(icfg.get("pmax", 99.0))
    patch_window = int(icfg.get("patch_window", 20))
    patch_stride = int(icfg.get("patch_stride", 1))
    cmap         = str(icfg.get("cmap", "rainbow"))

    # 优先：single 模式（一次性导出）
    single = icfg.get("single", {})
    x_single = _to_abs(single.get("x_npy")) if single else None
    y_single = _to_abs(single.get("y_npy")) if single else None
    out_single = _to_abs(single.get("out_dir")) if single else None
    max_single = single.get("max_samples", None) if single else None

    if x_single and y_single and out_single:
        _run_one_split(
            x_single, y_single, out_single,
            renderer, orientation, flip_v, flip_h,
            scale_mode, pmin, pmax, max_single, patch_window, patch_stride, cmap
        )
        print("[Imgify] Done (single split).")
        return

    # 否则：尝试 train/test 两套配置
    splits_conf = []
    for split in ("train", "test"):
        sc = icfg.get(split, {})
        x_npy = _to_abs(sc.get("x_npy"))
        y_npy = _to_abs(sc.get("y_npy"))
        out_dir = _to_abs(sc.get("out_dir"))
        max_samples = sc.get("max_samples", None)
        if x_npy and y_npy and out_dir:
            splits_conf.append((split, x_npy, y_npy, out_dir, max_samples))

    if not splits_conf:
        raise ValueError(
            "未检测到导出配置：请在 configs/imgify.yaml 中设置 "
            "data.imgify.single.{x_npy,y_npy,out_dir} 或 data.imgify.{train,test}.{x_npy,y_npy,out_dir}。"
        )

    for split_name, x_path, y_path, out_dir_root, max_samples in splits_conf:
        print(f"\n[Imgify] === Split: {split_name} ===")
        _run_one_split(
            x_path, y_path, out_dir_root,
            renderer, orientation, flip_v, flip_h,
            scale_mode, pmin, pmax, max_samples, patch_window, patch_stride, cmap
        )

    print("\n[Imgify] Done (multi-split).")


if __name__ == "__main__":
    main()
