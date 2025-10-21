# -*- coding: utf-8 -*-
"""将时序窗口转为按时间步顺序的 GASF 小块，并拼接为整图。

输入样本约定：sample 为 (T, F) 矩阵（时间步 T 在行，特征 F 在列）。

核心参数：
- grid_rows/grid_cols: 小块网格大小（行优先填充）
- patch_window/patch_stride/patch_overlap/patch_align_steps: 小块提取策略
- cmap: 可选伪彩上色（若为 None 则输出灰度）

注意：缩放由 ScaleSpec 负责，仅用于外层 sample；GASF 小块内部自归一化（-1..1）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from ..base import ImageRenderer, ScaleSpec, register_renderer


def _get_gasf(feature_vector: np.ndarray) -> np.ndarray:
    """计算 1D 向量的 GASF（返回形状: (F,F)，范围约 -1..1）。"""
    min_val = float(np.min(feature_vector))
    max_val = float(np.max(feature_vector))
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return np.zeros((feature_vector.shape[0], feature_vector.shape[0]), dtype=np.float32)
    scaled = 2.0 * (feature_vector - min_val) / (max_val - min_val + 1e-8) - 1.0
    phi = np.arccos(np.clip(scaled, -1.0, 1.0))
    gasf = np.cos(phi + phi[:, None])
    return gasf.astype(np.float32)


def _generate_patches(
    sample_tf: np.ndarray,
    *,
    patch_window: int,
    patch_stride: int,
    patch_overlap: int,
    patch_align_steps: bool,
) -> List[np.ndarray]:
    """基于 (T,F) 矩阵提取按时间步顺序的小块（每块为 F×F 的 GASF）。"""
    T, F = sample_tf.shape
    win = int(max(1, patch_window))
    over = int(max(0, patch_overlap))
    over = int(min(over, max(0, win - 1)))
    stride = int(max(1, win - over)) if over > 0 else int(max(1, patch_stride))

    if patch_align_steps:
        starts = list(range(0, T))
    else:
        if T < win:
            starts = [0]
        else:
            starts = list(range(0, max(T - win + 1, 1), stride))

    patches: List[np.ndarray] = []
    for s in starts:
        end = min(s + win, T) if patch_align_steps else s + (T if T < win else win)
        window_slice = sample_tf[s:end, :]
        # 将窗口内多个时间步的特征聚合为 1D（与 vis_window_step_mat.py 保持一致，按特征维求均值）
        feature_vector = window_slice.mean(axis=0)
        patches.append(_get_gasf(feature_vector))
    return patches


def _compose_grid(
    patches: List[np.ndarray],
    *,
    grid_rows: int,
    grid_cols: int,
    scale: int,
) -> np.ndarray:
    """将 F×F 小块按行优先填充至 (grid_rows*F, grid_cols*F) 的大矩阵。"""
    if not patches:
        raise ValueError("No patches to compose")
    needed = int(grid_rows) * int(grid_cols)
    if len(patches) < needed:
        patches = patches + [patches[-1]] * (needed - len(patches))
    elif len(patches) > needed:
        patches = patches[:needed]

    arr = np.asarray(patches, dtype=np.float32)  # (N, F, F)
    F = arr.shape[1]
    grid = arr.reshape(int(grid_rows), int(grid_cols), F, F)
    grid_swapped = grid.swapaxes(1, 2)
    img = grid_swapped.reshape(int(grid_rows) * F, int(grid_cols) * F)
    if scale > 1:
        img = np.kron(img, np.ones((scale, scale), dtype=np.float32))
    return img


@register_renderer
class GasfGridRenderer(ImageRenderer):
    """GASF 网格渲染器：将 (T,F) → 若干 F×F 小块 → 网格拼图。

    参数：
    - grid_rows: 网格行数
    - grid_cols: 网格列数
    - patch_window: 时间窗口长度
    - patch_stride: 窗口步长（当 patch_overlap>0 时被忽略）
    - patch_overlap: 窗口重叠（0..patch_window-1）
    - patch_align_steps: 是否一时刻一块（末尾允许半窗），保证块数≈T
    - scale: 整体最近邻放大倍数（整数）
    - cmap: 伪彩名称（None/"" 为灰度），例如 "viridis"、"plasma" 等
    """
    name: str = "gasf"

    def __init__(
        self,
        *,
        grid_rows: int = 8,
        grid_cols: int = 8,
        patch_window: int = 20,
        patch_stride: int = 1,
        patch_overlap: int = 0,
        patch_align_steps: bool = True,
        scale: int = 1,
        cmap: Optional[str] = "rainbow",
        # 为兼容 CLI 的通用参数，这里接受但不使用：
        orientation: Optional[str] = None,
        flip_vertical: Optional[bool] = None,
        flip_horizontal: Optional[bool] = None,
    ) -> None:
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.patch_window = int(patch_window)
        self.patch_stride = int(patch_stride)
        self.patch_overlap = int(patch_overlap)
        self.patch_align_steps = bool(patch_align_steps)
        self.scale = int(max(1, scale))
        self.cmap = (str(cmap).strip() or None) if cmap is not None else None

        o = (orientation or "row-major").strip().lower()
        if o not in {"row-major", "col-major"}:
            raise ValueError("orientation 必须为 'row-major' 或 'col-major'")
        self.orientation = o
        self.flip_vertical = bool(flip_vertical) if flip_vertical is not None else False
        self.flip_horizontal = bool(flip_horizontal) if flip_horizontal is not None else False

    def render(self, sample: np.ndarray, *, scaler: ScaleSpec) -> Image.Image:
        # ScaleSpec 在此无需改变时序结构，仅保证输入合法，做一次装配即可
        sx = np.asarray(sample, dtype=np.float32)
        if sx.ndim != 2:
            raise ValueError(f"Expect (T,F), got {sx.shape}")
        # 框架约定 sample 为 (T,F)；若上游给了 (F,T) 可在外层通过 orientation=col-major 处理。
        if self.orientation == "col-major":
            sx = sx.T
        if self.flip_vertical:
            sx = np.flipud(sx)
        if self.flip_horizontal:
            sx = np.fliplr(sx)

        patches = _generate_patches(
            sx,
            patch_window=self.patch_window,
            patch_stride=self.patch_stride,
            patch_overlap=self.patch_overlap,
            patch_align_steps=self.patch_align_steps,
        )
        img_mat = _compose_grid(
            patches,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            scale=self.scale,
        )

        # 将 -1..1 映射到 [0,255] 以构造图像；若指定 cmap，则转为伪彩 RGB
        img01 = (img_mat - (-1.0)) / (2.0)  # 线性映射 -1..1 → 0..1
        img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)
        arr_u8 = (img01 * 255.0 + 0.5).astype("uint8")
        if not self.cmap:
            return Image.fromarray(arr_u8, mode="L")

        # 简单伪彩：使用 matplotlib colormap（仅在需要时导入）
        import matplotlib.cm as cm
        cmap = cm.get_cmap(self.cmap)
        rgb = (cmap(img01)[..., :3] * 255.0 + 0.5).astype("uint8")
        return Image.fromarray(rgb, mode="RGB")
