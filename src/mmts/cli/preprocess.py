#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CMAPSS default preprocessing pipeline (Hydra-enabled).

Reads config from configs/default.yaml under:
  data:
    name: FD001
    preprocess:
      raw_dir: data/raw
      out_dir: data/processed
      rearley: 125
      window_size: auto      # or integer
      step: 1
      test_last_window: true
      scale_mode: minmax     # or none
      select14: false

Outputs:
  data/processed/<FD>/
    X_train.npy, y_train.npy, X_test.npy, y_test.npy
    min.npy, max.npy
    feature_cols.json
    meta.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd


# --------------------------
# Defaults & helpers
# --------------------------
ALL_FDS = ["FD001", "FD002", "FD003", "FD004"]

# Default window sizes (common practice)
DEFAULT_WINDOWS = {
    "FD001": 30,
    "FD002": 20,
    "FD003": 30,
    "FD004": 15,
}

# 14 commonly-used sensors (1-based indices in CMAPSS)
SENSOR_14_IDX_1BASED = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def load_cmapss_file(path: Path) -> pd.DataFrame:
    """Load CMAPSS space-separated file."""
    col_names = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=col_names)
    return df


def compute_rul_train(df_train: pd.DataFrame, rearley: int) -> pd.Series:
    """Train RUL = min(max_cycle - cycle, Rearly)."""
    max_cycle = df_train.groupby("unit")["cycle"].max().rename("max_cycle")
    tmp = df_train.merge(max_cycle, left_on="unit", right_index=True)
    rul_linear = tmp["max_cycle"] - tmp["cycle"]
    rul = np.minimum(rul_linear.values, rearley)
    return pd.Series(rul, index=df_train.index, name="RUL")


def compute_rul_test(df_test: pd.DataFrame, rul_path: Path, rearley: int) -> pd.Series:
    """Test RUL per row using RUL file (ordered by unit appearance).
       For unit u with last_cycle L and provided RUL_at_end r:
         RUL_linear(t) = r + (L - t); RUL = min(RUL_linear, Rearly)
    """
    rul_vec = pd.read_csv(rul_path, header=None).iloc[:, 0].values
    order = df_test[["unit"]].drop_duplicates().reset_index(drop=True)
    if len(rul_vec) < len(order):
        raise ValueError(
            f"RUL file has {len(rul_vec)} rows but test has {len(order)} units: {rul_path}"
        )
    rul_map = dict(zip(order["unit"].tolist(), rul_vec[: len(order)].tolist()))
    last_cycle = df_test.groupby("unit")["cycle"].max().rename("last_cycle")
    tmp = df_test.merge(last_cycle, left_on="unit", right_index=True)
    tmp["rul_end"] = tmp["unit"].map(rul_map)
    rul_linear = tmp["rul_end"] + (tmp["last_cycle"] - tmp["cycle"])
    rul = np.minimum(rul_linear.values, rearley)
    return pd.Series(rul, index=df_test.index, name="RUL")


def select_features(df: pd.DataFrame, select14: bool) -> Tuple[pd.DataFrame, List[str]]:
    """Return feature DataFrame + feature column names.
       - select14=True: keep 14 sensors only (no op1-3)
       - select14=False: keep op1-3 + all s1..s21
    """
    if select14:
        sensor_cols = [f"s{i}" for i in SENSOR_14_IDX_1BASED]
        feats = sensor_cols
    else:
        feats = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    return df[feats].copy(), feats


def fit_minmax(train_feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-feature min & max over TRAIN features only."""
    mins = train_feats.min(axis=0)
    maxs = train_feats.max(axis=0)
    return mins, maxs


def apply_minmax(feats: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                 lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Apply min-max scaling to [lo, hi], using provided mins/maxs."""
    scale = maxs - mins
    scale = np.where(scale == 0.0, 1.0, scale)  # avoid /0 -> becomes 0 after mapping
    norm01 = (feats - mins) / scale
    return (hi - lo) * norm01 + lo


def to_windows(df: pd.DataFrame,
               feature_cols: List[str],
               label_col: str,
               window: int,
               step: int = 1,
               only_last: bool = False,
               drop_short: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Per-unit sliding windows.
       - only_last=True: produce exactly one window per unit (last window).
       - drop_short=True: skip units shorter than window.
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for _, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        feats = sub[feature_cols].values
        labels = sub[label_col].values
        n = len(sub)

        if n < window:
            if drop_short:
                continue
            # left-pad with first row to reach window length (rarely needed here)
            pad = np.repeat(feats[[0], :], window - n, axis=0)
            feats = np.vstack([pad, feats])
            labels = np.concatenate([np.repeat(labels[0], window - n), labels])
            n = len(feats)

        if only_last:
            start = n - window
            end = n
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])
        else:
            for start in range(0, n - window + 1, step):
                end = start + window
                X_list.append(feats[start:end])
                y_list.append(labels[end - 1])

    if not X_list:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def resolve_window_size(fd: str, window_size_cfg: Union[str, int]) -> int:
    """Resolve window size: 'auto' -> preset, else int."""
    if isinstance(window_size_cfg, str) and window_size_cfg.lower() == "auto":
        return DEFAULT_WINDOWS[fd]
    try:
        return int(window_size_cfg)
    except Exception as e:
        raise ValueError(f"Invalid window_size: {window_size_cfg}") from e


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_meta(out_dir: Path,
              fd: str,
              feature_cols: List[str],
              mins: np.ndarray | None,
              maxs: np.ndarray | None,
              cfg: DictConfig,
              shapes: Dict[str, Tuple[int, ...]]) -> None:
    """Dump metadata and feature names."""
    (out_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "fd": fd,
        "feature_cols": feature_cols,
        "rearley": cfg.data.preprocess.rearley,
        "window_size": shapes.get("window", None),
        "step": cfg.data.preprocess.step,
        "test_last_window": bool(cfg.data.preprocess.test_last_window),
        "scale_mode": cfg.data.preprocess.scale_mode,
        "select14": bool(cfg.data.preprocess.select14),
        "X_train_shape": shapes.get("X_train"),
        "X_test_shape": shapes.get("X_test"),
        "y_train_shape": shapes.get("y_train"),
        "y_test_shape": shapes.get("y_test"),
        "config_snapshot": OmegaConf.to_container(cfg, resolve=True),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if mins is not None and maxs is not None:
        np.save(out_dir / "min.npy", mins.astype(np.float32))
        np.save(out_dir / "max.npy", maxs.astype(np.float32))


def preprocess_fd(fd: str, raw_dir: Path, out_root: Path, cfg: DictConfig) -> None:
    """Process one FD subset and write outputs to out_root/<FD>."""
    train_path = raw_dir / f"train_{fd}.txt"
    test_path = raw_dir / f"test_{fd}.txt"
    rul_path = raw_dir / f"RUL_{fd}.txt"
    if not train_path.exists() or not test_path.exists() or not rul_path.exists():
        raise FileNotFoundError(f"Missing CMAPSS files for {fd} in {raw_dir}")

    # Load
    train_df = load_cmapss_file(train_path)
    test_df = load_cmapss_file(test_path)

    # Labels with clipping to Rearly
    rearley = int(cfg.data.preprocess.rearley)
    train_df["RUL"] = compute_rul_train(train_df, rearley)
    test_df["RUL"] = compute_rul_test(test_df, rul_path, rearley)

    # Feature selection
    select14 = bool(cfg.data.preprocess.select14)
    train_feats_df, feature_cols = select_features(train_df, select14=select14)
    test_feats_df = test_df[feature_cols].copy()

    # Scaling
    scale_mode = str(cfg.data.preprocess.scale_mode).lower()
    mins = maxs = None
    if scale_mode == "minmax":
        mins, maxs = fit_minmax(train_feats_df.values)  # TRAIN ONLY
        train_scaled = apply_minmax(train_feats_df.values, mins, maxs, -1.0, 1.0)
        test_scaled = apply_minmax(test_feats_df.values, mins, maxs, -1.0, 1.0)
    elif scale_mode == "none":
        train_scaled = train_feats_df.values
        test_scaled = test_feats_df.values
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}. Use 'minmax' or 'none'.")

    # Replace scaled features
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_cols] = train_scaled
    test_df_scaled[feature_cols] = test_scaled

    # Windowing
    win = resolve_window_size(fd, cfg.data.preprocess.window_size)
    step = int(cfg.data.preprocess.step)
    only_last_test = bool(cfg.data.preprocess.test_last_window)

    X_train, y_train = to_windows(
        train_df_scaled, feature_cols, "RUL", window=win, step=step, only_last=False, drop_short=True
    )
    X_test, y_test = to_windows(
        test_df_scaled, feature_cols, "RUL", window=win, step=step, only_last=only_last_test, drop_short=True
    )

    # Save arrays
    out_dir = out_root / fd
    ensure_dir(out_dir)
    np.save(out_dir / "X_train.npy", X_train.astype(np.float32))
    np.save(out_dir / "y_train.npy", y_train.astype(np.float32))
    np.save(out_dir / "X_test.npy", X_test.astype(np.float32))
    np.save(out_dir / "y_test.npy", y_test.astype(np.float32))

    # Save metadata + scaler params
    shapes = {
        "window": win,
        "X_train": tuple(X_train.shape),
        "y_train": tuple(y_train.shape),
        "X_test": tuple(X_test.shape),
        "y_test": tuple(y_test.shape),
    }
    save_meta(out_dir, fd, feature_cols, mins, maxs, cfg, shapes)

    # Summary
    print(f"[{fd}] saved to: {out_dir}")
    for k, v in shapes.items():
        print(f"  {k}: {v}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    """
    Entry point with Hydra.
    - Hydra will change the working dir to a run directory under outputs_root by default if configured.
    - Use to_absolute_path to resolve paths declared in the config.
    """
    fd = str(cfg.data.name)
    if fd not in ALL_FDS:
        raise ValueError(f"data.name must be one of {ALL_FDS}, got: {fd}")

    raw_dir = Path(to_absolute_path(cfg.data.preprocess.raw_dir))
    out_root = Path(to_absolute_path(cfg.data.preprocess.out_dir))
    ensure_dir(out_root)

    print("=== Preprocess config ===")
    print(OmegaConf.to_yaml(cfg.data.preprocess, resolve=True))
    preprocess_fd(fd, raw_dir, out_root, cfg)
    print("=== Done ===")


if __name__ == "__main__":
    main()
