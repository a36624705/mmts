# src/mmts/configs/loader.py
# -*- coding: utf-8 -*-
"""
配置加载与合并工具：
- 从 YAML 文件读取 dict；
- 解析类似 Hydra 的 defaults 约定（可选）：在入口 YAML 顶层下有 `defaults:` 列表，
  其中每个元素可以是 "data: data/fd001.yaml" 这种形式，表示需要加载并合并该子配置；
- 递归深度合并多个 dict，后加载的覆盖先加载的同名键；
- 支持用 CLI 的扁平 key=val 覆盖合并结果（例如 train.epochs=5）。

用法（示例）
-----------
cfg = load_config_tree("configs/defaults.yaml")
# 访问：
#   cfg["data"]["dataset"]["train_dir"]
#   cfg["model"]["base_id"]  等等
#
# 若要应用 CLI 覆盖（如 ["train.epochs=4", "eval.max_new_tokens=64"]）：
#   cfg = apply_overrides(cfg, cli_overrides)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import copy
import re

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise ImportError("缺少 PyYAML，请先安装：pip install pyyaml") from e


# -----------------------------------------------------------------------------
# 基础：读 YAML
# -----------------------------------------------------------------------------
def read_yaml(path: str | Path) -> Dict[str, Any]:
    """读取 YAML 文件为 Python 字典。空文件/空映射返回 {}。"""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # type: ignore
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML 顶层应为 mapping(dict)，但 {p} 得到 {type(data)}")
    return data


# -----------------------------------------------------------------------------
# 深度合并（dict of dicts）
# -----------------------------------------------------------------------------
def deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典：同名键如果都是 dict 则递归合并；否则被 new 覆盖。
    返回一个**新的** dict，不修改原对象。
    """
    a = copy.deepcopy(base)
    for k, v in new.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


# -----------------------------------------------------------------------------
# 解析 defaults: 语义
# -----------------------------------------------------------------------------
def _parse_default_item(item: Any) -> Tuple[str, str]:
    """
    将 defaults 列表中的一项解析成 ("group", "path")。
    允许的形式：
      - "data: data/fd001.yaml"
      - {"data": "data/fd001.yaml"}
    """
    if isinstance(item, str):
        if ":" not in item:
            raise ValueError(f"defaults 项格式错误：{item!r}")
        group, path = item.split(":", 1)
        return group.strip(), path.strip()
    elif isinstance(item, dict):
        if len(item) != 1:
            raise ValueError(f"defaults 项必须只有一个键：{item!r}")
        (group, path), = item.items()
        return str(group).strip(), str(path).strip()
    else:
        raise ValueError(f"不支持的 defaults 项类型：{type(item)} -> {item!r}")


def _load_defaults(root_dir: Path, defaults: List[Any]) -> Dict[str, Any]:
    """
    依次加载 defaults 里的子配置（相对路径从 root_dir 解析），并深度合并。
    注意：并不强求按 group（如 data/model/train/eval）分门别类，只要文件能合并即可。
    """
    merged: Dict[str, Any] = {}
    for item in defaults:
        _group, rel = _parse_default_item(item)   # 目前不强制用 group 建层
        sub_path = (root_dir / rel).resolve()
        sub_cfg = read_yaml(sub_path)
        merged = deep_merge(merged, sub_cfg)
    return merged


def load_config_tree(entry_yaml: str | Path) -> Dict[str, Any]:
    """
    从入口 YAML（例如 configs/defaults.yaml）加载完整配置树：
    - 读取入口；
    - 若存在顶层键 `defaults`（list），则按顺序加载列表中的 YAML 并合并；
    - 将入口中除 `defaults` 外的其它键也合并进去（入口可作为最后一层覆盖）。
    """
    entry_path = Path(entry_yaml).resolve()
    entry_cfg = read_yaml(entry_path)

    defaults = entry_cfg.get("defaults", None)
    base_dir = entry_path.parent

    if isinstance(defaults, list) and defaults:
        merged = _load_defaults(base_dir, defaults)
    else:
        merged = {}

    # 入口中除 defaults 外的键，后应用（作为覆盖层）
    rest = {k: v for k, v in entry_cfg.items() if k != "defaults"}
    merged = deep_merge(merged, rest)
    return merged


# -----------------------------------------------------------------------------
# CLI 覆盖：train.epochs=5, eval.max_new_tokens=64 这样的键值
# -----------------------------------------------------------------------------
_DOT_RE = re.compile(r"\.")

def _set_by_dotpath(cfg: Dict[str, Any], key: str, value: Any) -> None:
    """
    在嵌套 dict 中用点路径设置值，例如：
      key="train.epochs", value=5  -> cfg["train"]["epochs"]=5
    如果中间层不存在，会自动创建 dict。
    """
    parts = _DOT_RE.split(key)
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]  # type: ignore
    cur[parts[-1]] = value


def _parse_scalar(s: str) -> Any:
    """
    将字符串转为更合适的 Python 类型：
    - "true"/"false" -> bool
    - 数字 -> int/float
    - 其他 -> 原样字符串
    """
    sl = s.strip().lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        if "." in sl or "e" in sl:
            return float(sl)
        return int(sl)
    except Exception:
        return s.strip()


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """
    应用 CLI 覆盖项：每项形如 "a.b.c=value"。
    返回新的 dict。
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项应形如 key=value：{item!r}")
        k, v = item.split("=", 1)
        _set_by_dotpath(out, k.strip(), _parse_scalar(v))
    return out
