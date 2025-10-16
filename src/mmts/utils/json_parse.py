# src/mmts/utils/json_parse.py
# -*- coding: utf-8 -*-
"""
JSON 片段抽取与安全解析；提供数值字段校验与 RUL 便捷校验。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple


# -------------------------
# JSON 片段抽取 / 解析
# -------------------------

def extract_first_json_block(text: str) -> Optional[str]:
    """从任意字符串中提取首个完整 JSON 对象的原始子串；失败返回 None。"""
    if not isinstance(text, str):
        return None

    t = text.strip()
    if not t:
        return None

    # 直接就是 JSON
    try:
        json.loads(t)
        return t
    except Exception:
        pass

    # 扫描首个 {..}，正确处理引号与转义
    start = t.find("{")
    if start == -1:
        return None

    brace_stack = 0
    in_string = False
    escape = False
    end: Optional[int] = None

    for i in range(start, len(t)):
        ch = t[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            brace_stack += 1
        elif ch == "}":
            brace_stack -= 1
            if brace_stack == 0:
                end = i
                break

    if end is None:
        return None

    candidate = t[start : end + 1].strip()
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None


def parse_json_safely(json_text: str) -> Optional[dict]:
    """解析 JSON 文本为 dict；解析失败或类型非 dict 时返回 None."""
    if not isinstance(json_text, str):
        return None
    try:
        obj = json.loads(json_text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def find_and_parse_first_json(text: str) -> Optional[dict]:
    """先抽取首个 JSON 对象，再解析为 dict；任一步失败返回 None。"""
    chunk = extract_first_json_block(text)
    if chunk is None:
        return None
    return parse_json_safely(chunk)


# -------------------------
# 数值字段校验与裁剪
# -------------------------

_NUMERIC_LIKE = re.compile(
    r"""
    ^\s*
    (?P<sign>[+-]?)              # 可选符号
    (?:
        (?P<int>\d+)             # 整数
        (?:\.(?P<frac>\d+))?     # 小数
      |
        \.(?P<onlyfrac>\d+)      # 仅小数部分
    )
    (?P<tail>.*)                 # 可选尾缀（单位等）
    \s*$
    """,
    re.VERBOSE,
)


def _to_float_maybe(x: Any) -> Optional[float]:
    """
    尝试将输入转为 float：
    - 原生数值直接转；
    - 字符串允许“数字 + 可选单位/尾缀”，如 "123", "12.3", "+.5", "123 cycles"；
    - 无法转换返回 None。
    """
    if isinstance(x, (int, float)):
        try:
            v = float(x)
            return v if v == v else None  # 过滤 NaN
        except Exception:
            return None

    if isinstance(x, str):
        s = x.strip()
        try:
            v = float(s)
            return v if v == v else None
        except Exception:
            pass
        m = _NUMERIC_LIKE.match(s)
        if m:
            sign = -1.0 if m.group("sign") == "-" else 1.0
            if m.group("onlyfrac") is not None:
                base = float("0." + m.group("onlyfrac"))
            else:
                whole = m.group("int") or "0"
                frac = m.group("frac")
                base = float(whole + ("." + frac if frac else ""))
            return sign * base

    return None


def validate_and_clip_numeric(
    obj: dict,
    key: str,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Optional[float]:
    """
    从 obj[key] 读取数值，转换为 float，并按 [low, high] 裁剪（端点可为 None）。
    失败返回 None。
    """
    if not isinstance(obj, dict) or key not in obj:
        return None

    v = _to_float_maybe(obj.get(key))
    if v is None:
        return None

    if (low is not None) and (v < low):
        v = float(low)
    if (high is not None) and (v > high):
        v = float(high)
    return v


# -------------------------
# RUL 便捷校验
# -------------------------

def validate_rul_object(
    obj: dict,
    low: float,
    high: float,
    reason_key: str = "reason",
) -> Tuple[Optional[float], str]:
    """
    针对形如 {"rul": <number>, "reason": "<text>"} 的对象：
    返回 (裁剪后的 rul 或 None, reason 文本)。
    """
    v = validate_and_clip_numeric(obj, key="rul", low=low, high=high)
    reason = ""
    if isinstance(obj, dict) and reason_key in obj:
        try:
            reason = str(obj.get(reason_key, "")).strip()
        except Exception:
            reason = ""
    return v, reason
