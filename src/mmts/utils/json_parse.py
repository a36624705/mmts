# src/mmts/utils/json_parse.py
# -*- coding: utf-8 -*-
"""
通用的 JSON 文本抽取与安全解析工具。

使用动机
--------
在多模态大模型（如 VL/LLM）生成文本时，常要求其输出严格 JSON，但实际经常会出现：
- 在 JSON 前后夹带自然语言或提示词残留；
- JSON 结构只有首个对象有效，其后还有额外内容；
- 数值字段是字符串（"123"）或带单位（不建议，但要尽量兼容）；
- 数值越界，需要裁剪到合法范围。

本模块提供以下能力：
1) 从任意字符串中**稳健定位首个 JSON 对象**的文本切片；
2) **安全解析**（失败返回 None，不抛异常）；
3) **数值字段校验与可选裁剪**（通用函数）；
4) **与 RUL 场景兼容**的便捷包装函数（仅作轻量封装，仍是可选/通用）。

注意事项
--------
- 我们只假设需要抽取的是一个 **对象（object, 以 { 开始，以 } 结束）**。
  若有数组/多对象等复杂场景，请根据需要扩展。
- 解析严格遵循 Python 标准库 `json` 的语法；不支持 JSON5 等扩展。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple


# ============================================================================
# 一、JSON 文本抽取
# ============================================================================

def extract_first_json_block(text: str) -> Optional[str]:
    """
    从任意字符串中抽取**首个完整 JSON 对象**的原始文本（substring）。

    设计要点：
    - 先尝试整体解析（字符串本身就是 JSON）；
    - 若失败，再进行“从第一个 '{' 起、基于括号配对与字符串转义的扫描”，
      在**保证引号内的花括号不计入配对**的前提下，寻找能闭合的第一个对象；
    - 成功则返回该子串；找不到返回 None。

    参数：
    - text: 任意字符串（可能包含 JSON + 杂质文本）

    返回：
    - JSON 子串（含首尾花括号）或 None
    """
    if not isinstance(text, str):
        return None

    t = text.strip()
    if not t:
        return None

    # 1) 尝试整体解析：若本身就是 JSON，则直接返回
    try:
        json.loads(t)
        return t
    except Exception:
        pass

    # 2) 扫描提取：定位第一个 '{'，并进行括号/引号配对扫描
    start = t.find("{")
    if start == -1:
        return None

    brace_stack = 0
    in_string = False
    escape = False
    end = None

    for i in range(start, len(t)):
        ch = t[i]

        if in_string:
            # 当前处于字符串字面量内
            if escape:
                # 处理转义字符：忽略当前字符的语义
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            # 字符串中出现的 '{' 或 '}' 不应参与计数
            continue

        # 不在字符串内
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
    # 再做一次严格校验
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None


def parse_json_safely(json_text: str) -> Optional[dict]:
    """
    安全地将 JSON 文本解析为 Python 字典。
    - 解析失败返回 None；
    - 仅接受对象类型（dict）；若解析为 list/int/str 等非对象，也返回 None。

    参数：
    - json_text: 严格的 JSON 文本（通常来自 extract_first_json_block）

    返回：
    - dict 或 None
    """
    if not isinstance(json_text, str):
        return None
    try:
        obj = json.loads(json_text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def find_and_parse_first_json(text: str) -> Optional[dict]:
    """
    组合式工具：先抽取首个 JSON 对象子串，再解析为 dict。
    任一步失败返回 None。
    """
    chunk = extract_first_json_block(text)
    if chunk is None:
        return None
    return parse_json_safely(chunk)


# ============================================================================
# 二、通用的数值字段校验与（可选）裁剪
# ============================================================================

_NUMERIC_LIKE = re.compile(
    r"""
    ^\s*
    (?P<sign>[+-]?)              # 可选正负号
    (?:
        (?P<int>\d+)             # 整数部分
        (?:\.(?P<frac>\d+))?     # 可选小数
      |
        \.(?P<onlyfrac>\d+)      # 或者只有 .xx 的形式
    )
    (?P<tail>.*)                 # 后缀（单位/额外字符），不做强校验
    \s*$
    """,
    re.VERBOSE,
)


def _to_float_maybe(x: Any) -> Optional[float]:
    """
    尝试将输入转换为浮点数：
    - 若是 int/float，直接返回；
    - 若是字符串，允许形如 "123", "12.3", "+4.5", ".5"，以及附带单位的形式（e.g. "123 cycles"）；
      * 对于带单位，会尝试提取开头的数字部分；
    - 其他类型/无法转换返回 None。
    """
    if isinstance(x, (int, float)):
        try:
            v = float(x)
            return v if v == v else None  # 过滤 NaN
        except Exception:
            return None

    if isinstance(x, str):
        s = x.strip()
        # 直接尝试 float()（覆盖标准数字字符串）
        try:
            v = float(s)
            return v if v == v else None
        except Exception:
            pass
        # 进一步尝试“数字+尾巴（单位）”的宽松解析
        m = _NUMERIC_LIKE.match(s)
        if m:
            sign = -1.0 if (m.group("sign") == "-") else 1.0
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
    通用数值字段校验函数：
    - 从字典 obj 中取出 key；
    - 尝试将其解析为 float；
    - 若给定范围 [low, high]，则对结果进行裁剪（clip）；
    - 任一步失败则返回 None。

    参数：
    - obj: 输入字典（通常为模型产出的 JSON 解析结果）
    - key: 需要校验的字段名
    - low/high: 允许范围的下/上界；任意一端为 None 时，不进行该端裁剪

    返回：
    - float 或 None
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


# ============================================================================
# 三、与 RUL 下游任务兼容的便捷函数（轻量包装，便于迁移旧代码）
# ============================================================================

def validate_rul_object(
    obj: dict,
    low: float,
    high: float,
    reason_key: str = "reason",
) -> Tuple[Optional[float], str]:
    """
    便捷包装：面向 RUL 任务的常见 JSON 结构：
        { "rul": <number>, "reason": "<短解释>" }

    - 从 obj 中解析 "rul" 字段的数值，按 [low, high] 进行裁剪；
    - 始终返回 (rul_value_or_None, reason_string)；
    - reason 若不存在则返回空字符串；
    - 若 "rul" 字段缺失或无法解析为数值，返回 (None, reason)。

    注意：
    - 该函数只是对通用函数 validate_and_clip_numeric 的薄封装，
      以便复用此前脚本中的 `validate_and_clip` 语义。
    """
    v = validate_and_clip_numeric(obj, key="rul", low=low, high=high)
    reason = ""
    if isinstance(obj, dict) and reason_key in obj:
        # 宽松取值：只要能转为字符串就接收
        try:
            reason = str(obj.get(reason_key, "")).strip()
        except Exception:
            reason = ""
    return v, reason
