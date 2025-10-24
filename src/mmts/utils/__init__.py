# src/mmts/utils/__init__.py
# -*- coding: utf-8 -*-
"""工具模块：随机性、I/O、提示模板等。"""

from .seed import set_global_seed, set_torch_backends
from .io import (
    build_output_paths,
    gen_run_name,
    save_json,
    load_json,
    save_text,
    load_text,
    save_matplotlib_figure,
    load_processor,
)
from .json_parse import (
    extract_first_json_block,
    find_and_parse_first_json,
    validate_rul_object,
)
from .prompts import (
    build_train_messages,
    build_infer_messages,
    apply_chat_template,
)

__all__ = [
    # seed
    "set_global_seed",
    "set_torch_backends",
    # io
    "build_output_paths",
    "gen_run_name",
    "save_json",
    "load_json",
    "save_text",
    "load_text",
    "save_matplotlib_figure",
    "load_processor",
    # json_parse
    "extract_first_json_block",
    "find_and_parse_first_json",
    "validate_rul_object",
    # prompts
    "build_train_messages",
    "build_infer_messages",
    "apply_chat_template",
]
