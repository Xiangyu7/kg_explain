"""
通用工具函数

文件读写、数据验证等
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Any

import pandas as pd

from .config import ensure_dir


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """读取CSV文件"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path.resolve()}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def require_cols(df: pd.DataFrame, cols: set[str], name: str) -> None:
    """检查必需的列"""
    miss = cols - set(df.columns)
    if miss:
        raise ValueError(f"{name} 缺少列: {sorted(miss)}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """写入JSONL文件"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False))
            w.write("\n")


def write_json(path: Path, data: dict) -> None:
    """写入JSON文件"""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_str(value: Any, default: str = "") -> str:
    """
    安全地将值转换为字符串

    处理 pandas NaN、None、float 等特殊情况

    Args:
        value: 需要转换的值
        default: 当值为 NaN/None 时返回的默认值

    Returns:
        字符串值（已去除首尾空格）
    """
    if pd.isna(value) or value is None:
        return default
    return str(value).strip()
