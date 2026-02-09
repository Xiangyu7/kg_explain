"""
通用工具函数 — 工业级, 含输入验证、NaN 保护和结构化日志

文件读写、数据验证、并发执行等

Improvements (v0.6.0):
    - concurrent_map: 失败任务计数和结构化日志
    - read_csv: 文件大小/行数日志
    - write_jsonl/write_json: NaN 安全序列化
    - require_cols: 更好的错误信息 (显示可用列)
    - safe_str: 增加 max_length 截断
"""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Any, TypeVar

import pandas as pd
from tqdm import tqdm

from .config import ensure_dir

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def concurrent_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 1,
    desc: str = "",
) -> list[R]:
    """
    对 items 并发执行 fn, 返回结果列表 (保持原始顺序).

    Args:
        fn: 对每个 item 执行的函数.
        items: 输入列表.
        max_workers: 并发线程数, ≤1 则顺序执行.
        desc: tqdm 进度条描述.

    Returns:
        与 items 同序的结果列表. 失败的任务返回 None.
    """
    items = list(items)
    if not items:
        return []

    max_workers = max(1, int(max_workers))

    if max_workers <= 1:
        results = []
        n_failed = 0
        for item in tqdm(items, desc=desc, disable=not desc):
            try:
                results.append(fn(item))
            except Exception:
                logger.warning("顺序任务失败", exc_info=True)
                results.append(None)
                n_failed += 1
        if n_failed > 0:
            logger.warning("%s: %d/%d 任务失败", desc or "concurrent_map", n_failed, len(items))
        return results

    results: list[R | None] = [None] * len(items)
    n_failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        with tqdm(total=len(futures), desc=desc, disable=not desc) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.warning("并发任务 #%d 失败", idx, exc_info=True)
                    results[idx] = None
                    n_failed += 1
                pbar.update(1)

    if n_failed > 0:
        logger.warning("%s: %d/%d 任务失败", desc or "concurrent_map", n_failed, len(items))
    return results


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    读取CSV文件, 含文件存在性校验.

    Args:
        path: CSV 文件路径.
        **kwargs: 透传给 pd.read_csv.

    Returns:
        DataFrame.

    Raises:
        FileNotFoundError: 文件不存在.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"文件不存在: {path.resolve()}\n"
            f"  提示: 检查数据目录或运行 fetch 命令"
        )
    df = pd.read_csv(path, low_memory=False, **kwargs)
    logger.debug("读取 CSV: %s (%d 行, %d 列)", path.name, len(df), len(df.columns))
    return df


def require_cols(df: pd.DataFrame, cols: set[str], name: str) -> None:
    """
    检查必需的列.

    Args:
        df: 要检查的 DataFrame.
        cols: 必需的列名集合.
        name: 数据集名称 (用于错误信息).

    Raises:
        ValueError: 缺少必需列.
    """
    miss = cols - set(df.columns)
    if miss:
        raise ValueError(
            f"{name} 缺少列: {sorted(miss)}\n"
            f"  可用列: {sorted(df.columns.tolist())}"
        )


def _sanitize_for_json(obj: Any) -> Any:
    """递归替换 NaN/Inf 为 None, 确保 JSON 合法."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """
    写入 JSONL 文件, NaN 安全.

    Args:
        path: 输出路径.
        rows: 行数据迭代器.

    Returns:
        写入的行数.
    """
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(_sanitize_for_json(r), ensure_ascii=False))
            w.write("\n")
            n += 1
    logger.debug("写入 JSONL: %s (%d 行)", path.name, n)
    return n


def write_json(path: Path, data: dict) -> None:
    """写入 JSON 文件, NaN 安全."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(_sanitize_for_json(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("写入 JSON: %s", path.name)


def load_canonical_map(data_dir: Path) -> dict[str, str]:
    """
    加载药物规范名称映射.

    Returns:
        dict: drug_raw (lowercase) → canonical_name
    """
    path = Path(data_dir) / "drug_canonical.csv"
    if not path.exists():
        logger.debug("规范名称映射不存在, 跳过: %s", path)
        return {}
    df = read_csv(path, dtype=str)
    mapping = {}
    for _, r in df.iterrows():
        raw = r.get("drug_raw")
        canon = r.get("canonical_name")
        if pd.notna(raw) and pd.notna(canon):
            mapping[str(raw).lower().strip()] = str(canon).lower().strip()
    logger.debug("加载规范名称映射: %d 条", len(mapping))
    return mapping


def safe_str(value: Any, default: str = "", max_length: int = 0) -> str:
    """
    安全地将值转换为字符串.

    处理 pandas NaN、None、float 等特殊情况.

    Args:
        value: 需要转换的值.
        default: 当值为 NaN/None 时返回的默认值.
        max_length: 最大长度, 0 表示不限制.

    Returns:
        字符串值 (已去除首尾空格).
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    result = str(value).strip()
    if max_length > 0 and len(result) > max_length:
        result = result[:max_length]
    return result
