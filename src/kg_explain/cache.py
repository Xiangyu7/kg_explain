"""
HTTP缓存与请求模块 — 工业级, 含统计、错误分类和可观测性

提供带重试机制的HTTP请求和基于文件的响应缓存

Improvements (v0.6.0):
    - HTTPCache.stats: 命中率/缺失率/过期率统计
    - HTTPCache.summary(): 缓存摘要 (条目数、大小、命中率)
    - HTTPCache.cleanup_expired(): 清理过期条目
    - 错误分类: HTTP 状态码感知的日志
    - cached_get_json/cached_post_json 支持统计
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import ensure_dir

logger = logging.getLogger(__name__)


def sha1(s: str) -> str:
    """计算SHA1哈希."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class HTTPCache:
    """HTTP响应文件缓存 (线程安全, 支持 TTL, 含统计)."""

    def __init__(self, cache_dir: Path, max_workers: int = 1, ttl_seconds: int = 0):
        """
        Args:
            cache_dir: 缓存根目录.
            max_workers: 并发线程数 (供 datasource 读取).
            ttl_seconds: 缓存过期时间 (秒), 0 表示永不过期.
        """
        self.root = ensure_dir(cache_dir / "http_json")
        self.max_workers = max(1, int(max_workers))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._lock = threading.Lock()

        # 统计
        self._hits = 0
        self._misses = 0
        self._expired = 0
        self._puts = 0
        self._errors = 0

    # ── 统计属性 ──

    @property
    def stats(self) -> dict[str, Any]:
        """返回缓存统计副本."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "expired": self._expired,
            "puts": self._puts,
            "errors": self._errors,
            "total_requests": total,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        """返回缓存摘要 (含磁盘使用情况)."""
        stats = self.stats
        n_files = 0
        total_bytes = 0
        try:
            for f in self.root.glob("*.json"):
                n_files += 1
                total_bytes += f.stat().st_size
        except OSError:
            pass

        stats["n_cached_files"] = n_files
        stats["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
        stats["ttl_seconds"] = self.ttl_seconds
        return stats

    # ── 核心操作 ──

    def _path(self, key: str) -> Path:
        return self.root / f"{sha1(key)}.json"

    def _is_expired(self, p: Path) -> bool:
        """检查缓存文件是否过期 (基于文件修改时间)."""
        if self.ttl_seconds <= 0:
            return False
        try:
            age = time.time() - p.stat().st_mtime
            return age > self.ttl_seconds
        except OSError:
            return True

    def get(self, key: str) -> Optional[dict]:
        """获取缓存 (过期条目视为未命中)."""
        p = self._path(key)
        with self._lock:
            if p.exists():
                if self._is_expired(p):
                    logger.debug("缓存已过期: %s", p.name)
                    self._expired += 1
                    self._misses += 1
                    return None
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    self._hits += 1
                    return data
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("缓存文件损坏, 将重新获取: %s (%s)", p.name, e)
                    self._errors += 1
                    self._misses += 1
                    return None
            self._misses += 1
        return None

    def set(self, key: str, value: dict) -> None:
        """设置缓存."""
        with self._lock:
            try:
                self._path(key).write_text(
                    json.dumps(value, ensure_ascii=False), encoding="utf-8"
                )
                self._puts += 1
            except (OSError, TypeError) as e:
                logger.warning("缓存写入失败: %s (%s)", self._path(key).name, e)
                self._errors += 1

    def has(self, key: str) -> bool:
        """检查是否有有效缓存 (未过期)."""
        p = self._path(key)
        if not p.exists():
            return False
        return not self._is_expired(p)

    def invalidate(self, key: str) -> bool:
        """删除缓存条目, 返回是否成功."""
        p = self._path(key)
        try:
            if p.exists():
                p.unlink()
                return True
        except OSError as e:
            logger.warning("缓存删除失败: %s (%s)", p.name, e)
        return False

    def cleanup_expired(self) -> int:
        """清理所有过期条目, 返回清理数量."""
        if self.ttl_seconds <= 0:
            return 0
        cleaned = 0
        try:
            for f in self.root.glob("*.json"):
                if self._is_expired(f):
                    try:
                        f.unlink()
                        cleaned += 1
                    except OSError:
                        pass
        except OSError:
            pass
        if cleaned > 0:
            logger.info("清理过期缓存: %d 个文件", cleaned)
        return cleaned


# ── HTTP 请求函数 (含重试) ──

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
)
def http_get_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 60,
) -> dict:
    """带重试的GET请求.

    Raises:
        requests.HTTPError: HTTP 错误 (4xx/5xx).
        requests.ConnectionError: 网络连接失败.
        requests.Timeout: 请求超时.
        ValueError: 响应不是合法 JSON.
    """
    logger.debug("GET %s params=%s", url, params)
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError as e:
        logger.warning("非 JSON 响应: GET %s → %d bytes", url, len(r.content))
        raise ValueError(f"GET {url} 响应不是合法 JSON: {e}") from e


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
)
def http_post_json(
    url: str,
    payload: dict,
    headers: dict | None = None,
    timeout: int = 60,
) -> dict:
    """带重试的POST请求.

    Raises:
        requests.HTTPError: HTTP 错误 (4xx/5xx).
        requests.ConnectionError: 网络连接失败.
        requests.Timeout: 请求超时.
        ValueError: 响应不是合法 JSON.
    """
    logger.debug("POST %s payload_keys=%s", url, list(payload.keys()) if payload else [])
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError as e:
        logger.warning("非 JSON 响应: POST %s → %d bytes", url, len(r.content))
        raise ValueError(f"POST {url} 响应不是合法 JSON: {e}") from e


# ── 带缓存的请求 ──

def cached_get_json(
    cache: HTTPCache,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 60,
) -> dict:
    """带缓存的GET请求."""
    key = url if not params else url + "?" + "&".join(
        [f"{k}={params[k]}" for k in sorted(params.keys())]
    )
    hit = cache.get(key)
    if hit is not None:
        return hit
    js = http_get_json(url, params=params, headers=headers, timeout=timeout)
    cache.set(key, js)
    return js


def cached_post_json(
    cache: HTTPCache,
    url: str,
    payload: dict,
    headers: dict | None = None,
    timeout: int = 60,
) -> dict:
    """带缓存的POST请求."""
    key = url + "::" + json.dumps(payload, sort_keys=True)
    hit = cache.get(key)
    if hit is not None:
        return hit
    js = http_post_json(url, payload=payload, headers=headers, timeout=timeout)
    cache.set(key, js)
    return js
