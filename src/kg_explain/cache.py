"""
HTTP缓存与请求模块

提供带重试机制的HTTP请求和基于文件的响应缓存
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import ensure_dir


def sha1(s: str) -> str:
    """计算SHA1哈希"""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class HTTPCache:
    """HTTP响应文件缓存"""

    def __init__(self, cache_dir: Path):
        self.root = ensure_dir(cache_dir / "http_json")

    def _path(self, key: str) -> Path:
        return self.root / f"{sha1(key)}.json"

    def get(self, key: str) -> Optional[dict]:
        """获取缓存"""
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, value: dict) -> None:
        """设置缓存"""
        self._path(key).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )


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
    """带重试的GET请求"""
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


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
    """带重试的POST请求"""
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def cached_get_json(
    cache: HTTPCache,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 60,
) -> dict:
    """带缓存的GET请求"""
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
    """带缓存的POST请求"""
    key = url + "::" + json.dumps(payload, sort_keys=True)
    hit = cache.get(key)
    if hit is not None:
        return hit
    js = http_post_json(url, payload=payload, headers=headers, timeout=timeout)
    cache.set(key, js)
    return js
