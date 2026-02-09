"""Unit tests for kg_explain.cache module.

Tests cover:
    - HTTPCache: get/set/has/invalidate/cleanup_expired
    - TTL expiration
    - Statistics tracking (hits, misses, expired, puts, errors)
    - Cache summary (disk usage)
    - Corrupt file handling
    - sha1 determinism
    - Thread safety basics
"""
import json
import time
import pytest
from pathlib import Path

from kg_explain.cache import HTTPCache, sha1


class TestSha1:
    def test_deterministic(self):
        assert sha1("hello") == sha1("hello")

    def test_different_inputs(self):
        assert sha1("hello") != sha1("world")

    def test_returns_hex(self):
        result = sha1("test")
        assert len(result) == 40
        assert all(c in "0123456789abcdef" for c in result)


class TestHTTPCacheBasic:
    def test_put_and_get(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        cache.set("key1", {"data": "value"})
        result = cache.get("key1")
        assert result == {"data": "value"}

    def test_miss(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        result = cache.get("nonexistent")
        assert result is None

    def test_has_exists(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        cache.set("key1", {"x": 1})
        assert cache.has("key1") is True

    def test_has_not_exists(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        assert cache.has("missing") is False

    def test_invalidate(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        cache.set("key1", {"x": 1})
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None

    def test_invalidate_missing(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        assert cache.invalidate("nonexistent") is False

    def test_overwrite(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        cache.set("key1", {"v": 1})
        cache.set("key1", {"v": 2})
        assert cache.get("key1") == {"v": 2}


class TestHTTPCacheTTL:
    def test_not_expired(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=3600)
        cache.set("key1", {"x": 1})
        assert cache.get("key1") == {"x": 1}

    def test_expired(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=1)
        cache.set("key1", {"x": 1})
        # Manually backdate the file
        p = cache._path("key1")
        import os
        os.utime(p, (time.time() - 100, time.time() - 100))
        assert cache.get("key1") is None

    def test_no_ttl_never_expires(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        cache.set("key1", {"x": 1})
        # Even with old mtime, should not expire
        p = cache._path("key1")
        import os
        os.utime(p, (time.time() - 999999, time.time() - 999999))
        assert cache.get("key1") == {"x": 1}

    def test_has_respects_ttl(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=1)
        cache.set("key1", {"x": 1})
        import os
        os.utime(cache._path("key1"), (time.time() - 100, time.time() - 100))
        assert cache.has("key1") is False


class TestHTTPCacheStats:
    def test_initial_stats(self, tmp_path):
        cache = HTTPCache(tmp_path)
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["puts"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_tracking(self, tmp_path):
        cache = HTTPCache(tmp_path)
        cache.set("k", {"v": 1})
        cache.get("k")
        assert cache.stats["hits"] == 1
        assert cache.stats["puts"] == 1

    def test_miss_tracking(self, tmp_path):
        cache = HTTPCache(tmp_path)
        cache.get("missing")
        assert cache.stats["misses"] == 1

    def test_expired_tracking(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=1)
        cache.set("k", {"v": 1})
        import os
        os.utime(cache._path("k"), (time.time() - 100, time.time() - 100))
        cache.get("k")
        assert cache.stats["expired"] == 1
        assert cache.stats["misses"] == 1

    def test_hit_rate(self, tmp_path):
        cache = HTTPCache(tmp_path)
        cache.set("k", {"v": 1})
        cache.get("k")  # hit
        cache.get("k")  # hit
        cache.get("miss")  # miss
        assert cache.stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_error_on_corrupt_file(self, tmp_path):
        cache = HTTPCache(tmp_path)
        # Write corrupt data
        cache.set("k", {"v": 1})
        cache._path("k").write_text("not json {{{", encoding="utf-8")
        cache.get("k")
        assert cache.stats["errors"] == 1


class TestHTTPCacheSummary:
    def test_summary_keys(self, tmp_path):
        cache = HTTPCache(tmp_path)
        summary = cache.summary()
        assert "hits" in summary
        assert "n_cached_files" in summary
        assert "total_size_mb" in summary
        assert "ttl_seconds" in summary

    def test_summary_counts_files(self, tmp_path):
        cache = HTTPCache(tmp_path)
        cache.set("k1", {"v": 1})
        cache.set("k2", {"v": 2})
        summary = cache.summary()
        assert summary["n_cached_files"] == 2
        assert summary["total_size_mb"] >= 0  # small JSON files may round to 0.0


class TestHTTPCacheCleanup:
    def test_cleanup_expired(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=1)
        cache.set("k1", {"v": 1})
        cache.set("k2", {"v": 2})
        # Backdate both
        import os
        for k in ["k1", "k2"]:
            os.utime(cache._path(k), (time.time() - 100, time.time() - 100))
        cleaned = cache.cleanup_expired()
        assert cleaned == 2

    def test_cleanup_no_ttl(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=0)
        cache.set("k1", {"v": 1})
        cleaned = cache.cleanup_expired()
        assert cleaned == 0

    def test_cleanup_keeps_fresh(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=3600)
        cache.set("fresh", {"v": 1})
        cache.set("old", {"v": 2})
        import os
        os.utime(cache._path("old"), (time.time() - 999999, time.time() - 999999))
        cleaned = cache.cleanup_expired()
        assert cleaned == 1
        assert cache.get("fresh") == {"v": 1}


class TestHTTPCacheInit:
    def test_max_workers_clamped(self, tmp_path):
        cache = HTTPCache(tmp_path, max_workers=-1)
        assert cache.max_workers == 1

    def test_ttl_clamped(self, tmp_path):
        cache = HTTPCache(tmp_path, ttl_seconds=-100)
        assert cache.ttl_seconds == 0

    def test_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "deep" / "nested"
        cache = HTTPCache(cache_dir)
        assert cache.root.exists()
