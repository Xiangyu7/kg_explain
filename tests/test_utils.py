"""Unit tests for kg_explain.utils module.

Tests cover:
    - concurrent_map: sequential, parallel, error handling
    - read_csv: valid, missing file
    - require_cols: valid, missing columns
    - _sanitize_for_json: NaN, Inf, nested structures
    - write_jsonl / write_json: basic, NaN safety
    - safe_str: None, NaN, max_length, normal values
    - load_canonical_map: valid, missing file
"""
import json
import math
import pytest
from pathlib import Path

import pandas as pd

from kg_explain.utils import (
    concurrent_map,
    read_csv,
    require_cols,
    _sanitize_for_json,
    write_jsonl,
    write_json,
    safe_str,
    load_canonical_map,
)


# ===== concurrent_map =====

class TestConcurrentMap:
    def test_empty_items(self):
        assert concurrent_map(lambda x: x, []) == []

    def test_sequential(self):
        result = concurrent_map(lambda x: x * 2, [1, 2, 3], max_workers=1)
        assert result == [2, 4, 6]

    def test_parallel(self):
        result = concurrent_map(lambda x: x * 2, [1, 2, 3], max_workers=2)
        assert result == [2, 4, 6]

    def test_order_preserved(self):
        import time
        def slow_fn(x):
            time.sleep(0.01 * (3 - x))  # reverse sleep
            return x
        result = concurrent_map(slow_fn, [1, 2, 3], max_workers=3)
        assert result == [1, 2, 3]

    def test_error_returns_none_sequential(self):
        def fail_on_two(x):
            if x == 2:
                raise ValueError("fail")
            return x
        result = concurrent_map(fail_on_two, [1, 2, 3], max_workers=1)
        assert result == [1, None, 3]

    def test_error_returns_none_parallel(self):
        def fail_on_two(x):
            if x == 2:
                raise ValueError("fail")
            return x
        result = concurrent_map(fail_on_two, [1, 2, 3], max_workers=2)
        assert result == [1, None, 3]

    def test_negative_workers_treated_as_1(self):
        result = concurrent_map(lambda x: x, [1, 2], max_workers=-1)
        assert result == [1, 2]


# ===== read_csv =====

class TestReadCsv:
    def test_valid_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b\n1,2\n3,4\n")
        df = read_csv(p)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            read_csv(tmp_path / "no.csv")


# ===== require_cols =====

class TestRequireCols:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        require_cols(df, {"a", "b"}, "test")

    def test_missing_cols(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="缺少列.*'b'"):
            require_cols(df, {"a", "b"}, "test")

    def test_shows_available(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="可用列"):
            require_cols(df, {"a"}, "test")


# ===== _sanitize_for_json =====

class TestSanitizeForJson:
    def test_nan(self):
        assert _sanitize_for_json(float("nan")) is None

    def test_inf(self):
        assert _sanitize_for_json(float("inf")) is None

    def test_neg_inf(self):
        assert _sanitize_for_json(float("-inf")) is None

    def test_normal_float(self):
        assert _sanitize_for_json(3.14) == 3.14

    def test_nested_dict(self):
        obj = {"a": float("nan"), "b": {"c": float("inf")}}
        result = _sanitize_for_json(obj)
        assert result == {"a": None, "b": {"c": None}}

    def test_list(self):
        obj = [1.0, float("nan"), "text", None]
        result = _sanitize_for_json(obj)
        assert result == [1.0, None, "text", None]

    def test_string_passthrough(self):
        assert _sanitize_for_json("hello") == "hello"

    def test_int_passthrough(self):
        assert _sanitize_for_json(42) == 42

    def test_none_passthrough(self):
        assert _sanitize_for_json(None) is None


# ===== write_jsonl =====

class TestWriteJsonl:
    def test_basic(self, tmp_path):
        p = tmp_path / "out.jsonl"
        n = write_jsonl(p, [{"a": 1}, {"b": 2}])
        assert n == 2
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

    def test_nan_safe(self, tmp_path):
        p = tmp_path / "nan.jsonl"
        write_jsonl(p, [{"val": float("nan")}])
        data = json.loads(p.read_text().strip())
        assert data == {"val": None}

    def test_empty(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        n = write_jsonl(p, [])
        assert n == 0

    def test_creates_parent_dir(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "out.jsonl"
        write_jsonl(p, [{"x": 1}])
        assert p.exists()


# ===== write_json =====

class TestWriteJson:
    def test_basic(self, tmp_path):
        p = tmp_path / "out.json"
        write_json(p, {"key": "value"})
        data = json.loads(p.read_text())
        assert data == {"key": "value"}

    def test_nan_safe(self, tmp_path):
        p = tmp_path / "nan.json"
        write_json(p, {"val": float("nan"), "nested": {"inf": float("inf")}})
        data = json.loads(p.read_text())
        assert data["val"] is None
        assert data["nested"]["inf"] is None


# ===== safe_str =====

class TestSafeStr:
    def test_none(self):
        assert safe_str(None) == ""

    def test_none_custom_default(self):
        assert safe_str(None, default="N/A") == "N/A"

    def test_nan(self):
        assert safe_str(float("nan")) == ""

    def test_normal_string(self):
        assert safe_str("hello") == "hello"

    def test_whitespace_stripped(self):
        assert safe_str("  hello  ") == "hello"

    def test_int(self):
        assert safe_str(42) == "42"

    def test_float(self):
        assert safe_str(3.14) == "3.14"

    def test_max_length(self):
        assert safe_str("abcdefghij", max_length=5) == "abcde"

    def test_max_length_zero_no_truncate(self):
        assert safe_str("abcdefghij", max_length=0) == "abcdefghij"

    def test_pandas_nat(self):
        assert safe_str(pd.NaT) == ""

    def test_list_passthrough(self):
        # Lists are not NaN, so should convert to string
        result = safe_str([1, 2, 3])
        assert result == "[1, 2, 3]"


# ===== load_canonical_map =====

class TestLoadCanonicalMap:
    def test_valid_map(self, tmp_path):
        p = tmp_path / "drug_canonical.csv"
        p.write_text("drug_raw,canonical_name\nAspirin,aspirin\nTylenol,acetaminophen\n")
        mapping = load_canonical_map(tmp_path)
        assert mapping["aspirin"] == "aspirin"
        assert mapping["tylenol"] == "acetaminophen"

    def test_missing_file(self, tmp_path):
        mapping = load_canonical_map(tmp_path)
        assert mapping == {}

    def test_skips_nan(self, tmp_path):
        p = tmp_path / "drug_canonical.csv"
        p.write_text("drug_raw,canonical_name\nAspirin,aspirin\n,\n")
        mapping = load_canonical_map(tmp_path)
        assert len(mapping) == 1
