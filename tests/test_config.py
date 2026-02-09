"""Unit tests for kg_explain.config module.

Tests cover:
    - Config default values and type safety
    - Property range clamping (http_timeout, max_workers, etc.)
    - Config validation (valid/invalid modes, URLs, parameters)
    - ConfigValidationError
    - Config.summary()
    - _deep_merge()
    - load_yaml() error handling
    - load_config() with layered YAML
"""
import pytest
import yaml
from pathlib import Path

from kg_explain.config import (
    Config,
    ConfigValidationError,
    _deep_merge,
    load_yaml,
    load_config,
    ensure_dir,
)


# ===== Config defaults =====

class TestConfigDefaults:
    def test_empty_config(self):
        cfg = Config(raw={})
        assert cfg.mode == "v5"
        assert cfg.condition == "atherosclerosis"
        assert isinstance(cfg.data_dir, Path)
        assert isinstance(cfg.output_dir, Path)

    def test_mode_normalization(self):
        cfg = Config(raw={"mode": "V3"})
        assert cfg.mode == "v3"

    def test_mode_with_whitespace(self):
        cfg = Config(raw={"mode": " v5 "})
        assert cfg.mode == "v5"

    def test_default_http_settings(self):
        cfg = Config(raw={})
        assert cfg.http_timeout == 60
        assert cfg.http_max_retries == 5
        assert cfg.http_page_size == 200
        assert cfg.http_max_workers == 8
        assert cfg.cache_ttl_hours == 0.0

    def test_default_trial_settings(self):
        cfg = Config(raw={})
        assert "TERMINATED" in cfg.trial_statuses
        assert cfg.trial_max_pages == 20

    def test_default_rank_settings(self):
        cfg = Config(raw={})
        assert cfg.topk_paths_per_pair == 10
        assert cfg.topk_pairs_per_drug == 50
        assert cfg.hub_penalty_lambda == 1.0
        assert cfg.support_gene_boost == 0.15


# ===== Range clamping =====

class TestRangeClamping:
    def test_timeout_clamped_high(self):
        cfg = Config(raw={"http": {"timeout": 99999}})
        assert cfg.http_timeout == 600

    def test_timeout_clamped_low(self):
        cfg = Config(raw={"http": {"timeout": -5}})
        assert cfg.http_timeout == 1

    def test_max_retries_clamped(self):
        cfg = Config(raw={"http": {"max_retries": 100}})
        assert cfg.http_max_retries == 20

    def test_max_workers_clamped(self):
        cfg = Config(raw={"http": {"max_workers": 1000}})
        assert cfg.http_max_workers == 64

    def test_page_size_clamped(self):
        cfg = Config(raw={"http": {"page_size": 0}})
        assert cfg.http_page_size == 1

    def test_hub_penalty_non_negative(self):
        cfg = Config(raw={"rank": {"hub_penalty_lambda": -0.5}})
        assert cfg.hub_penalty_lambda == 0.0


# ===== Validation =====

class TestConfigValidation:
    def test_valid_config(self):
        cfg = Config(raw={"mode": "v5", "disease": {"condition": "cancer"}})
        errors = cfg.validate()
        assert errors == []

    def test_invalid_mode(self):
        cfg = Config(raw={"mode": "v99"})
        errors = cfg.validate()
        assert any("mode" in e for e in errors)

    def test_empty_condition(self):
        cfg = Config(raw={"disease": {"condition": ""}})
        errors = cfg.validate()
        assert any("condition" in e for e in errors)

    def test_invalid_api_url(self):
        cfg = Config(raw={"api": {"chembl": "not-a-url"}})
        errors = cfg.validate()
        assert any("chembl" in e for e in errors)

    def test_valid_api_url(self):
        cfg = Config(raw={"api": {"chembl": "https://api.example.com"}})
        errors = cfg.validate()
        # Should have no api-related errors
        assert not any("chembl" in e for e in errors)

    def test_negative_timeout(self):
        cfg = Config(raw={"http": {"timeout": -1}})
        errors = cfg.validate()
        assert any("timeout" in e for e in errors)

    def test_invalid_weight(self):
        cfg = Config(raw={"rank": {"safety_penalty_weight": 1.5}})
        errors = cfg.validate()
        assert any("safety_penalty_weight" in e for e in errors)

    def test_validate_or_raise(self):
        cfg = Config(raw={"mode": "invalid_mode"})
        with pytest.raises(ConfigValidationError) as exc_info:
            cfg.validate_or_raise()
        assert len(exc_info.value.errors) >= 1


# ===== Config summary =====

class TestConfigSummary:
    def test_summary_keys(self):
        cfg = Config(raw={})
        summary = cfg.summary()
        assert "mode" in summary
        assert "condition" in summary
        assert "http_timeout" in summary
        assert "cache_ttl_hours" in summary

    def test_summary_values(self):
        cfg = Config(raw={"mode": "v3", "disease": {"condition": "diabetes"}})
        summary = cfg.summary()
        assert summary["mode"] == "v3"
        assert summary["condition"] == "diabetes"


# ===== _deep_merge =====

class TestDeepMerge:
    def test_simple_merge(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_non_dict_override(self):
        result = _deep_merge({"a": {"x": 1}}, {"a": "replaced"})
        assert result == {"a": "replaced"}

    def test_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}


# ===== load_yaml =====

class TestLoadYaml:
    def test_valid_yaml(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text("key: value\n")
        data = load_yaml(p)
        assert data == {"key": "value"}

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_empty_yaml(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        data = load_yaml(p)
        assert data == {}

    def test_non_dict_yaml(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="字典"):
            load_yaml(p)


# ===== load_config =====

class TestLoadConfig:
    def test_basic_load(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("mode: v3\npaths:\n  data_dir: ./mydata\n")
        cfg = load_config(base_path=str(base))
        assert cfg.mode == "v3"
        assert cfg.data_dir == Path("./mydata")

    def test_layered_load(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("mode: v3\nhttp:\n  timeout: 30\n")
        disease = tmp_path / "disease.yaml"
        disease.write_text("disease:\n  condition: cancer\n")
        version = tmp_path / "version.yaml"
        version.write_text("mode: v5\n")

        cfg = load_config(
            base_path=str(base),
            disease_path=str(disease),
            version_path=str(version),
        )
        assert cfg.mode == "v5"  # version overrides base
        assert cfg.condition == "cancer"
        assert cfg.http_timeout == 30


# ===== ensure_dir =====

class TestEnsureDir:
    def test_creates_dir(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_existing_dir(self, tmp_path):
        result = ensure_dir(tmp_path)
        assert result == tmp_path


# ===== Serious AE keywords =====

class TestSeriousAEKeywords:
    def test_default_keywords(self):
        cfg = Config(raw={})
        kw = cfg.serious_ae_keywords
        assert "death" in kw
        assert "fatal" in kw

    def test_custom_keywords(self):
        cfg = Config(raw={"serious_ae_keywords": ["custom1", "custom2"]})
        assert cfg.serious_ae_keywords == ["custom1", "custom2"]

    def test_non_list_fallback(self):
        cfg = Config(raw={"serious_ae_keywords": "not_a_list"})
        kw = cfg.serious_ae_keywords
        assert "death" in kw  # falls back to defaults


# ===== Trial statuses =====

class TestTrialStatuses:
    def test_default(self):
        cfg = Config(raw={})
        assert "TERMINATED" in cfg.trial_statuses

    def test_custom(self):
        cfg = Config(raw={"trial_filter": {"statuses": ["completed"]}})
        assert cfg.trial_statuses == ["COMPLETED"]

    def test_non_list_fallback(self):
        cfg = Config(raw={"trial_filter": {"statuses": "not_a_list"}})
        assert "TERMINATED" in cfg.trial_statuses
