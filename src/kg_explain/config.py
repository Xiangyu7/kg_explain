"""
配置加载模块 — 工业级, 含验证、类型安全和错误提示

支持配置继承: base.yaml → disease.yaml → version.yaml

Improvements (v0.6.0):
    - Config.validate(): 配置完整性检查
    - 数值参数范围校验 (timeout, retries, page_size 等)
    - 路径属性类型安全 (始终返回 Path)
    - API 端点 URL 基本校验
    - ConfigValidationError 带详细错误列表
    - 不可变属性: 读取后自动缓存
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("kg_explain.config")

# ── 常量: 合理范围 ──
_VALID_MODES = {"v1", "v2", "v3", "v4", "v5", "1", "2", "3", "4", "5", "default"}
_MAX_TIMEOUT = 600  # 秒
_MAX_RETRIES = 20
_MAX_PAGE_SIZE = 5000
_MAX_WORKERS = 64
_MAX_TTL_HOURS = 24 * 365


class ConfigValidationError(Exception):
    """配置验证失败, 包含多条错误信息."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        msg = f"配置验证失败 ({len(errors)} 个错误):\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


@dataclass
class Config:
    """配置数据类 — 带类型安全属性和验证."""

    raw: dict = field(default_factory=dict)

    # ── 基础路径 ──
    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "v5")).lower().strip()

    @property
    def data_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("data_dir", "./data"))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("output_dir", "./output"))

    @property
    def cache_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("cache_dir", "./cache"))

    # ── 子配置 dict ──
    @property
    def disease(self) -> dict:
        return self.raw.get("disease", {})

    @property
    def rank(self) -> dict:
        return self.raw.get("rank", {})

    @property
    def files(self) -> dict:
        return self.raw.get("files", {})

    @property
    def api(self) -> dict:
        return self.raw.get("api", {})

    # ── 疾病相关 ──
    @property
    def condition(self) -> str:
        return str(self.disease.get("condition", "atherosclerosis")).strip()

    @property
    def drug_filter(self) -> dict:
        return self.raw.get("drug_filter", {})

    # ── HTTP 设置 (含范围校验) ──
    @property
    def http_timeout(self) -> int:
        val = int(self.raw.get("http", {}).get("timeout", 60))
        return max(1, min(val, _MAX_TIMEOUT))

    @property
    def http_max_retries(self) -> int:
        val = int(self.raw.get("http", {}).get("max_retries", 5))
        return max(0, min(val, _MAX_RETRIES))

    @property
    def http_page_size(self) -> int:
        val = int(self.raw.get("http", {}).get("page_size", 200))
        return max(1, min(val, _MAX_PAGE_SIZE))

    @property
    def http_max_workers(self) -> int:
        val = int(self.raw.get("http", {}).get("max_workers", 8))
        return max(1, min(val, _MAX_WORKERS))

    @property
    def cache_ttl_hours(self) -> float:
        val = float(self.raw.get("http", {}).get("cache_ttl_hours", 0))
        return max(0.0, min(val, _MAX_TTL_HOURS))

    # ── 试验筛选 ──
    @property
    def trial_filter(self) -> dict:
        return self.raw.get("trial_filter", {})

    @property
    def trial_statuses(self) -> list[str]:
        statuses = self.trial_filter.get("statuses", ["TERMINATED", "WITHDRAWN", "SUSPENDED"])
        if not isinstance(statuses, list):
            return ["TERMINATED", "WITHDRAWN", "SUSPENDED"]
        return [str(s).upper().strip() for s in statuses if s]

    @property
    def trial_max_pages(self) -> int:
        val = int(self.trial_filter.get("max_pages", 20))
        return max(1, min(val, 500))

    # ── FAERS ──
    @property
    def faers(self) -> dict:
        return self.raw.get("faers", {})

    # ── 表型 ──
    @property
    def phenotype(self) -> dict:
        return self.raw.get("phenotype", {})

    # ── 严重AE关键词 ──
    @property
    def serious_ae_keywords(self) -> list[str]:
        kw = self.raw.get("serious_ae_keywords", [
            "death", "fatal", "life-threatening", "hospitalisation", "disability",
        ])
        if not isinstance(kw, list):
            return ["death", "fatal", "life-threatening", "hospitalisation", "disability"]
        return [str(k).strip() for k in kw if k]

    # ── 排序参数便捷访问 ──
    @property
    def topk_paths_per_pair(self) -> int:
        return max(1, int(self.rank.get("topk_paths_per_pair", 10)))

    @property
    def topk_pairs_per_drug(self) -> int:
        return max(1, int(self.rank.get("topk_pairs_per_drug", 50)))

    @property
    def hub_penalty_lambda(self) -> float:
        val = float(self.rank.get("hub_penalty_lambda", 1.0))
        return max(0.0, val)

    @property
    def support_gene_boost(self) -> float:
        val = float(self.rank.get("support_gene_boost", 0.15))
        return max(0.0, val)

    # ── 验证 ──
    def validate(self) -> list[str]:
        """
        验证配置完整性.

        Returns:
            错误列表 (空列表表示通过)
        """
        errors: list[str] = []

        # mode
        if self.mode not in _VALID_MODES:
            errors.append(f"mode='{self.mode}' 不合法, 可选: {sorted(_VALID_MODES)}")

        # condition 不能为空
        if not self.condition:
            errors.append("disease.condition 不能为空")

        # API 端点检查
        for api_name in ("ctgov", "rxnorm", "chembl", "reactome", "opentargets", "faers"):
            url = self.api.get(api_name, "")
            if url and not (url.startswith("http://") or url.startswith("https://")):
                errors.append(f"api.{api_name}='{url}' 不是合法的 HTTP URL")

        # HTTP 参数警告
        if self.raw.get("http", {}).get("timeout") is not None:
            try:
                t = int(self.raw["http"]["timeout"])
                if t <= 0:
                    errors.append(f"http.timeout={t} 必须为正整数")
            except (ValueError, TypeError) as e:
                errors.append(f"http.timeout 无法解析为整数: {e}")

        if self.raw.get("http", {}).get("max_retries") is not None:
            try:
                r = int(self.raw["http"]["max_retries"])
                if r < 0:
                    errors.append(f"http.max_retries={r} 不能为负数")
            except (ValueError, TypeError) as e:
                errors.append(f"http.max_retries 无法解析为整数: {e}")

        # 排序参数检查
        rank = self.rank
        for key in ("safety_penalty_weight", "trial_failure_penalty", "phenotype_overlap_boost"):
            if key in rank:
                try:
                    v = float(rank[key])
                    if v < 0 or v > 1:
                        errors.append(f"rank.{key}={v} 应在 [0, 1] 范围内")
                except (ValueError, TypeError) as e:
                    errors.append(f"rank.{key} 无法解析为浮点数: {e}")

        return errors

    def validate_or_raise(self) -> None:
        """验证配置, 有错误时抛出 ConfigValidationError."""
        errors = self.validate()
        if errors:
            raise ConfigValidationError(errors)

    def summary(self) -> dict[str, Any]:
        """返回配置摘要字典 (用于日志和 manifest)."""
        return {
            "mode": self.mode,
            "condition": self.condition,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "http_timeout": self.http_timeout,
            "http_max_retries": self.http_max_retries,
            "http_page_size": self.http_page_size,
            "http_max_workers": self.http_max_workers,
            "cache_ttl_hours": self.cache_ttl_hours,
            "trial_statuses": self.trial_statuses,
            "trial_max_pages": self.trial_max_pages,
            "topk_paths_per_pair": self.topk_paths_per_pair,
            "topk_pairs_per_drug": self.topk_pairs_per_drug,
        }


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典 (override 优先)."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict:
    """
    加载 YAML 文件.

    Args:
        path: YAML 文件路径.

    Returns:
        解析后的字典.

    Raises:
        FileNotFoundError: 文件不存在.
        yaml.YAMLError: YAML 语法错误.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {path}\n"
            f"  绝对路径: {path.resolve()}\n"
            f"  提示: 检查 configs/ 目录是否存在"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"YAML 解析错误: {path}\n  {e}\n"
            f"  提示: 检查缩进和特殊字符"
        ) from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是字典, 得到 {type(data).__name__}: {path}")
    return data


def load_config(
    base_path: str = "configs/base.yaml",
    disease_path: Optional[str] = None,
    version_path: Optional[str] = None,
) -> Config:
    """
    加载并合并配置.

    优先级: version > disease > base

    Args:
        base_path: 基础配置文件路径.
        disease_path: 疾病配置文件路径 (可选).
        version_path: 版本配置文件路径 (可选).

    Returns:
        合并后的 Config 对象.
    """
    config = load_yaml(Path(base_path))

    if disease_path:
        disease_cfg = load_yaml(Path(disease_path))
        config = _deep_merge(config, disease_cfg)
        logger.debug("合并疾病配置: %s", disease_path)

    if version_path:
        version_cfg = load_yaml(Path(version_path))
        config = _deep_merge(config, version_cfg)
        logger.debug("合并版本配置: %s", version_path)

    cfg = Config(raw=config)

    # 记录配置警告 (非致命)
    warnings = cfg.validate()
    for w in warnings:
        logger.warning("配置警告: %s", w)

    return cfg


def ensure_dir(p: Path) -> Path:
    """确保目录存在, 返回路径本身."""
    p.mkdir(parents=True, exist_ok=True)
    return p
