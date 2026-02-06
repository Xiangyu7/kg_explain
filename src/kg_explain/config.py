"""
配置加载模块

支持配置继承: base.yaml → disease.yaml → version.yaml
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class Config:
    """配置数据类"""
    raw: dict

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "v5")).lower()

    @property
    def data_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("data_dir", "./data"))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("output_dir", "./output"))

    @property
    def cache_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("cache_dir", "./cache"))

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


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict:
    """加载YAML文件"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    base_path: str = "configs/base.yaml",
    disease_path: Optional[str] = None,
    version_path: Optional[str] = None,
) -> Config:
    """
    加载并合并配置

    优先级: version > disease > base
    """
    # 加载基础配置
    config = load_yaml(Path(base_path))

    # 合并疾病配置
    if disease_path:
        disease_cfg = load_yaml(Path(disease_path))
        config = _deep_merge(config, disease_cfg)

    # 合并版本配置
    if version_path:
        version_cfg = load_yaml(Path(version_path))
        config = _deep_merge(config, version_cfg)

    return Config(raw=config)


def ensure_dir(p: Path) -> Path:
    """确保目录存在"""
    p.mkdir(parents=True, exist_ok=True)
    return p
