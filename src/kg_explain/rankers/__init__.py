"""
排序算法模块

支持的版本:
  - V1: Drug-Disease 直接关联 (CT.gov conditions)
  - V2: Drug-Target-Disease (ChEMBL + OpenTargets)
  - V3: Drug-Target-Pathway-Disease (+ Reactome)
  - V4: V3 + Evidence Pack (for RAG)
  - V5: 完整可解释路径 (+ FAERS + Phenotype)
"""
from .base import hub_penalty
from .v1 import run_v1
from .v2 import run_v2
from .v3 import run_v3
from .v4 import run_v4
from .v5 import run_v5

__all__ = ["hub_penalty", "run_v1", "run_v2", "run_v3", "run_v4", "run_v5"]


def run_pipeline(cfg) -> dict:
    """根据配置运行对应版本的排序"""
    m = cfg.mode
    if m in ("v1", "1"):
        return run_v1(cfg)
    if m in ("v2", "2"):
        return run_v2(cfg)
    if m in ("v3", "3"):
        return run_v3(cfg)
    if m in ("v4", "4"):
        return run_v4(cfg)
    if m in ("v5", "5", "default"):
        return run_v5(cfg)
    raise ValueError(f"未知模式: {m}")
