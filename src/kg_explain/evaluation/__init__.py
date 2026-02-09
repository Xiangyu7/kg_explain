"""
Benchmark 评估模块

用于评估药物重定位排序质量:
  - 标准 IR 指标: Hit@K, MRR, P@K, AP, NDCG@K, AUROC
  - 支持 gold-standard CSV 对照
"""
from .metrics import hit_at_k, reciprocal_rank, precision_at_k, average_precision, ndcg_at_k, auroc
from .benchmark import run_benchmark

__all__ = [
    "hit_at_k", "reciprocal_rank", "precision_at_k",
    "average_precision", "ndcg_at_k", "auroc",
    "run_benchmark",
]
