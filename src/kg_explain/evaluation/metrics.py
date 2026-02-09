"""
排序评估指标 — 工业级, 含输入验证和边界情况处理

纯函数, 输入为排序后的药物列表和已知正例集合
所有指标: 值域 [0, 1], 越大越好

Improvements (v0.6.0):
    - 所有函数增加输入验证 (k > 0, 非空列表)
    - 边界情况: 空列表、空正例集、k > len(ranked)
    - 类型安全: 确保 positives 为 set
"""
from __future__ import annotations
import math


def _validate_inputs(ranked: list[str], positives: set[str], k: int | None = None) -> set[str]:
    """验证通用输入, 返回确保为 set 的 positives."""
    if not isinstance(ranked, (list, tuple)):
        raise TypeError(f"ranked 必须是 list, 得到 {type(ranked).__name__}")
    if not isinstance(positives, (set, frozenset)):
        # 容忍 list/tuple, 自动转 set
        if isinstance(positives, (list, tuple)):
            positives = set(positives)
        else:
            raise TypeError(f"positives 必须是 set, 得到 {type(positives).__name__}")
    if k is not None:
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k 必须是正整数, 得到 {k}")
    return positives


def hit_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    """
    Hit@K: top-K 中是否包含至少一个正例.

    Args:
        ranked: 排序后的药物 ID 列表.
        positives: 已知正例集合.
        k: 截断位置.

    Returns:
        1.0 如果 top-K 中有正例, 否则 0.0.
    """
    positives = _validate_inputs(ranked, positives, k)
    if not ranked or not positives:
        return 0.0
    return float(any(d in positives for d in ranked[:k]))


def reciprocal_rank(ranked: list[str], positives: set[str]) -> float:
    """
    Reciprocal Rank: 第一个正例的排名倒数.

    Args:
        ranked: 排序后的药物 ID 列表.
        positives: 已知正例集合.

    Returns:
        1/rank (1-indexed), 无正例时返回 0.0.
    """
    positives = _validate_inputs(ranked, positives)
    if not ranked or not positives:
        return 0.0
    for i, d in enumerate(ranked):
        if d in positives:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    """
    Precision@K: top-K 中正例的比例.

    Args:
        ranked: 排序后的药物 ID 列表.
        positives: 已知正例集合.
        k: 截断位置.

    Returns:
        正例数 / K.
    """
    positives = _validate_inputs(ranked, positives, k)
    if not ranked or not positives:
        return 0.0
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in positives) / len(top_k)


def average_precision(ranked: list[str], positives: set[str]) -> float:
    """
    Average Precision (AP): 每个正例位置的 Precision 的均值.

    MAP = mean(AP) across queries

    Args:
        ranked: 排序后的药物 ID 列表.
        positives: 已知正例集合.

    Returns:
        AP 值, 无正例时返回 0.0.
    """
    positives = _validate_inputs(ranked, positives)
    if not positives or not ranked:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked):
        if d in positives:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(positives)


def _dcg_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    """Discounted Cumulative Gain at K (binary relevance)."""
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in positives:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 log2(1) = 0
    return dcg


def ndcg_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at K.

    Ideal DCG = 所有正例排在最前面

    Args:
        ranked: 排序后的药物 ID 列表.
        positives: 已知正例集合.
        k: 截断位置.

    Returns:
        NDCG 值 [0, 1], 无正例时返回 0.0.
    """
    positives = _validate_inputs(ranked, positives, k)
    if not ranked or not positives:
        return 0.0
    dcg = _dcg_at_k(ranked, positives, k)
    # Ideal: min(n_positive, k) 个正例排在最前
    n_ideal = min(len(positives), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def auroc(ranked: list[str], positives: set[str]) -> float:
    """
    AUROC: Area Under ROC Curve.

    等价于: 随机选一个正例和一个负例, 正例排名靠前的概率

    Args:
        ranked: 排序后的药物 ID 列表.
        positives: 已知正例集合.

    Returns:
        AUC 值 [0, 1], 全正或全负时返回 0.0.
    """
    positives = _validate_inputs(ranked, positives)
    if not ranked or not positives:
        return 0.0

    n_pos = sum(1 for d in ranked if d in positives)
    n_neg = len(ranked) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # 对每个正例, 统计排在它后面的负例数
    concordant = 0
    neg_below = n_neg
    for d in ranked:
        if d in positives:
            concordant += neg_below
        else:
            neg_below -= 1
    return concordant / (n_pos * n_neg)
