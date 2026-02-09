"""
Benchmark 评估器

加载 gold-standard CSV + 排序结果 CSV, 计算 per-disease 和聚合指标
"""
from __future__ import annotations
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .metrics import (
    hit_at_k, reciprocal_rank, precision_at_k,
    average_precision, ndcg_at_k, auroc,
)

logger = logging.getLogger(__name__)


def run_benchmark(
    rank_csv: Path,
    gold_csv: Path,
    ks: list[int] | None = None,
    score_col: str = "final_score",
) -> dict:
    """
    运行 benchmark 评估

    Args:
        rank_csv: 排序结果 CSV (需含 drug_normalized, diseaseId, final_score)
        gold_csv: Gold-standard CSV (需含 drug_normalized, diseaseId)
        ks: Hit@K / P@K / NDCG@K 的 K 值列表
        score_col: 用于排序的分数列名

    Returns:
        {
            "per_disease": {diseaseId: {metric: value, ...}, ...},
            "aggregate": {metric: mean_value, ...},
            "n_diseases_evaluated": int,
            "n_gold_pairs": int,
            "n_gold_found": int,
        }
    """
    if ks is None:
        ks = [5, 10, 20]

    rank_df = pd.read_csv(rank_csv, dtype=str)
    gold_df = pd.read_csv(gold_csv, dtype=str)

    rank_df[score_col] = pd.to_numeric(rank_df[score_col], errors="coerce").fillna(0)

    # 构建 gold set (per disease)
    gold_by_disease: dict[str, set[str]] = defaultdict(set)
    for _, r in gold_df.iterrows():
        drug = str(r["drug_normalized"]).lower().strip()
        disease = str(r["diseaseId"]).strip()
        gold_by_disease[disease].add(drug)

    n_gold_pairs = sum(len(v) for v in gold_by_disease.values())
    logger.info("Gold standard: %d 对 drug-disease, %d 个疾病",
                n_gold_pairs, len(gold_by_disease))

    # 对每个有 gold 数据的疾病计算指标
    per_disease: dict[str, dict] = {}
    n_gold_found = 0

    for did, positives in gold_by_disease.items():
        sub = rank_df[rank_df["diseaseId"] == did].sort_values(score_col, ascending=False)
        ranked_drugs = sub["drug_normalized"].tolist()

        if not ranked_drugs:
            logger.warning("疾病 %s 在排序结果中无记录, 跳过", did)
            continue

        found = positives & set(ranked_drugs)
        n_gold_found += len(found)

        metrics: dict[str, float] = {
            "n_ranked": float(len(ranked_drugs)),
            "n_positive": float(len(positives)),
            "n_found": float(len(found)),
            "mrr": reciprocal_rank(ranked_drugs, positives),
            "map": average_precision(ranked_drugs, positives),
            "auroc": auroc(ranked_drugs, positives),
        }
        for k in ks:
            metrics[f"hit@{k}"] = hit_at_k(ranked_drugs, positives, k)
            metrics[f"p@{k}"] = precision_at_k(ranked_drugs, positives, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_drugs, positives, k)

        per_disease[did] = metrics

    # 聚合 (macro average)
    aggregate: dict[str, float] = {}
    if per_disease:
        metric_keys = [k for k in next(iter(per_disease.values())) if k not in ("n_ranked", "n_positive", "n_found")]
        for key in metric_keys:
            vals = [m[key] for m in per_disease.values()]
            aggregate[key] = sum(vals) / len(vals)

    logger.info("评估完成: %d 个疾病, %d/%d gold pairs 在排序中找到",
                len(per_disease), n_gold_found, n_gold_pairs)

    return {
        "per_disease": per_disease,
        "aggregate": aggregate,
        "n_diseases_evaluated": len(per_disease),
        "n_gold_pairs": n_gold_pairs,
        "n_gold_found": n_gold_found,
    }


def format_report(result: dict) -> str:
    """格式化评估报告为可读文本"""
    lines = []
    lines.append("=" * 60)
    lines.append("Benchmark Evaluation Report")
    lines.append("=" * 60)
    lines.append(f"疾病数: {result['n_diseases_evaluated']}")
    lines.append(f"Gold pairs: {result['n_gold_pairs']} (找到 {result['n_gold_found']})")
    lines.append("")

    agg = result.get("aggregate", {})
    if agg:
        lines.append("Aggregate Metrics (macro avg):")
        lines.append("-" * 40)
        for k, v in sorted(agg.items()):
            lines.append(f"  {k:12s}: {v:.4f}")

    per = result.get("per_disease", {})
    if per:
        lines.append("")
        lines.append("Per-Disease Breakdown:")
        lines.append("-" * 40)
        for did in sorted(per):
            m = per[did]
            lines.append(f"  {did}:")
            lines.append(f"    ranked={int(m['n_ranked'])}, pos={int(m['n_positive'])}, found={int(m['n_found'])}")
            lines.append(f"    MRR={m['mrr']:.4f}  MAP={m['map']:.4f}  AUROC={m['auroc']:.4f}")
            hit_keys = sorted(k for k in m if k.startswith("hit@"))
            if hit_keys:
                hits = " ".join(f"{k}={m[k]:.2f}" for k in hit_keys)
                lines.append(f"    {hits}")

    lines.append("=" * 60)
    return "\n".join(lines)
