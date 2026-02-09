"""
FDA FAERS 数据源

用途: 获取药物不良事件报告，并计算 PRR 信号检测
API: https://api.fda.gov/drug/event.json

FAERS = FDA Adverse Event Reporting System

PRR (Proportional Reporting Ratio):
  PRR = P(AE|Drug) / P(AE|¬Drug)
      = (a / N_drug) / (c / N_other)

  其中:
    a       = 该药物报告该AE的数量
    N_drug  = 该药物所有AE报告总量
    c       = 其他所有药物报告该AE的数量 (= bg_count - a)
    N_other = 其他所有药物AE报告总量 (= bg_total - N_drug)

  PRR ≥ 2 且 N ≥ 3 通常视为安全信号
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

from ..cache import HTTPCache, cached_get_json
from ..utils import concurrent_map

logger = logging.getLogger(__name__)

# openFDA API端点
FAERS_API = "https://api.fda.gov/drug/event.json"


def _fetch_background_ae(cache: HTTPCache, limit: int = 1000) -> tuple[dict[str, int], int]:
    """
    获取全局 AE 背景发生率 (不限定药物)

    Returns:
        (ae_counts, total): AE术语→报告数映射, 总AE提及数
    """
    params = {
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    try:
        js = cached_get_json(cache, FAERS_API, params=params, timeout=60)
        results = js.get("results") or []
        ae_counts = {r["term"]: r["count"] for r in results if r.get("term")}
        total = sum(ae_counts.values())
        logger.info("FAERS 背景AE加载完成: %d 个AE术语, 总提及数=%d", len(ae_counts), total)
        return ae_counts, total
    except Exception as e:
        logger.warning("FAERS 背景AE获取失败: %s", e)
        return {}, 0


def _calc_prr(a: int, drug_total: int, bg_ae_count: int, bg_total: int) -> float:
    """
    计算 PRR (Proportional Reporting Ratio)

    Args:
        a: 该药物报告该AE的数量
        drug_total: 该药物所有AE报告总量
        bg_ae_count: 全局该AE报告总量
        bg_total: 全局所有AE报告总量

    Returns:
        PRR 值; 无法计算时返回 0.0
    """
    if drug_total <= 0 or bg_total <= drug_total:
        return 0.0

    p_drug = a / drug_total

    c = bg_ae_count - a  # 其他药物报告该AE的数量
    n_other = bg_total - drug_total  # 其他药物的AE总量

    if c <= 0 or n_other <= 0:
        return 0.0

    p_other = c / n_other
    return p_drug / p_other


def _faers_drug_ae(cache: HTTPCache, drug_name: str, limit: int = 100) -> list[dict]:
    """查询单个药物的不良事件"""
    # 搜索品牌名或通用名
    search = f'(patient.drug.openfda.brand_name:"{drug_name}"+patient.drug.openfda.generic_name:"{drug_name}")'
    params = {
        "search": search,
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    try:
        js = cached_get_json(cache, FAERS_API, params=params, timeout=30)
        return js.get("results") or []
    except Exception as e:
        logger.debug("FAERS 查询无结果, drug=%s: %s", drug_name, e)
        return []


def fetch_drug_ae(
    data_dir: Path,
    cache: HTTPCache,
    drug_list: list[str],
    min_count: int = 5,
    min_prr: float = 0.0,
    top_ae: int = 50,
    max_drugs: int = 500,
) -> Path:
    """
    获取药物-不良事件关系 (edge_drug_ae) 并计算 PRR

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        drug_list: 药物名称列表
        min_count: 最小报告数
        min_prr: 最小 PRR 阈值 (低于此值不视为信号，0 表示不过滤)
        top_ae: 每个药物保留的 top AE 数
        max_drugs: 最多处理的药物数

    Returns:
        输出文件路径
    """
    # 获取全局背景率 (只请求一次)
    bg_ae, bg_total = _fetch_background_ae(cache)
    has_bg = bg_total > 0

    def _fetch_one(drug):
        aes = _faers_drug_ae(cache, drug, limit=100)
        drug_total = sum(x.get("count", 0) for x in aes)

        drug_rows = []
        filtered = 0
        for ae in aes[:top_ae]:
            term = ae.get("term", "")
            count = ae.get("count", 0)
            if count < min_count or not term:
                continue

            prr = 0.0
            if has_bg:
                prr = _calc_prr(count, drug_total, bg_ae.get(term, 0), bg_total)

            if min_prr > 0 and has_bg and prr < min_prr:
                filtered += 1
                continue

            drug_rows.append({
                "drug_normalized": drug.lower().strip(),
                "ae_term": term,
                "report_count": count,
                "drug_total_reports": drug_total,
                "prr": round(prr, 4),
            })
        return drug_rows, filtered

    results = concurrent_map(
        _fetch_one, drug_list[:max_drugs],
        max_workers=cache.max_workers, desc="FAERS Drug→AE",
    )
    rows = [row for drug_rows, _ in results if drug_rows for row in drug_rows]
    n_prr_filtered = sum(f for _, f in results if f)

    out_df = pd.DataFrame(rows).drop_duplicates()
    n_signals = len(out_df[out_df["prr"] >= 2.0]) if not out_df.empty and "prr" in out_df.columns else 0
    logger.info("Drug→AE 关系: %d 条边, %d 个药物, %d 个信号(PRR≥2), %d 条被PRR过滤",
                len(out_df), out_df["drug_normalized"].nunique() if not out_df.empty else 0,
                n_signals, n_prr_filtered)

    out = data_dir / "edge_drug_ae_faers.csv"
    out_df.to_csv(out, index=False)
    return out
