"""
FDA FAERS 数据源

用途: 获取药物不良事件报告
API: https://api.fda.gov/drug/event.json

FAERS = FDA Adverse Event Reporting System
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..cache import HTTPCache, cached_get_json

# openFDA API端点
FAERS_API = "https://api.fda.gov/drug/event.json"


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
    except Exception:
        return []


def fetch_drug_ae(
    data_dir: Path,
    cache: HTTPCache,
    drug_list: list[str],
    min_count: int = 5,
    top_ae: int = 50,
) -> Path:
    """
    获取药物-不良事件关系 (edge_drug_ae)

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        drug_list: 药物名称列表
        min_count: 最小报告数
        top_ae: 每个药物保留的top AE数

    Returns:
        输出文件路径
    """
    rows = []

    for drug in tqdm(drug_list[:500], desc="FAERS Drug→AE"):
        aes = _faers_drug_ae(cache, drug, limit=100)
        drug_total = sum(x.get("count", 0) for x in aes)

        for ae in aes[:top_ae]:
            term = ae.get("term", "")
            count = ae.get("count", 0)
            if count < min_count or not term:
                continue

            rows.append({
                "drug_normalized": drug.lower().strip(),
                "ae_term": term,
                "report_count": count,
                "drug_total_reports": drug_total,
            })

    out = data_dir / "edge_drug_ae_faers.csv"
    pd.DataFrame(rows).drop_duplicates().to_csv(out, index=False)
    return out
