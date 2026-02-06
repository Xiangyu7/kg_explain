"""
RxNorm 数据源

用途: 药物名称标准化
API: https://rxnav.nlm.nih.gov/REST/approximateTerm.json
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..cache import HTTPCache, cached_get_json
from ..utils import read_csv

# RxNorm API端点
RX_API = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"


def rxnorm_map(
    data_dir: Path,
    cache: HTTPCache,
    max_entries: int = 3,
) -> Path:
    """
    将药物名称映射到RxNorm

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        max_entries: 每个药物最多返回的匹配数

    Returns:
        输出文件路径
    """
    df = read_csv(data_dir / "failed_trials_drug_rows.csv")
    drugs = sorted(set([
        x for x in df["drug_raw"].dropna().astype(str).tolist() if x.strip()
    ]))

    rows = []
    for d in tqdm(drugs, desc="RxNorm mapping"):
        try:
            js = cached_get_json(cache, RX_API, params={"term": d, "maxEntries": int(max_entries)})
            cand = ((js.get("approximateGroup") or {}).get("candidate") or [])[:max_entries]
            best = cand[0] if cand else {}
            rows.append({
                "drug_raw": d,
                "rxnorm_rxcui": best.get("rxcui"),
                "rxnorm_term": best.get("term"),
                "rxnorm_score": best.get("score"),
            })
        except Exception:
            rows.append({
                "drug_raw": d,
                "rxnorm_rxcui": None,
                "rxnorm_term": None,
                "rxnorm_score": None,
            })

    out = data_dir / "drug_rxnorm_map.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
