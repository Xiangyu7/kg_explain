"""
Reactome 数据源

用途: 获取靶点(UniProt) → 通路(Pathway)关系
API: https://reactome.org/ContentService
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

from ..cache import HTTPCache, cached_get_json
from ..utils import read_csv, concurrent_map

logger = logging.getLogger(__name__)

# Reactome API端点
REACTOME_API = "https://reactome.org/ContentService"


def _reactome_pathways_for_uniprot(cache: HTTPCache, uniprot: str) -> list[dict]:
    """获取UniProt蛋白参与的通路"""
    url = f"{REACTOME_API}/data/mapping/UniProt/{uniprot}/pathways"
    js = cached_get_json(cache, url, params=None)
    if isinstance(js, list):
        return js
    return js.get("pathways") or js.get("results") or []


def fetch_target_pathways(
    data_dir: Path,
    cache: HTTPCache,
) -> Path:
    """
    获取靶点-通路关系 (edge_target_pathway)

    通过UniProt ID从Reactome获取通路信息

    Returns:
        输出文件路径
    """
    xref = read_csv(data_dir / "target_xref.csv", dtype=str)
    up = xref[["target_chembl_id", "uniprot_accession"]].dropna().drop_duplicates()
    pair_list = [(r["target_chembl_id"], r["uniprot_accession"]) for _, r in up.iterrows()]

    def _fetch_one(pair):
        tid, u = pair
        try:
            ps = _reactome_pathways_for_uniprot(cache, u)
        except Exception as e:
            logger.warning("Reactome 查询失败, uniprot=%s: %s", u, e)
            return []
        return [{
            "target_chembl_id": tid,
            "uniprot_accession": u,
            "reactome_stid": p.get("stId") or p.get("stIdVersion") or p.get("id"),
            "reactome_name": p.get("displayName") or p.get("name"),
        } for p in ps]

    results = concurrent_map(
        _fetch_one, pair_list,
        max_workers=cache.max_workers, desc="Reactome Target→Pathway",
    )
    rows = [row for result in results for row in result]

    out_df = pd.DataFrame(rows).dropna(subset=["target_chembl_id", "reactome_stid"]).drop_duplicates()
    logger.info("Target→Pathway 关系: %d 条边, %d 个靶点, %d 个通路",
                len(out_df), out_df["target_chembl_id"].nunique(), out_df["reactome_stid"].nunique())

    out = data_dir / "edge_target_pathway_all.csv"
    out_df.to_csv(out, index=False)
    return out
