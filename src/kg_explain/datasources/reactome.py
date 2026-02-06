"""
Reactome 数据源

用途: 获取靶点(UniProt) → 通路(Pathway)关系
API: https://reactome.org/ContentService
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..cache import HTTPCache, cached_get_json
from ..utils import read_csv

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

    rows = []
    for _, r in tqdm(up.iterrows(), total=len(up), desc="Reactome Target→Pathway"):
        tid = r["target_chembl_id"]
        u = r["uniprot_accession"]

        try:
            ps = _reactome_pathways_for_uniprot(cache, u)
        except Exception:
            ps = []

        for p in ps:
            rows.append({
                "target_chembl_id": tid,
                "uniprot_accession": u,
                "reactome_stid": p.get("stId") or p.get("stIdVersion") or p.get("id"),
                "reactome_name": p.get("displayName") or p.get("name"),
            })

    out = data_dir / "edge_target_pathway_all.csv"
    pd.DataFrame(rows).dropna(subset=["target_chembl_id", "reactome_stid"]).drop_duplicates().to_csv(out, index=False)
    return out
