"""
OpenTargets 数据源

用途:
  1. 基因(Ensembl) → 疾病 关联
  2. 疾病 → 表型 关联

API: https://api.platform.opentargets.org/api/v4/graphql
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

from ..cache import HTTPCache, cached_post_json
from ..utils import read_csv, concurrent_map

logger = logging.getLogger(__name__)

# OpenTargets GraphQL端点
OT_API = "https://api.platform.opentargets.org/api/v4/graphql"


def fetch_gene_diseases(
    data_dir: Path,
    cache: HTTPCache,
    endpoint: str = OT_API,
    page_size: int = 200,
    max_pages: int = 5,
    max_diseases_per_gene: int = 200,
) -> Path:
    """
    获取基因-疾病关联 (edge_target_disease)

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        endpoint: GraphQL端点
        page_size: 每页大小
        max_pages: 每个基因最大页数
        max_diseases_per_gene: 每个基因最大疾病数

    Returns:
        输出文件路径
    """
    m = read_csv(data_dir / "target_chembl_to_ensembl_all.csv", dtype=str)
    genes = sorted(set([
        g for g in m["ensembl_gene_id"].dropna().astype(str).tolist()
        if g.startswith("ENSG")
    ]))

    headers = {"Content-Type": "application/json"}
    query = """
    query($ensg: String!, $size: Int!, $index: Int!) {
      target(ensemblId: $ensg) {
        associatedDiseases(page: {size: $size, index: $index}) {
          rows { score disease { id name } }
        }
      }
    }
    """

    def _fetch_one_gene(g):
        gene_rows = []
        got = 0
        for i in range(max_pages):
            payload = {
                "query": query,
                "variables": {"ensg": g, "size": int(page_size), "index": int(i)},
            }
            try:
                js = cached_post_json(cache, endpoint, payload, headers=headers, timeout=60)
            except Exception as e:
                logger.warning("OpenTargets Gene→Disease 查询失败, gene=%s page=%d: %s", g, i, e)
                break

            tgt = (js.get("data") or {}).get("target") or {}
            page_rows = (tgt.get("associatedDiseases") or {}).get("rows") or []
            if not page_rows:
                break

            for row in page_rows:
                dis = row.get("disease") or {}
                gene_rows.append({
                    "targetId": g,
                    "diseaseId": dis.get("id"),
                    "diseaseName": dis.get("name"),
                    "score": row.get("score"),
                })
                got += 1
                if got >= max_diseases_per_gene:
                    break

            if got >= max_diseases_per_gene:
                break
        return gene_rows

    results = concurrent_map(
        _fetch_one_gene, genes,
        max_workers=cache.max_workers, desc="OpenTargets Gene→Disease",
    )
    rows = [row for result in results for row in result]

    out_df = pd.DataFrame(rows).dropna(subset=["targetId", "diseaseId", "score"]).drop_duplicates()
    logger.info("Gene→Disease 关系: %d 条边, %d 个基因, %d 个疾病",
                len(out_df), out_df["targetId"].nunique(), out_df["diseaseId"].nunique())

    out = data_dir / "edge_target_disease_ot.csv"
    out_df.to_csv(out, index=False)
    return out


def fetch_disease_phenotypes(
    data_dir: Path,
    cache: HTTPCache,
    disease_ids: list[str],
    endpoint: str = OT_API,
    min_score: float = 0.3,
    max_phenotypes: int = 30,
) -> Path:
    """
    获取疾病-表型关联 (edge_disease_phenotype)

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        disease_ids: 疾病ID列表
        endpoint: GraphQL端点
        min_score: 最小分数阈值
        max_phenotypes: 每个疾病最大表型数

    Returns:
        输出文件路径
    """
    headers = {"Content-Type": "application/json"}
    query = """
    query($diseaseId: String!) {
      disease(efoId: $diseaseId) {
        id
        name
        phenotypes(page: {size: 100, index: 0}) {
          rows {
            phenotypeEFO { id name }
            evidence { score }
          }
        }
      }
    }
    """

    def _fetch_one(dis_id):
        payload = {"query": query, "variables": {"diseaseId": dis_id}}
        try:
            js = cached_post_json(cache, endpoint, payload, headers=headers, timeout=60)
        except Exception as e:
            logger.warning("OpenTargets Disease→Phenotype 查询失败, disease=%s: %s", dis_id, e)
            return []

        disease = (js.get("data") or {}).get("disease") or {}
        phenos = ((disease.get("phenotypes") or {}).get("rows") or [])[:max_phenotypes]

        result_rows = []
        for p in phenos:
            phe = p.get("phenotypeEFO") or {}
            evidence = p.get("evidence") or {}
            score = float(evidence.get("score", 0) or 0)
            if score < min_score:
                continue
            result_rows.append({
                "diseaseId": dis_id,
                "diseaseName": disease.get("name", ""),
                "phenotypeId": phe.get("id", ""),
                "phenotypeName": phe.get("name", ""),
                "score": score,
            })
        return result_rows

    results = concurrent_map(
        _fetch_one, disease_ids,
        max_workers=cache.max_workers, desc="OpenTargets Disease→Phenotype",
    )
    rows = [row for result in results for row in result]

    out_df = pd.DataFrame(rows).drop_duplicates()
    logger.info("Disease→Phenotype 关系: %d 条边, %d 个疾病",
                len(out_df), out_df["diseaseId"].nunique() if not out_df.empty else 0)

    out = data_dir / "edge_disease_phenotype.csv"
    out_df.to_csv(out, index=False)
    return out
