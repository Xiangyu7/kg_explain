"""
边数据构建

构建中间关系数据:
  - 基因-通路 (通过靶点-通路 + 靶点-基因 合并)
  - 通路-疾病 (聚合基因-疾病分数)
  - 试验-不良事件 (解析whyStopped字段)
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

from ..config import Config
from ..utils import read_csv, require_cols, safe_str, load_canonical_map

logger = logging.getLogger(__name__)


def build_gene_pathway(cfg: Config) -> Path:
    """
    构建基因-通路关系

    合并: target_pathway + target_to_ensembl → gene_pathway
    """
    data_dir = cfg.data_dir
    files = cfg.files

    tp = read_csv(data_dir / files.get("target_pathway", "edge_target_pathway_all.csv"), dtype=str)
    m = read_csv(data_dir / files.get("target_ensembl", "target_chembl_to_ensembl_all.csv"), dtype=str)

    require_cols(tp, {"target_chembl_id", "reactome_stid"}, "edge_target_pathway")
    require_cols(m, {"target_chembl_id", "ensembl_gene_id"}, "target_to_ensembl")

    cols = ["ensembl_gene_id", "reactome_stid"]
    if "reactome_name" in tp.columns:
        cols.append("reactome_name")

    gp = tp.merge(m, on="target_chembl_id", how="inner")[cols].dropna().drop_duplicates()
    logger.info("Gene→Pathway 构建完成: %d 条边, %d 个基因, %d 个通路",
                len(gp), gp["ensembl_gene_id"].nunique(), gp["reactome_stid"].nunique())

    out = data_dir / files.get("gene_pathway", "edge_gene_pathway.csv")
    gp.to_csv(out, index=False)
    return out


def build_pathway_disease(cfg: Config) -> Path:
    """
    构建通路-疾病关系

    聚合: gene_pathway + gene_disease → pathway_disease
    """
    data_dir = cfg.data_dir
    files = cfg.files

    gp = read_csv(data_dir / files.get("gene_pathway", "edge_gene_pathway.csv"), dtype=str)
    ot = read_csv(data_dir / files.get("gene_disease", "edge_target_disease_ot.csv"), dtype=str)

    require_cols(ot, {"targetId", "diseaseId", "score"}, "edge_gene_disease")

    genes = set(gp["ensembl_gene_id"].dropna().astype(str).tolist())
    ot = ot[ot["targetId"].isin(genes)].copy()
    ot["score_f"] = pd.to_numeric(ot["score"], errors="coerce").fillna(0.0)

    j = gp.merge(ot, left_on="ensembl_gene_id", right_on="targetId", how="inner")

    agg = j.groupby(["reactome_stid", "diseaseId"], as_index=False).agg(
        pathway_score=("score_f", "max"),
        support_genes=("ensembl_gene_id", "nunique"),
        diseaseName=("diseaseName", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
    )

    if "reactome_name" in gp.columns:
        pn = gp[["reactome_stid", "reactome_name"]].drop_duplicates()
        agg = agg.merge(pn, on="reactome_stid", how="left")

    logger.info("Pathway→Disease 构建完成: %d 条边, %d 个通路, %d 个疾病",
                len(agg), agg["reactome_stid"].nunique(), agg["diseaseId"].nunique())

    out = data_dir / files.get("pathway_disease", "edge_pathway_disease.csv")
    agg.to_csv(out, index=False)
    return out


def build_trial_ae(data_dir: Path) -> Path:
    """
    构建试验-不良事件关系

    从whyStopped字段解析安全相关和疗效相关的停止原因
    使用 canonical map 统一药物名称
    """
    ft = read_csv(data_dir / "failed_trials_drug_rows.csv", dtype=str)
    canonical = load_canonical_map(data_dir)

    rows = []
    for _, r in ft.iterrows():
        nct = safe_str(r.get("nctId"))
        drug_raw = safe_str(r.get("drug_raw")).lower()
        drug = canonical.get(drug_raw, drug_raw)
        why = safe_str(r.get("whyStopped"))
        status = safe_str(r.get("overallStatus"))

        if not nct or not drug:
            continue

        # 安全相关关键词
        is_safety = any(kw in why.lower() for kw in [
            "adverse", "toxicity", "safety", "death", "fatal", "serious"
        ])
        # 疗效相关关键词
        is_efficacy = any(kw in why.lower() for kw in [
            "efficacy", "futility", "ineffective", "no benefit"
        ])

        rows.append({
            "nctId": nct,
            "drug_normalized": drug,
            "overallStatus": status,
            "whyStopped": why,
            "is_safety_stop": int(is_safety),
            "is_efficacy_stop": int(is_efficacy),
        })

    _trial_ae_cols = ["nctId", "drug_normalized", "overallStatus",
                      "whyStopped", "is_safety_stop", "is_efficacy_stop"]
    out_df = (pd.DataFrame(rows, columns=_trial_ae_cols).drop_duplicates()
              if rows else pd.DataFrame(columns=_trial_ae_cols))
    n_safety = out_df["is_safety_stop"].astype(int).sum() if not out_df.empty else 0
    n_efficacy = out_df["is_efficacy_stop"].astype(int).sum() if not out_df.empty else 0
    logger.info("Trial→AE 构建完成: %d 条记录, %d 个安全停止, %d 个疗效停止",
                len(out_df), n_safety, n_efficacy)

    out = data_dir / "edge_trial_ae.csv"
    out_df.to_csv(out, index=False)
    return out
