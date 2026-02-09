"""
V2 排序器: Drug-Target-Disease

路径类型: Drug --[HAS_TARGET]--> Target --[MAP_GENE]--> Gene --[ASSOC_DISEASE]--> Disease
数据来源:
  - Drug-Target: ChEMBL mechanism
  - Target-Gene: ChEMBL xref (Ensembl)
  - Gene-Disease: OpenTargets

评分逻辑:
  path_score = ot_score * hub_penalty(target_degree)^lambda
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from ..config import Config, ensure_dir
from ..utils import read_csv, require_cols, write_jsonl
from .base import hub_penalty


def run_v2(cfg: Config) -> dict[str, Path]:
    """
    运行V2排序

    Returns:
        输出文件路径字典
    """
    output_dir = ensure_dir(cfg.output_dir)
    data_dir = cfg.data_dir
    files = cfg.files
    rank_cfg = cfg.rank

    # 加载数据
    dt = read_csv(data_dir / files.get("drug_target", "edge_drug_target.csv"), dtype=str)
    m = read_csv(data_dir / files.get("target_ensembl", "target_chembl_to_ensembl_all.csv"), dtype=str)
    ot = read_csv(data_dir / files.get("gene_disease", "edge_target_disease_ot.csv"), dtype=str)

    require_cols(dt, {"drug_normalized", "target_chembl_id"}, "edge_drug_target")
    require_cols(m, {"target_chembl_id", "ensembl_gene_id"}, "target_to_ensembl")
    require_cols(ot, {"targetId", "diseaseId", "score"}, "edge_gene_disease")

    ot["score_f"] = pd.to_numeric(ot["score"], errors="coerce").fillna(0.0)

    # 合并路径
    # 只保留路径核心列, 避免 drug_raw/mechanism_of_action 等额外列造成假性重复
    dt_core = dt[["drug_normalized", "target_chembl_id"]].drop_duplicates()
    dtg = dt_core.merge(m, on="target_chembl_id", how="inner")
    dtgd = dtg.merge(ot, left_on="ensembl_gene_id", right_on="targetId", how="inner")

    # 计算靶点度数
    tdeg = dtgd.groupby("target_chembl_id")["drug_normalized"].nunique().rename("target_deg")
    dtgd = dtgd.merge(tdeg, on="target_chembl_id", how="left")

    # 计算路径分数
    lam = float(rank_cfg.get("hub_penalty_lambda", 1.0))
    dtgd["path_score"] = dtgd["score_f"] * hub_penalty(dtgd["target_deg"]).pow(lam)

    # 每对取top K路径
    dtgd["pair_key"] = dtgd["drug_normalized"].astype(str) + "||" + dtgd["diseaseId"].astype(str)
    k = int(rank_cfg.get("topk_paths_per_pair", 10))
    top_paths = dtgd.sort_values("path_score", ascending=False).groupby("pair_key", as_index=False).head(k).copy()

    # 聚合
    pair = top_paths.groupby(["drug_normalized", "diseaseId"], as_index=False).agg(
        mechanism_score=("path_score", "sum"),
        diseaseName=("diseaseName", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
    )
    pair["final_score"] = pair["mechanism_score"]
    pair = pair.sort_values(["drug_normalized", "final_score"], ascending=[True, False])

    # 输出
    out_csv = output_dir / "drug_disease_rank_v2.csv"
    pair.to_csv(out_csv, index=False)

    # 证据路径
    ev_path = output_dir / "evidence_paths_v2.jsonl"
    write_jsonl(ev_path, [
        {
            "drug": r["drug_normalized"],
            "diseaseId": r["diseaseId"],
            "diseaseName": r.get("diseaseName", ""),
            "path_score": float(r["path_score"]),
            "nodes": [
                {"type": "Drug", "id": r["drug_normalized"]},
                {"type": "Target", "id": r["target_chembl_id"]},
                {"type": "Gene", "id": r["ensembl_gene_id"]},
                {"type": "Disease", "id": r["diseaseId"], "name": r.get("diseaseName", "")},
            ],
            "edges": [
                {"rel": "DRUG_HAS_TARGET", "src": r["drug_normalized"], "dst": r["target_chembl_id"], "source": "ChEMBL"},
                {"rel": "TARGET_MAP_GENE", "src": r["target_chembl_id"], "dst": r["ensembl_gene_id"], "source": "ChEMBL"},
                {"rel": "GENE_ASSOC_DISEASE", "src": r["ensembl_gene_id"], "dst": r["diseaseId"], "source": "OpenTargets", "score": float(r["score_f"])},
            ],
        }
        for _, r in top_paths.iterrows()
    ])

    return {"rank_csv": out_csv, "evidence_paths": ev_path}
