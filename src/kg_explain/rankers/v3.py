"""
V3 排序器: Drug-Target-Pathway-Disease

路径类型:
  Drug --[HAS_TARGET]--> Target --[IN_PATHWAY]--> Pathway --[ASSOC_DISEASE]--> Disease

数据来源:
  - Drug-Target: ChEMBL mechanism
  - Target-Pathway: Reactome (via UniProt)
  - Pathway-Disease: OpenTargets (聚合)

评分逻辑:
  path_score = pathway_score * hub_penalty(target_degree)^lambda * (1 + boost * log(support_genes))
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, ensure_dir
from ..utils import read_csv, require_cols, write_jsonl
from .base import hub_penalty


def run_v3(cfg: Config) -> dict[str, Path]:
    """
    运行V3排序

    Returns:
        输出文件路径字典
    """
    output_dir = ensure_dir(cfg.output_dir)
    data_dir = cfg.data_dir
    files = cfg.files
    rank_cfg = cfg.rank

    # 加载数据
    dt = read_csv(data_dir / files.get("drug_target", "edge_drug_target.csv"), dtype=str)
    tp = read_csv(data_dir / files.get("target_pathway", "edge_target_pathway_all.csv"), dtype=str)
    pd_edge = read_csv(data_dir / files.get("pathway_disease", "edge_pathway_disease.csv"), dtype=str)

    require_cols(dt, {"drug_normalized", "target_chembl_id"}, "edge_drug_target")
    require_cols(tp, {"target_chembl_id", "reactome_stid"}, "edge_target_pathway")
    require_cols(pd_edge, {"reactome_stid", "diseaseId", "pathway_score", "support_genes"}, "edge_pathway_disease")

    pd_edge["pathway_score_f"] = pd.to_numeric(pd_edge["pathway_score"], errors="coerce").fillna(0.0)
    pd_edge["support_genes_f"] = pd.to_numeric(pd_edge["support_genes"], errors="coerce").fillna(1.0)

    # 合并路径
    # 只保留路径核心列, 避免 drug_raw/mechanism_of_action 等额外列造成假性重复
    dt_core = dt[["drug_normalized", "target_chembl_id"]].drop_duplicates()
    dtp = dt_core.merge(tp, on="target_chembl_id", how="inner").dropna(
        subset=["drug_normalized", "target_chembl_id", "reactome_stid"]
    ).drop_duplicates()

    # 计算靶点度数
    tdeg = dtp.groupby("target_chembl_id")["drug_normalized"].nunique().rename("target_deg")
    dtp = dtp.merge(tdeg, on="target_chembl_id", how="left")

    # Hub惩罚
    lam = float(rank_cfg.get("hub_penalty_lambda", 1.0))
    dtp["w_hub_target"] = hub_penalty(dtp["target_deg"]).pow(lam)

    # 合并通路-疾病
    # tp 和 pd_edge 都有 reactome_name 列, 用后缀区分后保留 tp 侧的
    paths = dtp.merge(pd_edge, on="reactome_stid", how="inner", suffixes=("", "_pd"))
    # 优先使用 tp 侧的 reactome_name, 缺失时回退到 pd_edge 侧
    if "reactome_name_pd" in paths.columns:
        paths["reactome_name"] = paths["reactome_name"].fillna(paths["reactome_name_pd"])
        paths.drop(columns=["reactome_name_pd"], inplace=True)

    # 计算路径分数
    sb = float(rank_cfg.get("support_gene_boost", 0.15))
    paths["w_support"] = 1.0 + sb * np.log1p(paths["support_genes_f"])
    paths["path_score"] = paths["pathway_score_f"] * paths["w_hub_target"] * paths["w_support"]

    # 每对取top K路径
    paths["pair_key"] = paths["drug_normalized"].astype(str) + "||" + paths["diseaseId"].astype(str)
    k = int(rank_cfg.get("topk_paths_per_pair", 10))
    top_paths = paths.sort_values("path_score", ascending=False).groupby("pair_key", as_index=False).head(k).copy()

    # 聚合
    pair = top_paths.groupby(["drug_normalized", "diseaseId"], as_index=False).agg(
        mechanism_score=("path_score", "sum"),
        diseaseName=("diseaseName", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
    )
    pair["final_score"] = pair["mechanism_score"]
    pair = pair.sort_values(["drug_normalized", "final_score"], ascending=[True, False])

    # 输出
    out_csv = output_dir / "drug_disease_rank_v3.csv"
    pair.to_csv(out_csv, index=False)

    # 证据路径
    ev_path = output_dir / "evidence_paths_v3.jsonl"
    write_jsonl(ev_path, [
        {
            "drug": r["drug_normalized"],
            "diseaseId": r["diseaseId"],
            "diseaseName": r.get("diseaseName", ""),
            "path_score": float(r["path_score"]),
            "nodes": [
                {"type": "Drug", "id": r["drug_normalized"]},
                {"type": "Target", "id": r["target_chembl_id"]},
                {"type": "Pathway", "id": r["reactome_stid"], "name": r.get("reactome_name", "")},
                {"type": "Disease", "id": r["diseaseId"], "name": r.get("diseaseName", "")},
            ],
            "edges": [
                {"rel": "DRUG_HAS_TARGET", "src": r["drug_normalized"], "dst": r["target_chembl_id"], "source": "ChEMBL"},
                {"rel": "TARGET_IN_PATHWAY", "src": r["target_chembl_id"], "dst": r["reactome_stid"], "source": "Reactome"},
                {"rel": "PATHWAY_ASSOC_DISEASE", "src": r["reactome_stid"], "dst": r["diseaseId"], "source": "OpenTargets(agg)",
                 "pathway_score": float(r.get("pathway_score_f", 0)), "support_genes": int(float(r["support_genes_f"]))},
            ],
        }
        for _, r in top_paths.iterrows()
    ])

    return {"rank_csv": out_csv, "evidence_paths": ev_path}
