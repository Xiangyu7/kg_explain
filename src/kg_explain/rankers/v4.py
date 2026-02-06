"""
V4 排序器: V3 + Evidence Pack

基于V3的排序结果，生成用于RAG的证据包

输出:
  - drug_disease_rank_v4.csv (与V3相同)
  - evidence_pack/: 每个(drug, disease)对的完整证据JSON
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import pandas as pd

from ..config import Config, ensure_dir
from ..utils import read_csv
from .v3 import run_v3


def run_v4(cfg: Config) -> dict[str, Path]:
    """
    运行V4排序 (V3 + Evidence Pack)

    Returns:
        输出文件路径字典
    """
    # 先运行V3
    run_v3(cfg)

    output_dir = ensure_dir(cfg.output_dir)
    rank_cfg = cfg.rank

    # 创建证据包目录
    ep_dir = ensure_dir(output_dir / "evidence_pack")

    # 加载V3结果
    pair = read_csv(output_dir / "drug_disease_rank_v3.csv", dtype=str)
    paths = pd.read_json(output_dir / "evidence_paths_v3.jsonl", lines=True)

    # 为每对生成证据包
    for _, pr in pair.iterrows():
        drug = pr["drug_normalized"]
        dis = pr["diseaseId"]
        sub = paths[(paths["drug"] == drug) & (paths["diseaseId"] == dis)].copy()

        pack = {
            "drug": drug,
            "disease": {"id": dis, "name": pr.get("diseaseName", "")},
            "final_score": float(pd.to_numeric(pr.get("final_score", 0), errors="coerce") or 0),
            "top_paths": [],
        }

        for _, row in sub.head(int(rank_cfg.get("topk_paths_per_pair", 10))).iterrows():
            pn = ""
            for n in row.get("nodes") or []:
                if n.get("type") == "Pathway":
                    pn = n.get("name", "") or n.get("id", "")
                    break

            pack["top_paths"].append({
                "type": "DTPD",
                "score": float(row.get("path_score", 0)),
                "nodes": row.get("nodes", []),
                "edges": row.get("edges", []),
                "rag_queries": [
                    f"{drug} {pn} {pr.get('diseaseName', '')}".strip(),
                    f"{drug} {dis} {pn}".strip(),
                ],
            })

        safe = (drug + "__" + dis).replace("/", "_").replace(":", "_")
        (ep_dir / f"{safe}.json").write_text(
            json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 复制V3结果作为V4
    v4_csv = output_dir / "drug_disease_rank_v4.csv"
    shutil.copyfile(output_dir / "drug_disease_rank_v3.csv", v4_csv)

    return {
        "rank_csv": v4_csv,
        "evidence_paths": output_dir / "evidence_paths_v3.jsonl",
        "evidence_pack_dir": ep_dir,
    }
