"""
V1 排序器: Drug-Disease 直接关联

路径类型: Drug --[ASSOC_DISEASE]--> Disease
数据来源: CT.gov failed trials 的 conditions 字段

评分逻辑:
  final_score = n_trials - 0.5 * safety_flags

其中:
  - n_trials: 该药物在该疾病的失败试验数
  - safety_flags: 因安全问题停止的试验数
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from ..config import Config, ensure_dir
from ..utils import read_csv, write_jsonl, safe_str


def run_v1(cfg: Config) -> dict[str, Path]:
    """
    运行V1排序

    Returns:
        输出文件路径字典
    """
    output_dir = ensure_dir(cfg.output_dir)
    data_dir = cfg.data_dir
    files = cfg.files

    ft = read_csv(data_dir / files.get("failed_trials", "failed_trials_drug_rows.csv"), dtype=str)
    # 安全处理 NaN: 先填充空字符串再转小写
    ft["drug_normalized"] = ft["drug_raw"].fillna("").astype(str).str.strip().str.lower()

    # 构建 drug-disease 关系
    rows = []
    for _, r in ft.iterrows():
        drug = r["drug_normalized"]
        why = safe_str(r.get("whyStopped"))
        conditions = safe_str(r.get("conditions"))
        for c in conditions.split(" | "):
            c = c.strip()
            if c:
                rows.append({
                    "drug_normalized": drug,
                    "diseaseName": c,
                    "nctId": r["nctId"],
                    "whyStopped": why,
                })

    dd = pd.DataFrame(rows)
    dd["is_safety"] = dd["whyStopped"].str.contains(
        "adverse|tox|safety|death", case=False, na=False
    ).astype(int)

    # 聚合评分
    agg = dd.groupby(["drug_normalized", "diseaseName"], as_index=False).agg(
        n_trials=("nctId", "nunique"),
        safety_flags=("is_safety", "sum"),
    )
    agg["final_score"] = agg["n_trials"] - 0.5 * agg["safety_flags"]
    agg = agg.sort_values(["drug_normalized", "final_score"], ascending=[True, False])

    # 输出
    out_csv = output_dir / "drug_disease_rank_v1.csv"
    agg.to_csv(out_csv, index=False)

    # 证据路径
    ev_path = output_dir / "evidence_paths_v1.jsonl"
    write_jsonl(ev_path, [
        {
            "drug": r["drug_normalized"],
            "diseaseName": r["diseaseName"],
            "score": float(r["final_score"]),
            "edges": [{
                "rel": "DRUG_ASSOC_DISEASE",
                "src": r["drug_normalized"],
                "dst": r["diseaseName"],
                "source": "CT.gov conditions",
            }],
        }
        for _, r in agg.head(2000).iterrows()
    ])

    return {"rank_csv": out_csv, "evidence_paths": ev_path}
