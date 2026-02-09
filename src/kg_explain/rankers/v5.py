"""
V5 排序器: 完整可解释路径

路径类型:
  1. DTPD: Drug → Target → Pathway → Disease (机制路径)
  2. Drug → AE (FAERS安全信号)
  3. Drug → Trial (失败试验证据)
  4. Disease → Phenotype (表型关联)

评分公式 (Drug Repurposing场景):
  final_score = mechanism_score * (1 - safety_penalty - trial_penalty) + phenotype_boost

其中:
  - mechanism_score: V3的路径分数
  - safety_penalty: FAERS不良事件惩罚 (严重AE权重更高)
  - trial_penalty: 因安全原因停止的试验惩罚
  - phenotype_boost: 疾病表型数量加分

输出:
  - drug_disease_rank_v5.csv: 排序结果
  - evidence_paths_v5.jsonl: 所有证据
  - evidence_pack_v5/: 每对的完整证据包JSON
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import Config, ensure_dir
from ..utils import read_csv, write_jsonl, safe_str
from .v3 import run_v3


def _is_serious_ae(ae_term: str, serious_keywords: list[str]) -> bool:
    """判断是否为严重不良事件"""
    ae_lower = ae_term.lower()
    return any(kw.lower() in ae_lower for kw in serious_keywords)


def _generate_path_explanation(path_row) -> str:
    """生成路径的自然语言解释"""
    nodes = path_row.get("nodes", [])
    if len(nodes) < 4:
        return ""

    drug = nodes[0].get("id", "") if len(nodes) > 0 else ""
    target = nodes[1].get("id", "") if len(nodes) > 1 else ""
    pathway = nodes[2].get("name", "") or nodes[2].get("id", "") if len(nodes) > 2 else ""
    disease = nodes[3].get("name", "") or nodes[3].get("id", "") if len(nodes) > 3 else ""

    return (
        f"{drug} targets {target}, which participates in the {pathway} pathway. "
        f"This pathway is associated with {disease}."
    )


def run_v5(cfg: Config) -> dict[str, Path]:
    """
    运行V5排序: 完整可解释路径

    Returns:
        输出文件路径字典
    """
    # 先运行V3获取基础路径
    run_v3(cfg)

    output_dir = ensure_dir(cfg.output_dir)
    data_dir = cfg.data_dir
    rank_cfg = cfg.rank

    # 加载配置参数
    safety_penalty_w = float(rank_cfg.get("safety_penalty_weight", 0.3))
    trial_penalty_w = float(rank_cfg.get("trial_failure_penalty", 0.2))
    phenotype_boost_w = float(rank_cfg.get("phenotype_overlap_boost", 0.1))
    serious_ae_kw = cfg.serious_ae_keywords
    min_prr = float(cfg.faers.get("min_prr", 0))

    # 加载V3结果
    pair_v3 = read_csv(output_dir / "drug_disease_rank_v3.csv", dtype=str)
    paths_v3 = pd.read_json(output_dir / "evidence_paths_v3.jsonl", lines=True)

    # 加载FAERS数据 (可选)
    ae_df = None
    has_prr = False
    ae_path = data_dir / "edge_drug_ae_faers.csv"
    if ae_path.exists():
        ae_df = read_csv(ae_path, dtype=str)
        ae_df["report_count"] = pd.to_numeric(ae_df["report_count"], errors="coerce").fillna(0)
        if "prr" in ae_df.columns:
            ae_df["prr"] = pd.to_numeric(ae_df["prr"], errors="coerce").fillna(0.0)
            has_prr = True

    # 加载表型数据 (可选)
    phe_df = None
    phe_path = data_dir / "edge_disease_phenotype.csv"
    if phe_path.exists() and phe_path.stat().st_size > 1:
        phe_df = read_csv(phe_path, dtype=str)

    # 加载试验AE数据 (可选)
    trial_ae_df = None
    trial_path = data_dir / "edge_trial_ae.csv"
    if trial_path.exists() and trial_path.stat().st_size > 1:
        trial_ae_df = read_csv(trial_path, dtype=str)

    def calc_safety_penalty(drug: str) -> tuple[float, list[dict]]:
        """计算安全惩罚 (用 PRR 做信号门槛)"""
        if ae_df is None:
            return 0.0, []
        drug_aes = ae_df[ae_df["drug_normalized"] == drug.lower().strip()]
        if drug_aes.empty:
            return 0.0, []

        penalty = 0.0
        ae_evidence = []
        for _, ae in drug_aes.head(10).iterrows():
            term = ae.get("ae_term", "")
            count = float(ae.get("report_count", 0))
            prr = float(ae.get("prr", 0)) if has_prr else 0.0
            is_serious = _is_serious_ae(term, serious_ae_kw)

            # PRR 信号门槛: 低于阈值的不视为真实信号，跳过
            if has_prr and min_prr > 0 and prr < min_prr:
                continue

            ae_penalty = np.log1p(count) / 10.0
            if is_serious:
                ae_penalty *= 2.0
            penalty += ae_penalty

            ae_evidence.append({
                "ae_term": term,
                "report_count": int(count),
                "prr": round(prr, 4),
                "is_serious": is_serious,
            })

        return min(penalty, 1.0), ae_evidence

    def calc_trial_penalty(drug: str) -> tuple[float, list[dict]]:
        """计算试验失败惩罚"""
        if trial_ae_df is None:
            return 0.0, []
        drug_trials = trial_ae_df[trial_ae_df["drug_normalized"] == drug.lower().strip()]
        if drug_trials.empty:
            return 0.0, []

        safety_stops = len(drug_trials[drug_trials["is_safety_stop"].astype(str) == "1"])
        efficacy_stops = len(drug_trials[drug_trials["is_efficacy_stop"].astype(str) == "1"])
        penalty = 0.1 * safety_stops + 0.05 * efficacy_stops

        trial_evidence = []
        for _, t in drug_trials.head(5).iterrows():
            trial_evidence.append({
                "nctId": t.get("nctId", ""),
                "status": t.get("overallStatus", ""),
                "whyStopped": t.get("whyStopped", ""),
                "is_safety_stop": str(t.get("is_safety_stop", "0")) == "1",
            })

        return min(penalty, 1.0), trial_evidence

    def get_phenotypes(disease_id: str) -> list[dict]:
        """获取疾病表型"""
        if phe_df is None:
            return []
        disease_phes = phe_df[phe_df["diseaseId"] == disease_id]
        return [
            {"id": p.get("phenotypeId", ""), "name": p.get("phenotypeName", ""), "score": float(p.get("score", 0))}
            for _, p in disease_phes.head(10).iterrows()
        ]

    # 计算最终分数
    final_rows = []
    evidence_packs = []

    for _, pr in tqdm(pair_v3.iterrows(), total=len(pair_v3), desc="V5 ranking"):
        drug = safe_str(pr.get("drug_normalized"))
        disease_id = safe_str(pr.get("diseaseId"))
        disease_name = safe_str(pr.get("diseaseName"))
        base_score = float(pd.to_numeric(pr.get("final_score", 0), errors="coerce") or 0)

        # 获取机制路径
        sub_paths = paths_v3[(paths_v3["drug"] == drug) & (paths_v3["diseaseId"] == disease_id)]

        # 计算惩罚
        safety_pen, ae_evidence = calc_safety_penalty(drug)
        trial_pen, trial_evidence = calc_trial_penalty(drug)
        phenotypes = get_phenotypes(disease_id)

        # 最终分数
        phenotype_score = phenotype_boost_w * len(phenotypes) if phenotypes else 0
        final_score = base_score * (1 - safety_penalty_w * safety_pen - trial_penalty_w * trial_pen) + phenotype_score

        final_rows.append({
            "drug_normalized": drug,
            "diseaseId": disease_id,
            "diseaseName": disease_name,
            "mechanism_score": base_score,
            "safety_penalty": round(safety_pen, 4),
            "trial_penalty": round(trial_pen, 4),
            "phenotype_boost": round(phenotype_score, 4),
            "final_score": round(final_score, 4),
        })

        # 构建证据包
        pack = {
            "drug": drug,
            "disease": {"id": disease_id, "name": disease_name},
            "scores": {
                "final": round(final_score, 4),
                "mechanism": round(base_score, 4),
                "safety_penalty": round(safety_pen, 4),
                "trial_penalty": round(trial_pen, 4),
            },
            "explainable_paths": [],
            "safety_signals": ae_evidence,
            "trial_evidence": trial_evidence,
            "phenotypes": phenotypes,
        }

        for _, row in sub_paths.head(int(rank_cfg.get("topk_paths_per_pair", 10))).iterrows():
            pack["explainable_paths"].append({
                "type": "DTPD",
                "path_score": float(row.get("path_score", 0)),
                "nodes": row.get("nodes", []),
                "edges": row.get("edges", []),
                "explanation": _generate_path_explanation(row),
            })

        evidence_packs.append(pack)

    # 排序并保留top K
    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(["drug_normalized", "final_score"], ascending=[True, False])
    topk = int(rank_cfg.get("topk_pairs_per_drug", 50))
    final_df = final_df.groupby("drug_normalized", as_index=False).head(topk)

    # 输出
    out_csv = output_dir / "drug_disease_rank_v5.csv"
    final_df.to_csv(out_csv, index=False)

    ev_path = output_dir / "evidence_paths_v5.jsonl"
    write_jsonl(ev_path, evidence_packs)

    ep_dir = ensure_dir(output_dir / "evidence_pack_v5")
    for pack in evidence_packs:
        safe = (pack["drug"] + "__" + pack["disease"]["id"]).replace("/", "_").replace(":", "_")
        (ep_dir / f"{safe}.json").write_text(
            json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {
        "rank_csv": out_csv,
        "evidence_paths": ev_path,
        "evidence_pack_dir": ep_dir,
    }
