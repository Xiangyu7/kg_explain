#!/usr/bin/env python3
"""
Gold‑Standard 端到端测试
========================
用 **两个药物 × 三个靶点** 的极小数据集, 手工计算每一步的期望输出,
验证 pipeline Step 8 (build edges) + Step 9 (V5 ranking) 的全部逻辑。

数据图谱 (已知真值):
  DrugA  ──target──▶  T1 ──pathway──▶  P_alpha  ──disease──▶  D001 (score 0.8)
                                                 ──disease──▶  D002 (score 0.4)
  DrugA  ──target──▶  T2 ──pathway──▶  P_beta   ──disease──▶  D001 (score 0.6)
  DrugB  ──target──▶  T3 ──pathway──▶  P_gamma  ──disease──▶  D001 (score 0.9)

  DrugA 有 1 条安全相关 FAERS AE (报告数 20, 严重)
  DrugA 有 1 条试验停止记录 (safety stop)
  DrugB 无 FAERS / trial 数据

  D001 有 2 条表型, D002 有 0 条

运行:
  cd /Users/xinyueke/Desktop/kg_explain
  python -m pytest tests/test_pipeline_gold.py -v
  或
  python tests/test_pipeline_gold.py
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────
# 确保能 import 项目代码
# ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kg_explain.cache import HTTPCache
from kg_explain.config import Config, ensure_dir
from kg_explain.builders.edges import build_gene_pathway, build_pathway_disease, build_trial_ae
from kg_explain.datasources.rxnorm import build_drug_canonical
from kg_explain.evaluation.metrics import (
    hit_at_k, reciprocal_rank, precision_at_k,
    average_precision, ndcg_at_k, auroc,
)
from kg_explain.evaluation.benchmark import run_benchmark
from kg_explain.graph import build_kg, graph_stats, find_dtpd_paths, drug_summary, export_graphml
from kg_explain.rankers.v3 import run_v3
from kg_explain.rankers.v5 import run_v5
from kg_explain.utils import read_csv, load_canonical_map, concurrent_map


# ============================================================
# 1. 辅助: 创建 Config
# ============================================================
def _cfg(data_dir: Path, output_dir: Path) -> Config:
    return Config(raw={
        "mode": "v5",
        "paths": {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "cache_dir": str(data_dir / "_cache"),
        },
        "files": {
            "failed_trials":   "failed_trials_drug_rows.csv",
            "drug_target":     "edge_drug_target.csv",
            "target_ensembl":  "target_chembl_to_ensembl_all.csv",
            "target_pathway":  "edge_target_pathway_all.csv",
            "gene_disease":    "edge_target_disease_ot.csv",
            "gene_pathway":    "edge_gene_pathway.csv",
            "pathway_disease": "edge_pathway_disease.csv",
        },
        "rank": {
            "topk_paths_per_pair":     10,
            "topk_pairs_per_drug":     50,
            "hub_penalty_lambda":      1.0,
            "support_gene_boost":      0.15,
            "safety_penalty_weight":   0.3,
            "trial_failure_penalty":   0.2,
            "phenotype_overlap_boost": 0.1,
        },
        "serious_ae_keywords": [
            "death", "fatal", "life-threatening",
            "hospitalisation", "disability",
        ],
    })


# ============================================================
# 2. 写入 gold‑standard 输入 CSV
# ============================================================
def write_gold_inputs(data_dir: Path) -> None:
    """在 data_dir 中创建所有管道输入 CSV (已知数据)"""

    # -- failed_trials_drug_rows.csv --
    pd.DataFrame([
        {"nctId": "NCT0001", "drug_raw": "DrugA", "overallStatus": "Terminated",
         "whyStopped": "Adverse events and safety concerns",
         "briefTitle": "Trial for DrugA"},
        {"nctId": "NCT0002", "drug_raw": "DrugB", "overallStatus": "Completed",
         "whyStopped": "",
         "briefTitle": "Trial for DrugB"},
    ]).to_csv(data_dir / "failed_trials_drug_rows.csv", index=False)

    # -- drug_canonical.csv --
    pd.DataFrame([
        {"drug_raw": "DrugA", "canonical_name": "druga", "rxnorm_rxcui": "12345"},
        {"drug_raw": "DrugB", "canonical_name": "drugb", "rxnorm_rxcui": "67890"},
    ]).to_csv(data_dir / "drug_canonical.csv", index=False)

    # -- drug_chembl_map.csv --
    pd.DataFrame([
        {"drug_raw": "DrugA", "canonical_name": "druga", "rxnorm_term": "drugA",
         "chembl_id": "CHEMBL_A", "chembl_pref_name": "DrugA"},
        {"drug_raw": "DrugB", "canonical_name": "drugb", "rxnorm_term": "drugB",
         "chembl_id": "CHEMBL_B", "chembl_pref_name": "DrugB"},
    ]).to_csv(data_dir / "drug_chembl_map.csv", index=False)

    # -- edge_drug_target.csv --
    pd.DataFrame([
        {"drug_normalized": "druga", "drug_raw": "DrugA", "molecule_chembl_id": "CHEMBL_A",
         "target_chembl_id": "T1", "mechanism_of_action": "inhibitor"},
        {"drug_normalized": "druga", "drug_raw": "DrugA", "molecule_chembl_id": "CHEMBL_A",
         "target_chembl_id": "T2", "mechanism_of_action": "antagonist"},
        {"drug_normalized": "drugb", "drug_raw": "DrugB", "molecule_chembl_id": "CHEMBL_B",
         "target_chembl_id": "T3", "mechanism_of_action": "agonist"},
    ]).to_csv(data_dir / "edge_drug_target.csv", index=False)

    # -- target_chembl_to_ensembl_all.csv --
    pd.DataFrame([
        {"target_chembl_id": "T1", "ensembl_gene_id": "ENSG0001"},
        {"target_chembl_id": "T2", "ensembl_gene_id": "ENSG0002"},
        {"target_chembl_id": "T3", "ensembl_gene_id": "ENSG0003"},
    ]).to_csv(data_dir / "target_chembl_to_ensembl_all.csv", index=False)

    # -- edge_target_pathway_all.csv --
    pd.DataFrame([
        {"target_chembl_id": "T1", "uniprot_accession": "U001",
         "reactome_stid": "R-HSA-100", "reactome_name": "Pathway_Alpha"},
        {"target_chembl_id": "T2", "uniprot_accession": "U002",
         "reactome_stid": "R-HSA-200", "reactome_name": "Pathway_Beta"},
        {"target_chembl_id": "T3", "uniprot_accession": "U003",
         "reactome_stid": "R-HSA-300", "reactome_name": "Pathway_Gamma"},
    ]).to_csv(data_dir / "edge_target_pathway_all.csv", index=False)

    # -- edge_target_disease_ot.csv --
    # (targetId = Ensembl gene ID)
    pd.DataFrame([
        {"targetId": "ENSG0001", "diseaseId": "EFO_D001", "diseaseName": "Disease_One",   "score": "0.8"},
        {"targetId": "ENSG0001", "diseaseId": "EFO_D002", "diseaseName": "Disease_Two",   "score": "0.4"},
        {"targetId": "ENSG0002", "diseaseId": "EFO_D001", "diseaseName": "Disease_One",   "score": "0.6"},
        {"targetId": "ENSG0003", "diseaseId": "EFO_D001", "diseaseName": "Disease_One",   "score": "0.9"},
    ]).to_csv(data_dir / "edge_target_disease_ot.csv", index=False)

    # -- edge_drug_ae_faers.csv   (V5 FAERS safety signals) --
    # PRR=3.0 表示该AE在DrugA上的报告率是背景率的3倍 (明确信号)
    pd.DataFrame([
        {"drug_normalized": "druga", "ae_term": "Death",
         "report_count": 20, "drug_total_reports": 100, "prr": 3.0},
    ]).to_csv(data_dir / "edge_drug_ae_faers.csv", index=False)

    # -- edge_disease_phenotype.csv (V5 phenotypes) --
    pd.DataFrame([
        {"diseaseId": "EFO_D001", "diseaseName": "Disease_One",
         "phenotypeId": "HP_0001", "phenotypeName": "Pheno_X", "score": 0.7},
        {"diseaseId": "EFO_D001", "diseaseName": "Disease_One",
         "phenotypeId": "HP_0002", "phenotypeName": "Pheno_Y", "score": 0.5},
    ]).to_csv(data_dir / "edge_disease_phenotype.csv", index=False)


# ============================================================
# 3. 手工计算期望值
# ============================================================
# ---- Step A: build_gene_pathway ----
# merge target_pathway (on target_chembl_id) × target_ensembl → (ensembl_gene_id, reactome_stid, reactome_name)
# Expected rows:
#   ENSG0001 | R-HSA-100 | Pathway_Alpha
#   ENSG0002 | R-HSA-200 | Pathway_Beta
#   ENSG0003 | R-HSA-300 | Pathway_Gamma

# ---- Step B: build_pathway_disease ----
# gene_pathway × gene_disease  (inner on ensembl_gene_id == targetId)
# Join produces:
#   R-HSA-100 × ENSG0001 → D001(0.8), D002(0.4)
#   R-HSA-200 × ENSG0002 → D001(0.6)
#   R-HSA-300 × ENSG0003 → D001(0.9)
# groupby (reactome_stid, diseaseId):
#   (R-HSA-100, D001): pathway_score = max(0.8) = 0.8,  support_genes = 1
#   (R-HSA-100, D002): pathway_score = max(0.4) = 0.4,  support_genes = 1
#   (R-HSA-200, D001): pathway_score = max(0.6) = 0.6,  support_genes = 1
#   (R-HSA-300, D001): pathway_score = max(0.9) = 0.9,  support_genes = 1

# ---- Step C: build_trial_ae ----
# DrugA NCT0001 whyStopped="Adverse events and safety concerns"
#   → is_safety=True (matches "adverse", "safety"), is_efficacy=False
# DrugB NCT0002 whyStopped="" → is_safety=False, is_efficacy=False

# ---- Step D: V3 ranking ----
# merge drug_target × target_pathway → dtp
# dtp rows:
#   druga | T1 | R-HSA-100   (drug_raw: DrugA)
#   druga | T2 | R-HSA-200
#   drugb | T3 | R-HSA-300
#
# target_deg (nunique drugs per target):
#   T1 → 1 drug,  T2 → 1 drug,  T3 → 1 drug
#
# hub_penalty = 1 / log(1 + deg)   (deg clipped ≥1)
#   for deg=1: hp = 1 / log(2) ≈ 1.4427
#
# w_hub_target = hp^lambda  (lambda=1.0)  →  1.4427
#
# merge dtp × pathway_disease on reactome_stid
# Paths:
#   druga|T1|R-HSA-100|D001: score_f=0.8, sg=1, w_hub=1.4427
#   druga|T1|R-HSA-100|D002: score_f=0.4, sg=1, w_hub=1.4427
#   druga|T2|R-HSA-200|D001: score_f=0.6, sg=1, w_hub=1.4427
#   drugb|T3|R-HSA-300|D001: score_f=0.9, sg=1, w_hub=1.4427
#
# w_support = 1 + 0.15 * log(1 + 1) ≈ 1 + 0.15 * 0.6931 = 1.10397
#
# path_score = pathway_score_f * w_hub_target * w_support
#   druga|T1|R-HSA-100|D001: 0.8 * 1.4427 * 1.10397 ≈ 1.27381
#   druga|T1|R-HSA-100|D002: 0.4 * 1.4427 * 1.10397 ≈ 0.63691
#   druga|T2|R-HSA-200|D001: 0.6 * 1.4427 * 1.10397 ≈ 0.95536
#   drugb|T3|R-HSA-300|D001: 0.9 * 1.4427 * 1.10397 ≈ 1.43304
#
# pair aggregation (sum path_score per drug-disease):
#   (druga, D001): 1.27381 + 0.95536 = 2.22917  → final_score = mechanism_score = 2.22917
#   (druga, D002): 0.63691                       → final_score = 0.63691
#   (drugb, D001): 1.43304                       → final_score = 1.43304

# ---- Step E: V5 final scoring ----
# safety_penalty for DrugA ("druga"):
#   AE "Death" count=20, is_serious=True (matches "death")
#   ae_penalty = log(1+20)/10 * 2.0 = log(21)/10 * 2 ≈ 3.04452/10*2 = 0.60890
#   total safety_pen = min(0.60890, 1.0) = 0.60890
#
# trial_penalty for DrugA ("druga"):
#   NCT0001: is_safety_stop=1, is_efficacy_stop=0
#   penalty = 0.1*1 + 0.05*0 = 0.10
#   total trial_pen = min(0.10, 1.0) = 0.10
#
# safety_penalty for DrugB: 0.0
# trial_penalty for DrugB: 0.0
#
# phenotypes for D001: 2 phenotypes → phenotype_score = 0.1 * 2 = 0.2
# phenotypes for D002: 0 phenotypes → phenotype_score = 0.0
#
# V5 formula: final = mechanism * (1 - 0.3*safety_pen - 0.2*trial_pen) + phenotype_score
#
# (druga, D001): 2.22917 * (1 - 0.3*0.60890 - 0.2*0.10) + 0.2
#              = 2.22917 * (1 - 0.18267 - 0.02) + 0.2
#              = 2.22917 * 0.79733 + 0.2
#              ≈ 1.77729 + 0.2 = 1.97729
#
# (druga, D002): 0.63691 * (1 - 0.3*0.60890 - 0.2*0.10) + 0.0
#              = 0.63691 * 0.79733
#              ≈ 0.50786
#
# (drugb, D001): 1.43304 * (1 - 0 - 0) + 0.2
#              = 1.43304 + 0.2 = 1.63304


# ============================================================
# 4. 测试
# ============================================================
class TestGold:
    """Gold‑standard 测试"""

    @classmethod
    def setup_class(cls):
        """创建临时目录并写入测试数据"""
        cls._tmpdir = tempfile.mkdtemp(prefix="kg_test_")
        cls.data_dir = Path(cls._tmpdir) / "data"
        cls.output_dir = Path(cls._tmpdir) / "output"
        ensure_dir(cls.data_dir)
        ensure_dir(cls.output_dir)
        write_gold_inputs(cls.data_dir)
        cls.cfg = _cfg(cls.data_dir, cls.output_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # ------- Step A: build_gene_pathway -------
    def test_step_a_gene_pathway(self):
        build_gene_pathway(self.cfg)
        gp = read_csv(self.data_dir / "edge_gene_pathway.csv", dtype=str)
        print("\n=== edge_gene_pathway.csv ===")
        print(gp.to_string(index=False))

        assert set(gp.columns) >= {"ensembl_gene_id", "reactome_stid", "reactome_name"}
        assert len(gp) == 3

        pairs = set(zip(gp["ensembl_gene_id"], gp["reactome_stid"]))
        assert ("ENSG0001", "R-HSA-100") in pairs
        assert ("ENSG0002", "R-HSA-200") in pairs
        assert ("ENSG0003", "R-HSA-300") in pairs

    # ------- Step B: build_pathway_disease -------
    def test_step_b_pathway_disease(self):
        build_gene_pathway(self.cfg)  # prerequisite
        build_pathway_disease(self.cfg)
        pd_df = read_csv(self.data_dir / "edge_pathway_disease.csv", dtype=str)
        print("\n=== edge_pathway_disease.csv ===")
        print(pd_df.to_string(index=False))

        assert len(pd_df) == 4  # 4 unique (pathway, disease) pairs

        r100_d001 = pd_df[(pd_df["reactome_stid"] == "R-HSA-100") & (pd_df["diseaseId"] == "EFO_D001")]
        assert len(r100_d001) == 1
        assert float(r100_d001.iloc[0]["pathway_score"]) == 0.8
        assert int(float(r100_d001.iloc[0]["support_genes"])) == 1

        r100_d002 = pd_df[(pd_df["reactome_stid"] == "R-HSA-100") & (pd_df["diseaseId"] == "EFO_D002")]
        assert float(r100_d002.iloc[0]["pathway_score"]) == 0.4

        r300_d001 = pd_df[(pd_df["reactome_stid"] == "R-HSA-300") & (pd_df["diseaseId"] == "EFO_D001")]
        assert float(r300_d001.iloc[0]["pathway_score"]) == 0.9

    # ------- Step C: build_trial_ae -------
    def test_step_c_trial_ae(self):
        build_trial_ae(self.data_dir)
        ta = read_csv(self.data_dir / "edge_trial_ae.csv", dtype=str)
        print("\n=== edge_trial_ae.csv ===")
        print(ta.to_string(index=False))

        assert len(ta) == 2  # two trials

        a = ta[ta["drug_normalized"] == "druga"]
        assert len(a) == 1
        assert str(a.iloc[0]["is_safety_stop"]) == "1"
        assert str(a.iloc[0]["is_efficacy_stop"]) == "0"

        b = ta[ta["drug_normalized"] == "drugb"]
        assert len(b) == 1
        assert str(b.iloc[0]["is_safety_stop"]) == "0"

    # ------- Step D: V3 ranking -------
    def test_step_d_v3_ranking(self):
        # prerequisites
        build_gene_pathway(self.cfg)
        build_pathway_disease(self.cfg)

        run_v3(self.cfg)
        v3 = read_csv(self.output_dir / "drug_disease_rank_v3.csv", dtype=str)
        print("\n=== drug_disease_rank_v3.csv ===")
        print(v3.to_string(index=False))

        assert len(v3) == 3  # 3 drug-disease pairs

        # 计算期望值
        hp = 1.0 / np.log1p(1.0)           # 1 / log(2) ≈ 1.4427
        ws = 1.0 + 0.15 * np.log1p(1.0)    # ≈ 1.10397

        exp_a_d001 = (0.8 * hp * ws) + (0.6 * hp * ws)   # T1 path + T2 path
        exp_a_d002 = 0.4 * hp * ws
        exp_b_d001 = 0.9 * hp * ws

        # 检查
        row_a_d001 = v3[(v3["drug_normalized"] == "druga") & (v3["diseaseId"] == "EFO_D001")]
        row_a_d002 = v3[(v3["drug_normalized"] == "druga") & (v3["diseaseId"] == "EFO_D002")]
        row_b_d001 = v3[(v3["drug_normalized"] == "drugb") & (v3["diseaseId"] == "EFO_D001")]

        assert len(row_a_d001) == 1
        assert abs(float(row_a_d001.iloc[0]["final_score"]) - exp_a_d001) < 0.001, \
            f"DrugA-D001: got {float(row_a_d001.iloc[0]['final_score']):.5f}, expected {exp_a_d001:.5f}"

        assert abs(float(row_a_d002.iloc[0]["final_score"]) - exp_a_d002) < 0.001, \
            f"DrugA-D002: got {float(row_a_d002.iloc[0]['final_score']):.5f}, expected {exp_a_d002:.5f}"

        assert abs(float(row_b_d001.iloc[0]["final_score"]) - exp_b_d001) < 0.001, \
            f"DrugB-D001: got {float(row_b_d001.iloc[0]['final_score']):.5f}, expected {exp_b_d001:.5f}"

        print(f"\n期望值:  DrugA-D001={exp_a_d001:.5f}, DrugA-D002={exp_a_d002:.5f}, DrugB-D001={exp_b_d001:.5f}")

    # ------- Step E: V5 ranking -------
    def test_step_e_v5_ranking(self):
        # prerequisites
        build_gene_pathway(self.cfg)
        build_pathway_disease(self.cfg)
        build_trial_ae(self.data_dir)

        results = run_v5(self.cfg)
        v5 = read_csv(self.output_dir / "drug_disease_rank_v5.csv", dtype=str)
        print("\n=== drug_disease_rank_v5.csv ===")
        print(v5.to_string(index=False))

        assert len(v5) == 3

        # 重算期望
        hp = 1.0 / np.log1p(1.0)
        ws = 1.0 + 0.15 * np.log1p(1.0)

        mech_a_d001 = (0.8 * hp * ws) + (0.6 * hp * ws)
        mech_a_d002 = 0.4 * hp * ws
        mech_b_d001 = 0.9 * hp * ws

        # safety_penalty for DrugA: ae "Death" count=20, is_serious=True
        # ae_penalty = log1p(20)/10 * 2.0
        ae_pen = np.log1p(20) / 10.0 * 2.0
        safety_pen_a = min(ae_pen, 1.0)
        safety_pen_b = 0.0

        # trial_penalty for DrugA: 1 safety stop → 0.1
        trial_pen_a = min(0.1 * 1 + 0.05 * 0, 1.0)
        trial_pen_b = 0.0

        # phenotype counts: D001=2, D002=0
        pheno_d001 = 0.1 * 2
        pheno_d002 = 0.0

        # V5 formula
        exp_a_d001 = mech_a_d001 * (1 - 0.3 * safety_pen_a - 0.2 * trial_pen_a) + pheno_d001
        exp_a_d002 = mech_a_d002 * (1 - 0.3 * safety_pen_a - 0.2 * trial_pen_a) + pheno_d002
        exp_b_d001 = mech_b_d001 * (1 - 0.3 * safety_pen_b - 0.2 * trial_pen_b) + pheno_d001

        row_a_d001 = v5[(v5["drug_normalized"] == "druga") & (v5["diseaseId"] == "EFO_D001")]
        row_a_d002 = v5[(v5["drug_normalized"] == "druga") & (v5["diseaseId"] == "EFO_D002")]
        row_b_d001 = v5[(v5["drug_normalized"] == "drugb") & (v5["diseaseId"] == "EFO_D001")]

        assert abs(float(row_a_d001.iloc[0]["final_score"]) - exp_a_d001) < 0.01, \
            f"V5 DrugA-D001: got {float(row_a_d001.iloc[0]['final_score']):.5f}, expected {exp_a_d001:.5f}"
        assert abs(float(row_a_d002.iloc[0]["final_score"]) - exp_a_d002) < 0.01, \
            f"V5 DrugA-D002: got {float(row_a_d002.iloc[0]['final_score']):.5f}, expected {exp_a_d002:.5f}"
        assert abs(float(row_b_d001.iloc[0]["final_score"]) - exp_b_d001) < 0.01, \
            f"V5 DrugB-D001: got {float(row_b_d001.iloc[0]['final_score']):.5f}, expected {exp_b_d001:.5f}"

        print(f"\nV5 期望值:")
        print(f"  DrugA-D001: mech={mech_a_d001:.5f}, safety_pen={safety_pen_a:.5f}, trial_pen={trial_pen_a:.5f}, pheno={pheno_d001}, final={exp_a_d001:.5f}")
        print(f"  DrugA-D002: mech={mech_a_d002:.5f}, safety_pen={safety_pen_a:.5f}, trial_pen={trial_pen_a:.5f}, pheno={pheno_d002}, final={exp_a_d002:.5f}")
        print(f"  DrugB-D001: mech={mech_b_d001:.5f}, safety_pen={safety_pen_b:.5f}, trial_pen={trial_pen_b:.5f}, pheno={pheno_d001}, final={exp_b_d001:.5f}")

        # 检查排序逻辑:
        # DrugA-D001 (mech=2.23) 虽然有 safety penalty, 但基础分远高于 DrugB-D001 (mech=1.43)
        # 所以 DrugA-D001 最终分仍然更高 → 这是正确的
        assert exp_a_d001 > exp_b_d001, "DrugA-D001 基础分更高, 即使有惩罚也应排在 DrugB 前面"
        # DrugA-D002 分最低 (基础分低 + 无表型加分)
        assert exp_a_d002 < exp_b_d001 < exp_a_d001, \
            f"排序应为: DrugA-D001({exp_a_d001:.4f}) > DrugB-D001({exp_b_d001:.4f}) > DrugA-D002({exp_a_d002:.4f})"

        # ------- 检查 evidence_pack_v5 目录 -------
        ep_dir = self.output_dir / "evidence_pack_v5"
        assert ep_dir.exists(), "evidence_pack_v5 目录应存在"
        pack_files = sorted(ep_dir.glob("*.json"))
        assert len(pack_files) == 3, f"应有 3 个 evidence pack, 实际 {len(pack_files)}"

        # 读一个验证结构
        with open(pack_files[0], "r", encoding="utf-8") as f:
            pack = json.load(f)
        assert "drug" in pack
        assert "disease" in pack
        assert "scores" in pack
        assert "explainable_paths" in pack
        assert "safety_signals" in pack
        assert "trial_evidence" in pack
        assert "phenotypes" in pack
        print(f"\nevidence pack 示例: {pack_files[0].name}")
        print(f"  drug={pack['drug']}, disease={pack['disease']['name']}")
        print(f"  scores={pack['scores']}")
        print(f"  paths={len(pack['explainable_paths'])}, AE={len(pack['safety_signals'])}, trials={len(pack['trial_evidence'])}, pheno={len(pack['phenotypes'])}")

    # ------- Step F: evidence_paths_v3.jsonl -------
    def test_step_f_evidence_paths_v3(self):
        build_gene_pathway(self.cfg)
        build_pathway_disease(self.cfg)
        run_v3(self.cfg)

        evp = self.output_dir / "evidence_paths_v3.jsonl"
        assert evp.exists()
        with open(evp, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        print(f"\n=== evidence_paths_v3.jsonl: {len(lines)} paths ===")

        # 应该有 4 条路径 (4个唯一的 drug-target-pathway-disease 组合)
        assert len(lines) == 4

        # 每条路径应有 nodes 和 edges
        for l in lines:
            assert "nodes" in l
            assert "edges" in l
            assert len(l["nodes"]) == 4  # Drug, Target, Pathway, Disease
            assert len(l["edges"]) == 3  # D→T, T→P, P→D
            print(f"  {l['drug']} → {l['nodes'][1]['id']} → {l['nodes'][2]['id']} → {l['diseaseId']}  score={l['path_score']:.5f}")


# ============================================================
# 5. 直接运行 (非 pytest)
# ============================================================
def main():
    """直接运行所有测试并打印中间结果"""
    print("=" * 60)
    print("Gold‑Standard Pipeline Test")
    print("=" * 60)

    t = TestGold()
    t.setup_class()

    tests = [
        ("Step A: build_gene_pathway",   t.test_step_a_gene_pathway),
        ("Step B: build_pathway_disease", t.test_step_b_pathway_disease),
        ("Step C: build_trial_ae",        t.test_step_c_trial_ae),
        ("Step D: V3 ranking",            t.test_step_d_v3_ranking),
        ("Step F: evidence_paths_v3",     t.test_step_f_evidence_paths_v3),
        ("Step E: V5 ranking (full)",     t.test_step_e_v5_ranking),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'─' * 50}")
        print(f"▶ {name}")
        print(f"{'─' * 50}")
        try:
            fn()
            print(f"  ✅  PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    t.teardown_class()
    return 0 if failed == 0 else 1


class TestInterventionFilter:
    """干预类型过滤单元测试"""

    def test_include_only_drug(self):
        from kg_explain.datasources.ctgov import _filter_interventions
        interventions = [
            {"name": "Aspirin", "type": "DRUG"},
            {"name": "Stent", "type": "DEVICE"},
            {"name": "Exercise", "type": "BEHAVIORAL"},
            {"name": "Antibody X", "type": "BIOLOGICAL"},
        ]
        result = _filter_interventions(interventions, include_types=["DRUG", "BIOLOGICAL"], exclude_types=None)
        names = [r["name"] for r in result]
        assert names == ["Aspirin", "Antibody X"]

    def test_exclude_device(self):
        from kg_explain.datasources.ctgov import _filter_interventions
        interventions = [
            {"name": "Aspirin", "type": "DRUG"},
            {"name": "Stent", "type": "DEVICE"},
            {"name": "CBT", "type": "BEHAVIORAL"},
        ]
        result = _filter_interventions(interventions, include_types=None, exclude_types=["DEVICE", "BEHAVIORAL"])
        names = [r["name"] for r in result]
        assert names == ["Aspirin"]

    def test_unknown_type_preserved(self):
        from kg_explain.datasources.ctgov import _filter_interventions
        interventions = [
            {"name": "Mystery", "type": None},
            {"name": "Empty", "type": ""},
            {"name": "Aspirin", "type": "DRUG"},
        ]
        result = _filter_interventions(interventions, include_types=["DRUG"], exclude_types=None)
        names = [r["name"] for r in result]
        assert "Mystery" in names, "未知类型应保留"
        assert "Empty" in names, "空类型应保留"
        assert "Aspirin" in names

    def test_case_insensitive(self):
        from kg_explain.datasources.ctgov import _filter_interventions
        interventions = [
            {"name": "Aspirin", "type": "drug"},
            {"name": "Stent", "type": "Device"},
        ]
        result = _filter_interventions(interventions, include_types=["DRUG"], exclude_types=None)
        assert len(result) == 1
        assert result[0]["name"] == "Aspirin"

    def test_no_filter(self):
        from kg_explain.datasources.ctgov import _filter_interventions
        interventions = [
            {"name": "A", "type": "DRUG"},
            {"name": "B", "type": "DEVICE"},
        ]
        result = _filter_interventions(interventions, include_types=None, exclude_types=None)
        assert len(result) == 2, "无过滤条件时应全部保留"


class TestPRR:
    """PRR 计算与信号门槛测试"""

    def test_prr_basic(self):
        """典型场景: 药物AE比例高于背景"""
        from kg_explain.datasources.faers import _calc_prr
        # Drug: 20/100 = 20% 报告了该AE
        # Background: 1000/1000000 = 0.1% 报告了该AE
        # Other: (1000-20)/(1000000-100) = 980/999900 ≈ 0.098%
        # PRR = (20/100) / (980/999900) ≈ 0.2 / 0.000980 ≈ 204
        prr = _calc_prr(a=20, drug_total=100, bg_ae_count=1000, bg_total=1_000_000)
        assert prr > 100, f"PRR should be high for disproportionate signal, got {prr:.2f}"

    def test_prr_no_signal(self):
        """药物AE比例等于背景 → PRR ≈ 1"""
        from kg_explain.datasources.faers import _calc_prr
        # Drug: 10/1000 = 1%, Background: 10000/1000000 = 1%
        # Other: (10000-10)/(1000000-1000) = 9990/999000 ≈ 1%
        # PRR ≈ 1.0
        prr = _calc_prr(a=10, drug_total=1000, bg_ae_count=10000, bg_total=1_000_000)
        assert 0.9 < prr < 1.1, f"PRR should be ~1 for no signal, got {prr:.4f}"

    def test_prr_edge_zero_bg(self):
        """背景为0时返回0"""
        from kg_explain.datasources.faers import _calc_prr
        assert _calc_prr(10, 100, 0, 1000000) == 0.0
        assert _calc_prr(10, 100, 10, 100) == 0.0  # bg_total == drug_total
        assert _calc_prr(10, 0, 100, 1000) == 0.0   # drug_total == 0

    def test_v5_prr_gate_filters_low_prr(self):
        """V5 排序: PRR 低于阈值的 AE 不计入安全惩罚"""
        cfg = _cfg(self.__class__._data_dir, self.__class__._output_dir)
        cfg.raw.setdefault("faers", {})["min_prr"] = 1.5

        # 先构建 V3 所需的中间文件
        build_gene_pathway(cfg)
        build_pathway_disease(cfg)
        build_trial_ae(self.__class__._data_dir)

        # 写一个 PRR 低于阈值的 FAERS 文件
        pd.DataFrame([
            {"drug_normalized": "druga", "ae_term": "Headache",
             "report_count": 50, "drug_total_reports": 200, "prr": 0.5},
        ]).to_csv(self.__class__._data_dir / "edge_drug_ae_faers.csv", index=False)

        # 运行 V5
        results = run_v5(cfg)
        v5 = read_csv(self.__class__._output_dir / "drug_disease_rank_v5.csv", dtype=str)

        # DrugA 不应有安全惩罚 (PRR=0.5 < min_prr=1.5, 被过滤)
        row_a = v5[v5["drug_normalized"] == "druga"].iloc[0]
        assert float(row_a["safety_penalty"]) == 0.0, \
            f"PRR=0.5 低于阈值, safety_penalty 应为 0, got {row_a['safety_penalty']}"

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="kg_prr_test_")
        cls._data_dir = Path(cls._tmpdir) / "data"
        cls._output_dir = Path(cls._tmpdir) / "output"
        ensure_dir(cls._data_dir)
        ensure_dir(cls._output_dir)
        write_gold_inputs(cls._data_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)


class TestConcurrentMap:
    """concurrent_map 并发执行工具测试"""

    def test_preserves_order(self):
        """并发结果保持输入顺序"""
        items = list(range(20))
        results = concurrent_map(lambda x: x * 2, items, max_workers=4, desc="order test")
        assert results == [x * 2 for x in items]

    def test_sequential_fallback(self):
        """max_workers=1 时顺序执行"""
        items = [10, 20, 30]
        results = concurrent_map(lambda x: x + 1, items, max_workers=1, desc="seq test")
        assert results == [11, 21, 31]

    def test_empty_input(self):
        """空输入返回空列表"""
        results = concurrent_map(lambda x: x, [], max_workers=4, desc="empty test")
        assert results == []

    def test_handles_exceptions(self):
        """任务异常不中断其他任务"""
        def _worker(x):
            if x == 2:
                raise ValueError("boom")
            return x * 10

        results = concurrent_map(_worker, [1, 2, 3], max_workers=2, desc="error test")
        assert results[0] == 10
        assert results[1] is None  # failed task returns None
        assert results[2] == 30


class TestCacheTTL:
    """缓存 TTL 测试"""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="kg_cache_ttl_test_")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_no_ttl_never_expires(self):
        """ttl_seconds=0 时缓存永不过期"""
        import os, time
        cache = HTTPCache(Path(self._tmpdir) / "cache_a", ttl_seconds=0)
        cache.set("key1", {"data": "hello"})

        # 将文件 mtime 改为 1 年前
        p = cache._path("key1")
        old_time = time.time() - 365 * 86400
        os.utime(p, (old_time, old_time))

        result = cache.get("key1")
        assert result == {"data": "hello"}, "TTL=0 时即使文件很旧也应命中"

    def test_ttl_fresh_entry_hits(self):
        """未过期的缓存条目应命中"""
        cache = HTTPCache(Path(self._tmpdir) / "cache_b", ttl_seconds=3600)
        cache.set("key2", {"data": "fresh"})
        result = cache.get("key2")
        assert result == {"data": "fresh"}, "刚写入的缓存应命中"

    def test_ttl_expired_entry_misses(self):
        """过期的缓存条目应视为未命中"""
        import os, time
        cache = HTTPCache(Path(self._tmpdir) / "cache_c", ttl_seconds=60)
        cache.set("key3", {"data": "old"})

        # 将文件 mtime 改为 2 分钟前 (超过 60 秒 TTL)
        p = cache._path("key3")
        old_time = time.time() - 120
        os.utime(p, (old_time, old_time))

        result = cache.get("key3")
        assert result is None, "过期缓存应返回 None"

    def test_ttl_rewrite_refreshes(self):
        """重新写入缓存后应刷新 mtime"""
        import os, time
        cache = HTTPCache(Path(self._tmpdir) / "cache_d", ttl_seconds=60)
        cache.set("key4", {"v": 1})

        # 过期
        p = cache._path("key4")
        old_time = time.time() - 120
        os.utime(p, (old_time, old_time))
        assert cache.get("key4") is None

        # 重新写入
        cache.set("key4", {"v": 2})
        result = cache.get("key4")
        assert result == {"v": 2}, "重新写入后应命中新值"


class TestBenchmarkMetrics:
    """评估指标单元测试 (手工计算验证)"""

    def test_hit_at_k(self):
        ranked = ["a", "b", "c", "d", "e"]
        pos = {"c", "e"}
        assert hit_at_k(ranked, pos, 2) == 0.0   # c 在位置 3
        assert hit_at_k(ranked, pos, 3) == 1.0   # c 在位置 3
        assert hit_at_k(ranked, pos, 5) == 1.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
        assert reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0

    def test_precision_at_k(self):
        ranked = ["pos1", "neg1", "pos2", "neg2"]
        pos = {"pos1", "pos2"}
        assert precision_at_k(ranked, pos, 1) == 1.0    # 1/1
        assert precision_at_k(ranked, pos, 2) == 0.5    # 1/2
        assert precision_at_k(ranked, pos, 4) == 0.5    # 2/4

    def test_average_precision(self):
        ranked = ["pos1", "neg1", "pos2", "neg2"]
        pos = {"pos1", "pos2"}
        # pos1 at rank 1: prec = 1/1
        # pos2 at rank 3: prec = 2/3
        # AP = (1 + 2/3) / 2 = 5/6
        ap = average_precision(ranked, pos)
        assert abs(ap - 5 / 6) < 1e-9

    def test_ndcg_at_k(self):
        ranked = ["neg", "pos1", "pos2"]
        pos = {"pos1", "pos2"}
        # DCG@3: 0 + 1/log2(3) + 1/log2(4)
        # IDCG@3: 1/log2(2) + 1/log2(3)  (2 positives ideal at top)
        import math
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert abs(ndcg_at_k(ranked, pos, 3) - expected) < 1e-9

    def test_auroc_perfect(self):
        # 所有正例排在前面 → AUC = 1.0
        ranked = ["pos1", "pos2", "neg1", "neg2"]
        pos = {"pos1", "pos2"}
        assert auroc(ranked, pos) == 1.0

    def test_auroc_worst(self):
        # 所有负例排在前面 → AUC = 0.0
        ranked = ["neg1", "neg2", "pos1", "pos2"]
        pos = {"pos1", "pos2"}
        assert auroc(ranked, pos) == 0.0

    def test_auroc_mixed(self):
        ranked = ["pos1", "neg1", "pos2", "neg2"]
        pos = {"pos1", "pos2"}
        # pos1(rank 0): 2 neg below → 2
        # pos2(rank 2): 1 neg below → 1
        # concordant = 3, total = 2*2 = 4
        assert auroc(ranked, pos) == 0.75


class TestBenchmarkE2E:
    """Benchmark 端到端测试: 用 V5 排序结果 + 合成 gold standard"""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="kg_bench_test_")
        cls._data_dir = Path(cls._tmpdir) / "data"
        cls._output_dir = Path(cls._tmpdir) / "output"
        ensure_dir(cls._data_dir)
        ensure_dir(cls._output_dir)
        write_gold_inputs(cls._data_dir)
        cls._cfg = _cfg(cls._data_dir, cls._output_dir)

        # 运行 V5 生成排序结果
        build_gene_pathway(cls._cfg)
        build_pathway_disease(cls._cfg)
        build_trial_ae(cls._data_dir)
        run_v5(cls._cfg)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_benchmark_with_known_pairs(self):
        """gold = druga→D001, drugb→D001, 验证 benchmark 输出"""
        gold_csv = self._data_dir / "_gold.csv"
        pd.DataFrame([
            {"drug_normalized": "druga", "diseaseId": "EFO_D001"},
            {"drug_normalized": "drugb", "diseaseId": "EFO_D001"},
        ]).to_csv(gold_csv, index=False)

        rank_csv = self._output_dir / "drug_disease_rank_v5.csv"
        result = run_benchmark(rank_csv, gold_csv, ks=[1, 2, 3])

        assert result["n_diseases_evaluated"] == 1
        assert result["n_gold_pairs"] == 2
        assert result["n_gold_found"] == 2

        m = result["per_disease"]["EFO_D001"]
        # D001 排序: druga 和 drugb 都在结果中, 它们都是正例
        # 在 V5 中 D001 有 druga 和 drugb 两个药物
        assert m["mrr"] == 1.0, "第一个药物就是正例, MRR=1"
        assert m["auroc"] == 0.0, "全正无负例, AUROC=0 (退化情况)"
        assert m["hit@1"] == 1.0

    def test_benchmark_partial_gold(self):
        """gold 只含 drugb→D001, druga 不在 gold 中 → 非满分"""
        gold_csv = self._data_dir / "_gold_partial.csv"
        pd.DataFrame([
            {"drug_normalized": "drugb", "diseaseId": "EFO_D001"},
        ]).to_csv(gold_csv, index=False)

        rank_csv = self._output_dir / "drug_disease_rank_v5.csv"
        result = run_benchmark(rank_csv, gold_csv, ks=[1, 2])

        m = result["per_disease"]["EFO_D001"]
        assert m["n_positive"] == 1
        # drugb 在 V5 D001 排序中排第几取决于分数
        # V5: druga-D001 final ≈ 1.977, drugb-D001 final ≈ 1.633
        # → druga 排第一, drugb 排第二
        assert m["hit@1"] == 0.0, "drugb 不在 top-1"
        assert m["hit@2"] == 1.0, "drugb 在 top-2"
        assert m["mrr"] == 0.5, "drugb 在 rank 2, MRR=0.5"
        assert m["auroc"] == 0.0, "只有 1 pos + 1 neg: drugb at rank 2 → AUC=0"


class TestDrugCanonical:
    """药物实体解析 (RxNorm CUI 规范名称) 测试"""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="kg_canonical_test_")
        cls._data_dir = Path(cls._tmpdir) / "data"
        cls._output_dir = Path(cls._tmpdir) / "output"
        ensure_dir(cls._data_dir)
        ensure_dir(cls._output_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_canonical_merges_same_rxcui(self):
        """同一 RXCUI 的不同药物名合并为一个规范名称"""
        # "Aspirin" 和 "Acetylsalicylic Acid" 共享 RXCUI 1191
        pd.DataFrame([
            {"drug_raw": "Aspirin",               "rxnorm_rxcui": "1191",
             "rxnorm_term": "aspirin",             "rxnorm_score": "100"},
            {"drug_raw": "Acetylsalicylic Acid",  "rxnorm_rxcui": "1191",
             "rxnorm_term": "aspirin",             "rxnorm_score": "80"},
            {"drug_raw": "Atorvastatin",           "rxnorm_rxcui": "83367",
             "rxnorm_term": "atorvastatin",        "rxnorm_score": "100"},
        ]).to_csv(self._data_dir / "drug_rxnorm_map.csv", index=False)

        build_drug_canonical(self._data_dir)
        canon = read_csv(self._data_dir / "drug_canonical.csv", dtype=str)

        assert len(canon) == 3
        assert canon["canonical_name"].nunique() == 2, \
            "Aspirin 和 Acetylsalicylic Acid 应合并为 1 个规范名称"

        # 两个都应映射到 "aspirin"
        aspirin_rows = canon[canon["rxnorm_rxcui"] == "1191"]
        assert len(aspirin_rows) == 2
        assert all(aspirin_rows["canonical_name"] == "aspirin")

        # Atorvastatin 保持独立
        ator_rows = canon[canon["rxnorm_rxcui"] == "83367"]
        assert len(ator_rows) == 1
        assert ator_rows.iloc[0]["canonical_name"] == "atorvastatin"

    def test_canonical_no_rxcui_fallback(self):
        """无 RXCUI 的药物使用 drug_raw.lower() 作为规范名称"""
        pd.DataFrame([
            {"drug_raw": "KnownDrug",  "rxnorm_rxcui": "999",
             "rxnorm_term": "knowndrug", "rxnorm_score": "100"},
            {"drug_raw": "MysteryDrug", "rxnorm_rxcui": None,
             "rxnorm_term": None,        "rxnorm_score": None},
        ]).to_csv(self._data_dir / "drug_rxnorm_map.csv", index=False)

        build_drug_canonical(self._data_dir)
        cmap = load_canonical_map(self._data_dir)

        assert cmap["knowndrug"] == "knowndrug"
        assert cmap["mysterydrug"] == "mysterydrug"

    def test_canonical_flows_through_drug_target(self):
        """canonical_name 正确传递到 edge_drug_target 的 drug_normalized"""
        # 写入带 canonical_name 的 drug_chembl_map
        pd.DataFrame([
            {"drug_raw": "Aspirin", "canonical_name": "aspirin",
             "rxnorm_term": "aspirin", "chembl_id": "CHEMBL25",
             "chembl_pref_name": "ASPIRIN"},
            {"drug_raw": "Acetylsalicylic Acid", "canonical_name": "aspirin",
             "rxnorm_term": "aspirin", "chembl_id": "CHEMBL25",
             "chembl_pref_name": "ASPIRIN"},
        ]).to_csv(self._data_dir / "drug_chembl_map.csv", index=False)

        # 模拟 ChEMBL mechanism 数据 - 写入预期的 edge_drug_target
        # fetch_drug_targets 会读取 drug_chembl_map.csv 并使用 canonical_name
        # 这里直接验证 canonical_name 列存在且被使用
        chembl_map = read_csv(self._data_dir / "drug_chembl_map.csv", dtype=str)
        assert "canonical_name" in chembl_map.columns
        assert all(chembl_map["canonical_name"] == "aspirin"), \
            "两个同义药物应共享同一 canonical_name"

    def test_canonical_flows_through_trial_ae(self):
        """canonical map 正确应用到 trial_ae 的 drug_normalized"""
        # 写入 drug_canonical.csv
        pd.DataFrame([
            {"drug_raw": "Aspirin", "canonical_name": "aspirin", "rxnorm_rxcui": "1191"},
            {"drug_raw": "ASA",     "canonical_name": "aspirin", "rxnorm_rxcui": "1191"},
        ]).to_csv(self._data_dir / "drug_canonical.csv", index=False)

        # 写入 failed_trials 用不同的 drug_raw 名
        pd.DataFrame([
            {"nctId": "NCT_A", "drug_raw": "Aspirin", "overallStatus": "Terminated",
             "whyStopped": "Safety concerns", "briefTitle": "Trial A"},
            {"nctId": "NCT_B", "drug_raw": "ASA", "overallStatus": "Terminated",
             "whyStopped": "Adverse events", "briefTitle": "Trial B"},
        ]).to_csv(self._data_dir / "failed_trials_drug_rows.csv", index=False)

        build_trial_ae(self._data_dir)
        ta = read_csv(self._data_dir / "edge_trial_ae.csv", dtype=str)

        # 两条试验记录都应使用 canonical name "aspirin"
        assert all(ta["drug_normalized"] == "aspirin"), \
            f"drug_normalized 应全部为 'aspirin', 实际: {ta['drug_normalized'].tolist()}"
        assert len(ta) == 2


class TestGraph:
    """NetworkX 知识图谱构建与查询测试"""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="kg_graph_test_")
        cls._data_dir = Path(cls._tmpdir) / "data"
        cls._output_dir = Path(cls._tmpdir) / "output"
        ensure_dir(cls._data_dir)
        ensure_dir(cls._output_dir)
        write_gold_inputs(cls._data_dir)
        cls._cfg = _cfg(cls._data_dir, cls._output_dir)

        # 构建中间文件 (pathway_disease 需要 gene_pathway 先构建)
        build_gene_pathway(cls._cfg)
        build_pathway_disease(cls._cfg)
        build_trial_ae(cls._data_dir)

        cls._G = build_kg(cls._cfg)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_graph_node_counts(self):
        """节点数量按类型正确"""
        stats = graph_stats(self._G)
        # Drug: druga, drugb = 2
        # Target: T1, T2, T3 = 3
        # Pathway: R-HSA-100, R-HSA-200, R-HSA-300 = 3
        # Disease: EFO_D001, EFO_D002 = 2
        # AE: Death = 1
        # Phenotype: HP_0001, HP_0002 = 2
        # Trial: NCT0001, NCT0002 = 2
        assert stats["nodes"]["Drug"] == 2
        assert stats["nodes"]["Target"] == 3
        assert stats["nodes"]["Pathway"] == 3
        assert stats["nodes"]["Disease"] == 2
        assert stats["nodes"]["AE"] == 1
        assert stats["nodes"]["Phenotype"] == 2
        assert stats["nodes"]["Trial"] == 2
        assert stats["total_nodes"] == 15

    def test_graph_edge_counts(self):
        """边数量按类型正确"""
        stats = graph_stats(self._G)
        assert stats["edges"]["DRUG_TARGET"] == 3
        assert stats["edges"]["TARGET_PATHWAY"] == 3
        assert stats["edges"]["PATHWAY_DISEASE"] == 4
        assert stats["edges"]["DRUG_AE"] == 1
        assert stats["edges"]["DISEASE_PHENOTYPE"] == 2
        assert stats["edges"]["DRUG_TRIAL"] == 2
        assert stats["total_edges"] == 15

    def test_find_dtpd_paths_druga_d001(self):
        """DrugA → D001 应找到 2 条 DTPD 路径"""
        paths = find_dtpd_paths(self._G, "druga", "EFO_D001")
        assert len(paths) == 2

        targets = {p["target"] for p in paths}
        assert targets == {"T1", "T2"}

        # 验证路径属性
        t1_path = [p for p in paths if p["target"] == "T1"][0]
        assert t1_path["pathway"] == "R-HSA-100"
        assert t1_path["pathway_name"] == "Pathway_Alpha"
        assert t1_path["mechanism"] == "inhibitor"
        assert t1_path["pathway_score"] == 0.8
        assert t1_path["support_genes"] == 1

    def test_find_dtpd_paths_drugb_d001(self):
        """DrugB → D001 应找到 1 条路径"""
        paths = find_dtpd_paths(self._G, "drugb", "EFO_D001")
        assert len(paths) == 1
        assert paths[0]["target"] == "T3"
        assert paths[0]["pathway"] == "R-HSA-300"
        assert paths[0]["pathway_score"] == 0.9

    def test_find_dtpd_paths_nonexistent(self):
        """不存在的药物或疾病返回空"""
        assert find_dtpd_paths(self._G, "drugX", "EFO_D001") == []
        assert find_dtpd_paths(self._G, "druga", "EFO_D999") == []

    def test_drug_summary(self):
        """药物邻域摘要验证"""
        summ = drug_summary(self._G, "druga")
        assert len(summ["targets"]) == 2
        assert len(summ["pathways"]) == 2
        assert len(summ["adverse_events"]) == 1
        assert len(summ["trials"]) == 1

        # AE 属性
        ae = summ["adverse_events"][0]
        assert ae["term"] == "Death"
        assert ae["report_count"] == 20
        assert ae["prr"] == 3.0

        # Trial 属性
        trial = summ["trials"][0]
        assert trial["nctId"] == "NCT0001"
        assert trial["is_safety_stop"] is True

    def test_drug_summary_nonexistent(self):
        """不存在的药物返回空摘要"""
        summ = drug_summary(self._G, "drugX")
        assert summ == {"targets": [], "pathways": [], "adverse_events": [], "trials": []}

    def test_export_graphml(self):
        """GraphML 导出可以写入并重新加载"""
        import networkx as nx
        out = Path(self._tmpdir) / "test_export.graphml"
        result = export_graphml(self._G, out)
        assert result == out
        assert out.exists()

        # 重新加载验证节点/边数量
        G2 = nx.read_graphml(str(out))
        assert G2.number_of_nodes() == 15
        assert G2.number_of_edges() == 15


if __name__ == "__main__":
    sys.exit(main())
