"""
数据源模块

支持的数据源:
  - CT.gov: 临床试验数据
  - RxNorm: 药物名称标准化
  - ChEMBL: 药物-靶点关系
  - Reactome: 通路数据
  - OpenTargets: 基因-疾病关联
  - FAERS: FDA不良事件报告
"""
from .ctgov import fetch_failed_trials
from .rxnorm import rxnorm_map, build_drug_canonical
from .chembl import chembl_map, fetch_drug_targets, fetch_target_xrefs, target_to_ensembl
from .reactome import fetch_target_pathways
from .opentargets import fetch_gene_diseases, fetch_disease_phenotypes
from .faers import fetch_drug_ae

__all__ = [
    "fetch_failed_trials",
    "rxnorm_map",
    "build_drug_canonical",
    "chembl_map",
    "fetch_drug_targets",
    "fetch_target_xrefs",
    "target_to_ensembl",
    "fetch_target_pathways",
    "fetch_gene_diseases",
    "fetch_disease_phenotypes",
    "fetch_drug_ae",
]
