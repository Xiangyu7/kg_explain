"""
KG Explain - 药物重定位知识图谱可解释路径系统

疾病方向: 动脉粥样硬化 (Atherosclerosis) 及相关心血管疾病
药物来源: ClinicalTrials.gov 失败/终止的临床试验

数据流:
  Drug → Target → Pathway → Disease → Phenotype
    ↓                         ↑
  Trial ←─────────────────────┘
    ↓
   AE (FAERS)

版本:
  - V1: Drug-Disease (CT.gov conditions)
  - V2: Drug-Target-Disease (ChEMBL + OpenTargets)
  - V3: Drug-Target-Pathway-Disease (+ Reactome)
  - V4: V3 + Evidence Pack
  - V5: 完整可解释路径 (+ FAERS + Phenotype)
"""

__version__ = "0.6.0"
