"""
数据构建模块

中间数据构建：
  - gene_pathway: 基因-通路关系
  - pathway_disease: 通路-疾病关系
  - trial_ae: 试验-不良事件关系
"""
from .edges import build_gene_pathway, build_pathway_disease, build_trial_ae

__all__ = ["build_gene_pathway", "build_pathway_disease", "build_trial_ae"]
