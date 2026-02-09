"""
知识图谱 NetworkX 层

将 CSV 边数据加载为 nx.DiGraph, 支持:
  - 图构建 (build_kg)
  - DTPD 路径枚举 (find_dtpd_paths)
  - 图统计 (graph_stats)
  - GraphML 导出 (export_graphml)

节点类型: Drug, Target, Pathway, Disease, AE, Phenotype
边类型:   DRUG_TARGET, TARGET_PATHWAY, PATHWAY_DISEASE,
          DRUG_AE, DISEASE_PHENOTYPE, DRUG_TRIAL

后续可平滑迁移至 Neo4j (节点标签=type, 关系类型=edge type)
"""
from __future__ import annotations
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from .config import Config
from .utils import safe_str

logger = logging.getLogger(__name__)


def _load_csv(path: Path) -> pd.DataFrame:
    """加载 CSV, 文件不存在返回空 DataFrame"""
    if not path.exists():
        logger.debug("文件不存在, 跳过: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, dtype=str)


def build_kg(cfg: Config) -> nx.DiGraph:
    """
    从 CSV 边数据构建 KG

    Args:
        cfg: 配置 (用于 data_dir 和 files 映射)

    Returns:
        nx.DiGraph 带有类型化节点和边
    """
    G = nx.DiGraph()
    data_dir = cfg.data_dir
    files = cfg.files

    # ── Drug → Target ──
    dt = _load_csv(data_dir / files.get("drug_target", "edge_drug_target.csv"))
    for _, r in dt.iterrows():
        drug = safe_str(r.get("drug_normalized"))
        target = safe_str(r.get("target_chembl_id"))
        if not drug or not target:
            continue
        G.add_node(drug, type="Drug")
        G.add_node(target, type="Target")
        G.add_edge(drug, target, type="DRUG_TARGET",
                   mechanism=safe_str(r.get("mechanism_of_action")))

    # ── Target → Pathway ──
    tp = _load_csv(data_dir / files.get("target_pathway", "edge_target_pathway_all.csv"))
    for _, r in tp.iterrows():
        target = safe_str(r.get("target_chembl_id"))
        pathway = safe_str(r.get("reactome_stid"))
        if not target or not pathway:
            continue
        G.add_node(target, type="Target")
        G.add_node(pathway, type="Pathway",
                   name=safe_str(r.get("reactome_name")))
        G.add_edge(target, pathway, type="TARGET_PATHWAY")

    # ── Pathway → Disease ──
    pd_edge = _load_csv(data_dir / files.get("pathway_disease", "edge_pathway_disease.csv"))
    for _, r in pd_edge.iterrows():
        pathway = safe_str(r.get("reactome_stid"))
        disease = safe_str(r.get("diseaseId"))
        if not pathway or not disease:
            continue
        G.add_node(pathway, type="Pathway",
                   name=safe_str(r.get("reactome_name")))
        G.add_node(disease, type="Disease",
                   name=safe_str(r.get("diseaseName")))
        score = float(pd.to_numeric(r.get("pathway_score", 0), errors="coerce") or 0)
        support = int(float(pd.to_numeric(r.get("support_genes", 1), errors="coerce") or 1))
        G.add_edge(pathway, disease, type="PATHWAY_DISEASE",
                   pathway_score=score, support_genes=support)

    # ── Drug → AE (FAERS) ──
    ae = _load_csv(data_dir / "edge_drug_ae_faers.csv")
    for _, r in ae.iterrows():
        drug = safe_str(r.get("drug_normalized"))
        ae_term = safe_str(r.get("ae_term"))
        if not drug or not ae_term:
            continue
        G.add_node(drug, type="Drug")
        G.add_node(ae_term, type="AE")
        count = int(float(pd.to_numeric(r.get("report_count", 0), errors="coerce") or 0))
        prr = float(pd.to_numeric(r.get("prr", 0), errors="coerce") or 0)
        G.add_edge(drug, ae_term, type="DRUG_AE",
                   report_count=count, prr=round(prr, 4))

    # ── Disease → Phenotype ──
    phe = _load_csv(data_dir / "edge_disease_phenotype.csv")
    for _, r in phe.iterrows():
        disease = safe_str(r.get("diseaseId"))
        pheno = safe_str(r.get("phenotypeId"))
        if not disease or not pheno:
            continue
        G.add_node(disease, type="Disease",
                   name=safe_str(r.get("diseaseName")))
        G.add_node(pheno, type="Phenotype",
                   name=safe_str(r.get("phenotypeName")))
        score = float(pd.to_numeric(r.get("score", 0), errors="coerce") or 0)
        G.add_edge(disease, pheno, type="DISEASE_PHENOTYPE", score=score)

    # ── Drug → Trial (safety/efficacy stops) ──
    trial = _load_csv(data_dir / "edge_trial_ae.csv")
    for _, r in trial.iterrows():
        drug = safe_str(r.get("drug_normalized"))
        nct = safe_str(r.get("nctId"))
        if not drug or not nct:
            continue
        G.add_node(drug, type="Drug")
        G.add_node(nct, type="Trial")
        G.add_edge(drug, nct, type="DRUG_TRIAL",
                   is_safety_stop=safe_str(r.get("is_safety_stop")) == "1",
                   is_efficacy_stop=safe_str(r.get("is_efficacy_stop")) == "1",
                   status=safe_str(r.get("overallStatus")))

    stats = graph_stats(G)
    logger.info("KG 构建完成: %d 节点, %d 边", stats["total_nodes"], stats["total_edges"])
    for ntype, count in sorted(stats["nodes"].items()):
        logger.info("  节点 %-12s: %d", ntype, count)
    for etype, count in sorted(stats["edges"].items()):
        logger.info("  边   %-20s: %d", etype, count)

    return G


def graph_stats(G: nx.DiGraph) -> dict:
    """
    图统计: 按类型计数节点和边

    Returns:
        {"nodes": {type: count}, "edges": {type: count},
         "total_nodes": int, "total_edges": int}
    """
    node_types: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "Unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        t = d.get("type", "Unknown")
        edge_types[t] = edge_types.get(t, 0) + 1

    return {
        "nodes": node_types,
        "edges": edge_types,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
    }


def find_dtpd_paths(
    G: nx.DiGraph,
    drug: str,
    disease: str,
    max_paths: int = 100,
) -> list[dict]:
    """
    枚举 Drug → Target → Pathway → Disease 路径

    Args:
        G: 知识图谱
        drug: 药物 ID (drug_normalized)
        disease: 疾病 ID (diseaseId)
        max_paths: 最大返回路径数

    Returns:
        路径列表, 每条路径含 drug/target/pathway/disease 及边属性
    """
    if drug not in G or disease not in G:
        return []

    paths = []
    for target in G.successors(drug):
        if G.nodes[target].get("type") != "Target":
            continue
        dt_edge = G.edges[drug, target]

        for pathway in G.successors(target):
            if G.nodes[pathway].get("type") != "Pathway":
                continue

            if not G.has_edge(pathway, disease):
                continue
            pd_edge = G.edges[pathway, disease]
            if pd_edge.get("type") != "PATHWAY_DISEASE":
                continue

            paths.append({
                "drug": drug,
                "target": target,
                "pathway": pathway,
                "pathway_name": G.nodes[pathway].get("name", ""),
                "disease": disease,
                "disease_name": G.nodes[disease].get("name", ""),
                "mechanism": dt_edge.get("mechanism", ""),
                "pathway_score": pd_edge.get("pathway_score", 0),
                "support_genes": pd_edge.get("support_genes", 0),
            })
            if len(paths) >= max_paths:
                return paths

    return paths


def drug_summary(G: nx.DiGraph, drug: str) -> dict:
    """
    药物邻域摘要: 靶点、通路、AE、试验

    Args:
        G: 知识图谱
        drug: 药物 ID

    Returns:
        {targets: [...], pathways: [...], adverse_events: [...], trials: [...]}
    """
    if drug not in G:
        return {"targets": [], "pathways": [], "adverse_events": [], "trials": []}

    targets = []
    pathways = set()
    aes = []
    trials = []

    for nbr in G.successors(drug):
        edge = G.edges[drug, nbr]
        ntype = G.nodes[nbr].get("type")

        if ntype == "Target":
            targets.append({
                "id": nbr,
                "mechanism": edge.get("mechanism", ""),
            })
            # 二级: Target → Pathway
            for pw in G.successors(nbr):
                if G.nodes[pw].get("type") == "Pathway":
                    pathways.add((pw, G.nodes[pw].get("name", "")))

        elif ntype == "AE":
            aes.append({
                "term": nbr,
                "report_count": edge.get("report_count", 0),
                "prr": edge.get("prr", 0),
            })

        elif ntype == "Trial":
            trials.append({
                "nctId": nbr,
                "is_safety_stop": edge.get("is_safety_stop", False),
                "is_efficacy_stop": edge.get("is_efficacy_stop", False),
            })

    return {
        "targets": targets,
        "pathways": [{"id": pid, "name": pname} for pid, pname in sorted(pathways)],
        "adverse_events": aes,
        "trials": trials,
    }


def export_graphml(G: nx.DiGraph, path: Path) -> Path:
    """
    导出为 GraphML (可用 Cytoscape / Gephi 可视化)

    Returns:
        输出文件路径
    """
    # GraphML 不支持 bool, 转为字符串
    G_copy = G.copy()
    for _, _, d in G_copy.edges(data=True):
        for k, v in list(d.items()):
            if isinstance(v, bool):
                d[k] = str(v)

    nx.write_graphml(G_copy, str(path))
    logger.info("GraphML 导出完成: %s (%d 节点, %d 边)",
                path, G.number_of_nodes(), G.number_of_edges())
    return path
