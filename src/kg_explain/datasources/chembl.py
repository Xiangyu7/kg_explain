"""
ChEMBL 数据源

用途:
  1. 药物名称 → ChEMBL ID 映射
  2. 药物 → 靶点 (mechanism of action)
  3. 靶点 → UniProt/Ensembl 交叉引用

API: https://www.ebi.ac.uk/chembl/api/data
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

from ..cache import HTTPCache, cached_get_json
from ..utils import read_csv, safe_str, load_canonical_map, concurrent_map

logger = logging.getLogger(__name__)

# ChEMBL API端点
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def _chembl_molecule_search(cache: HTTPCache, q: str, max_hits: int = 5) -> list[dict]:
    """搜索分子"""
    url = f"{CHEMBL_API}/molecule/search.json"
    js = cached_get_json(cache, url, params={"q": q, "limit": int(max_hits)})
    return js.get("molecules") or []


def chembl_map(
    data_dir: Path,
    cache: HTTPCache,
    max_hits: int = 5,
) -> Path:
    """
    将药物映射到ChEMBL ID

    使用 canonical_name 去重: 共享同一规范名称的药物只搜索一次 ChEMBL

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        max_hits: 搜索最大命中数

    Returns:
        输出文件路径
    """
    rx = read_csv(data_dir / "drug_rxnorm_map.csv", dtype=str)
    canonical = load_canonical_map(data_dir)

    # 按 canonical name 去重, 每组只搜索一次
    unique_queries: dict[str, str] = {}  # canonical → query_string
    for _, r in rx.iterrows():
        raw = safe_str(r.get("drug_raw"))
        canon = canonical.get(raw.lower(), raw.lower())
        if canon not in unique_queries:
            term = safe_str(r.get("rxnorm_term"))
            unique_queries[canon] = term if term else raw

    def _search_one(item):
        canon, q = item
        if not q:
            return canon, None, None
        try:
            hits = _chembl_molecule_search(cache, q, max_hits=max_hits)
            for h in hits:
                if h.get("molecule_chembl_id"):
                    return canon, h.get("molecule_chembl_id"), h.get("pref_name")
        except Exception as e:
            logger.warning("ChEMBL 分子搜索失败, query=%s: %s", q, e)
        return canon, None, None

    results = concurrent_map(
        _search_one, list(unique_queries.items()),
        max_workers=cache.max_workers, desc="ChEMBL mapping",
    )
    chembl_lookup = {canon: (cid, pname) for canon, cid, pname in results}

    # 展开到所有药物行
    rows = []
    for _, r in rx.iterrows():
        raw = safe_str(r.get("drug_raw"))
        term = safe_str(r.get("rxnorm_term"))
        canon = canonical.get(raw.lower(), raw.lower())
        chembl_id, pref_name = chembl_lookup.get(canon, (None, None))
        rows.append({
            "drug_raw": raw,
            "canonical_name": canon,
            "rxnorm_term": term,
            "chembl_id": chembl_id,
            "chembl_pref_name": pref_name,
        })

    result_df = pd.DataFrame(rows)
    n_mapped = result_df["chembl_id"].notna().sum()
    n_dedup = len(rx) - len(unique_queries)
    logger.info("ChEMBL 映射完成: %d 个药物, %d 个成功映射 (%.1f%%), %d 个去重跳过",
                len(result_df), n_mapped, 100 * n_mapped / max(len(result_df), 1), n_dedup)

    out = data_dir / "drug_chembl_map.csv"
    result_df.to_csv(out, index=False)
    return out


def _chembl_mechanisms(cache: HTTPCache, molecule_chembl_id: str, limit: int = 1000) -> list[dict]:
    """获取分子的作用机制"""
    url = f"{CHEMBL_API}/mechanism.json"
    js = cached_get_json(cache, url, params={"molecule_chembl_id": molecule_chembl_id, "limit": limit})
    return js.get("mechanisms") or []


def fetch_drug_targets(
    data_dir: Path,
    cache: HTTPCache,
) -> Path:
    """
    获取药物-靶点关系 (edge_drug_target)

    Returns:
        输出文件路径
    """
    mp = read_csv(data_dir / "drug_chembl_map.csv", dtype=str)

    # 获取唯一 molecule ID, 并发查询 mechanisms
    unique_mols = sorted({
        safe_str(r.get("chembl_id"))
        for _, r in mp.iterrows()
        if safe_str(r.get("chembl_id"))
    })

    def _fetch_mech(mol):
        try:
            return mol, _chembl_mechanisms(cache, mol)
        except Exception as e:
            logger.warning("ChEMBL mechanism 查询失败, molecule=%s: %s", mol, e)
            return mol, []

    mech_results = concurrent_map(
        _fetch_mech, unique_mols,
        max_workers=cache.max_workers, desc="ChEMBL Drug→Target",
    )
    mech_lookup = {mol: mechs for mol, mechs in mech_results}

    # 展开到所有药物行
    rows = []
    for _, r in mp.iterrows():
        mol = safe_str(r.get("chembl_id"))
        drug_raw = safe_str(r.get("drug_raw"))
        if not mol or not drug_raw:
            continue

        drug_norm = safe_str(r.get("canonical_name")) or drug_raw.lower()
        for m in mech_lookup.get(mol, []):
            rows.append({
                "drug_normalized": drug_norm,
                "drug_raw": drug_raw,
                "molecule_chembl_id": mol,
                "target_chembl_id": m.get("target_chembl_id"),
                "mechanism_of_action": m.get("mechanism_of_action"),
            })

    out_df = pd.DataFrame(rows).dropna(subset=["drug_normalized", "target_chembl_id"]).drop_duplicates()
    logger.info("Drug→Target 关系: %d 条边, %d 个药物, %d 个靶点",
                len(out_df), out_df["drug_normalized"].nunique(), out_df["target_chembl_id"].nunique())

    out = data_dir / "edge_drug_target.csv"
    out_df.to_csv(out, index=False)
    return out


def _chembl_target(cache: HTTPCache, target_chembl_id: str) -> dict:
    """获取靶点详情"""
    url = f"{CHEMBL_API}/target/{target_chembl_id}.json"
    return cached_get_json(cache, url)


# NOTE: The /target_component_xref.json endpoint has been removed from ChEMBL API.
# Cross-references are now embedded in target detail responses under
# target_components[].target_component_xrefs.


def fetch_target_xrefs(
    data_dir: Path,
    cache: HTTPCache,
) -> tuple[Path, Path]:
    """
    获取靶点交叉引用 (UniProt, Ensembl等)

    Returns:
        (node_path, xref_path): 靶点节点和交叉引用
    """
    dt = read_csv(data_dir / "edge_drug_target.csv", dtype=str)
    targets = sorted(set(dt["target_chembl_id"].dropna().astype(str).tolist()))

    def _fetch_one(tid):
        try:
            t = _chembl_target(cache, tid)
        except Exception as e:
            logger.warning("ChEMBL target 详情获取失败, target=%s: %s", tid, e)
            return None, []

        node = {
            "target_chembl_id": tid,
            "target_type": t.get("target_type"),
            "pref_name": t.get("pref_name"),
            "organism": t.get("organism"),
        }
        xrefs = []
        for comp in t.get("target_components") or []:
            tcid = comp.get("component_id")
            acc = comp.get("accession")
            if tcid is None:
                continue
            for xr in comp.get("target_component_xrefs") or []:
                xrefs.append({
                    "target_chembl_id": tid,
                    "target_component_id": tcid,
                    "uniprot_accession": acc,
                    "xref_src_db": xr.get("xref_src_db"),
                    "xref_id": xr.get("xref_id"),
                })
        return node, xrefs

    results = concurrent_map(
        _fetch_one, targets,
        max_workers=cache.max_workers, desc="ChEMBL Target Xref",
    )

    node_rows = [node for node, _ in results if node is not None]
    xref_rows = [xr for _, xrefs in results for xr in xrefs]

    node_path = data_dir / "node_target.csv"
    xref_path = data_dir / "target_xref.csv"
    pd.DataFrame(node_rows).drop_duplicates().to_csv(node_path, index=False)
    pd.DataFrame(xref_rows).drop_duplicates().to_csv(xref_path, index=False)

    return node_path, xref_path


def _uniprot_ensembl(cache: HTTPCache, accession: str) -> list[str]:
    """从UniProt获取Ensembl基因ID列表"""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        data = cached_get_json(cache, url)
    except Exception as e:
        logger.warning("UniProt→Ensembl 查询失败, accession=%s: %s", accession, e)
        return []
    gene_ids: list[str] = []
    for xr in data.get("uniProtKBCrossReferences") or []:
        if xr.get("database") == "Ensembl":
            for prop in xr.get("properties") or []:
                val = prop.get("value", "")
                if val.startswith("ENSG"):
                    # strip version suffix (e.g. ENSG00000173198.7 → ENSG00000173198)
                    gene_ids.append(val.split(".")[0])
    return list(set(gene_ids))


def target_to_ensembl(data_dir: Path, cache: HTTPCache | None = None) -> Path:
    """
    将靶点映射到Ensembl基因ID

    优先从target_xref.csv的ChEMBL交叉引用中提取Ensembl；
    若无Ensembl记录，则通过UniProt API用uniprot_accession查询。

    Returns:
        输出文件路径
    """
    path = data_dir / "target_chembl_to_ensembl_all.csv"
    empty = pd.DataFrame(columns=["target_chembl_id", "ensembl_gene_id"])

    xref_path = data_dir / "target_xref.csv"
    if not xref_path.exists() or xref_path.stat().st_size <= 1:
        xref = pd.DataFrame()
    else:
        xref = read_csv(xref_path, dtype=str)

    rows: list[dict] = []

    # ---- Strategy 1: ChEMBL xref contains Ensembl directly ----
    if not xref.empty:
        m = xref[
            (xref["xref_src_db"].fillna("").str.contains("Ensembl", case=False))
            | (xref["xref_id"].fillna("").str.startswith("ENSG"))
        ]
        for _, r in m.iterrows():
            gid = str(r["xref_id"]).split(".")[0]
            if gid.startswith("ENSG"):
                rows.append({"target_chembl_id": r["target_chembl_id"], "ensembl_gene_id": gid})

    # ---- Strategy 2: UniProt → Ensembl via UniProt REST API ----
    if not rows and cache is not None and not xref.empty:
        pairs = (
            xref[["target_chembl_id", "uniprot_accession"]]
            .dropna()
            .drop_duplicates()
        )
        pair_list = [(str(r["target_chembl_id"]), str(r["uniprot_accession"])) for _, r in pairs.iterrows()]

        def _fetch_ensembl(pair):
            tid, acc = pair
            return [{"target_chembl_id": tid, "ensembl_gene_id": gid} for gid in _uniprot_ensembl(cache, acc)]

        results = concurrent_map(
            _fetch_ensembl, pair_list,
            max_workers=cache.max_workers, desc="UniProt→Ensembl",
        )
        for result_rows in results:
            rows.extend(result_rows)

    if rows:
        out = pd.DataFrame(rows).drop_duplicates()
    else:
        out = empty
    out.to_csv(path, index=False)
    return path
