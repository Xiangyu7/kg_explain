"""
ChEMBL 数据源

用途:
  1. 药物名称 → ChEMBL ID 映射
  2. 药物 → 靶点 (mechanism of action)
  3. 靶点 → UniProt/Ensembl 交叉引用

API: https://www.ebi.ac.uk/chembl/api/data
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..cache import HTTPCache, cached_get_json
from ..utils import read_csv, safe_str

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

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        max_hits: 搜索最大命中数

    Returns:
        输出文件路径
    """
    rx = read_csv(data_dir / "drug_rxnorm_map.csv", dtype=str)

    rows = []
    for _, r in tqdm(rx.iterrows(), total=len(rx), desc="ChEMBL mapping"):
        raw = safe_str(r.get("drug_raw"))
        term = safe_str(r.get("rxnorm_term"))
        q = term if term else raw
        chembl_id = None
        pref_name = None

        if q:
            try:
                hits = _chembl_molecule_search(cache, q, max_hits=max_hits)
                for h in hits:
                    if h.get("molecule_chembl_id"):
                        chembl_id = h.get("molecule_chembl_id")
                        pref_name = h.get("pref_name")
                        break
            except Exception:
                pass

        rows.append({
            "drug_raw": raw,
            "rxnorm_term": term,
            "chembl_id": chembl_id,
            "chembl_pref_name": pref_name,
        })

    out = data_dir / "drug_chembl_map.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
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

    rows = []
    for _, r in tqdm(mp.iterrows(), total=len(mp), desc="ChEMBL Drug→Target"):
        mol = safe_str(r.get("chembl_id"))
        drug_raw = safe_str(r.get("drug_raw"))
        if not mol or not drug_raw:
            continue

        drug_norm = drug_raw.lower()
        try:
            mechs = _chembl_mechanisms(cache, mol)
        except Exception:
            mechs = []

        for m in mechs:
            rows.append({
                "drug_normalized": drug_norm,
                "drug_raw": drug_raw,
                "molecule_chembl_id": mol,
                "target_chembl_id": m.get("target_chembl_id"),
                "mechanism_of_action": m.get("mechanism_of_action"),
            })

    out = data_dir / "edge_drug_target.csv"
    pd.DataFrame(rows).dropna(subset=["drug_normalized", "target_chembl_id"]).drop_duplicates().to_csv(out, index=False)
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

    node_rows = []
    xref_rows = []

    for tid in tqdm(targets, desc="ChEMBL Target Xref"):
        try:
            t = _chembl_target(cache, tid)
        except Exception:
            continue

        node_rows.append({
            "target_chembl_id": tid,
            "target_type": t.get("target_type"),
            "pref_name": t.get("pref_name"),
            "organism": t.get("organism"),
        })

        for comp in t.get("target_components") or []:
            tcid = comp.get("component_id")
            acc = comp.get("accession")
            if tcid is None:
                continue

            xrefs = comp.get("target_component_xrefs") or []

            for xr in xrefs:
                xref_rows.append({
                    "target_chembl_id": tid,
                    "target_component_id": tcid,
                    "uniprot_accession": acc,
                    "xref_src_db": xr.get("xref_src_db"),
                    "xref_id": xr.get("xref_id"),
                })

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
    except Exception:
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
        # Build target→UniProt mapping from xref
        pairs = (
            xref[["target_chembl_id", "uniprot_accession"]]
            .dropna()
            .drop_duplicates()
        )
        for _, r in tqdm(pairs.iterrows(), total=len(pairs), desc="UniProt→Ensembl"):
            acc = str(r["uniprot_accession"])
            tid = str(r["target_chembl_id"])
            for gid in _uniprot_ensembl(cache, acc):
                rows.append({"target_chembl_id": tid, "ensembl_gene_id": gid})

    if rows:
        out = pd.DataFrame(rows).drop_duplicates()
    else:
        out = empty
    out.to_csv(path, index=False)
    return path
