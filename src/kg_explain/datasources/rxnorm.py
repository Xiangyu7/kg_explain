"""
RxNorm 数据源

用途: 药物名称标准化
API: https://rxnav.nlm.nih.gov/REST/approximateTerm.json
"""
from __future__ import annotations
import logging
import re
from pathlib import Path

import pandas as pd

from ..cache import HTTPCache, cached_get_json
from ..utils import read_csv, safe_str, concurrent_map

logger = logging.getLogger(__name__)

# RxNorm API端点
RX_API = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"


def rxnorm_map(
    data_dir: Path,
    cache: HTTPCache,
    max_entries: int = 3,
) -> Path:
    """
    将药物名称映射到RxNorm

    Args:
        data_dir: 数据目录
        cache: HTTP缓存
        max_entries: 每个药物最多返回的匹配数

    Returns:
        输出文件路径
    """
    df = read_csv(data_dir / "failed_trials_drug_rows.csv")
    drugs = sorted(set([
        x for x in df["drug_raw"].dropna().astype(str).tolist() if x.strip()
    ]))

    def _fetch_one(d):
        try:
            js = cached_get_json(cache, RX_API, params={"term": d, "maxEntries": int(max_entries)})
            cand = ((js.get("approximateGroup") or {}).get("candidate") or [])[:max_entries]
            best = cand[0] if cand else {}
            return {
                "drug_raw": d,
                "rxnorm_rxcui": best.get("rxcui"),
                "rxnorm_term": best.get("term"),
                "rxnorm_score": best.get("score"),
            }
        except Exception as e:
            logger.warning("RxNorm 查询失败, drug=%s: %s", d, e)
            return {"drug_raw": d, "rxnorm_rxcui": None, "rxnorm_term": None, "rxnorm_score": None}

    rows = concurrent_map(_fetch_one, drugs, max_workers=cache.max_workers, desc="RxNorm mapping")

    result_df = pd.DataFrame(rows)
    n_mapped = result_df["rxnorm_rxcui"].notna().sum()
    logger.info("RxNorm 映射完成: %d 个药物, %d 个成功映射 (%.1f%%)",
                len(result_df), n_mapped, 100 * n_mapped / max(len(result_df), 1))

    out = data_dir / "drug_rxnorm_map.csv"
    result_df.to_csv(out, index=False)
    return out


def _clean_drug_name(name: str) -> str:
    """
    清洗药物名称: 去除剂量、剂型、给药方式等后缀

    Examples:
        "cilostazol 100 mg"      → "cilostazol"
        "aspirin 25 mg bid"      → "aspirin"
        "ezetimibe 10mg"         → "ezetimibe"
        "fentanyl injection"     → "fentanyl"
        "aspirin tablet"         → "aspirin"
        "nitroglycerin 0.4 mg sublingual" → "nitroglycerin"
        "alirocumab 150 mg/ml subcutaneous injection" → "alirocumab"
        "dipyridamole 200mg and aspirin 25mg bid:" → "dipyridamole + aspirin"
        "atorvastatin, aspirin, losartan, amlodipine"
            → "atorvastatin + aspirin + losartan + amlodipine"
    """
    s = name.strip().lower()
    # 去掉尾部冒号
    s = s.rstrip(":")

    # 把 "X and Y" / "X, Y, Z" 拆成多个成分, 分别清洗后用 " + " 连接
    # 先试 " and " 分割
    if " and " in s:
        parts = [p.strip() for p in s.split(" and ")]
    elif ", " in s:
        parts = [p.strip() for p in s.split(", ")]
    else:
        parts = [s]

    cleaned = []
    for p in parts:
        p = _strip_dosage(p)
        if p:
            cleaned.append(p)

    return " + ".join(cleaned) if cleaned else name.strip().lower()


# 剂量/剂型/给药方式后缀模式
_DOSAGE_RE = re.compile(
    r"\s+\d[\d.,/]*\s*"             # "100 mg", "0.4 mg", "200mg", "150 mg/ml"
    r"(?:mg|ml|mcg|iu|units?|%|"
    r"mg/ml|mcg/ml|mg/kg)"
    r"(?:/\w+)?"                     # "/ml", "/day" 等
    r"(?:\s+.*)?$",                  # 后面的 "bid", "sublingual" 等全部丢弃
    re.I,
)

_FORMULATION_SUFFIXES = re.compile(
    r"\s+(?:tablet|tablets|capsule|capsules|injection|infusion|"
    r"oral|sublingual|subcutaneous|intravenous|iv|im|sc|"
    r"bid|tid|qd|qid|daily|per\s+day)s?\s*$",
    re.I,
)


def _strip_dosage(s: str) -> str:
    """去除单个成分的剂量和剂型后缀"""
    s = _DOSAGE_RE.sub("", s).strip()
    # 可能还有残留的剂型词
    s = _FORMULATION_SUFFIXES.sub("", s).strip()
    return s


def build_drug_canonical(data_dir: Path) -> Path:
    """
    根据 RxNorm RXCUI 构建药物规范名称映射

    共享同一 RXCUI 的药物会被合并为同一规范名称 (rxnorm_term)
    无 RXCUI 的药物使用 drug_raw.lower() 作为规范名称
    最后统一清洗剂量/剂型后缀

    输出: drug_canonical.csv (drug_raw, canonical_name, rxnorm_rxcui)
    """
    rx = read_csv(data_dir / "drug_rxnorm_map.csv", dtype=str)

    rows = []
    has_cui = rx[rx["rxnorm_rxcui"].notna() & (rx["rxnorm_rxcui"] != "")]
    no_cui = rx[rx["rxnorm_rxcui"].isna() | (rx["rxnorm_rxcui"] == "")]

    for _, grp in has_cui.groupby("rxnorm_rxcui"):
        canonical = safe_str(grp.iloc[0].get("rxnorm_term"))
        if not canonical:
            canonical = safe_str(grp.iloc[0].get("drug_raw"))
        canonical = canonical.lower().strip()

        for _, r in grp.iterrows():
            rows.append({
                "drug_raw": safe_str(r["drug_raw"]),
                "canonical_name": canonical,
                "rxnorm_rxcui": safe_str(r["rxnorm_rxcui"]),
            })

    for _, r in no_cui.iterrows():
        raw = safe_str(r["drug_raw"])
        rows.append({
            "drug_raw": raw,
            "canonical_name": raw.lower().strip(),
            "rxnorm_rxcui": "",
        })

    out_df = pd.DataFrame(rows).drop_duplicates()

    # 统一清洗: 去除剂量/剂型后缀
    out_df["canonical_name"] = out_df["canonical_name"].apply(_clean_drug_name)

    n_canonical = out_df["canonical_name"].nunique()
    n_merged = len(out_df) - n_canonical
    logger.info("Drug canonical 映射: %d 个药物 → %d 个规范名称 (%d 个合并)",
                len(out_df), n_canonical, n_merged)

    out = data_dir / "drug_canonical.csv"
    out_df.to_csv(out, index=False)
    return out
