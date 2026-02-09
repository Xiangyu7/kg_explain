"""
ClinicalTrials.gov 数据源

用途: 获取指定疾病的失败/终止临床试验
当前配置: 动脉粥样硬化 (Atherosclerosis) 及相关心血管疾病

API文档: https://clinicaltrials.gov/data-api/api
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..cache import HTTPCache, cached_get_json
from ..config import ensure_dir

logger = logging.getLogger(__name__)

# ClinicalTrials.gov API端点
CTG_API = "https://clinicaltrials.gov/api/v2/studies"


def _extract_basic(study: dict) -> dict:
    """提取试验基本信息"""
    ps = (study or {}).get("protocolSection", {}) or {}
    idm = ps.get("identificationModule") or {}
    sm = ps.get("statusModule") or {}
    cm = ps.get("conditionsModule") or {}
    dm = ps.get("designModule") or {}
    return {
        "nctId": idm.get("nctId"),
        "briefTitle": idm.get("briefTitle") or idm.get("officialTitle"),
        "overallStatus": sm.get("overallStatus"),
        "whyStopped": sm.get("whyStopped"),
        "phases": "|".join(dm.get("phases") or []),
        "conditions": " | ".join(cm.get("conditions") or []),
    }


def _extract_interventions(study: dict) -> list[dict]:
    """提取干预措施(药物)"""
    ps = (study or {}).get("protocolSection", {}) or {}
    out = []
    im = ps.get("interventionsModule") or {}
    for it in im.get("interventions", []) or []:
        out.append({"name": it.get("name"), "type": it.get("type")})
    aim = ps.get("armsInterventionsModule") or {}
    for it in aim.get("interventions", []) or []:
        out.append({"name": it.get("name"), "type": it.get("type")})

    # 去重
    seen = set()
    dedup = []
    for x in out:
        if not x.get("name"):
            continue
        key = (x.get("name") or "", x.get("type") or "")
        if key not in seen:
            seen.add(key)
            dedup.append(x)
    return dedup


def _filter_interventions(
    interventions: list[dict],
    include_types: list[str] | None,
    exclude_types: list[str] | None,
) -> list[dict]:
    """
    按干预类型过滤

    Args:
        interventions: 干预列表 [{"name": ..., "type": ...}, ...]
        include_types: 仅保留这些类型 (如 ["DRUG", "BIOLOGICAL"])，None 表示不限
        exclude_types: 排除这些类型 (如 ["DEVICE", "PROCEDURE"])，None 表示不排除

    Returns:
        过滤后的干预列表
    """
    if not include_types and not exclude_types:
        return interventions

    inc = {t.upper() for t in include_types} if include_types else None
    exc = {t.upper() for t in exclude_types} if exclude_types else set()

    filtered = []
    for it in interventions:
        itype = (it.get("type") or "").upper()
        if not itype:
            # 类型未知的干预保留 (不误杀)
            filtered.append(it)
            continue
        if inc is not None and itype not in inc:
            continue
        if itype in exc:
            continue
        filtered.append(it)
    return filtered


def fetch_failed_trials(
    condition: str,
    data_dir: Path,
    cache: HTTPCache,
    statuses: list[str] = None,
    page_size: int = 200,
    max_pages: int = 20,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> tuple[Path, Path]:
    """
    从CT.gov获取失败的临床试验

    Args:
        condition: 疾病条件 (如 "atherosclerosis")
        data_dir: 数据输出目录
        cache: HTTP缓存
        statuses: 试验状态 (默认: TERMINATED, WITHDRAWN, SUSPENDED)
        page_size: 每页大小
        max_pages: 最大页数
        include_types: 仅保留的干预类型 (如 ["DRUG", "BIOLOGICAL"])
        exclude_types: 排除的干预类型 (如 ["DEVICE", "PROCEDURE"])

    Returns:
        (rows_path, summary_path): 试验行数据和药物汇总
    """
    if statuses is None:
        statuses = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]

    ensure_dir(data_dir)
    rows = []
    n_filtered = 0
    page_token = None

    for _ in tqdm(range(max_pages), desc=f"CT.gov [{condition}]"):
        params = {
            "query.cond": condition,
            "pageSize": min(int(page_size), 1000),
            "countTotal": "true",
        }
        if statuses:
            params["filter.overallStatus"] = ",".join(statuses)
        if page_token:
            params["pageToken"] = page_token

        js = cached_get_json(cache, CTG_API, params=params)

        for st in js.get("studies") or []:
            b = _extract_basic(st)
            ints_raw = _extract_interventions(st)
            ints = _filter_interventions(ints_raw, include_types, exclude_types)
            n_filtered += len(ints_raw) - len(ints)

            if not ints:
                rows.append({**b, "drug_raw": None, "intervention_type": None})
            else:
                for it in ints:
                    rows.append({
                        **b,
                        "drug_raw": it.get("name"),
                        "intervention_type": it.get("type"),
                    })

        page_token = js.get("nextPageToken")
        if not page_token:
            break

    df = pd.DataFrame(rows)
    n_trials = df["nctId"].nunique() if not df.empty else 0
    n_drugs = df["drug_raw"].dropna().nunique() if not df.empty else 0
    logger.info("CT.gov 获取完成: %d 行, %d 个试验, %d 个药物, %d 个干预被过滤",
                len(df), n_trials, n_drugs, n_filtered)

    # 保存行数据
    rows_path = data_dir / "failed_trials_drug_rows.csv"
    df.to_csv(rows_path, index=False)

    # 生成药物汇总
    d = df[df["drug_raw"].notna()].copy()
    d["drug_normalized"] = d["drug_raw"].fillna("").astype(str).str.strip().str.lower()
    summ = d.groupby("drug_normalized", as_index=False).agg(
        n_trials=("nctId", "nunique"),
        example_name=("drug_raw", "first"),
        example_status=("overallStatus", "first"),
        example_condition=("conditions", "first"),
        example_whyStopped=("whyStopped", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
    )
    summ_path = data_dir / "failed_drugs_summary.csv"
    summ.to_csv(summ_path, index=False)

    return rows_path, summ_path
