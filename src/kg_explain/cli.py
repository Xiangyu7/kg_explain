"""
KG Explain 命令行接口 — v0.6.0, 含步骤计时和管道 manifest

用法:
  # 运行完整管道
  python -m kg_explain pipeline --disease atherosclerosis --version v5

  # 仅运行排序
  python -m kg_explain rank --version v5

  # 获取数据
  python -m kg_explain fetch ctgov --condition atherosclerosis

Improvements (v0.6.0):
    - _timed_step(): 统一步骤计时
    - 管道完成后输出 manifest.json
    - 配置验证 (加载后自动检查)
    - 缓存统计记录
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ensure_dir, Config, load_config
from .cache import HTTPCache
from .utils import read_csv, write_json
from . import datasources
from . import builders
from . import rankers
from . import evaluation

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """配置全局日志."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _timed_step(name: str, fn, *args, **kwargs):
    """
    执行带计时的管道步骤.

    Args:
        name: 步骤名称.
        fn: 要执行的函数.
        *args, **kwargs: 函数参数.

    Returns:
        (result, timing_dict): 函数返回值和计时信息.
    """
    logger.info("▶ %s ...", name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info("✓ %s 完成 (%.1fs)", name, elapsed)
        return result, {"step": name, "elapsed_sec": round(elapsed, 2), "status": "ok"}
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("✗ %s 失败 (%.1fs): %s", name, elapsed, e)
        return None, {"step": name, "elapsed_sec": round(elapsed, 2), "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        prog="kg_explain",
        description="""
KG Explain v0.6.0 - 药物重定位知识图谱可解释路径系统

疾病方向: 动脉粥样硬化 (Atherosclerosis) 及相关心血管疾病
数据来源: CT.gov, RxNorm, ChEMBL, Reactome, OpenTargets, FAERS
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示调试日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pipeline: 运行完整管道
    p_pipeline = subparsers.add_parser("pipeline", help="运行完整管道")
    p_pipeline.add_argument("--disease", default="atherosclerosis", help="疾病名称")
    p_pipeline.add_argument("--version", default="v5", choices=["v1", "v2", "v3", "v4", "v5"], help="排序版本")
    p_pipeline.add_argument("--skip-fetch", action="store_true", help="跳过数据获取")

    # rank: 仅运行排序
    p_rank = subparsers.add_parser("rank", help="运行排序算法")
    p_rank.add_argument("--version", default="v5", choices=["v1", "v2", "v3", "v4", "v5"], help="排序版本")
    p_rank.add_argument("--config", help="配置文件路径")

    # fetch: 数据获取
    p_fetch = subparsers.add_parser("fetch", help="获取数据")
    fetch_sub = p_fetch.add_subparsers(dest="source", required=True)

    # fetch ctgov
    p_ctgov = fetch_sub.add_parser("ctgov", help="从CT.gov获取失败试验")
    p_ctgov.add_argument("--condition", required=True, help="疾病条件")
    p_ctgov.add_argument("--max-pages", type=int, default=20, help="最大页数")

    # fetch rxnorm
    fetch_sub.add_parser("rxnorm", help="RxNorm药物映射")

    # fetch chembl
    fetch_sub.add_parser("chembl", help="ChEMBL药物映射")

    # fetch targets
    fetch_sub.add_parser("targets", help="获取药物靶点")

    # fetch pathways
    fetch_sub.add_parser("pathways", help="获取靶点通路")

    # fetch diseases
    fetch_sub.add_parser("diseases", help="获取基因-疾病关联")

    # fetch faers
    fetch_sub.add_parser("faers", help="获取FAERS不良事件")

    # fetch phenotypes
    fetch_sub.add_parser("phenotypes", help="获取疾病表型")

    # build: 构建中间数据
    p_build = subparsers.add_parser("build", help="构建中间数据")
    build_sub = p_build.add_subparsers(dest="target", required=True)
    build_sub.add_parser("gene-pathway", help="构建基因-通路关系")
    build_sub.add_parser("pathway-disease", help="构建通路-疾病关系")
    build_sub.add_parser("trial-ae", help="构建试验-AE关系")

    # benchmark: 评估排序质量
    p_bench = subparsers.add_parser("benchmark", help="评估排序结果 (需提供 gold-standard CSV)")
    p_bench.add_argument("--version", default="v5", choices=["v1", "v2", "v3", "v4", "v5"], help="排序版本")
    p_bench.add_argument("--gold", required=True, help="Gold-standard CSV (需含 drug_normalized, diseaseId)")
    p_bench.add_argument("--ks", default="5,10,20", help="K 值列表, 逗号分隔 (默认: 5,10,20)")

    # graph: 知识图谱构建与查询
    p_graph = subparsers.add_parser("graph", help="构建/查询/导出知识图谱 (NetworkX)")
    p_graph.add_argument("--export", metavar="PATH", help="导出 GraphML 文件路径")
    p_graph.add_argument("--drug", help="查询指定药物的 DTPD 路径")
    p_graph.add_argument("--disease", help="查询指定疾病的 DTPD 路径")

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    # 加载配置
    cfg = _load_pipeline_config(
        disease=getattr(args, "disease", "atherosclerosis"),
        version=getattr(args, "version", "v5"),
    )
    logger.info("配置加载完成: mode=%s, data_dir=%s", cfg.mode, cfg.data_dir)

    # 确保目录存在
    ensure_dir(cfg.data_dir)
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)
    ttl_sec = int(cfg.cache_ttl_hours * 3600)
    cache = HTTPCache(cfg.cache_dir, max_workers=cfg.http_max_workers, ttl_seconds=ttl_sec)
    if ttl_sec > 0:
        logger.info("缓存 TTL: %.1f 小时", cfg.cache_ttl_hours)

    # 处理命令
    if args.command == "pipeline":
        run_pipeline(args, cfg, cache)
    elif args.command == "rank":
        run_rank(args, cfg)
    elif args.command == "fetch":
        run_fetch(args, cfg, cache)
    elif args.command == "build":
        run_build(args, cfg)
    elif args.command == "benchmark":
        run_benchmark_cmd(args, cfg)
    elif args.command == "graph":
        run_graph_cmd(args, cfg)


def _load_pipeline_config(disease: str, version: str) -> Config:
    """
    从 YAML 加载配置.

    加载顺序: configs/base.yaml → configs/diseases/{disease}.yaml → configs/versions/{version}.yaml
    缺失的 disease/version 文件会被跳过 (使用 base 默认值)
    """
    base = Path("configs/base.yaml")
    disease_path = Path(f"configs/diseases/{disease}.yaml")
    version_path = Path(f"configs/versions/{version}.yaml")

    if not base.exists():
        logger.warning("基础配置不存在: %s, 使用内置默认值", base)
        return Config(raw={"mode": version})

    cfg = load_config(
        base_path=str(base),
        disease_path=str(disease_path) if disease_path.exists() else None,
        version_path=str(version_path) if version_path.exists() else None,
    )

    # 确保 mode 与 CLI 参数一致 (CLI 优先)
    cfg.raw["mode"] = version

    return cfg


def run_pipeline(args, cfg: Config, cache: HTTPCache):
    """运行完整管道 (含计时和 manifest)."""
    pipeline_start = time.time()
    step_timings = []

    logger.info("=" * 60)
    logger.info("KG Explain v0.6.0 - Drug Repurposing Pipeline")
    logger.info("疾病: %s (condition=%s)", args.disease, cfg.condition)
    logger.info("版本: %s", args.version)
    logger.info("=" * 60)

    data_dir = cfg.data_dir

    if not args.skip_fetch:
        # Step 1: CT.gov
        drug_filter = cfg.drug_filter
        _, t = _timed_step("[1/10] CT.gov 失败试验",
            datasources.fetch_failed_trials,
            cfg.condition, data_dir, cache,
            statuses=cfg.trial_statuses,
            page_size=cfg.http_page_size,
            max_pages=cfg.trial_max_pages,
            include_types=drug_filter.get("include_types"),
            exclude_types=drug_filter.get("exclude_types"),
        )
        step_timings.append(t)

        # Step 2: RxNorm
        _, t = _timed_step("[2/10] RxNorm 映射",
            datasources.rxnorm_map, data_dir, cache)
        step_timings.append(t)

        # Step 3: Canonical drug names
        _, t = _timed_step("[3/10] 药物规范名称",
            datasources.build_drug_canonical, data_dir)
        step_timings.append(t)

        # Step 4: ChEMBL mapping
        _, t = _timed_step("[4/10] ChEMBL 药物映射",
            datasources.chembl_map, data_dir, cache)
        step_timings.append(t)

        # Step 5: Drug-Target
        _, t = _timed_step("[5/10] Drug→Target",
            datasources.fetch_drug_targets, data_dir, cache)
        step_timings.append(t)

        # Step 6: Target Xref
        def _step6():
            datasources.fetch_target_xrefs(data_dir, cache)
            datasources.target_to_ensembl(data_dir, cache)
        _, t = _timed_step("[6/10] Target Xref + Ensembl", _step6)
        step_timings.append(t)

        # Step 7: Target-Pathway
        _, t = _timed_step("[7/10] Target→Pathway",
            datasources.fetch_target_pathways, data_dir, cache)
        step_timings.append(t)

        # Step 8: Gene-Disease
        _, t = _timed_step("[8/10] Gene→Disease",
            datasources.fetch_gene_diseases, data_dir, cache)
        step_timings.append(t)

        # Step 9: Build edges
        def _step9():
            builders.build_gene_pathway(cfg)
            builders.build_pathway_disease(cfg)
        _, t = _timed_step("[9/10] 构建中间边", _step9)
        step_timings.append(t)

        # Step 10: V5 extras
        if args.version == "v5":
            def _step10():
                builders.build_trial_ae(data_dir)

                faers_cfg = cfg.faers
                drugs_df = read_csv(data_dir / "drug_chembl_map.csv", dtype=str)
                if "canonical_name" in drugs_df.columns:
                    drugs = drugs_df["canonical_name"].dropna().unique().tolist()
                else:
                    drugs = drugs_df["drug_raw"].dropna().tolist()
                datasources.fetch_drug_ae(
                    data_dir, cache, drugs,
                    min_count=int(faers_cfg.get("min_report_count", 5)),
                    min_prr=float(faers_cfg.get("min_prr", 0)),
                    top_ae=int(faers_cfg.get("top_ae_per_drug", 50)),
                    max_drugs=int(faers_cfg.get("max_drugs", 500)),
                )

                phe_cfg = cfg.phenotype
                diseases = read_csv(data_dir / "edge_target_disease_ot.csv", dtype=str)["diseaseId"].dropna().unique().tolist()
                datasources.fetch_disease_phenotypes(
                    data_dir, cache, diseases,
                    min_score=float(phe_cfg.get("min_score", 0.3)),
                    max_phenotypes=int(phe_cfg.get("max_phenotypes_per_disease", 30)),
                )
            _, t = _timed_step("[10/10] V5 额外数据 (FAERS, Phenotype)", _step10)
            step_timings.append(t)

    # 运行排序
    result, t = _timed_step("[Final] %s 排序" % args.version,
        rankers.run_pipeline, cfg)
    step_timings.append(t)

    # 输出 manifest
    pipeline_elapsed = time.time() - pipeline_start
    manifest = {
        "kg_explain_version": "0.6.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disease": args.disease,
        "condition": cfg.condition,
        "ranker_version": args.version,
        "pipeline_elapsed_sec": round(pipeline_elapsed, 2),
        "config_summary": cfg.summary(),
        "cache_stats": cache.summary(),
        "step_timings": step_timings,
        "outputs": {k: str(v) for k, v in (result or {}).items()},
    }

    manifest_path = cfg.output_dir / "pipeline_manifest.json"
    write_json(manifest_path, manifest)
    logger.info("Manifest 写入: %s", manifest_path)

    logger.info("=" * 60)
    logger.info("完成! 总耗时: %.1fs", pipeline_elapsed)
    logger.info("输出文件:")
    for k, v in (result or {}).items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)


def run_rank(args, cfg: Config):
    """仅运行排序."""
    if args.config:
        logger.warning("--config 参数目前不支持，使用默认配置")
    result = rankers.run_pipeline(cfg)

    logger.info("输出文件:")
    for k, v in result.items():
        logger.info("  %s: %s", k, v)


def run_fetch(args, cfg: Config, cache: HTTPCache):
    """获取数据."""
    data_dir = cfg.data_dir

    if args.source == "ctgov":
        drug_filter = cfg.drug_filter
        result = datasources.fetch_failed_trials(
            args.condition, data_dir, cache,
            statuses=cfg.trial_statuses,
            page_size=cfg.http_page_size,
            max_pages=args.max_pages,
            include_types=drug_filter.get("include_types"),
            exclude_types=drug_filter.get("exclude_types"),
        )
        logger.info("Wrote: %s", result)
    elif args.source == "rxnorm":
        result = datasources.rxnorm_map(data_dir, cache)
        logger.info("Wrote: %s", result)
    elif args.source == "chembl":
        result = datasources.chembl_map(data_dir, cache)
        logger.info("Wrote: %s", result)
    elif args.source == "targets":
        result = datasources.fetch_drug_targets(data_dir, cache)
        logger.info("Wrote: %s", result)
        result = datasources.fetch_target_xrefs(data_dir, cache)
        logger.info("Wrote: %s", result)
        result = datasources.target_to_ensembl(data_dir)
        logger.info("Wrote: %s", result)
    elif args.source == "pathways":
        result = datasources.fetch_target_pathways(data_dir, cache)
        logger.info("Wrote: %s", result)
    elif args.source == "diseases":
        result = datasources.fetch_gene_diseases(data_dir, cache)
        logger.info("Wrote: %s", result)
    elif args.source == "faers":
        faers_cfg = cfg.faers
        drugs_df = read_csv(data_dir / "drug_chembl_map.csv", dtype=str)
        if "canonical_name" in drugs_df.columns:
            drugs = drugs_df["canonical_name"].dropna().unique().tolist()
        else:
            drugs = drugs_df["drug_raw"].dropna().tolist()
        result = datasources.fetch_drug_ae(
            data_dir, cache, drugs,
            min_count=int(faers_cfg.get("min_report_count", 5)),
            min_prr=float(faers_cfg.get("min_prr", 0)),
            top_ae=int(faers_cfg.get("top_ae_per_drug", 50)),
            max_drugs=int(faers_cfg.get("max_drugs", 500)),
        )
        logger.info("Wrote: %s", result)
    elif args.source == "phenotypes":
        phe_cfg = cfg.phenotype
        diseases = read_csv(data_dir / "edge_target_disease_ot.csv", dtype=str)["diseaseId"].dropna().unique().tolist()
        result = datasources.fetch_disease_phenotypes(
            data_dir, cache, diseases,
            min_score=float(phe_cfg.get("min_score", 0.3)),
            max_phenotypes=int(phe_cfg.get("max_phenotypes_per_disease", 30)),
        )
        logger.info("Wrote: %s", result)


def run_build(args, cfg: Config):
    """构建中间数据."""
    if args.target == "gene-pathway":
        result = builders.build_gene_pathway(cfg)
        logger.info("Wrote: %s", result)
    elif args.target == "pathway-disease":
        result = builders.build_pathway_disease(cfg)
        logger.info("Wrote: %s", result)
    elif args.target == "trial-ae":
        result = builders.build_trial_ae(cfg.data_dir)
        logger.info("Wrote: %s", result)


def run_graph_cmd(args, cfg: Config):
    """构建/查询/导出知识图谱."""
    from .graph import build_kg, graph_stats, find_dtpd_paths, drug_summary, export_graphml

    G = build_kg(cfg)
    stats = graph_stats(G)

    # 打印统计
    print(f"知识图谱: {stats['total_nodes']} 节点, {stats['total_edges']} 边")
    print("节点类型:")
    for t, c in sorted(stats["nodes"].items()):
        print(f"  {t:12s}: {c}")
    print("边类型:")
    for t, c in sorted(stats["edges"].items()):
        print(f"  {t:20s}: {c}")

    # 路径查询
    if args.drug and args.disease:
        paths = find_dtpd_paths(G, args.drug.lower().strip(), args.disease.strip())
        print(f"\nDTPD 路径 ({args.drug} → {args.disease}): {len(paths)} 条")
        for i, p in enumerate(paths[:20], 1):
            print(f"  {i}. {p['drug']} → {p['target']} → {p['pathway']} ({p['pathway_name']}) → {p['disease']}")
            print(f"     mechanism={p['mechanism']}, score={p['pathway_score']}, genes={p['support_genes']}")
    elif args.drug:
        summ = drug_summary(G, args.drug.lower().strip())
        print(f"\n药物摘要: {args.drug}")
        print(f"  靶点: {len(summ['targets'])}")
        for t in summ["targets"][:10]:
            print(f"    {t['id']} ({t['mechanism']})")
        print(f"  通路: {len(summ['pathways'])}")
        for p in summ["pathways"][:10]:
            print(f"    {p['id']} ({p['name']})")
        print(f"  不良事件: {len(summ['adverse_events'])}")
        for a in summ["adverse_events"][:5]:
            print(f"    {a['term']} (n={a['report_count']}, PRR={a['prr']})")
        print(f"  试验: {len(summ['trials'])}")

    # 导出
    if args.export:
        export_graphml(G, Path(args.export))
        print(f"\nGraphML 已导出: {args.export}")


def run_benchmark_cmd(args, cfg: Config):
    """运行 benchmark 评估."""
    from pathlib import Path
    from .evaluation.benchmark import run_benchmark, format_report

    version = args.version
    rank_csv = cfg.output_dir / f"drug_disease_rank_{version}.csv"
    gold_csv = Path(args.gold)
    ks = [int(k.strip()) for k in args.ks.split(",")]

    if not rank_csv.exists():
        logger.error("排序结果不存在: %s (请先运行 rank --version %s)", rank_csv, version)
        return
    if not gold_csv.exists():
        logger.error("Gold-standard 文件不存在: %s", gold_csv)
        return

    result = run_benchmark(rank_csv, gold_csv, ks=ks)
    report = format_report(result)
    print(report)

    # 同时输出 JSON
    out_json = cfg.output_dir / f"benchmark_{version}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("评估结果已保存: %s", out_json)


if __name__ == "__main__":
    main()
