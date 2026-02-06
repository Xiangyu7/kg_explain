"""
KG Explain 命令行接口

用法:
  # 运行完整管道
  python -m kg_explain pipeline --disease atherosclerosis --version v5

  # 仅运行排序
  python -m kg_explain rank --version v5

  # 获取数据
  python -m kg_explain fetch ctgov --condition atherosclerosis
"""
from __future__ import annotations
import argparse
from pathlib import Path

from .config import ensure_dir, Config
from .cache import HTTPCache
from .utils import read_csv
from . import datasources
from . import builders
from . import rankers


def main():
    parser = argparse.ArgumentParser(
        prog="kg_explain",
        description="""
KG Explain - 药物重定位知识图谱可解释路径系统

疾病方向: 动脉粥样硬化 (Atherosclerosis) 及相关心血管疾病
数据来源: CT.gov, RxNorm, ChEMBL, Reactome, OpenTargets, FAERS
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    args = parser.parse_args()

    # 设置目录
    data_dir = Path("data")
    output_dir = Path("output")
    cache_dir = Path("cache")
    ensure_dir(data_dir)
    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    cache = HTTPCache(cache_dir)

    # 处理命令
    if args.command == "pipeline":
        run_pipeline(args, data_dir, output_dir, cache)
    elif args.command == "rank":
        run_rank(args, data_dir, output_dir)
    elif args.command == "fetch":
        run_fetch(args, data_dir, cache)
    elif args.command == "build":
        run_build(args, data_dir, output_dir)


def run_pipeline(args, data_dir: Path, output_dir: Path, cache: HTTPCache):
    """运行完整管道"""
    print(f"========================================")
    print(f"KG Explain - Drug Repurposing Pipeline")
    print(f"疾病: {args.disease}")
    print(f"版本: {args.version}")
    print(f"========================================")

    if not args.skip_fetch:
        # Step 1: CT.gov
        print("\n[1/9] CT.gov 失败试验...")
        datasources.fetch_failed_trials(args.disease, data_dir, cache)

        # Step 2: RxNorm
        print("\n[2/9] RxNorm 映射...")
        datasources.rxnorm_map(data_dir, cache)

        # Step 3: ChEMBL mapping
        print("\n[3/9] ChEMBL 药物映射...")
        datasources.chembl_map(data_dir, cache)

        # Step 4: Drug-Target
        print("\n[4/9] Drug→Target...")
        datasources.fetch_drug_targets(data_dir, cache)

        # Step 5: Target Xref
        print("\n[5/9] Target Xref...")
        datasources.fetch_target_xrefs(data_dir, cache)
        datasources.target_to_ensembl(data_dir)

        # Step 6: Target-Pathway
        print("\n[6/9] Target→Pathway...")
        datasources.fetch_target_pathways(data_dir, cache)

        # Step 7: Gene-Disease
        print("\n[7/9] Gene→Disease...")
        datasources.fetch_gene_diseases(data_dir, cache)

        # Step 8: Build edges
        print("\n[8/9] 构建中间边...")
        cfg = _make_config(args.version, data_dir, output_dir)
        builders.build_gene_pathway(cfg)
        builders.build_pathway_disease(cfg)

        # Step 9: V5 extras
        if args.version == "v5":
            print("\n[9/9] V5 额外数据 (FAERS, Phenotype)...")
            builders.build_trial_ae(data_dir)

            drugs = read_csv(data_dir / "drug_chembl_map.csv", dtype=str)["drug_raw"].dropna().tolist()
            datasources.fetch_drug_ae(data_dir, cache, drugs)

            diseases = read_csv(data_dir / "edge_target_disease_ot.csv", dtype=str)["diseaseId"].dropna().unique().tolist()
            datasources.fetch_disease_phenotypes(data_dir, cache, diseases)

    # 运行排序
    print(f"\n[Final] 运行 {args.version} 排序...")
    cfg = _make_config(args.version, data_dir, output_dir)
    result = rankers.run_pipeline(cfg)

    print(f"\n========================================")
    print(f"完成! 输出文件:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"========================================")


def run_rank(args, data_dir: Path, output_dir: Path):
    """仅运行排序"""
    if args.config:
        print(f"警告: --config 参数目前不支持，使用默认配置")
    cfg = _make_config(args.version, data_dir, output_dir)
    result = rankers.run_pipeline(cfg)

    print("输出文件:")
    for k, v in result.items():
        print(f"  {k}: {v}")


def run_fetch(args, data_dir: Path, cache: HTTPCache):
    """获取数据"""
    if args.source == "ctgov":
        result = datasources.fetch_failed_trials(args.condition, data_dir, cache, max_pages=args.max_pages)
        print(f"Wrote: {result}")
    elif args.source == "rxnorm":
        result = datasources.rxnorm_map(data_dir, cache)
        print(f"Wrote: {result}")
    elif args.source == "chembl":
        result = datasources.chembl_map(data_dir, cache)
        print(f"Wrote: {result}")
    elif args.source == "targets":
        result = datasources.fetch_drug_targets(data_dir, cache)
        print(f"Wrote: {result}")
        result = datasources.fetch_target_xrefs(data_dir, cache)
        print(f"Wrote: {result}")
        result = datasources.target_to_ensembl(data_dir)
        print(f"Wrote: {result}")
    elif args.source == "pathways":
        result = datasources.fetch_target_pathways(data_dir, cache)
        print(f"Wrote: {result}")
    elif args.source == "diseases":
        result = datasources.fetch_gene_diseases(data_dir, cache)
        print(f"Wrote: {result}")
    elif args.source == "faers":
        drugs = read_csv(data_dir / "drug_chembl_map.csv", dtype=str)["drug_raw"].dropna().tolist()
        result = datasources.fetch_drug_ae(data_dir, cache, drugs)
        print(f"Wrote: {result}")
    elif args.source == "phenotypes":
        diseases = read_csv(data_dir / "edge_target_disease_ot.csv", dtype=str)["diseaseId"].dropna().unique().tolist()
        result = datasources.fetch_disease_phenotypes(data_dir, cache, diseases)
        print(f"Wrote: {result}")


def run_build(args, data_dir: Path, output_dir: Path):
    """构建中间数据"""
    cfg = _make_config("v5", data_dir, output_dir)

    if args.target == "gene-pathway":
        result = builders.build_gene_pathway(cfg)
        print(f"Wrote: {result}")
    elif args.target == "pathway-disease":
        result = builders.build_pathway_disease(cfg)
        print(f"Wrote: {result}")
    elif args.target == "trial-ae":
        result = builders.build_trial_ae(data_dir)
        print(f"Wrote: {result}")


def _make_config(version: str, data_dir: Path, output_dir: Path) -> Config:
    """创建配置对象"""
    return Config(raw={
        "mode": version,
        "paths": {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "cache_dir": "cache",
        },
        "files": {
            "failed_trials": "failed_trials_drug_rows.csv",
            "drug_target": "edge_drug_target.csv",
            "target_ensembl": "target_chembl_to_ensembl_all.csv",
            "target_pathway": "edge_target_pathway_all.csv",
            "gene_disease": "edge_target_disease_ot.csv",
            "gene_pathway": "edge_gene_pathway.csv",
            "pathway_disease": "edge_pathway_disease.csv",
        },
        "rank": {
            "topk_paths_per_pair": 10,
            "topk_pairs_per_drug": 50,
            "hub_penalty_lambda": 1.0,
            "support_gene_boost": 0.15,
            "safety_penalty_weight": 0.3,
            "trial_failure_penalty": 0.2,
            "phenotype_overlap_boost": 0.1,
        },
        "serious_ae_keywords": [
            "death", "fatal", "life-threatening", "hospitalisation", "disability"
        ],
    })


if __name__ == "__main__":
    main()
