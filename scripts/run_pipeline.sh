#!/usr/bin/env bash
# ============================================
# KG Explain - 完整管道运行脚本
# ============================================
#
# 疾病方向: 动脉粥样硬化 (Atherosclerosis)
# 药物来源: ClinicalTrials.gov 失败/终止的临床试验
#
# 数据流程:
#   1. CT.gov → 失败试验 + 药物列表
#   2. RxNorm → 药物名称标准化
#   3. ChEMBL → 药物-靶点关系
#   4. ChEMBL → 靶点-基因映射
#   5. Reactome → 靶点-通路关系
#   6. OpenTargets → 基因-疾病关联
#   7. [V5] FAERS → 药物-不良事件
#   8. [V5] OpenTargets → 疾病-表型
#   9. 运行排序算法
#
# 用法:
#   bash scripts/run_pipeline.sh                    # 默认: atherosclerosis, v5
#   bash scripts/run_pipeline.sh diabetes v3       # 指定疾病和版本
#
# ============================================

set -euo pipefail

DISEASE="${1:-atherosclerosis}"
VERSION="${2:-v5}"

echo "============================================"
echo "KG Explain - Drug Repurposing Pipeline"
echo "============================================"
echo "疾病: $DISEASE"
echo "版本: $VERSION"
echo "============================================"

cd "$(dirname "$0")/.."

mkdir -p data output cache

# 运行完整管道
python -m src.kg_explain.cli pipeline --disease "$DISEASE" --version "$VERSION"

echo "============================================"
echo "完成! 输出文件在 ./output/ 目录"
echo "============================================"
