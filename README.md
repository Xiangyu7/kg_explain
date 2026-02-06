# KG Explain - 药物重定位知识图谱可解释路径系统

## 概述

本项目通过构建**可解释的知识图谱路径**，分析失败临床试验药物的潜在重定位机会。

**疾病方向**: 动脉粥样硬化 (Atherosclerosis) 及相关心血管疾病

**输出**: Top diseases + 可解释路径（而非黑盒分数）

## 数据流

```
Drug → Target → Pathway → Disease → Phenotype
  ↓                         ↑
Trial ←─────────────────────┘
  ↓
 AE (FAERS)
```

## 数据来源

| 数据源 | 用途 | API |
|--------|------|-----|
| CT.gov | 失败临床试验 | ClinicalTrials.gov API |
| RxNorm | 药物名称标准化 | RxNav API |
| ChEMBL | 药物-靶点关系 | ChEMBL API |
| Reactome | 靶点-通路关系 | Reactome API |
| OpenTargets | 基因-疾病关联 | GraphQL API |
| FDA FAERS | 药物不良事件 | openFDA API |

## 版本说明

| 版本 | 路径类型 | 说明 |
|------|----------|------|
| V1 | Drug → Disease | CT.gov conditions直接关联 |
| V2 | Drug → Target → Disease | 加入ChEMBL机制 |
| V3 | Drug → Target → Pathway → Disease | 加入Reactome通路 |
| V4 | V3 + Evidence Pack | 输出RAG证据包 |
| **V5** | 完整可解释路径 | **+ FAERS安全信号 + 表型** |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
mkdir -p data output cache

# 运行完整管道 (V5)
python -m src.kg_explain.cli pipeline --disease atherosclerosis --version v5

# 或使用脚本
bash scripts/run_pipeline.sh atherosclerosis v5
```

## 命令行用法

```bash
# 完整管道
python -m src.kg_explain.cli pipeline --disease atherosclerosis --version v5

# 仅运行排序 (假设数据已存在)
python -m src.kg_explain.cli rank --version v5

# 分步获取数据
python -m src.kg_explain.cli fetch ctgov --condition atherosclerosis
python -m src.kg_explain.cli fetch rxnorm
python -m src.kg_explain.cli fetch chembl
python -m src.kg_explain.cli fetch targets
python -m src.kg_explain.cli fetch pathways
python -m src.kg_explain.cli fetch diseases
python -m src.kg_explain.cli fetch faers
python -m src.kg_explain.cli fetch phenotypes

# 构建中间数据
python -m src.kg_explain.cli build gene-pathway
python -m src.kg_explain.cli build pathway-disease
python -m src.kg_explain.cli build trial-ae
```

## 输出文件

```
output/
├── drug_disease_rank_v5.csv       # 排序结果
├── evidence_paths_v5.jsonl        # 所有证据路径
└── evidence_pack_v5/              # 每对的完整证据JSON
    ├── colchicine__EFO_0003914.json
    └── ...
```

## V5 输出格式

```json
{
  "drug": "colchicine",
  "disease": {"id": "EFO_0003914", "name": "Atherosclerosis"},
  "scores": {
    "final": 0.85,
    "mechanism": 1.2,
    "safety_penalty": 0.15,
    "trial_penalty": 0.1
  },
  "explainable_paths": [
    {
      "type": "DTPD",
      "path_score": 0.45,
      "nodes": [
        {"type": "Drug", "id": "colchicine"},
        {"type": "Target", "id": "CHEMBL4523"},
        {"type": "Pathway", "id": "R-HSA-168256", "name": "Immune System"},
        {"type": "Disease", "id": "EFO_0003914", "name": "Atherosclerosis"}
      ],
      "explanation": "colchicine targets CHEMBL4523, which participates in the Immune System pathway..."
    }
  ],
  "safety_signals": [
    {"ae_term": "Diarrhoea", "report_count": 1200, "is_serious": false}
  ],
  "trial_evidence": [
    {"nctId": "NCT001234", "status": "TERMINATED", "whyStopped": "lack of efficacy"}
  ],
  "phenotypes": [
    {"id": "HP:0001658", "name": "Myocardial infarction", "score": 0.8}
  ]
}
```

## 项目结构

```
kg_explain/
├── src/kg_explain/           # 源代码 (新)
│   ├── config.py            # 配置加载
│   ├── cache.py             # HTTP缓存
│   ├── utils.py             # 工具函数
│   ├── cli.py               # 命令行入口
│   ├── datasources/         # 数据源模块
│   │   ├── ctgov.py        # CT.gov
│   │   ├── rxnorm.py       # RxNorm
│   │   ├── chembl.py       # ChEMBL
│   │   ├── reactome.py     # Reactome
│   │   ├── opentargets.py  # OpenTargets
│   │   └── faers.py        # FAERS
│   ├── builders/            # 数据构建
│   │   └── edges.py        # 边构建
│   └── rankers/             # 排序算法
│       ├── v1.py → v5.py   # V1-V5排序器
│       └── base.py         # 共享工具
├── configs/                  # 配置文件
│   ├── base.yaml            # 基础配置
│   ├── diseases/            # 疾病配置
│   │   └── atherosclerosis.yaml
│   └── versions/            # 版本配置
│       └── v5.yaml
├── scripts/                  # 运行脚本
│   └── run_pipeline.sh
├── data/                     # 数据目录
├── cache/                    # HTTP缓存
└── output/                   # 输出目录
```

## V5 评分公式

```
final_score = mechanism_score × (1 - safety_penalty × w1 - trial_penalty × w2) + phenotype_boost × w3

其中:
- mechanism_score: V3的Drug-Target-Pathway-Disease路径分数
- safety_penalty: FAERS不良事件惩罚 (严重AE权重×2)
- trial_penalty: 因安全原因停止的试验惩罚
- phenotype_boost: 疾病表型数量加分
- w1, w2, w3: 可配置的权重 (见 configs/versions/v5.yaml)
```

## 配置说明

配置文件在 `configs/` 目录:

- `base.yaml`: 基础配置 (API端点、文件名等)
- `diseases/atherosclerosis.yaml`: 疾病方向说明
- `versions/v5.yaml`: V5特定参数 (安全权重等)

## 常见问题

**Q: 药物名称匹配不到ChEMBL ID?**

A: 手动补齐 `data/drug_chembl_map.csv` 中的 `chembl_id`，然后从 `fetch targets` 开始重跑。

**Q: 如何添加新的疾病方向?**

A: 在 `configs/diseases/` 下创建新的配置文件，指定 `condition` 字段。
