# data/ 中间表（全自动重跑会生成）

- failed_trials_drug_rows.csv         (CT.gov 拉取：失败/终止试验 + 干预名)
- failed_drugs_summary.csv            (按 drug 汇总)
- drug_rxnorm_map.csv                 (RxNorm 近似匹配)
- drug_chembl_map.csv                 (药名 -> ChEMBL molecule)
- edge_drug_target.csv                (ChEMBL mechanism: Drug -> Target)
- target_xref.csv                     (ChEMBL target xref：含 UniProt / Ensembl)
- target_chembl_to_ensembl_all.csv    (Target -> ENSG)
- edge_target_pathway_all.csv         (Reactome: UniProt -> Pathway)
- edge_target_disease_ot.csv          (OpenTargets: ENSG -> Disease)
- edge_gene_pathway.csv               (join)
- edge_pathway_disease.csv            (aggregate: Pathway -> Disease)
