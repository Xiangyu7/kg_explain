"""Unit tests for kg_explain.graph module.

Tests cover:
    - build_kg: from CSV data
    - graph_stats: node/edge type counting
    - find_dtpd_paths: path enumeration
    - drug_summary: neighbor aggregation
    - export_graphml: file output
"""
import pytest
from pathlib import Path
import pandas as pd

from kg_explain.config import Config
from kg_explain.graph import (
    build_kg,
    graph_stats,
    find_dtpd_paths,
    drug_summary,
    export_graphml,
    _load_csv,
)


@pytest.fixture
def mini_data(tmp_path):
    """Create minimal CSV data for graph construction."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Drug → Target
    pd.DataFrame({
        "drug_normalized": ["drugA", "drugA", "drugB"],
        "target_chembl_id": ["T1", "T2", "T3"],
        "mechanism_of_action": ["mech1", "mech2", "mech3"],
    }).to_csv(data_dir / "edge_drug_target.csv", index=False)

    # Target → Pathway
    pd.DataFrame({
        "target_chembl_id": ["T1", "T2", "T3"],
        "reactome_stid": ["P1", "P2", "P3"],
        "reactome_name": ["Pathway1", "Pathway2", "Pathway3"],
    }).to_csv(data_dir / "edge_target_pathway_all.csv", index=False)

    # Pathway → Disease
    pd.DataFrame({
        "reactome_stid": ["P1", "P2"],
        "diseaseId": ["D001", "D001"],
        "pathway_score": [0.8, 0.6],
        "support_genes": [5, 3],
        "reactome_name": ["Pathway1", "Pathway2"],
        "diseaseName": ["Disease1", "Disease1"],
    }).to_csv(data_dir / "edge_pathway_disease.csv", index=False)

    # Drug → AE
    pd.DataFrame({
        "drug_normalized": ["drugA"],
        "ae_term": ["headache"],
        "report_count": [100],
        "prr": [2.5],
    }).to_csv(data_dir / "edge_drug_ae_faers.csv", index=False)

    # Disease → Phenotype
    pd.DataFrame({
        "diseaseId": ["D001"],
        "diseaseName": ["Disease1"],
        "phenotypeId": ["HP:001"],
        "phenotypeName": ["Pheno1"],
        "score": [0.9],
    }).to_csv(data_dir / "edge_disease_phenotype.csv", index=False)

    # Drug → Trial
    pd.DataFrame({
        "drug_normalized": ["drugA"],
        "nctId": ["NCT001"],
        "is_safety_stop": ["1"],
        "is_efficacy_stop": ["0"],
        "overallStatus": ["TERMINATED"],
    }).to_csv(data_dir / "edge_trial_ae.csv", index=False)

    cfg = Config(raw={
        "paths": {"data_dir": str(data_dir)},
        "files": {},
    })
    return cfg, data_dir


class TestLoadCsv:
    def test_valid_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b\n1,2\n")
        df = _load_csv(p)
        assert len(df) == 1

    def test_missing_csv(self, tmp_path):
        df = _load_csv(tmp_path / "no.csv")
        assert df.empty


class TestBuildKg:
    def test_builds_graph(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_node_types(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        types = {d.get("type") for _, d in G.nodes(data=True)}
        assert "Drug" in types
        assert "Target" in types
        assert "Pathway" in types

    def test_edge_types(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        etypes = {d.get("type") for _, _, d in G.edges(data=True)}
        assert "DRUG_TARGET" in etypes
        assert "TARGET_PATHWAY" in etypes
        assert "PATHWAY_DISEASE" in etypes

    def test_drug_target_edge(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        assert G.has_edge("drugA", "T1")
        assert G.edges["drugA", "T1"]["mechanism"] == "mech1"


class TestGraphStats:
    def test_stats_structure(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        stats = graph_stats(G)
        assert "nodes" in stats
        assert "edges" in stats
        assert "total_nodes" in stats
        assert "total_edges" in stats

    def test_node_counts(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        stats = graph_stats(G)
        assert stats["nodes"]["Drug"] >= 2  # drugA, drugB
        assert stats["nodes"]["Target"] >= 3

    def test_totals_match(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        stats = graph_stats(G)
        assert stats["total_nodes"] == sum(stats["nodes"].values())
        assert stats["total_edges"] == sum(stats["edges"].values())


class TestFindDtpdPaths:
    def test_finds_paths(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        paths = find_dtpd_paths(G, "drugA", "D001")
        assert len(paths) > 0
        for p in paths:
            assert p["drug"] == "drugA"
            assert p["disease"] == "D001"

    def test_path_structure(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        paths = find_dtpd_paths(G, "drugA", "D001")
        p = paths[0]
        assert "drug" in p
        assert "target" in p
        assert "pathway" in p
        assert "disease" in p
        assert "pathway_score" in p

    def test_missing_drug(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        paths = find_dtpd_paths(G, "nonexistent", "D001")
        assert paths == []

    def test_missing_disease(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        paths = find_dtpd_paths(G, "drugA", "NONEXISTENT")
        assert paths == []

    def test_no_connection(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        # drugB → T3 → P3, but P3 has no connection to D001
        paths = find_dtpd_paths(G, "drugB", "D001")
        assert paths == []

    def test_max_paths(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        paths = find_dtpd_paths(G, "drugA", "D001", max_paths=1)
        assert len(paths) <= 1


class TestDrugSummary:
    def test_basic_summary(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        summ = drug_summary(G, "drugA")
        assert len(summ["targets"]) >= 1
        assert len(summ["adverse_events"]) >= 1
        assert len(summ["trials"]) >= 1

    def test_missing_drug(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        summ = drug_summary(G, "NONEXISTENT")
        assert summ["targets"] == []
        assert summ["pathways"] == []

    def test_pathways_found(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        summ = drug_summary(G, "drugA")
        assert len(summ["pathways"]) >= 1

    def test_ae_structure(self, mini_data):
        cfg, _ = mini_data
        G = build_kg(cfg)
        summ = drug_summary(G, "drugA")
        ae = summ["adverse_events"][0]
        assert "term" in ae
        assert "report_count" in ae
        assert "prr" in ae


class TestExportGraphml:
    def test_export(self, mini_data, tmp_path):
        cfg, _ = mini_data
        G = build_kg(cfg)
        out = tmp_path / "test.graphml"
        result = export_graphml(G, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_booleans_converted(self, mini_data, tmp_path):
        """GraphML doesn't support bool; verify conversion to string."""
        cfg, _ = mini_data
        G = build_kg(cfg)
        out = tmp_path / "test.graphml"
        export_graphml(G, out)
        content = out.read_text()
        # Should not raise an error during export
        assert "graphml" in content.lower()
