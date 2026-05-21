"""Smoke tests for vsleakkg.v2.build_graph.

These run against tiny synthetic v1-shaped parquets so we can validate
the schema mapping without needing the unpacked dataset archive.
"""
from __future__ import annotations

import polars as pl
import pytest

from vsleakkg.v2 import build_graph as bg
from vsleakkg.v2.schema import EdgeType, HubMitigationConfig, NodeType


def _write_v1_minimal(tmp_path):
    """Build a v1-shaped (nodes, edges) parquet pair."""
    nodes = pl.DataFrame(
        {
            "node_id": [
                "Example::ex1", "Example::ex2",
                "Ligand::L1", "Ligand::L2",
                "Scaffold::S1", "Scaffold::trivial",
                "Protein::P1",
                "ChEMBLActivity::A1",  # to be dropped
                "Complex::C1",          # to be dropped
            ],
            "node_type": [
                "Example", "Example",
                "Ligand", "Ligand",
                "Scaffold", "Scaffold",
                "Protein",
                "ChEMBLActivity",
                "Complex",
            ],
            "label": [
                "ex1", "ex2",
                "L1", "L2",
                "C1CCCCC1c1ccccc1", "CC",  # phenyl-cyclohexane (12) vs trivial (2)
                "MKTAYIAKQRQ",
                "A1",
                "C1",
            ],
            "props": ["{}"] * 9,
        }
    )
    edges = pl.DataFrame(
        {
            "src": [
                "Example::ex1", "Example::ex2",  # examples to ligands
                "Example::ex1", "Example::ex2",
                "Ligand::L1", "Ligand::L2",
                "Complex::C1",     # to be dropped
            ],
            "dst": [
                "Ligand::L1", "Ligand::L2",
                "Protein::P1", "Protein::P1",
                "Scaffold::S1", "Scaffold::S1",
                "Protein::P1",
            ],
            "edge_type": [
                "example_has_ligand", "example_has_ligand",
                "example_targets_protein", "example_targets_protein",
                "ligand_has_scaffold", "ligand_has_scaffold",
                "complex_has_protein",
            ],
            "props": ["{}"] * 7,
        }
    )
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    nodes.write_parquet(proc / "mvp2_nodes.parquet")
    edges.write_parquet(proc / "mvp2_edges.parquet")
    return proc


def test_build_graph_maps_types(tmp_path, monkeypatch):
    proc = _write_v1_minimal(tmp_path)
    monkeypatch.setenv("VSLEAKKG_V1_ROOT", str(tmp_path))

    out = tmp_path / "out"
    stats = bg.build_graph(out, corpus="all")

    assert stats.n_edges_in == 7
    # complex_has_protein dropped + nothing else dropped
    assert stats.n_edges_dropped == 1
    # ChEMBLActivity + Complex dropped
    assert stats.n_nodes_dropped == 2

    nodes_df = pl.read_parquet(out / "v2_nodes.parquet")
    types = set(nodes_df["node_type"].unique().to_list())
    assert NodeType.EXAMPLE.value in types
    assert NodeType.LIGAND.value in types
    assert NodeType.PROTEIN.value in types
    assert NodeType.SCAFFOLD.value in types
    # No leftover v1 scaffolding types
    assert "ChEMBLActivity" not in types
    assert "Complex" not in types

    edges_df = pl.read_parquet(out / "v2_edges.parquet")
    etypes = set(edges_df["edge_type"].unique().to_list())
    assert EdgeType.EXAMPLE_HAS_LIGAND.value in etypes
    assert EdgeType.EXAMPLE_HAS_PROTEIN.value in etypes
    assert EdgeType.LIGAND_SCAFFOLD.value in etypes
    # `complex_has_protein` doesn't map to a v2 edge — should be gone
    assert "complex_has_protein" not in etypes


def test_build_graph_drops_trivial_scaffolds(tmp_path, monkeypatch):
    proc = _write_v1_minimal(tmp_path)
    monkeypatch.setenv("VSLEAKKG_V1_ROOT", str(tmp_path))

    out = tmp_path / "out"
    stats = bg.build_graph(
        out, corpus="all", hub_cfg=HubMitigationConfig(trivial_scaffold_max_atoms=6)
    )
    assert stats.n_trivial_scaffolds_dropped == 1   # Scaffold::trivial = "CC"

    nodes_df = pl.read_parquet(out / "v2_nodes.parquet")
    assert "Scaffold::trivial" not in set(nodes_df["node_id"].to_list())


def test_build_graph_writes_stats(tmp_path, monkeypatch):
    proc = _write_v1_minimal(tmp_path)
    monkeypatch.setenv("VSLEAKKG_V1_ROOT", str(tmp_path))

    out = tmp_path / "out"
    bg.build_graph(out, corpus="all")
    stats_df = pl.read_csv(out / "stats.csv")
    keys = set(stats_df["key"].to_list())
    assert "n_nodes_in" in keys
    assert "n_edges_out" in keys
    # deferred markers present
    assert any(k.startswith("deferred::") for k in keys)
