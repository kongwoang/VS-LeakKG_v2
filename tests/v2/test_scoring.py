"""Unit tests for v2 contamination scoring."""
from __future__ import annotations

import math

import polars as pl
import pytest

from vsleakkg.v2.scoring import (
    contamination_nn_label,
    score_axis,
    score_overall,
)


def _edges(tuples):
    return pl.DataFrame(
        [{"src": s, "dst": d, "edge_type": t} for s, d, t in tuples],
        schema={"src": pl.Utf8, "dst": pl.Utf8, "edge_type": pl.Utf8},
    )


def test_single_exact_ligand_edge_gives_score_one():
    # Two examples sharing the same ligand exactly.
    edges = _edges([
        ("ex:1", "lig:A", "example_has_ligand"),
        ("ex:2", "lig:A", "example_has_ligand"),
    ])
    out = score_axis(edges, reference={"ex:1"}, queries={"ex:2"}, axis="ligand")
    val = out["contamination"].to_list()[0]
    # ex:1 -> lig:A (w=1.0) -> ex:2 (w=1.0). Product = 1.0.
    assert math.isclose(val, 1.0, abs_tol=1e-9)


def test_scaffold_axis_uses_only_scaffold_edges():
    # Two ligands share a scaffold; no direct ligand-similarity edge.
    edges = _edges([
        ("ex:1", "lig:A", "example_has_ligand"),
        ("ex:2", "lig:B", "example_has_ligand"),
        ("lig:A", "sca:S", "ligand_scaffold"),
        ("lig:B", "sca:S", "ligand_scaffold"),
    ])
    out = score_axis(edges, reference={"ex:1"}, queries={"ex:2"}, axis="scaffold")
    val = out["contamination"].to_list()[0]
    # Product = 1.0 * 0.70 * 0.70 * 1.0 = 0.49
    assert math.isclose(val, 1.0 * 0.7 * 0.7 * 1.0, rel_tol=1e-9)


def test_axis_decomposition_isolates_axes():
    # ex:1 and ex:2 share a scaffold AND a protein cluster. Scaffold axis
    # should not see the protein edges.
    edges = _edges([
        ("ex:1", "lig:A", "example_has_ligand"),
        ("ex:2", "lig:B", "example_has_ligand"),
        ("lig:A", "sca:S", "ligand_scaffold"),
        ("lig:B", "sca:S", "ligand_scaffold"),
        ("ex:1", "prot:P", "example_has_protein"),
        ("ex:2", "prot:P", "example_has_protein"),
    ])
    overall = score_overall(edges, reference={"ex:1"}, queries={"ex:2"})
    row = overall.row(0, named=True)
    assert math.isclose(row["C_scaffold"], 0.49, rel_tol=1e-9)
    assert math.isclose(row["C_protein"], 1.0, rel_tol=1e-9)
    assert row["dominant_axis"] == "protein"
    assert math.isclose(row["C_overall"], 1.0, rel_tol=1e-9)


def test_unreached_query_scores_zero():
    edges = _edges([
        ("ex:1", "lig:A", "example_has_ligand"),
    ])
    out = score_axis(edges, reference={"ex:1"}, queries={"ex:2"}, axis="ligand")
    assert out["contamination"].to_list()[0] == 0.0


def test_contamination_nn_records_argmax_source():
    edges = _edges([
        ("ex:1", "lig:A", "example_has_ligand"),
        ("ex:2", "lig:B", "example_has_ligand"),
        ("ex:t", "lig:A", "example_has_ligand"),
    ])
    out = contamination_nn_label(
        edges,
        reference_labels={"ex:1": 1, "ex:2": 0},
        queries={"ex:t"},
    )
    row = out.row(0, named=True)
    assert row["source_example"] == "ex:1"
    assert row["predicted_label"] == 1
    assert math.isclose(row["contamination"], 1.0, abs_tol=1e-9)


def test_max_hops_blocks_longer_paths():
    # Two examples bridged by a long ligand-similarity chain. With max_hops=2
    # the bridge is unreachable.
    edges = _edges([
        ("ex:1", "lig:A", "example_has_ligand"),
        ("lig:A", "lig:B", "ligand_similar"),
        ("lig:B", "lig:C", "ligand_similar"),
        ("ex:2", "lig:C", "example_has_ligand"),
    ])
    out_short = score_axis(
        edges, reference={"ex:1"}, queries={"ex:2"},
        axis="ligand", max_hops=2,
    )
    out_long = score_axis(
        edges, reference={"ex:1"}, queries={"ex:2"},
        axis="ligand", max_hops=10,
    )
    assert out_short["contamination"].to_list()[0] == 0.0
    assert out_long["contamination"].to_list()[0] > 0.0
