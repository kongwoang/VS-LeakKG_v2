"""Smoke test for vsleakkg.v2.pipeline.

Build a tiny synthetic v2 graph + side-table, run the pipeline driver
end-to-end, assert that per-regime splits + summary CSVs land in the
expected places.
"""
from __future__ import annotations

import polars as pl

from vsleakkg.v2 import pipeline as pp
from vsleakkg.v2.hydrate import SIDE_TABLE_COLUMNS, SIDE_TABLE_SCHEMA
from vsleakkg.v2.schema import EdgeType, NodeType


def _make_graph(tmp_path):
    """A minimal v2 graph: 6 examples, 3 ligands, 2 proteins, 3 scaffolds."""
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True)
    examples = ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"]
    ligands = ["L1", "L1", "L2", "L2", "L3", "L3"]  # paired ligands
    proteins = ["P1", "P1", "P1", "P2", "P2", "P2"]
    scaffolds = ["S1", "S1", "S2", "S2", "S3", "S3"]
    nodes_rows = []
    for e in examples:
        nodes_rows.append({"node_id": e, "node_type": NodeType.EXAMPLE.value,
                           "label": e, "props": "{}"})
    for n in set(ligands):
        nodes_rows.append({"node_id": n, "node_type": NodeType.LIGAND.value,
                           "label": n, "props": "{}"})
    for n in set(proteins):
        nodes_rows.append({"node_id": n, "node_type": NodeType.PROTEIN.value,
                           "label": n, "props": "{}"})
    for n in set(scaffolds):
        nodes_rows.append({"node_id": n, "node_type": NodeType.SCAFFOLD.value,
                           "label": n, "props": "{}"})
    pl.DataFrame(nodes_rows).write_parquet(graph_dir / "v2_nodes.parquet")

    edges_rows = []
    for e, lg, pr, sc in zip(examples, ligands, proteins, scaffolds):
        edges_rows.append({"src": e, "dst": lg,
                           "edge_type": EdgeType.EXAMPLE_HAS_LIGAND.value,
                           "props": "{}"})
        edges_rows.append({"src": e, "dst": pr,
                           "edge_type": EdgeType.EXAMPLE_HAS_PROTEIN.value,
                           "props": "{}"})
        edges_rows.append({"src": lg, "dst": sc,
                           "edge_type": EdgeType.LIGAND_SCAFFOLD.value,
                           "props": "{}"})
    pl.DataFrame(edges_rows).write_parquet(graph_dir / "v2_edges.parquet")
    return graph_dir, examples


def _make_side_table(tmp_path, example_ids):
    rows = []
    for i, eid in enumerate(example_ids):
        row = {c: None for c in SIDE_TABLE_COLUMNS}
        row.update({
            "example_id": eid,
            "source": "litpcba",
            "source_id": eid,
            "smiles": "CCO" if i % 2 == 0 else "c1ccccc1",
            "smiles_canonical": "CCO" if i % 2 == 0 else "c1ccccc1",
            "label": float(i % 2),
            "label_kind": "binary",
        })
        rows.append(row)
    df = pl.DataFrame(rows, schema=SIDE_TABLE_SCHEMA)
    p = tmp_path / "side_table.parquet"
    df.write_parquet(p)
    return p


def test_pipeline_runs_all_regimes(tmp_path):
    graph_dir, examples = _make_graph(tmp_path)
    side_p = _make_side_table(tmp_path, examples)
    out = tmp_path / "out"

    results = pp.run_corpus(
        graph_dir=graph_dir,
        side_table_path=side_p,
        output_dir=out,
        corpus_tag="test",
        # Relax constraints — synthetic graph is too small for the defaults
        constraints=pp.SplitConstraints(
            min_targets_per_partition=1,
            min_actives_per_partition=1,
            label_balance_tol=1.0,
        ),
    )
    assert set(results.keys()) == set(pp.REGIMES.keys())
    # At least the ligand/protein/scaffold regimes should be feasible —
    # we have edges for all three. Pocket/assay/source/time have no edges
    # so they're correctly infeasible.
    for ax in ("ligand", "protein", "scaffold"):
        rr = results[ax]
        assert rr.notes == "" or "infeasible" not in rr.notes, \
            f"{ax} should have edges, got notes={rr.notes!r}"
    # Pocket has no edges in our synthetic graph
    assert "infeasible" in results["pocket"].notes

    # Outputs land in the expected layout
    assert (out / "splits" / "test" / "ligand.parquet").exists()
    assert (out / "phase1" / "test_summary.csv").exists()
    summary = pl.read_csv(out / "phase1" / "test_summary.csv")
    assert summary.height == len(pp.REGIMES)
    assert "feasible" in summary.columns
    assert "baseline_auroc" in summary.columns
