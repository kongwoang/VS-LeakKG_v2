"""Smoke test for vsleakkg.v2.final_figures.

Builds a tiny fake outputs/v2 tree and verifies the tables get rendered.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from vsleakkg.v2 import final_figures as ff


def _make_outputs(tmp_path: Path) -> Path:
    out = tmp_path / "outputs" / "v2"
    (out / "graph_litpcba_ave").mkdir(parents=True)
    pl.DataFrame({"key": ["n_nodes_in", "n_edges_in", "n_nodes_out"],
                  "value": [10, 20, 8]}).write_csv(
        out / "graph_litpcba_ave" / "stats.csv"
    )
    (out / "phase1").mkdir()
    pl.DataFrame({
        "corpus": ["litpcba", "litpcba"],
        "regime": ["ligand", "scaffold"],
        "feasible": [True, False],
        "n_groups": [3, 1],
        "rho_max": [0.3, 0.95],
        "size_train": [70, 100],
        "size_val": [15, 0],
        "size_test": [15, 0],
        "actives_train": [10, 50],
        "actives_test": [3, 0],
        "baseline_auroc": [0.6, float("nan")],
        "baseline_auprc": [0.4, float("nan")],
        "n_pos_test": [3, 0],
        "n_neg_test": [12, 0],
    }).write_csv(out / "phase1" / "phase1_combined.csv")
    # one matrix CSV
    vc = out / "phase1" / "validation_contamination" / "litpcba"
    vc.mkdir(parents=True)
    pl.DataFrame({
        "direction": ["train->test", "train->val", "val->test"],
        "n": [10, 5, 2],
        "mean": [0.1, 0.2, 0.05],
        "median": [0.0, 0.1, 0.0],
        "p90": [0.5, 0.6, 0.1],
        "p99": [0.9, 0.95, 0.2],
        "frac_gt_0.5": [0.1, 0.2, 0.0],
        "frac_gt_0.8": [0.05, 0.05, 0.0],
    }).write_csv(vc / "ligand.csv")
    return tmp_path


def test_render_all(tmp_path):
    repo = _make_outputs(tmp_path)
    paths = ff.render_all(repo)
    assert paths["table1"].exists()
    assert paths["table2"].exists()
    assert paths["table5"].exists()
    # Figure may or may not exist depending on matplotlib presence

    t2 = pl.read_csv(paths["table2"])
    assert t2.height == 2
    assert "corpus" in t2.columns
    t5 = pl.read_csv(paths["table5"])
    assert t5.height == 3
    assert {"corpus", "regime", "direction"} <= set(t5.columns)
