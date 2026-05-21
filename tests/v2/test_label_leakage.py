"""Unit tests for label-leakage exact-row detection."""
from __future__ import annotations

import polars as pl

from vsleakkg.v2.label_leakage import exact_row_overlap, label_leakage_report


def _df(rows):
    return pl.DataFrame(rows, schema={
        "example_id": pl.Utf8, "protein_id": pl.Utf8, "ligand_id": pl.Utf8,
        "label": pl.Int64,
    })


def test_same_label_leak_is_detected():
    a = _df([{"example_id": "a1", "protein_id": "P", "ligand_id": "L", "label": 1}])
    b = _df([{"example_id": "b1", "protein_id": "P", "ligand_id": "L", "label": 1}])
    out = exact_row_overlap(a, b, require_same_label=True)
    assert out.height == 1
    assert out["leak_kind"].to_list() == ["same_label"]


def test_conflicting_label_flagged_when_lenient():
    a = _df([{"example_id": "a1", "protein_id": "P", "ligand_id": "L", "label": 1}])
    b = _df([{"example_id": "b1", "protein_id": "P", "ligand_id": "L", "label": 0}])
    strict = exact_row_overlap(a, b, require_same_label=True)
    assert strict.height == 0
    lenient = exact_row_overlap(a, b, require_same_label=False)
    assert lenient.height == 1
    assert lenient["leak_kind"].to_list() == ["conflicting_label"]


def test_no_overlap():
    a = _df([{"example_id": "a1", "protein_id": "P", "ligand_id": "L1", "label": 1}])
    b = _df([{"example_id": "b1", "protein_id": "P", "ligand_id": "L2", "label": 1}])
    assert exact_row_overlap(a, b, require_same_label=False).height == 0


def test_report_three_directions():
    train = _df([{"example_id": "t1", "protein_id": "P", "ligand_id": "L", "label": 1}])
    val   = _df([{"example_id": "v1", "protein_id": "P", "ligand_id": "L", "label": 1}])
    test  = _df([{"example_id": "x1", "protein_id": "P", "ligand_id": "L", "label": 0}])
    rep = label_leakage_report(train, val, test)
    rows = {(r["source_partition"], r["target_partition"]): r for r in rep.iter_rows(named=True)}
    assert rows[("train", "val")]["n_same_label_leak"] == 1
    assert rows[("train", "test")]["n_conflicting_label"] == 1
    assert rows[("val", "test")]["n_conflicting_label"] == 1
