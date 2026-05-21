"""Unit tests for the leakage-hub diagnostic."""
from __future__ import annotations

import polars as pl

from vsleakkg.v2.hubs import hub_scores, leakage_concentration


def _nn(rows):
    return pl.DataFrame(rows, schema={
        "example_id": pl.Utf8, "predicted_label": pl.Int64,
        "source_example": pl.Utf8, "contamination": pl.Float64,
    })


def test_hub_counts_concentrate_on_argmax_sources():
    table = _nn([
        {"example_id": "q1", "predicted_label": 1, "source_example": "t1", "contamination": 0.9},
        {"example_id": "q2", "predicted_label": 1, "source_example": "t1", "contamination": 0.8},
        {"example_id": "q3", "predicted_label": 0, "source_example": "t2", "contamination": 0.5},
    ])
    out = hub_scores(table)
    rows = {r["training_example"]: r for r in out.iter_rows(named=True)}
    assert rows["t1"]["n_test_rows_leaked"] == 2
    assert rows["t2"]["n_test_rows_leaked"] == 1


def test_unreached_queries_excluded():
    table = _nn([
        {"example_id": "q1", "predicted_label": 1, "source_example": "t1", "contamination": 0.9},
        {"example_id": "q2", "predicted_label": -1, "source_example": "", "contamination": 0.0},
    ])
    out = hub_scores(table)
    assert set(out["training_example"].to_list()) == {"t1"}


def test_top_k_truncates():
    rows = [
        {"example_id": f"q{i}", "predicted_label": 1,
         "source_example": f"t{i//10}", "contamination": 0.5}
        for i in range(50)
    ]
    out = hub_scores(_nn(rows), top_k=2)
    assert out.height == 2


def test_concentration_metrics_sane():
    rows = [
        {"example_id": f"q{i}", "predicted_label": 1,
         "source_example": "t1" if i < 90 else f"t{i}", "contamination": 0.5}
        for i in range(100)
    ]
    stats = leakage_concentration(_nn(rows))
    assert 0.0 <= stats["top_1pct_share"] <= 1.0
    assert stats["top_10pct_share"] >= 0.9  # 90 of 100 leaks land on t1
