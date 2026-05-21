"""Leakage-hub diagnostic: which training rows cause the most cross-partition leakage?

Proposal section 5.7. For each training example x_i, the hub score is the
number of validation/test examples for which x_i is the argmax contamination
match:

    H(x_i) = |{ x_t in val | test : x_i in argmax_{x_j in train} C(x_t, {x_j}) }|

Top-K hubs identify a small set of training rows responsible for a
disproportionate share of leakage. Trimming them is often a more practical
mitigation than reconfiguring the split.

Input is the output of scoring.contamination_nn_label, which already records
the argmax source per query. We just aggregate.
"""
from __future__ import annotations

import polars as pl


def hub_scores(nn_table: pl.DataFrame, *, top_k: int | None = None) -> pl.DataFrame:
    """Aggregate hub counts from a contamination-NN table.

    Args:
        nn_table: output of scoring.contamination_nn_label, must contain
                  columns (example_id, source_example, contamination).
        top_k:    if set, return only the top-K hubs by H(x_i).

    Returns:
        DataFrame with columns (training_example, n_test_rows_leaked,
        mean_contamination, max_contamination).
    """
    if "source_example" not in nn_table.columns:
        raise KeyError("nn_table missing 'source_example' column")

    df = nn_table.filter(pl.col("source_example") != "")
    if df.height == 0:
        return pl.DataFrame(schema={
            "training_example": pl.Utf8,
            "n_test_rows_leaked": pl.Int64,
            "mean_contamination": pl.Float64,
            "max_contamination": pl.Float64,
        })

    agg = (
        df.group_by("source_example")
        .agg([
            pl.len().alias("n_test_rows_leaked"),
            pl.col("contamination").mean().alias("mean_contamination"),
            pl.col("contamination").max().alias("max_contamination"),
        ])
        .rename({"source_example": "training_example"})
        .sort("n_test_rows_leaked", descending=True)
    )
    if top_k is not None:
        agg = agg.head(top_k)
    return agg


def leakage_concentration(nn_table: pl.DataFrame) -> dict[str, float]:
    """How concentrated is leakage? Returns Gini-like statistics.

    Useful to motivate hub trimming:
      - if `top_1pct_share` is high, a tiny minority of train rows accounts
        for most leakage and trimming is effective.
      - if it's low, leakage is distributed and trimming won't help.
    """
    hubs = hub_scores(nn_table)
    if hubs.height == 0:
        return {"top_1pct_share": 0.0, "top_10pct_share": 0.0, "n_unique_sources": 0}

    counts = hubs["n_test_rows_leaked"].to_numpy()
    counts_sorted = sorted(counts, reverse=True)
    total = sum(counts_sorted)
    n = len(counts_sorted)

    def share(frac: float) -> float:
        k = max(1, int(round(n * frac)))
        return sum(counts_sorted[:k]) / total if total else 0.0

    return {
        "top_1pct_share": share(0.01),
        "top_10pct_share": share(0.10),
        "n_unique_sources": n,
    }
