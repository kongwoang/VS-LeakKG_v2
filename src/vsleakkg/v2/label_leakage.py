"""Exact-row label-leakage check (orthogonal to path-based contamination).

A label leak occurs when an identical (protein, pocket, ligand, label) tuple
appears in two partitions, or when the same (protein, ligand) pair appears
with conflicting labels in two partitions. Path-based contamination cannot
detect this directly because the path strength only measures feature
similarity, not label identity.

Proposal section 5.5: feature leakage and label leakage are reported
separately. This module produces the label-leakage report.
"""
from __future__ import annotations

import polars as pl

# Columns the examples DataFrame must contain. label/pocket may be null.
REQUIRED_COLS = ("example_id", "protein_id", "ligand_id", "label")
OPTIONAL_COLS = ("pocket_id",)


def _key_cols(df: pl.DataFrame, include_label: bool) -> list[str]:
    cols = ["protein_id", "ligand_id"]
    if "pocket_id" in df.columns:
        cols.append("pocket_id")
    if include_label:
        cols.append("label")
    return cols


def exact_row_overlap(
    a: pl.DataFrame,
    b: pl.DataFrame,
    *,
    require_same_label: bool = True,
) -> pl.DataFrame:
    """Identify rows in `a` that have an identical key in `b`.

    Args:
        a, b:               two partitions (e.g. train and test) with the
                            REQUIRED_COLS columns.
        require_same_label: if True, a leak requires identical labels too.
                            If False, any (protein, ligand) overlap is reported
                            (which captures label-conflicting duplicates).

    Returns:
        DataFrame with the leaked rows from `a` plus a `leak_kind` column
        ('same_label' or 'conflicting_label').
    """
    for col in REQUIRED_COLS:
        if col not in a.columns or col not in b.columns:
            raise KeyError(f"missing column: {col}")

    key = _key_cols(a, include_label=False)
    # Inner-join on the feature key only, then split by label agreement.
    b_keys = b.select(key + ["label"]).rename({"label": "label_b"})
    joined = a.join(b_keys, on=key, how="inner")
    if joined.height == 0:
        return joined.with_columns(pl.lit("").alias("leak_kind")).head(0)
    joined = joined.with_columns(
        pl.when(pl.col("label") == pl.col("label_b"))
        .then(pl.lit("same_label"))
        .otherwise(pl.lit("conflicting_label"))
        .alias("leak_kind")
    )
    if require_same_label:
        joined = joined.filter(pl.col("leak_kind") == "same_label")
    return joined.drop("label_b")


def label_leakage_report(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
) -> pl.DataFrame:
    """Summary report of label leakage across the three partitions.

    Returns one row per (source -> target) direction with counts of:
        - same_label leaks (identical row in both partitions)
        - conflicting_label rows (same protein+ligand, different label)
    """
    rows = []
    pairs = [
        ("train", "test", train, test),
        ("train", "val", train, val),
        ("val", "test", val, test),
    ]
    for src_name, dst_name, src_df, dst_df in pairs:
        overlap = exact_row_overlap(src_df, dst_df, require_same_label=False)
        same = int((overlap["leak_kind"] == "same_label").sum()) if overlap.height else 0
        conflict = int((overlap["leak_kind"] == "conflicting_label").sum()) if overlap.height else 0
        rows.append({
            "source_partition": src_name,
            "target_partition": dst_name,
            "n_source": src_df.height,
            "n_target": dst_df.height,
            "n_same_label_leak": same,
            "n_conflicting_label": conflict,
            "frac_target_leaked": (same / dst_df.height) if dst_df.height else 0.0,
        })
    return pl.DataFrame(rows)
