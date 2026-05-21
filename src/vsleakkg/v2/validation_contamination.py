"""Three validation-contamination matrices (proposal section 5.13).

For every clean regime, we report contamination separately in three
directions:

    C(train -> test)   inflates fit
    C(train -> val)    inflates checkpoint selection
    C(val   -> test)   inflates test scores via val-driven model selection

In Mode B (model audit), we additionally compute C(D_train^m -> D_val^paper)
and C(D_train^m -> D_test^paper) for the model's published splits.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .scoring import score_overall


@dataclass(frozen=True)
class ContaminationMatrixResult:
    direction: str  # e.g. "train->test"
    per_example: pl.DataFrame  # output of score_overall on the target queries
    summary: dict[str, float]


def _summary(scores: pl.DataFrame) -> dict[str, float]:
    if scores.height == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0,
                "frac_gt_0.5": 0.0, "frac_gt_0.8": 0.0}
    s = scores["C_overall"].to_numpy()
    s_sorted = sorted(s)
    n = len(s_sorted)
    return {
        "n": n,
        "mean": float(sum(s) / n),
        "median": float(s_sorted[n // 2]),
        "p90": float(s_sorted[int(0.90 * (n - 1))]),
        "p99": float(s_sorted[int(0.99 * (n - 1))]),
        "frac_gt_0.5": float(sum(1 for x in s if x > 0.5) / n),
        "frac_gt_0.8": float(sum(1 for x in s if x > 0.8) / n),
    }


def three_way_contamination(
    edges: pl.DataFrame,
    *,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    weights: dict[str, float] | None = None,
    max_hops: int = 6,
) -> dict[str, ContaminationMatrixResult]:
    """Compute the three contamination matrices for a single regime.

    Returns:
        dict with keys 'train->test', 'train->val', 'val->test', each mapping
        to a ContaminationMatrixResult with per-example scores and a summary.
    """
    results: dict[str, ContaminationMatrixResult] = {}

    directions = [
        ("train->test", train_ids, test_ids),
        ("train->val", train_ids, val_ids),
        ("val->test", val_ids, test_ids),
    ]
    for direction, ref, queries in directions:
        scores = score_overall(
            edges, reference=ref, queries=queries,
            weights=weights, max_hops=max_hops,
        )
        results[direction] = ContaminationMatrixResult(
            direction=direction,
            per_example=scores,
            summary=_summary(scores),
        )
    return results


def to_summary_table(results: dict[str, ContaminationMatrixResult]) -> pl.DataFrame:
    """Flatten the three matrices into one summary DataFrame for reporting."""
    rows = []
    for direction, r in results.items():
        row = {"direction": direction}
        row.update(r.summary)
        rows.append(row)
    return pl.DataFrame(rows)


def validation_leakage_effect(
    test_metric_leaky_val: float,
    test_metric_clean_val: float,
) -> dict[str, float]:
    """Quantify the validation-leakage effect on the final test score.

    Args:
        test_metric_leaky_val:  test metric for checkpoint selected on the
                                model's published (potentially leaky) val.
        test_metric_clean_val:  test metric for checkpoint selected on the
                                VS-LeakKG-clean val.

    Returns:
        dict with absolute and relative drop.
    """
    drop_abs = test_metric_leaky_val - test_metric_clean_val
    drop_rel = drop_abs / test_metric_leaky_val if test_metric_leaky_val else 0.0
    return {
        "test_metric_leaky_val": test_metric_leaky_val,
        "test_metric_clean_val": test_metric_clean_val,
        "drop_absolute": drop_abs,
        "drop_relative": drop_rel,
    }
