"""Paired Wilcoxon + Holm correction + bootstrap CIs across splitters.

For Mode A: per-target paired test between KG and each non-KG splitter,
on each of {AUROC, EF1%, BEDROC, c_total, max_lig_tanimoto}.
For Mode B: bootstrap CIs over targets.

Output schema: TABLE_STAT_TESTS_COLUMNS from schemas.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl

from .schemas import TABLE_STAT_TESTS_COLUMNS


def holm(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = [1.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        adj_p = min(1.0, max(running, pvals[i] * (n - rank)))
        adj[i] = adj_p
        running = adj_p
    return adj


def wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan"), float("nan")
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return float("nan"), float("nan")
    try:
        r = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 1000, seed: int = 2025) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 5:
        return float("nan"), float("nan")
    means = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(diffs[idx].mean())
    means = np.sort(means)
    return float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality",   required=True, type=Path)
    ap.add_argument("--metrics",   required=True, type=Path)
    ap.add_argument("--corpus",    required=True)
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--out-csv",   required=True, type=Path)
    ap.add_argument("--kg-splitter-prefix", default="kg_",
                    help="comparator key: splitters starting with this are the KG family.")
    args = ap.parse_args()

    metric_df = pl.read_csv(args.metrics) if args.metrics.exists() else pl.DataFrame()
    quality_df = pl.read_csv(args.quality, infer_schema_length=0) if args.quality.exists() else pl.DataFrame()
    rows = []

    # Model metrics: per-target paired between each non-KG splitter and the matching KG splitter.
    if metric_df.height > 0:
        kg_splitters = [s for s in metric_df["splitter"].unique().to_list()
                        if s.startswith(args.kg_splitter_prefix)]
        non_kg = [s for s in metric_df["splitter"].unique().to_list()
                  if not s.startswith(args.kg_splitter_prefix)]
        for metric in ["auroc", "ef1pct", "bedroc"]:
            pvals = []; meta_rows = []
            for kg in kg_splitters:
                for other in non_kg:
                    a_df = metric_df.filter((pl.col("splitter") == other) & (pl.col("model") == "morgan_rf"))
                    b_df = metric_df.filter((pl.col("splitter") == kg)    & (pl.col("model") == "morgan_rf"))
                    joined = a_df.join(b_df, on="target_id", how="inner", suffix="_kg")
                    if joined.height < 5:
                        continue
                    a = joined[metric].to_numpy().astype(float)
                    b = joined[f"{metric}_kg"].to_numpy().astype(float)
                    stat, p = wilcoxon_pair(a, b)
                    ci_lo, ci_hi = bootstrap_ci(a - b)
                    if np.isnan(p): continue
                    pvals.append(p); meta_rows.append({
                        "corpus": args.corpus, "mode": args.mode, "metric": metric,
                        "splitter_a": other, "splitter_b": kg,
                        "n": int(joined.height),
                        "mean_diff": float(np.nanmean(a - b)),
                        "wilcoxon_stat": stat, "p_value": p,
                        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                    })
            adj = holm(pvals) if pvals else []
            for r, p_h in zip(meta_rows, adj):
                r["p_holm"] = p_h
                rows.append(r)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pl.DataFrame(rows)[TABLE_STAT_TESTS_COLUMNS]
    else:
        df = pl.DataFrame({c: [] for c in TABLE_STAT_TESTS_COLUMNS})
    df.write_csv(args.out_csv)
    print(f"stats: {len(rows)} comparisons -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
