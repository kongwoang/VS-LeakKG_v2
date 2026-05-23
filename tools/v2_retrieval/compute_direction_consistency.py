"""Compute direction-consistency across methods for each (corpus, metric, regime) pair.

For each method, compute (target_random - <regime>) for each clean regime.
Then count: how many of N methods showed a drop (random > clean)?

If the leakage signal is real and model-invariant, we expect MOST methods
to show drops on the clean regimes (consistent direction).
"""
import polars as pl
import sys
from pathlib import Path

corpus = sys.argv[1]  # dude / dekois / litpcba
root = Path("/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v2/outputs/v2_retrieval/results")
summary_path = root / f"{corpus}_cross_method" / f"{corpus.replace('litpcba','pcba')}_long_summary.parquet"
df = pl.read_parquet(summary_path)

regimes_to_compare = ["target_clean", "active_clean", "scaffold_clean", "dual_clean"]

for metric in ("BEDROC", "AUROC"):
    print(f"\n=== {corpus.upper()} / {metric} — direction consistency ===")
    sub = df.filter(pl.col("metric") == metric)
    # Pivot: rows=method, cols=regime
    pivot = sub.pivot(values="mean", index="method", on="regime", aggregate_function="first")
    if "target_random" not in pivot.columns:
        print("  no target_random column; skip")
        continue
    n_methods = pivot.shape[0]
    print(f"  total methods: {n_methods}")
    for regime in regimes_to_compare:
        if regime not in pivot.columns:
            continue
        df_r = pivot.select(["method", "target_random", regime]).drop_nulls()
        if df_r.shape[0] == 0:
            continue
        n_drop = df_r.filter(pl.col(regime) < pl.col("target_random")).shape[0]
        n_up = df_r.filter(pl.col(regime) > pl.col("target_random")).shape[0]
        n_eq = df_r.filter(pl.col(regime) == pl.col("target_random")).shape[0]
        n = df_r.shape[0]
        # Compute mean delta
        diff = df_r.select((pl.col(regime) - pl.col("target_random")).mean().alias("d"))[0, "d"]
        # Sign test p-value (binomial test): probability of seeing n_drop or more
        # heads with p=0.5 under null
        try:
            from scipy.stats import binomtest
            res = binomtest(n_drop, n, p=0.5, alternative="two-sided")
            p = res.pvalue
        except Exception:
            p = float("nan")
        print(f"  random→{regime:14s}: n={n}  drop={n_drop}  up={n_up}  flat={n_eq}  mean Δ={diff:+.4f}  sign-test p={p:.3g}")
