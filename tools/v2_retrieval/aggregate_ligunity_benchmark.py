"""Apply v2 KG target-level splits to LigUnity's published per-target VS benchmark.

LigUnity ships per-target BEDROC/AUROC/EF1 for 26 methods on DUD-E, 18 on
DEKOIS, 11 on LIT-PCBA. Each row is a target, each column is a method's
score on that target.

For each KG split regime (target_random / target_clean / active_clean /
dual_clean / scaffold_clean), we filter the benchmark table to the test
targets defined by that regime, then aggregate per method (mean ± std).

Output:
  <out-dir>/<corpus>/<metric>_per_regime.csv
    rows = methods, cols = regimes; each cell = "mean ± std (n=N)"
  <out-dir>/<corpus>/<metric>_summary.parquet
    long form (method, regime, metric, mean, std, n)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ligunity-results", required=True, type=Path,
                   help="LigUnity/results/VS_results directory")
    p.add_argument("--splits-root", required=True, type=Path,
                   help="outputs/v2_retrieval/splits root (contains <corpus>/<regime>.parquet)")
    p.add_argument("--corpus", choices=["DUDE", "DEKOIS", "PCBA"], required=True)
    p.add_argument("--splits-corpus", required=True, type=str,
                   help="splits subdir name e.g. 'dude' or 'dekois'")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    regimes_dir = args.splits_root / args.splits_corpus
    regimes = []
    for f in sorted(regimes_dir.iterdir()):
        if f.suffix == ".parquet":
            regime = f.stem
            df = pl.read_parquet(f)
            test = df.filter(pl.col("partition") == "test")["target_id"].to_list()
            # target_id format: "tgt:<SOURCE>:<name>" → name (lowercase)
            test_names = [t.split(":")[-1].lower() for t in test]
            regimes.append((regime, test_names))
    print(f"loaded {len(regimes)} regimes for {args.splits_corpus}")

    summary_rows = []
    for metric in ("AUROC", "BEDROC", "EF1"):
        path = args.ligunity_results / f"{args.corpus}_{metric}.csv"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        bench = pl.read_csv(path)
        # Normalize target names to lowercase for matching
        bench = bench.with_columns(pl.col("tid").str.to_lowercase().alias("_tid_lc"))
        method_cols = [c for c in bench.columns if c not in ("tid", "_tid_lc")]
        print(f"\n=== {args.corpus} / {metric} — {len(method_cols)} methods, {bench.shape[0]} targets ===")

        # Build "method × regime" table of "mean ± std (n)"
        out_rows = []
        for method in method_cols:
            row: dict[str, object] = {"method": method}
            for regime, test_names in regimes:
                sub = bench.filter(pl.col("_tid_lc").is_in(test_names))
                vals = sub[method].drop_nulls().to_numpy()
                if len(vals) == 0:
                    row[regime] = "n=0"
                    continue
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                row[regime] = f"{mean:.3f} ± {std:.3f} (n={len(vals)})"
                summary_rows.append({
                    "corpus": args.corpus,
                    "metric": metric,
                    "method": method,
                    "regime": regime,
                    "mean": mean,
                    "std": std,
                    "n": len(vals),
                })
            out_rows.append(row)
        out_df = pl.DataFrame(out_rows)
        out_csv = args.out_dir / f"{args.corpus.lower()}_{metric}_per_regime.csv"
        out_df.write_csv(out_csv)
        print(f"wrote {out_csv}")
        print(out_df)

    summary_df = pl.DataFrame(summary_rows)
    summary_path = args.out_dir / f"{args.corpus.lower()}_long_summary.parquet"
    summary_df.write_parquet(summary_path)
    print(f"\nwrote {summary_path}")

    # Headline: BEDROC method × regime, sorted by random AUROC descending
    if summary_df.shape[0] > 0:
        bedroc_random = (summary_df
            .filter((pl.col("metric") == "BEDROC") & (pl.col("regime") == "target_random"))
            .select(["method", "mean"])
            .sort("mean", descending=True))
        print()
        print("=== Top methods by BEDROC on target_random (DrugCLIP, LigUnity, etc.) ===")
        print(bedroc_random.head(10))


if __name__ == "__main__":
    main()
