"""Build a RANDOM control split of PDBBind v2 — same examples, same train/val/test
sizes as protein-clean, but assignment is uniform-random instead of leakage-aware.

Purpose: when SPRINT trained on the v2 protein-clean split scores AUROC 0.59 vs
ligand-clean's 0.76, a skeptic could argue the drop is sample-size noise or a
weirdness of the v2 split sizes. Training the same SPRINT on a random split of
the same size lets us refute that:

  - If random ≈ ligand-clean (~0.76)  → the v2 protein-clean drop is real
                                         (the KG is removing genuine leakage)
  - If random ≈ protein-clean (~0.59) → the v2 framework is not doing anything
                                         beyond what random sampling does

We match protein-clean's sizes (7337/5856/5844) because that's the regime with
the strongest drop, so it's the regime that needs the cleanest control.

Usage:
    python build_random_pdbbind_split.py \
        --base outputs/v2/phase1_full/splits/pdbbind/protein.parquet \
        --out  outputs/v2/phase1_full/splits/pdbbind/random.parquet \
        --seed 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=Path,
                   help="A v2 split parquet whose train/val/test sizes we should match")
    p.add_argument("--out", required=True, type=Path,
                   help="Where to write the random split parquet")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    base = pl.read_parquet(args.base)
    n_train = base.filter(pl.col("partition") == "train").shape[0]
    n_val = base.filter(pl.col("partition") == "val").shape[0]
    n_test = base.filter(pl.col("partition") == "test").shape[0]
    total = base.shape[0]
    print(f"Base split: {args.base.name}")
    print(f"  total={total}  train={n_train}  val={n_val}  test={n_test}")
    assert n_train + n_val + n_test == total, "base sizes don't sum to total"

    rng = np.random.default_rng(args.seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    partitions = np.empty(total, dtype=object)
    partitions[indices[:n_train]] = "train"
    partitions[indices[n_train:n_train + n_val]] = "val"
    partitions[indices[n_train + n_val:]] = "test"

    out = base.drop("partition").with_columns(pl.Series("partition", partitions))
    out = out.select(base.columns)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    print(f"Wrote {args.out}")
    print()
    print("Sanity:")
    print(out.group_by("partition").agg(
        pl.len().alias("n_total"),
        pl.col("label").sum().alias("n_pos"),
    ).sort("partition").to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
