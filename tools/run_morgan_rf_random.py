"""Run the Morgan-RF ligand-only baseline on the v2 random control split.

Mirrors what the Phase 1 pipeline did for ligand/protein/dual but pointed at
random.parquet. Outputs a single-row CSV that we can append into the audit
table.

Usage:
    python run_morgan_rf_random.py \
        --split outputs/v2/phase1_full/splits/pdbbind/random.parquet \
        --out  outputs/v2/phase1_full/baselines/pdbbind_random_morgan_rf.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v2/src")

from vsleakkg.v2.baselines.ligand_only import evaluate_ligand_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--corpus", default="pdbbind",
                   help="Corpus label for the output row (e.g. dekois, dude, litpcba, pdbbind)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pl.read_parquet(args.split)
    train = df.filter(pl.col("partition") == "train")
    test = df.filter(pl.col("partition") == "test")

    print(f"Train: {train.shape[0]}  Test: {test.shape[0]}")
    print(f"Train pos rate: {train.filter(pl.col('label')==1.0).shape[0] / train.shape[0]:.3f}")
    print(f"Test  pos rate: {test.filter(pl.col('label')==1.0).shape[0] / test.shape[0]:.3f}")

    result = evaluate_ligand_only(train, test, random_state=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    row = pl.DataFrame({
        "corpus": [args.corpus],
        "regime": ["random"],
        "n_train": [train.shape[0]],
        "n_test": [test.shape[0]],
        "n_pos_test": [int(result.n_pos)],
        "n_neg_test": [int(result.n_neg)],
        "baseline_auroc": [result.auroc],
        "baseline_auprc": [result.auprc],
        "used_rdkit": [result.used_rdkit],
    })
    row.write_csv(args.out)
    print()
    print(f"AUROC: {result.auroc:.4f}")
    print(f"AUPRC: {result.auprc:.4f}")
    print(f"used_rdkit: {result.used_rdkit}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
