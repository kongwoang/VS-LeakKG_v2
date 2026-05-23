"""Third Group A model: Logistic Regression on Morgan fingerprints.

Adds a third ligand-only baseline (alongside Morgan-RF and SPRINT) to
strengthen the model-invariance claim on Group A. LR is a linear model;
RF is non-parametric tree ensemble; SPRINT is a deep dual-tower neural
net. If all three show the same leakage gap, the gap is shape-agnostic.

Usage:
    python run_lr_baseline.py \
        --split outputs/v2/phase1_full/splits/pdbbind/random.parquet \
        --out outputs/v2/phase1_full/baselines/pdbbind_random_morgan_lr.csv \
        --corpus pdbbind --regime random
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v2/src")
from vsleakkg.v2.baselines.ligand_only import featurise_ligands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--corpus", required=True)
    p.add_argument("--regime", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pl.read_parquet(args.split)
    train = df.filter(pl.col("partition") == "train")
    test = df.filter(pl.col("partition") == "test")
    print(f"Train: {train.shape[0]}  Test: {test.shape[0]}")

    X_train = featurise_ligands(train["smiles"].to_list())
    X_test = featurise_ligands(test["smiles"].to_list())
    y_train = train["label"].to_numpy().astype(np.int64)
    y_test = test["label"].to_numpy().astype(np.int64)

    clf = LogisticRegression(max_iter=200, n_jobs=-1, random_state=args.seed, solver="liblinear")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, proba)
    aupr = average_precision_score(y_test, proba)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    row = pl.DataFrame({
        "corpus": [args.corpus],
        "regime": [args.regime],
        "n_train": [train.shape[0]],
        "n_test": [test.shape[0]],
        "n_pos_test": [int(y_test.sum())],
        "n_neg_test": [int((1 - y_test).sum())],
        "baseline_auroc_lr": [auroc],
        "baseline_auprc_lr": [aupr],
    })
    row.write_csv(args.out)
    print(f"AUROC = {auroc:.4f}, AUPR = {aupr:.4f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
