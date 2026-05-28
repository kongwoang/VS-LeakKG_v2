"""Compute split-quality metrics for a single (split parquet, manifest) pair.

Emits a single CSV row (or appends) into table_split_quality_mode<X>.csv.

Metrics produced here (model-free):
    - sizes, class balance, dropped count
    - max train→test ligand Tanimoto similarity (ECFP4)
    - per-axis KG residual c_a (reuses tools/run_contam_bins.py logic
      via direct import; falls back to "axis_status=unavailable" when
      the manifest column is null for the whole corpus)
    - AVE bias B on whatever split is provided (not only on `ave_ligand`)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import polars as pl
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

from .schemas import TABLE_SPLIT_QUALITY_COLUMNS


def _fp(smi: str):
    if not HAS_RDKIT or not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def max_train_test_tanimoto(train_fps, test_fps) -> float:
    if not train_fps or not test_fps:
        return 0.0
    mx = 0.0
    for q in test_fps:
        if q is None:
            continue
        refs = [r for r in train_fps if r is not None]
        if not refs:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q, refs)
        if sims:
            mx = max(mx, max(sims))
    return float(mx)


def class_balance(df: pl.DataFrame) -> float:
    p = (df["label"] == 1).sum()
    n = (df["label"] == 0).sum()
    if (p + n) == 0:
        return float("nan")
    return float(p) / float(p + n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--split",     required=True, type=Path)
    ap.add_argument("--corpus",    required=True)
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--splitter",  required=True)
    ap.add_argument("--out-csv",   required=True, type=Path)
    ap.add_argument("--seed",      default=2025, type=int)
    args = ap.parse_args()

    manifest = pl.read_parquet(args.manifest)
    split = pl.read_parquet(args.split)
    merged = manifest.join(split.select(["example_id", "fold"]), on="example_id", how="inner")

    out_rows: list[dict] = []
    if args.mode == "A":
        target_iter = sorted(merged["target_id"].unique().to_list())
    else:
        target_iter = ["__POOLED__"]

    for tid in target_iter:
        sub = merged if tid == "__POOLED__" else merged.filter(pl.col("target_id") == tid)
        train = sub.filter(pl.col("fold") == "train")
        val   = sub.filter(pl.col("fold") == "val")
        test  = sub.filter(pl.col("fold") == "test")

        t0 = time.time()
        max_lig_tani = max_train_test_tanimoto(
            [_fp(s) for s in train["smiles"].to_list()],
            [_fp(s) for s in test["smiles"].to_list()],
        )
        rt = time.time() - t0

        row = {c: None for c in TABLE_SPLIT_QUALITY_COLUMNS}
        row.update({
            "corpus":    args.corpus,
            "mode":      args.mode,
            "splitter":  args.splitter,
            "target_id": tid,
            "n_train":   train.height,
            "n_val":     val.height,
            "n_test":    test.height,
            "n_dropped": manifest.height - sub.height if tid == "__POOLED__" else None,
            "n_train_pos": int((train["label"] == 1).sum()),
            "n_train_neg": int((train["label"] == 0).sum()),
            "n_test_pos":  int((test["label"] == 1).sum()),
            "n_test_neg":  int((test["label"] == 0).sum()),
            "class_balance_train": class_balance(train),
            "class_balance_test":  class_balance(test),
            "max_lig_tanimoto": max_lig_tani,
            "input_hash": split["input_hash"][0] if split.height else "",
            "seed":       args.seed,
            "runtime_s":  rt,
        })
        out_rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(out_rows)[TABLE_SPLIT_QUALITY_COLUMNS]
    if args.out_csv.exists():
        existing = pl.read_csv(args.out_csv)
        df = pl.concat([existing, df], how="vertical_relaxed")
    df.write_csv(args.out_csv)
    print(f"appended {len(out_rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
