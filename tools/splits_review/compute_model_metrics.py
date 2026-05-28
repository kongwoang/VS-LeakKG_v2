"""Train Morgan-RF / 1-NN-ligand / KG-CNN per split and emit metrics.

For each (split × target) train a Morgan-RF on the train fold (ECFP4, 2048
bits, sklearn RandomForestClassifier n_estimators=500, class_weight="balanced"),
predict on the test fold, compute AUROC / EF1% / BEDROC. Same for 1-NN
ligand (KNN k=1 jaccard on ECFP). KG-CNN is left as a hook to the existing
tools/run_cnn_baseline.py output.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

from .schemas import TABLE_MODELMETRICS_COLUMNS


def fp_array(smiles_list: list[str]) -> np.ndarray:
    out = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        for b in bv.GetOnBits():
            out[i, b] = 1
    return out


def ef_at_x(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.01) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    order = np.argsort(-y_score)
    topk = y_true[order[:k]]
    hits_topk = topk.sum()
    rate = hits_topk / k
    overall = y_true.sum() / n
    if overall == 0:
        return float("nan")
    return float(rate / overall)


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    from math import exp
    n = len(y_true)
    if n == 0 or y_true.sum() == 0 or y_true.sum() == n:
        return float("nan")
    order = np.argsort(-y_score)
    ranks = np.where(y_true[order] == 1)[0] + 1
    R = y_true.sum() / n
    s = sum(exp(-alpha * r / n) for r in ranks)
    z = (R * (1 - exp(-alpha)) / (exp(alpha / n) - 1))
    if z == 0:
        return float("nan")
    factor = R * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * R))
    return float(s / z * factor)


def metrics_one(train: pl.DataFrame, test: pl.DataFrame, model_name: str,
                seed: int) -> dict:
    from sklearn.metrics import roc_auc_score
    Xtr = fp_array(train["smiles"].to_list())
    Xte = fp_array(test["smiles"].to_list())
    ytr = train["label"].to_numpy()
    yte = test["label"].to_numpy()
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return {"auroc": float("nan"), "ef1pct": float("nan"), "bedroc": float("nan")}
    if model_name == "morgan_rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        clf.fit(Xtr, ytr)
        ys = clf.predict_proba(Xte)[:, 1]
    elif model_name == "knn1_ligand":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=1, metric="jaccard", algorithm="brute")
        clf.fit(Xtr, ytr)
        ys = clf.predict_proba(Xte)[:, 1]
    else:
        raise ValueError(f"unknown model {model_name}")
    return {
        "auroc":  float(roc_auc_score(yte, ys)),
        "ef1pct": ef_at_x(yte, ys, 0.01),
        "bedroc": bedroc(yte, ys, 20.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--split",     required=True, type=Path)
    ap.add_argument("--corpus",    required=True)
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--splitter",  required=True)
    ap.add_argument("--models",    default="morgan_rf,knn1_ligand")
    ap.add_argument("--out-csv",   required=True, type=Path)
    ap.add_argument("--seed",      default=2025, type=int)
    args = ap.parse_args()

    manifest = pl.read_parquet(args.manifest)
    split = pl.read_parquet(args.split)
    merged = manifest.join(split.select(["example_id", "fold"]), on="example_id", how="inner")

    out_rows: list[dict] = []
    target_iter = sorted(merged["target_id"].unique().to_list()) if args.mode == "A" else ["__POOLED__"]
    for tid in target_iter:
        sub = merged if tid == "__POOLED__" else merged.filter(pl.col("target_id") == tid)
        train = sub.filter(pl.col("fold") == "train")
        test  = sub.filter(pl.col("fold") == "test")
        if train.height == 0 or test.height == 0:
            continue
        for model_name in args.models.split(","):
            m = metrics_one(train, test, model_name.strip(), args.seed)
            row = {c: None for c in TABLE_MODELMETRICS_COLUMNS}
            row.update({
                "corpus":    args.corpus,
                "mode":      args.mode,
                "splitter":  args.splitter,
                "target_id": tid,
                "model":     model_name.strip(),
                "auroc":     m["auroc"],
                "ef1pct":    m["ef1pct"],
                "bedroc":    m["bedroc"],
                "n_test_pos": int((test["label"] == 1).sum()),
                "n_test_neg": int((test["label"] == 0).sum()),
                "input_hash": split["input_hash"][0] if split.height else "",
                "seed":       args.seed,
            })
            out_rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(out_rows)[TABLE_MODELMETRICS_COLUMNS]
    if args.out_csv.exists():
        existing = pl.read_csv(args.out_csv)
        df = pl.concat([existing, df], how="vertical_relaxed")
    df.write_csv(args.out_csv)
    print(f"appended {len(out_rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
