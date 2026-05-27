"""Build outputs/reports/data/table_split_sensitivity.csv from existing artefacts.

Sources (already in repo / outputs/v2):
  - outputs/v2/phase1/phase1_combined.csv     -> morgan_rf-style Phase 1 baselines
  - outputs/reports/data/table_cnn_baseline.csv -> C-NN (label-copying) AUROC

Output columns: corpus,regime,model,auroc,n_pos_test,n_neg_test,source
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def read_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out",  required=True, type=Path)
    args = ap.parse_args()

    rows: list[dict] = []

    # 1) Morgan baseline rows from phase1_combined.csv
    p1 = args.repo / "outputs/v2/phase1/phase1_combined.csv"
    if p1.exists():
        for r in read_csv(p1):
            if r.get("feasible", "").lower() != "true":
                continue
            auroc = r.get("baseline_auroc", "")
            if not auroc or auroc.lower() in {"nan", ""}:
                continue
            rows.append({
                "corpus": r["corpus"],
                "regime": r["regime"],
                "model": "morgan_rf",
                "auroc": auroc,
                "n_pos_test": r.get("n_pos_test", ""),
                "n_neg_test": r.get("n_neg_test", ""),
                "source": "outputs/v2/phase1/phase1_combined.csv",
            })

    # 2) C-NN label-copying rows from table_cnn_baseline.csv (variant=all_axis)
    cnn = args.repo / "outputs/reports/data/table_cnn_baseline.csv"
    if cnn.exists():
        for r in read_csv(cnn):
            if r.get("variant") != "all_axis":
                continue
            try:
                auroc = float(r["auroc"])
            except (ValueError, KeyError):
                continue
            rows.append({
                "corpus": r["corpus"],
                "regime": r["regime"],
                "model": "cnn",
                "auroc": f"{auroc:.6f}",
                "n_pos_test": "",
                "n_neg_test": "",
                "source": "outputs/reports/data/table_cnn_baseline.csv",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["corpus", "regime", "model", "auroc", "n_pos_test", "n_neg_test", "source"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out}  rows={len(rows)}")
    n_morgan = sum(1 for r in rows if r["model"] == "morgan_rf")
    n_cnn = sum(1 for r in rows if r["model"] == "cnn")
    print(f"  morgan_rf={n_morgan}  cnn={n_cnn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
