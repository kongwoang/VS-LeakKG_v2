"""DrugOOD-style domain splitter.

Implements the explicit procedure from the protocol (Section 4):
    1. compute per-domain frequency table on the chosen axis
    2. sort domains by frequency descending
    3. greedy bin-packing — walk down the sorted list and assign each whole
       domain to the fold whose current cumulative count is furthest below
       its 80/10/10 target. Domains are never split.
    4. class balance is not enforced inside the loop; it is reported afterwards.

Supported axes: scaffold, size, protein, protein_family.
The `assay` axis is **not** offered (unavailable on DUD-E/DEKOIS, degenerate
with target on LIT-PCBA).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl

from .common import write_split, fold_quotas
from .schemas import hash_manifest_slice


def _size_bucket(smiles: str) -> str:
    n = len(smiles)
    if n <= 20:  return "tiny"
    if n <= 35:  return "small"
    if n <= 50:  return "medium"
    if n <= 75:  return "large"
    return "very_large"


def domain_column(df: pl.DataFrame, axis: str) -> pl.Series:
    if axis == "scaffold":
        return df["scaffold_smiles"].fill_null("")
    if axis == "size":
        return pl.Series("dom", [_size_bucket(s or "") for s in df["smiles"].to_list()])
    if axis == "protein":
        return df["target_id"]
    if axis == "protein_family":
        return df["protein_family"].fill_null("UNK")
    raise ValueError(f"unsupported axis {axis}")


def domain_split(df: pl.DataFrame, axis: str) -> list[dict]:
    h = hash_manifest_slice(df)
    df = df.with_columns(domain_column(df, axis).alias("_domain"))
    sizes = (df.group_by("_domain")
               .agg(pl.len().alias("n"))
               .sort("n", descending=True))
    n_tr, n_va, n_te = fold_quotas(df.height)
    quota = {"train": n_tr, "val": n_va, "test": n_te}
    have  = {"train": 0,   "val": 0,    "test": 0}
    g_to_fold: dict[str, str] = {}
    for row in sizes.iter_rows(named=True):
        dom, n = row["_domain"], row["n"]
        deficits = {f: quota[f] - have[f] for f in quota}
        choice = max(deficits, key=deficits.get)
        g_to_fold[dom] = choice
        have[choice] += n

    rows = []
    for r in df.iter_rows(named=True):
        rows.append({
            "example_id": r["example_id"], "target_id": r["target_id"],
            "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
            "fold":       g_to_fold[r["_domain"]],
            "input_hash": h,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--subset-dir", required=False, type=Path, default=None)
    ap.add_argument("--axis",      required=True,
                    choices=["scaffold", "size", "protein", "protein_family"])
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--seed",      default=2025, type=int)
    args = ap.parse_args()

    manifest = pl.read_parquet(args.manifest)
    # Mode B is the natural mode for DrugOOD; Mode A is allowed only for
    # ligand-side axes (scaffold, size).
    if args.mode == "A" and args.axis in {"protein", "protein_family"}:
        raise SystemExit(f"DrugOOD {args.axis} is undefined per-target; use --mode B.")

    if args.mode == "A":
        rows: list[dict] = []
        for tid in sorted(manifest["target_id"].unique().to_list()):
            slc = manifest.filter(pl.col("target_id") == tid)
            if args.subset_dir is not None:
                sub = args.subset_dir / f"subset_{tid}.parquet"
                if sub.exists():
                    slc = pl.read_parquet(sub)
            rows.extend(domain_split(slc, args.axis))
        write_split(rows, args.out, input_hash="mode_A_per_target")
    else:
        rows = domain_split(manifest, args.axis)
        write_split(rows, args.out, input_hash=hash_manifest_slice(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
