"""Bemis-Murcko scaffold splitter (naive grouping).

Mode A: per target, group ligands by scaffold_smiles; assign whole groups to
folds by greedy fill toward (0.8, 0.1, 0.1).

Mode B: same procedure but across the whole pooled manifest.
"""
from __future__ import annotations
import polars as pl

from .common import (
    parse_common_args, load_manifest, materialise_per_target,
    write_split, fold_quotas,
)
from .schemas import hash_manifest_slice


def split_slice_by_groupkey(slice_df: pl.DataFrame, group_col: str) -> list[dict]:
    h = hash_manifest_slice(slice_df)
    # Group sizes desc; greedy bin-pack toward 80/10/10.
    sizes = (slice_df.group_by(group_col)
                     .agg(pl.len().alias("n"))
                     .sort("n", descending=True))
    n_tr, n_va, n_te = fold_quotas(slice_df.height)
    quota = {"train": n_tr, "val": n_va, "test": n_te}
    have  = {"train": 0,   "val": 0,    "test": 0}
    g_to_fold: dict[str, str] = {}
    for row in sizes.iter_rows(named=True):
        gk, n = row[group_col], row["n"]
        # Assign to the fold furthest below its quota.
        deficits = {f: quota[f] - have[f] for f in quota}
        choice = max(deficits, key=deficits.get)
        g_to_fold[gk] = choice
        have[choice] += n

    rows = []
    for r in slice_df.iter_rows(named=True):
        rows.append({
            "example_id": r["example_id"], "target_id": r["target_id"],
            "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
            "fold":       g_to_fold[r[group_col]],
            "input_hash": h,
        })
    return rows


def main() -> int:
    args = parse_common_args(__doc__).parse_args()
    manifest = load_manifest(args.manifest).with_columns(
        pl.col("scaffold_smiles").fill_null("")
    )

    if args.mode == "A":
        rows: list[dict] = []
        per_target = materialise_per_target(manifest, args.subset_dir)
        for tid in sorted(per_target):
            rows.extend(split_slice_by_groupkey(per_target[tid], "scaffold_smiles"))
        write_split(rows, args.out, input_hash="mode_A_per_target")
    else:
        rows = split_slice_by_groupkey(manifest, "scaffold_smiles")
        write_split(rows, args.out, input_hash=hash_manifest_slice(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
