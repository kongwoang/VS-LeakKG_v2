"""Random per-target splitter (Mode A) or random pooled splitter (Mode B).

Stratifies by label so that train/val/test preserve the class ratio.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from .common import (
    parse_common_args, load_manifest, materialise_per_target,
    write_split, fold_quotas,
)
from .schemas import hash_manifest_slice


def split_one_slice(slice_df: pl.DataFrame, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    h = hash_manifest_slice(slice_df)
    for lab in (1, 0):
        part = slice_df.filter(pl.col("label") == lab)
        idx = np.arange(part.height)
        rng.shuffle(idx)
        n_tr, n_va, _ = fold_quotas(part.height)
        for k, i in enumerate(idx):
            fold = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
            r = part.row(int(i), named=True)
            rows.append({
                "example_id": r["example_id"], "target_id": r["target_id"],
                "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
                "fold": fold, "input_hash": h,
            })
    return rows


def main() -> int:
    args = parse_common_args(__doc__).parse_args()
    manifest = load_manifest(args.manifest)
    rng = np.random.default_rng(args.seed)

    if args.mode == "A":
        rows: list[dict] = []
        per_target = materialise_per_target(manifest, args.subset_dir)
        for tid in sorted(per_target):
            rows.extend(split_one_slice(per_target[tid], rng))
        # input_hash for the whole file is the concat of slice hashes; each row carries the per-slice hash.
        write_split(rows, args.out, input_hash="mode_A_per_target")
    else:
        rows = split_one_slice(manifest, rng)
        write_split(rows, args.out, input_hash=hash_manifest_slice(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
