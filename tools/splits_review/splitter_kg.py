"""KG multi-axis splitter (passthrough on existing Phase 1 v2 artefacts).

The KG splits already exist at:
    outputs/v2/phase1/splits/<corpus>/<axis>.parquet
with the schema:
    example_id, partition, ligand_id, protein_id, smiles, label
where partition ∈ {train, val, test} after a remap (paper labels them
"train"/"valid"/"test" or "train"/"test").

This tool reads the requested axis-split, joins onto the manifest's
(example_id) so we keep the same provenance columns, and emits a parquet
matching SPLIT_SCHEMA. The supported axes are:
    --axis {ligand, scaffold, protein, dual}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl

from .common import write_split
from .schemas import hash_manifest_slice


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--kg-splits", required=True, type=Path,
                    help="outputs/v2/phase1/splits/<corpus>/")
    ap.add_argument("--axis",      required=True,
                    choices=["ligand", "scaffold", "protein", "dual"])
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--seed",      default=2025, type=int)
    ap.add_argument("--subset-dir", required=False, type=Path, default=None)
    args = ap.parse_args()

    manifest = pl.read_parquet(args.manifest)
    kg = pl.read_parquet(args.kg_splits / f"{args.axis}.parquet")
    # Remap "valid" -> "val" if needed.
    kg = kg.with_columns(
        pl.when(pl.col("partition") == "valid").then(pl.lit("val"))
          .otherwise(pl.col("partition")).alias("fold")
    )

    join_cols = ["example_id"]
    merged = manifest.join(kg.select(["example_id", "fold"]),
                            on=join_cols, how="inner")
    rows = [
        {
            "example_id": r["example_id"], "target_id": r["target_id"],
            "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
            "fold":       r["fold"],       "input_hash": "kg_phase1_passthrough",
        }
        for r in merged.iter_rows(named=True)
    ]
    write_split(rows, args.out, input_hash=hash_manifest_slice(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
