"""Emit SPRINT-shaped CSVs from a v2 split + protein-sequence lookup.

SPRINT's `--task custom` path expects:

    data/<task>/train.csv
    data/<task>/val.csv
    data/<task>/test.csv

with columns `SMILES, Target Sequence, Label` (the columns shipped by
DAVIS / BIOSNAP / etc. are exactly this set, optionally plus
`drug_encoding, target_encoding, uniprot_id`).

This script reads:
- v2 split parquet: (example_id, partition, ligand_id, protein_id, smiles, label)
- protein_seq_lookup parquet from `tools/build_protein_seq_lookup.py`:
  (target_id, target_sequence, source)

and emits one CSV per partition, dropping rows where SMILES or
Target Sequence are missing.

CAVEAT: protein_id in our v2 split is the v1 Protein node_id format
(`Protein::<seq_hash>` for PDBBind), which equals `seq_sha256` from
the proteins parquet. But the lookup table is keyed by `pdb_id`
because that's the v1 join key visible in the rest of the graph.
This tool exposes a `--protein-key` flag so the user can pick which
join to use.

Usage:

    python tools/v2_to_sprint_csv.py \
        --split outputs/v2/phase1_full/splits/pdbbind/protein.parquet \
        --protein-seq outputs/v2/graph/protein_seq_lookup.parquet \
        --sprint-data-dir ~/SPRINT/data/custom_pdbbind_protein \
        [--protein-key pdb_id|seq_sha256]

The script is fair-comparison-friendly: SPRINT can then be run with
`ultrafast-train --task custom --config configs/saprot_agg_config.yaml`
which uses the published hyperparameters verbatim.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def split_to_sprint_csvs(
    split: pl.DataFrame,
    protein_seq: pl.DataFrame,
    output_dir: Path,
    *,
    protein_key: str = "pdb_id",
) -> dict[str, int]:
    """Write train.csv/val.csv/test.csv under output_dir.

    The split frame must have `example_id`, `partition`, `protein_id`,
    `smiles`, `label`. We resolve `Target Sequence` via a left-join on
    `protein_seq` keyed by `protein_key`.

    Rows missing either Drug or Target are dropped (SPRINT can't consume
    them) and counted in the return dict under `dropped_<partition>`.
    """
    needed = {"example_id", "partition", "protein_id", "smiles", "label"}
    missing = needed - set(split.columns)
    if missing:
        raise ValueError(f"split missing required columns: {sorted(missing)}")
    if "target_id" not in protein_seq.columns or "target_sequence" not in protein_seq.columns:
        raise ValueError(
            f"protein_seq missing columns; have {protein_seq.columns}, "
            f"need at least (target_id, target_sequence)"
        )

    # The split's protein_id may be in any of these formats observed
    # across v1 corpora:
    #   "Protein::5za2"                  (synthetic test data)
    #   "prot:PDBBind:220c836fee080836"  (real v1 PDBBind: 16-char seq hash)
    #   "tgt:DEKOIS:pim-2"               (DEKOIS - target name, no sequence)
    # Take the rightmost colon-delimited piece. That's the part that
    # matches our protein_seq_lookup target_id (which carries both
    # pdb_id rows and seq_sha256-prefix rows).
    if protein_key in ("pdb_id", "seq_sha256"):
        split = split.with_columns(
            pl.col("protein_id")
            .str.split(":").list.last()
            .alias("target_lookup_key")
        )
    else:
        raise ValueError(f"unknown protein_key={protein_key}")

    joined = split.join(
        protein_seq.rename({"target_id": "target_lookup_key",
                            "target_sequence": "Target Sequence"}),
        on="target_lookup_key",
        how="left",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for part in ("train", "val", "test"):
        sub = joined.filter(pl.col("partition") == part)
        before = sub.height
        sub = sub.filter(
            pl.col("smiles").is_not_null()
            & pl.col("Target Sequence").is_not_null()
        )
        # SPRINT's DTIDataModule reads CSVs with pandas's default
        # index_col semantics, which assumes the first unnamed column
        # is the index. DAVIS/BIOSNAP/BindingDB CSVs all ship with an
        # explicit `,SMILES,Target Sequence,Label,...` header (note
        # leading comma). We must match that to avoid SMILES being
        # consumed as the dataframe index, which would manifest as
        # `KeyError: 'SMILES'` downstream.
        rows = sub.with_row_index(name="").select(
            pl.col(""),
            pl.col("smiles").alias("SMILES"),
            pl.col("Target Sequence"),
            pl.col("label").alias("Label"),
        )
        out_csv = output_dir / f"{part}.csv"
        rows.write_csv(out_csv)
        counts[part] = rows.height
        counts[f"dropped_{part}"] = before - rows.height
        log.info("partition=%s rows_in=%d rows_out=%d -> %s",
                 part, before, rows.height, out_csv)
    return counts


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True,
                   help="v2 split parquet (outputs/v2/phase1_*/splits/<corpus>/<regime>.parquet)")
    p.add_argument("--protein-seq", type=Path, required=True,
                   help="parquet from tools/build_protein_seq_lookup.py")
    p.add_argument("--sprint-data-dir", type=Path, required=True)
    p.add_argument("--protein-key", default="pdb_id",
                   choices=["pdb_id", "seq_sha256"])
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s")
    split = pl.read_parquet(args.split)
    protein_seq = pl.read_parquet(args.protein_seq)
    counts = split_to_sprint_csvs(
        split=split,
        protein_seq=protein_seq,
        output_dir=args.sprint_data_dir,
        protein_key=args.protein_key,
    )
    print(f"v2_to_sprint_csv: {counts}")


if __name__ == "__main__":
    _cli()
