"""Emit SPRINT-shaped CSVs from a v2 split + hydrate side-table.

SPRINT (`https://github.com/ml-jku/sprint`) consumes
`data/<task>/{train,val,test}.csv` with columns:
    Drug,Target,Y                  (the AA-sequence variant)
    Drug,Target,Y                  (the SaProt variant; columns identical
                                    but `Target` is the 3Di-interleaved seq
                                    and the file is named `*_foldseek.csv`).

We map:
    Drug    = side_table.smiles_canonical (fallback: smiles)
    Target  = side_table.target_sequence   (AA)
            or side_table.target_sequence_saprot (foldseek variant)
    Y       = side_table.label             (binary or pAct)

Usage:
    python tools/sprint_csv_from_v2.py \
        --split outputs/v2/phase1/splits/litpcba/strict.parquet \
        --side-table outputs/v2/graph/side_table.parquet \
        --sprint-data-dir ~/SPRINT/data/custom \
        [--foldseek]      # emit *_foldseek.csv (SaProt path)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

# Allow running this script from a checkout without installing the package.
import sys as _sys
_PKG_ROOT = Path(__file__).resolve().parent.parent / "src"
if _PKG_ROOT.exists() and str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))

from vsleakkg.v2.hydrate import Hydrator

log = logging.getLogger(__name__)


def split_to_sprint_csvs(
    split_parquet: Path,
    hydrator: Hydrator,
    output_dir: Path,
    *,
    foldseek: bool = False,
) -> dict[str, int]:
    """Write {train,val,test}{,_foldseek}.csv into output_dir.

    Returns a dict of {partition: rows_written}.
    """
    df = pl.read_parquet(split_parquet)
    if "example_id" not in df.columns or "partition" not in df.columns:
        raise ValueError(
            f"split parquet missing required columns; got {df.columns}"
        )
    counts: dict[str, int] = {}
    suffix = "_foldseek" if foldseek else ""
    target_col = "target_sequence_saprot" if foldseek else "target_sequence"
    output_dir.mkdir(parents=True, exist_ok=True)

    for part in ("train", "val", "test"):
        ex_ids = df.filter(pl.col("partition") == part)["example_id"].to_list()
        if not ex_ids:
            log.warning("partition=%s is empty", part)
            counts[part] = 0
            continue
        hyd = hydrator.hydrate(ex_ids)
        rows = hyd.rows.with_columns(
            pl.coalesce([pl.col("smiles_canonical"), pl.col("smiles")]).alias("Drug"),
            pl.col(target_col).alias("Target"),
            pl.col("label").alias("Y"),
        ).select(["Drug", "Target", "Y"])
        # Drop rows where Drug or Target is null - SPRINT can't consume them
        rows = rows.filter(pl.col("Drug").is_not_null() & pl.col("Target").is_not_null())
        out = output_dir / f"{part}{suffix}.csv"
        rows.write_csv(out)
        counts[part] = rows.height
        log.info(
            "partition=%s rows_in=%d missing_in_side_table=%d rows_out=%d -> %s",
            part, len(ex_ids), len(hyd.missing), rows.height, out,
        )
    return counts


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--side-table", type=Path, required=True)
    p.add_argument("--sprint-data-dir", type=Path, required=True,
                   help="Path to <SPRINT_ROOT>/data/<task_name>/")
    p.add_argument("--foldseek", action="store_true",
                   help="Emit *_foldseek.csv with SaProt 3Di-interleaved sequences")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    hydrator = Hydrator.from_parquet(args.side_table)
    counts = split_to_sprint_csvs(
        split_parquet=args.split,
        hydrator=hydrator,
        output_dir=args.sprint_data_dir,
        foldseek=args.foldseek,
    )
    print(f"counts: {counts}")


if __name__ == "__main__":
    _cli()
