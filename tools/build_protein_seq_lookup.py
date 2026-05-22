"""Build a protein-sequence lookup parquet from v1's per-corpus sources.

Currently supports PDBBind (v1 pdbbind_proteins.parquet has the AA
sequences keyed by `pdb_ids` list, plus `seq_sha256` for unique
sequence identification).

For DEKOIS/DUD-E we'd need a UniProt-name -> sequence mapping which v1
doesn't directly ship; that's a future enhancement. For LIT-PCBA-AVE
we'd need ChEMBL targets join.

Output schema (matches the side-table's `target_sequence` field):

    target_id, target_sequence, source

Usage:

    python tools/build_protein_seq_lookup.py \
        --v1-processed <v1_repo>/data/processed \
        --output outputs/v2/graph/protein_seq_lookup.parquet
        [--sources pdbbind]

Result can be joined into v2_nodes (where Protein node_id == "Protein::<seq_sha256>" or "Protein::<pdb_id>")
or piped into a future v2_to_sprint_csv.py tool.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def _load_pdbbind(processed: Path) -> pl.DataFrame:
    """Return (target_id, target_sequence, source='pdbbind').

    Emits one row per PDB id AND one row per truncated 16-char seq_sha256
    prefix. The latter is the format used by the v1 graph builder's
    Protein node_ids (`prot:PDBBind:<16-hex>`), so the side-table can
    join either way.
    """
    f = processed / "pdbbind_proteins.parquet"
    if not f.exists():
        log.warning("pdbbind_proteins.parquet missing")
        return pl.DataFrame()
    df = pl.read_parquet(f)

    # (a) pdb_id-keyed view: 19037 rows
    by_pdb = df.select(
        pl.col("pdb_ids").alias("pdb_id"),
        pl.col("sequence_concat").alias("target_sequence"),
    ).explode("pdb_id").rename({"pdb_id": "target_id"})
    by_pdb = by_pdb.with_columns(pl.lit("pdbbind").alias("source"))

    # (b) seq_sha256-prefix-keyed view: ~thousands of unique sequences
    by_seq = df.select(
        pl.col("seq_sha256").str.slice(0, 16).alias("target_id"),
        pl.col("sequence_concat").alias("target_sequence"),
    )
    by_seq = by_seq.with_columns(pl.lit("pdbbind").alias("source"))

    combined = pl.concat([by_pdb, by_seq], how="vertical_relaxed").unique(
        subset=["target_id"], keep="first"
    )
    return combined.select(["target_id", "target_sequence", "source"])


SOURCE_LOADERS = {
    "pdbbind": _load_pdbbind,
}


def build_lookup(processed: Path, sources: list[str], output: Path) -> dict[str, int]:
    """Build the lookup parquet. Returns per-source row counts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    counts: dict[str, int] = {}
    for src in sources:
        loader = SOURCE_LOADERS.get(src)
        if not loader:
            log.warning("unknown source %s", src)
            counts[src] = 0
            continue
        df = loader(processed)
        counts[src] = df.height
        if df.height:
            frames.append(df)
            log.info("source=%s rows=%d", src, df.height)
    if frames:
        merged = pl.concat(frames, how="vertical_relaxed").unique(subset=["target_id"], keep="first")
    else:
        merged = pl.DataFrame({"target_id": [], "target_sequence": [], "source": []})
    merged.write_parquet(output)
    counts["total"] = merged.height
    return counts


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1-processed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sources", default="pdbbind",
                   help="comma-separated source list (default: pdbbind)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s")
    counts = build_lookup(
        processed=args.v1_processed,
        sources=[s.strip() for s in args.sources.split(",") if s.strip()],
        output=args.output,
    )
    print(f"protein_seq_lookup: {counts} -> {args.output}")


if __name__ == "__main__":
    _cli()
