"""Build a unified corpus manifest from the existing v2 KG nodes.

The v2 KG already contains Example nodes whose `props` JSON carries label
and target; Ligand nodes carry SMILES and scaffold; protein metadata can
be joined from KG edges. This script reduces those nodes to the
benchmark's canonical CORPUS_MANIFEST_SCHEMA so every splitter sees the
same input.

Three corpora supported: dekois, dude, litpcba. The PDBBind appendix is
out of scope.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import polars as pl

from .schemas import CORPUS_MANIFEST_SCHEMA


CORPUS_TO_GRAPH = {
    "dekois":  "graph_dekois/v2_nodes.parquet",
    "dude":    "graph_dude/v2_nodes.parquet",
    "litpcba": "graph_litpcba_ave/v2_nodes.parquet",
}


def parse_props(props_str: str) -> dict:
    if not props_str:
        return {}
    try:
        return json.loads(props_str)
    except Exception:
        return {}


def build_manifest(v2_root: Path, corpus: str) -> pl.DataFrame:
    nodes = pl.read_parquet(v2_root / CORPUS_TO_GRAPH[corpus])

    # Examples carry label + target + ligand id; we want a row per Example.
    ex = nodes.filter(pl.col("node_type") == "Example")
    print(f"[{corpus}] Examples: {ex.height}")

    # Expand props json
    parsed = [parse_props(p) for p in ex["props"].to_list()]
    target_ids = []
    labels = []
    for p in parsed:
        target_ids.append(p.get("target", ""))
        # label may be int or str
        lab = p.get("label", 0)
        try:
            labels.append(int(lab))
        except Exception:
            labels.append(0)

    # Look up the ligand_id and smiles by Example -> Ligand edges.
    # We rely on the convention that example_id encodes target in the id string,
    # but the canonical mapping is in props["ligand_id"] when present, otherwise
    # we'll need to load the edges. Inspect first to know.
    ligand_ids = []
    smiles_list = []
    scaffolds = []
    for p in parsed:
        ligand_ids.append(p.get("ligand_id", ""))
        smiles_list.append(p.get("smiles", ""))
        scaffolds.append(p.get("scaffold_smiles", None))

    df = pl.DataFrame({
        "example_id":      ex["node_id"].to_list(),
        "target_id":       target_ids,
        "ligand_id":       ligand_ids,
        "smiles":          smiles_list,
        "label":           labels,
        "scaffold_smiles": scaffolds,
    })

    # Fill the remaining columns with nulls (corpus-dependent enrichment lives in
    # later passes; we keep them nullable up front).
    for col, dtype in [
        ("uniprot",        pl.Utf8),
        ("protein_family", pl.Utf8),
        ("pdb_id",         pl.Utf8),
        ("assay_id",       pl.Utf8),
        ("source",         pl.Utf8),
        ("timestamp_year", pl.Int64),
    ]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Coerce label to int.
    df = df.with_columns(pl.col("label").cast(pl.Int64))
    df = df.select(list(CORPUS_MANIFEST_SCHEMA.keys()))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-root", required=True, type=Path,
                    help="path to outputs/v2/")
    ap.add_argument("--corpus",  required=True, choices=["dekois", "dude", "litpcba"])
    ap.add_argument("--out",     required=True, type=Path)
    args = ap.parse_args()

    df = build_manifest(args.v2_root, args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"wrote {args.out}  rows={df.height}  targets={df['target_id'].n_unique()}  "
          f"pos={(df['label'] == 1).sum()}  neg={(df['label'] == 0).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
