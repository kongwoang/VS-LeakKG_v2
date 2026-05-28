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
    "dekois":  "graph_dekois",
    "dude":    "graph_dude",
    "litpcba": "graph_litpcba_ave",
}


def parse_props(props_str: str) -> dict:
    if not props_str:
        return {}
    try:
        return json.loads(props_str)
    except Exception:
        return {}


def build_manifest(v2_root: Path, corpus: str) -> pl.DataFrame:
    graph_dir = v2_root / CORPUS_TO_GRAPH[corpus]
    nodes = pl.read_parquet(graph_dir / "v2_nodes.parquet")
    edges = pl.read_parquet(graph_dir / "v2_edges.parquet")

    # Examples carry label + target.
    ex = nodes.filter(pl.col("node_type") == "Example")
    print(f"[{corpus}] Examples: {ex.height}")

    # Parse Example props for label + target.
    ex_props = [parse_props(p) for p in ex["props"].to_list()]
    ex_df = pl.DataFrame({
        "example_id": ex["node_id"].to_list(),
        "target_id":  [p.get("target", "") for p in ex_props],
        "label":      [int(p.get("label", 0) or 0) for p in ex_props],
    })

    # Edge Example -> Ligand (edge_type='example_has_ligand'). Ligand node carries
    # the canonical SMILES (in `label`) and scaffold_smiles (in `props`).
    e_lig = edges.filter(pl.col("edge_type") == "example_has_ligand")\
                 .select(pl.col("src").alias("example_id"),
                         pl.col("dst").alias("ligand_id"))
    lig_nodes = nodes.filter(pl.col("node_type") == "Ligand")
    lig_props = [parse_props(p) for p in lig_nodes["props"].to_list()]
    lig_df = pl.DataFrame({
        "ligand_id":       lig_nodes["node_id"].to_list(),
        "smiles":          lig_nodes["label"].to_list(),
        "scaffold_smiles": [p.get("scaffold_smiles") for p in lig_props],
    })

    # Join Example -> Ligand -> SMILES/scaffold.
    df = ex_df.join(e_lig, on="example_id", how="left")\
              .join(lig_df, on="ligand_id", how="left")

    # Fill the remaining manifest columns with nulls; corpus-specific enrichment
    # (uniprot, family, pdb_id, assay_id, source, year) lands in a follow-up
    # `enrich_manifest.py` pass that joins on protein_meta.parquet.

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
    ap.add_argument("--protein-meta", required=False, type=Path, default=None,
                    help="protein_meta.parquet from fetch_sequences.py "
                         "(adds uniprot, family).")
    ap.add_argument("--out",     required=True, type=Path)
    args = ap.parse_args()

    df = build_manifest(args.v2_root, args.corpus)
    if args.protein_meta is not None and args.protein_meta.exists():
        meta = pl.read_parquet(args.protein_meta).select(["target_id", "uniprot", "family"])\
                  .rename({"family": "protein_family"})
        df = df.drop("uniprot", "protein_family").join(meta, on="target_id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"wrote {args.out}  rows={df.height}  targets={df['target_id'].n_unique()}  "
          f"pos={(df['label'] == 1).sum()}  neg={(df['label'] == 0).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
