"""Build target-level KG view for LIT-PCBA (15 targets).

Mirrors build_dude_target_kg.py but adapted:
  - 15 targets: ADRB2, ALDH1, ESR1_ago, ESR1_ant, FEN1, GBA, IDH1, KAT2A,
    MAPK1, MTORC1, OPRK1, PKM2, PPARG, TP53, VDR
  - No reference PDB code (LIT-PCBA targets multiple PDBs per assay)
  - Family clustering: each target is its own singleton (no FASTA available
    + targets unlikely to share Pfam given diverse enzyme classes)
  - Pocket PDB availability is not needed for the Group C++ cross-method
    audit (which reuses published per-target benchmark numbers).

Outputs:
  v2_target_node.parquet
  v2_active_of_target.parquet
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl


def canonical_ligand_id(smi: str) -> str:
    return "lig:" + hashlib.sha256(smi.encode()).hexdigest()[:24]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples-parquet", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.examples_parquet)
    df = df.filter(pl.col("source") == "LIT-PCBA")
    print(f"LIT-PCBA examples: {df.shape[0]} rows, {df['target'].n_unique()} targets")

    per_target = (
        df.group_by("target")
          .agg([
              pl.col("label").sum().cast(pl.Int64).alias("n_actives"),
              pl.col("label").count().cast(pl.Int64).alias("n_total"),
          ])
          .with_columns([(pl.col("n_total") - pl.col("n_actives")).alias("n_decoys")])
    )

    target_rows = []
    for row in per_target.iter_rows(named=True):
        name = row["target"]
        target_rows.append({
            "target_id": f"tgt:LIT-PCBA:{name}",
            "target_name": name,
            "source": "LIT-PCBA",
            "uniprot": "",
            "pdb_code": "",
            "n_actives": row["n_actives"],
            "n_decoys": row["n_decoys"],
            "has_pocket_pdb": False,  # not relevant; cross-method audit uses paper benchmarks
        })
    target_node_df = pl.DataFrame(target_rows)
    out_node = args.out_dir / "v2_target_node.parquet"
    target_node_df.write_parquet(out_node)
    print(f"wrote {out_node} ({target_node_df.shape[0]} targets)")

    actives_df = (
        df.filter(pl.col("label") == 1)
          .select(["smiles_canonical", "scaffold_smiles", "target"])
    )
    actives_df = actives_df.with_columns([
        pl.col("smiles_canonical").map_elements(canonical_ligand_id, return_dtype=pl.Utf8).alias("ligand_id"),
        (pl.lit("tgt:LIT-PCBA:") + pl.col("target")).alias("target_id"),
    ])
    active_edges = actives_df.select(
        ["ligand_id", "target_id", "smiles_canonical", "scaffold_smiles"]
    ).unique()
    out_act = args.out_dir / "v2_active_of_target.parquet"
    active_edges.write_parquet(out_act)
    print(f"wrote {out_act} ({active_edges.shape[0]} active-target edges)")

    # Each target as its own family (no protein-side family proxy here)
    family_rows = [
        {"target_id": r["target_id"], "family_id": f"fam:{r['target_name']}"}
        for r in target_rows
    ]
    family_df = pl.DataFrame(family_rows, schema={"target_id": pl.Utf8, "family_id": pl.Utf8})
    out_fam = args.out_dir / "v2_target_in_family.parquet"
    family_df.write_parquet(out_fam)
    print(f"wrote {out_fam} ({family_df.shape[0]} singleton-family edges)")

    print()
    print("=" * 60)
    print("Group C++ — LIT-PCBA target-level KG summary")
    print("=" * 60)
    print(f"  targets:                {target_node_df.shape[0]}")
    print(f"  total actives:          {int(target_node_df['n_actives'].sum())}")
    print(f"  total decoys:           {int(target_node_df['n_decoys'].sum())}")
    print(f"  unique active ligands:  {active_edges['ligand_id'].n_unique()}")
    print(f"  families:               {target_node_df.shape[0]} (singletons)")


if __name__ == "__main__":
    main()
