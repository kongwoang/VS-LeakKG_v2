"""Build target-level KG view for DEKOIS 2.0 (Group C — retrieval audit).

Mirrors build_dude_target_kg.py but adapted to DEKOIS conventions:
  - Target name comes from dekois_examples.target (e.g. "11betahsd1")
  - Pocket PDB lives at: <raw>/DEKOIS/extracted/DEKOIS2/<target>/<target>_pocket_ligH12A.pdb
  - Protein PDB lives alongside as <target>_protein.pdb — we extract the
    longest chain's sequence for the family proxy
  - No published target → UniProt mapping; we keep uniprot blank

Outputs (additive, alongside outputs/v2_retrieval/graph_dekois/):
  v2_target_node.parquet       (target metadata)
  v2_active_of_target.parquet  (ligand-active-of-target edges)
  v2_target_in_family.parquet  (≥40% seq-identity clusters from pocket protein PDBs)
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
from biopandas.pdb import PandasPdb


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def canonical_ligand_id(smi: str) -> str:
    return "lig:" + hashlib.sha256(smi.encode()).hexdigest()[:24]


def pdb_to_sequence(pdb_path: Path) -> str:
    """Extract the longest chain's amino-acid sequence as a 1-letter string."""
    try:
        df = PandasPdb().read_pdb(str(pdb_path)).df["ATOM"]
    except Exception:
        return ""
    if len(df) == 0:
        return ""
    df = df[df["atom_name"] == "CA"]
    if len(df) == 0:
        return ""
    # Group by chain → take longest
    by_chain = df.groupby("chain_id")
    best_seq = ""
    for _, sub in by_chain:
        seq = "".join(THREE_TO_ONE.get(r, "X") for r in sub["residue_name"].tolist())
        if len(seq) > len(best_seq):
            best_seq = seq
    return best_seq


def pairwise_seq_identity(seqs: list[tuple[str, str]]) -> np.ndarray:
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5
    n = len(seqs)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            si, sj = seqs[i][1], seqs[j][1]
            if not si or not sj:
                continue
            try:
                score = aligner.score(si, sj)
                ident = score / max(len(si), len(sj))
            except Exception:
                ident = 0.0
            mat[i, j] = ident
            mat[j, i] = ident
    return mat


def cluster_by_identity(names: list[str], mat: np.ndarray, threshold: float) -> dict[str, str]:
    n = len(names)
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] >= threshold:
                union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    name_to_family: dict[str, str] = {}
    for _, members in clusters.items():
        family_id = "fam:" + min(names[i] for i in members)
        for idx in members:
            name_to_family[names[idx]] = family_id
    return name_to_family


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples-parquet", required=True, type=Path)
    p.add_argument("--dekois-root", required=True, type=Path,
                   help="DEKOIS2 raw root containing one dir per target")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seq-id-threshold", type=float, default=0.40)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.examples_parquet)
    df = df.filter(pl.col("source") == "DEKOIS")
    print(f"DEKOIS examples: {df.shape[0]} rows, {df['target'].n_unique()} targets")

    per_target = (
        df.group_by("target")
          .agg([
              pl.col("label").sum().cast(pl.Int64).alias("n_actives"),
              pl.col("label").count().cast(pl.Int64).alias("n_total"),
          ])
          .with_columns([(pl.col("n_total") - pl.col("n_actives")).alias("n_decoys")])
    )

    # Build target metadata + check pocket existence + extract sequence
    target_rows = []
    target_to_seq: dict[str, str] = {}
    for row in per_target.iter_rows(named=True):
        name = row["target"]
        tdir = args.dekois_root / name
        pocket_pdb = tdir / "protein" / f"{name}_pocket_ligH12A.pdb"
        protein_pdb = tdir / "protein" / f"{name}_protein.pdb"

        has_pocket = pocket_pdb.exists()
        seq = pdb_to_sequence(protein_pdb) if protein_pdb.exists() else ""
        if seq:
            target_to_seq[name] = seq

        target_rows.append({
            "target_id": f"tgt:DEKOIS:{name}",
            "target_name": name,
            "source": "DEKOIS",
            "uniprot": "",
            "pdb_code": "",   # DEKOIS doesn't ship reference PDB codes
            "n_actives": row["n_actives"],
            "n_decoys": row["n_decoys"],
            "has_pocket_pdb": has_pocket,
        })

    target_node_df = pl.DataFrame(target_rows)
    out_node = args.out_dir / "v2_target_node.parquet"
    target_node_df.write_parquet(out_node)
    print(f"wrote {out_node}  ({target_node_df.shape[0]} targets, "
          f"{int(target_node_df['has_pocket_pdb'].sum())} with pocket pdb)")

    # Active-of-target edges
    actives_df = (
        df.filter(pl.col("label") == 1)
          .select(["smiles_canonical", "scaffold_smiles", "target"])
    )
    actives_df = actives_df.with_columns([
        pl.col("smiles_canonical").map_elements(canonical_ligand_id, return_dtype=pl.Utf8).alias("ligand_id"),
        (pl.lit("tgt:DEKOIS:") + pl.col("target")).alias("target_id"),
    ])
    active_edges = actives_df.select(
        ["ligand_id", "target_id", "smiles_canonical", "scaffold_smiles"]
    ).unique()
    out_act = args.out_dir / "v2_active_of_target.parquet"
    active_edges.write_parquet(out_act)
    print(f"wrote {out_act}  ({active_edges.shape[0]} active-target edges)")

    # Pfam-proxy family edges
    print(f"target_to_seq: {len(target_to_seq)} of {target_node_df.shape[0]}")
    names = sorted(target_to_seq.keys())
    seqs = [(n, target_to_seq[n]) for n in names]
    print(f"pairwise seq identity for {len(names)} targets…")
    mat = pairwise_seq_identity(seqs)
    family = cluster_by_identity(names, mat, args.seq_id_threshold)
    family_rows = [
        {"target_id": f"tgt:DEKOIS:{n}", "family_id": family[n]}
        for n in names
    ]
    family_df = pl.DataFrame(family_rows, schema={"target_id": pl.Utf8, "family_id": pl.Utf8})
    out_fam = args.out_dir / "v2_target_in_family.parquet"
    family_df.write_parquet(out_fam)
    n_fam = family_df["family_id"].n_unique() if family_df.shape[0] > 0 else 0
    print(f"wrote {out_fam}  ({family_df.shape[0]} edges, "
          f"{n_fam} families at ID≥{args.seq_id_threshold})")

    print()
    print("=" * 60)
    print("Group C — DEKOIS target-level KG summary")
    print("=" * 60)
    print(f"  targets:                 {target_node_df.shape[0]}")
    print(f"  targets with pocket PDB: {int(target_node_df['has_pocket_pdb'].sum())}")
    print(f"  total actives (rows):    {int(target_node_df['n_actives'].sum())}")
    print(f"  total decoys (rows):     {int(target_node_df['n_decoys'].sum())}")
    print(f"  unique active ligands:   {active_edges['ligand_id'].n_unique()}")
    print(f"  families (ID≥{args.seq_id_threshold}):       {n_fam}")


if __name__ == "__main__":
    main()
