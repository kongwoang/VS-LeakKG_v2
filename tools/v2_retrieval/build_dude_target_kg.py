"""Build target-level KG view for DUD-E (Group C — retrieval audit).

Additive: writes new parquets under outputs/v2_retrieval/graph_dude/ but
does NOT modify the existing outputs/v2/graph_dude/ files.

What gets built:
  v2_target_node.parquet      — one row per DUD-E target with metadata
  v2_active_of_target.parquet — typed edges: ligand-is-active-of-target
  v2_target_in_family.parquet — Pfam-family-style equivalence classes

Inputs:
  --examples-parquet  flat row table (smiles_canonical, target, label, scaffold_smiles)
  --target-mapping    LigUnity dude.json: [[uniprot, pdb_code, target_name], ...]
  --protein-fasta     conglude protein_sequences.fasta (one chain per >id)
  --pocket-root       PDBBind extracted/P-L (year-bucketed pdb_code dirs)
  --out-dir           outputs/v2_retrieval/graph_dude/

Pfam family proxy (for target-clean splits):
  We don't fetch live Pfam annotations. Instead, we compute pairwise
  sequence identity between target chains (via Biopython global align)
  and cluster targets with identity >= --seq-id-threshold (default 0.40,
  the conventional Pfam-superfamily-ish boundary). Family ID = lowest
  target name in cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


def canonical_ligand_id(smi: str) -> str:
    return "lig:" + hashlib.sha256(smi.encode()).hexdigest()[:24]


def parse_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    current_id, current_seq = None, []
    for raw in path.read_text().splitlines():
        if not raw:
            continue
        if raw.startswith(">"):
            if current_id is not None:
                out[current_id] = "".join(current_seq)
            current_id = raw[1:].strip()
            current_seq = []
        else:
            current_seq.append(raw.strip())
    if current_id is not None:
        out[current_id] = "".join(current_seq)
    return out


def pairwise_seq_identity(seqs: list[tuple[str, str]]) -> np.ndarray:
    """All-vs-all sequence identity (n × n matrix).

    Cheap O(L1*L2) per pair via simple aligned-length normalization on the
    longest common subsequence approximation. For O(10K) pairs we use a
    fast Biopython.Align local alignment.
    """
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
    """Single-linkage clustering at the given identity threshold."""
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
    p.add_argument("--target-mapping", required=True, type=Path)
    p.add_argument("--protein-fasta", required=True, type=Path)
    p.add_argument("--pocket-root", required=True, type=Path,
                   help="PDBBind extracted P-L root (year/<pdb>/<pdb>_pocket.pdb layout)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seq-id-threshold", type=float, default=0.40)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load target mapping (LigUnity dude.json) ----
    mapping = json.loads(args.target_mapping.read_text())
    # mapping rows: [uniprot, pdb_code, target_name]
    target_meta: dict[str, dict[str, str]] = {}
    for uniprot, pdb_code, target_name in mapping:
        target_meta[target_name.lower()] = {
            "uniprot": uniprot,
            "pdb_code": pdb_code.lower(),
            "target_name_upper": target_name,
        }

    # ---- Load flat examples ----
    df = pl.read_parquet(args.examples_parquet)
    df = df.filter(pl.col("source") == "DUD-E")
    print(f"DUD-E examples: {df.shape[0]} rows, "
          f"{df['target'].n_unique()} targets")

    per_target = (
        df.group_by("target")
          .agg([
              pl.col("label").sum().cast(pl.Int64).alias("n_actives"),
              pl.col("label").count().cast(pl.Int64).alias("n_total"),
          ])
          .with_columns([
              (pl.col("n_total") - pl.col("n_actives")).alias("n_decoys"),
          ])
    )

    # ---- Pocket availability ----
    pocket_root = args.pocket_root
    has_pocket: dict[str, bool] = {}
    for name, meta in target_meta.items():
        pdb = meta["pdb_code"]
        found = False
        for year_dir in pocket_root.iterdir():
            if not year_dir.is_dir():
                continue
            cand = year_dir / pdb / f"{pdb}_pocket.pdb"
            if cand.exists():
                found = True
                break
        has_pocket[name] = found

    # ---- Build target node table ----
    target_rows = []
    for row in per_target.iter_rows(named=True):
        name = row["target"].lower()
        meta = target_meta.get(name, {})
        target_rows.append({
            "target_id": f"tgt:DUD-E:{name}",
            "target_name": name,
            "source": "DUD-E",
            "uniprot": meta.get("uniprot", ""),
            "pdb_code": meta.get("pdb_code", ""),
            "n_actives": row["n_actives"],
            "n_decoys": row["n_decoys"],
            "has_pocket_pdb": has_pocket.get(name, False),
        })
    target_node_df = pl.DataFrame(target_rows)
    out_node = args.out_dir / "v2_target_node.parquet"
    target_node_df.write_parquet(out_node)
    print(f"wrote {out_node}  ({target_node_df.shape[0]} targets, "
          f"{int(target_node_df['has_pocket_pdb'].sum())} with extracted pocket)")

    # ---- Active-of-target edge ----
    actives_df = (
        df.filter(pl.col("label") == 1)
          .select(["smiles_canonical", "scaffold_smiles", "target"])
    )
    actives_df = actives_df.with_columns([
        pl.col("smiles_canonical").map_elements(canonical_ligand_id, return_dtype=pl.Utf8).alias("ligand_id"),
        (pl.lit("tgt:DUD-E:") + pl.col("target")).alias("target_id"),
    ])
    active_edges = actives_df.select(["ligand_id", "target_id", "smiles_canonical", "scaffold_smiles"]).unique()
    out_act = args.out_dir / "v2_active_of_target.parquet"
    active_edges.write_parquet(out_act)
    print(f"wrote {out_act}  ({active_edges.shape[0]} active-target edges)")

    # ---- Pfam-style family edges via pairwise sequence identity ----
    fasta = parse_fasta(args.protein_fasta)
    # FASTA ids look like "3LPB_A" — map to lowercase target by pdb code
    pdb_to_target = {meta["pdb_code"]: name for name, meta in target_meta.items()}
    target_to_seq: dict[str, str] = {}
    for chain_id, seq in fasta.items():
        pdb = chain_id.split("_")[0].lower()
        tname = pdb_to_target.get(pdb)
        if tname is None:
            continue
        # Keep the longest chain per target
        if tname not in target_to_seq or len(seq) > len(target_to_seq[tname]):
            target_to_seq[tname] = seq
    print(f"target_to_seq: {len(target_to_seq)} (of {len(target_meta)} mapping)")

    names = sorted(target_to_seq.keys())
    seqs = [(n, target_to_seq[n]) for n in names]
    print("computing pairwise sequence identity for "
          f"{len(names)} targets ({len(names)*(len(names)-1)//2} pairs)…")
    mat = pairwise_seq_identity(seqs)
    family = cluster_by_identity(names, mat, args.seq_id_threshold)

    family_rows = [
        {"target_id": f"tgt:DUD-E:{n}", "family_id": family[n]}
        for n in names
    ]
    family_df = pl.DataFrame(family_rows)
    out_fam = args.out_dir / "v2_target_in_family.parquet"
    family_df.write_parquet(out_fam)
    print(f"wrote {out_fam}  ({family_df.shape[0]} edges, "
          f"{family_df['family_id'].n_unique()} families at ID≥{args.seq_id_threshold})")

    # ---- Summary stats ----
    print()
    print("=" * 60)
    print("Group C — DUD-E target-level KG summary")
    print("=" * 60)
    print(f"  targets:                 {target_node_df.shape[0]}")
    print(f"  targets with pocket PDB: {int(target_node_df['has_pocket_pdb'].sum())}")
    print(f"  total actives (rows):    {int(target_node_df['n_actives'].sum())}")
    print(f"  total decoys (rows):     {int(target_node_df['n_decoys'].sum())}")
    print(f"  unique active ligands:   {active_edges['ligand_id'].n_unique()}")
    print(f"  families (ID≥{args.seq_id_threshold}):       {family_df['family_id'].n_unique()}")


if __name__ == "__main__":
    main()
