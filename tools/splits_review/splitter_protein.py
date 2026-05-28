"""Protein-sequence cluster splitter (cross-target, Mode B only).

Clusters target sequences with MMseqs2 at a chosen identity threshold and
assigns whole sequence-clusters to folds with greedy bin-pack toward
(0.8, 0.1, 0.1). When no protein sequence is available, falls back to
target_id as the cluster key (degenerate, equivalent to per-target).

Requires `mmseqs` on PATH. If absent, the script raises with a hint to
install it via the datasail_env or bioconda.
"""
from __future__ import annotations
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
import polars as pl

from .common import write_split, fold_quotas
from .schemas import hash_manifest_slice


def cluster_sequences(uniprot_to_seq: dict[str, str], identity: float, tmpdir: Path) -> dict[str, str]:
    """Return uniprot -> representative-uniprot (cluster id)."""
    mm = shutil.which("mmseqs")
    if mm is None:
        raise SystemExit("mmseqs not on PATH; install bioconda's mmseqs2 in datasail_env.")
    fasta = tmpdir / "input.fasta"
    with fasta.open("w") as f:
        for uid, seq in uniprot_to_seq.items():
            if seq:
                f.write(f">{uid}\n{seq}\n")
    out_prefix = tmpdir / "clu"
    subprocess.run([mm, "easy-cluster", str(fasta), str(out_prefix),
                    str(tmpdir / "mm_tmp"), "--min-seq-id", str(identity),
                    "-c", "0.8", "--cov-mode", "0"],
                   check=True, capture_output=True)
    rep_map: dict[str, str] = {}
    cluster_tsv = Path(str(out_prefix) + "_cluster.tsv")
    for line in cluster_tsv.read_text().splitlines():
        rep, member = line.split("\t")
        rep_map[member] = rep
    return rep_map


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--identity",  default=0.3, type=float)
    ap.add_argument("--seed",      default=2025, type=int)
    ap.add_argument("--subset-dir", required=False, type=Path, default=None)
    ap.add_argument("--protein-meta", required=False, type=Path, default=None,
                    help="protein_meta.parquet with target_id+sequence; required "
                         "for real protein-similarity splits.")
    args = ap.parse_args()
    if args.mode != "B":
        raise SystemExit("protein splitter is Mode B only (cross-target).")

    manifest = pl.read_parquet(args.manifest)
    # Build uniprot -> seq table by joining with protein_meta if available.
    protein_meta = getattr(args, "protein_meta", None)
    seq_lookup: dict[str, str] = {}
    is_target_control = False
    if protein_meta is not None and protein_meta.exists():
        meta = pl.read_parquet(protein_meta)
        for r in meta.iter_rows(named=True):
            if r.get("sequence"):
                seq_lookup[r["target_id"]] = r["sequence"]

    if seq_lookup:
        with tempfile.TemporaryDirectory() as td:
            rep_map = cluster_sequences(seq_lookup, args.identity, Path(td))
        cluster_col = pl.Series("_clu",
            [rep_map.get(t, t) for t in manifest["target_id"].to_list()])
        print(f"  protein splitter: clustered {len(seq_lookup)} sequences -> "
              f"{len(set(rep_map.values()))} clusters at identity {args.identity}")
    else:
        # No sequences — explicitly label as target_split_control rather than
        # silently degenerate. Output file naming should reflect this externally.
        cluster_col = manifest["target_id"].alias("_clu")
        is_target_control = True
        print("  WARN: no sequences available — this is a TARGET_SPLIT_CONTROL, "
              "not a protein-similarity split. Mark as control in the report.")

    manifest = manifest.with_columns(cluster_col)
    sizes = manifest.group_by("_clu").agg(pl.len().alias("n")).sort("n", descending=True)
    n_tr, n_va, n_te = fold_quotas(manifest.height)
    quota = {"train": n_tr, "val": n_va, "test": n_te}
    have  = {"train": 0,   "val": 0,    "test": 0}
    c_to_fold: dict[str, str] = {}
    for row in sizes.iter_rows(named=True):
        deficits = {f: quota[f] - have[f] for f in quota}
        choice = max(deficits, key=deficits.get)
        c_to_fold[row["_clu"]] = choice
        have[choice] += row["n"]
    rows = []
    for r in manifest.iter_rows(named=True):
        rows.append({
            "example_id": r["example_id"], "target_id": r["target_id"],
            "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
            "fold":       c_to_fold[r["_clu"]],
            "input_hash": hash_manifest_slice(manifest),
        })
    write_split(rows, args.out, input_hash=hash_manifest_slice(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
