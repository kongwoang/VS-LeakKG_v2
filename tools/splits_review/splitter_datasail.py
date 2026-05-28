"""DataSAIL wrapper for S1-ligand / S1-protein / S2.

The wrapper:
  1. Reads the corpus manifest + optional protein_meta.parquet for sequences.
  2. Writes the three DataSAIL input files (e_data, f_data, inter) into a
     tmp dir.
  3. Calls `datasail.sail.datasail(...)` with SCIP solver.
  4. Maps the returned cluster->fold assignments back to (example_id, fold).
  5. Writes a SPLIT_SCHEMA parquet.

datasail call returns three structures; the actual shape is technique- and
version-specific. We use defensive parsing — accept either a dict-of-folds
or a list parallel to the input order — and log the realised shape.

DataSAIL uses external clusterers for proteins (`mmseqs`); for the ligand
side we rely on its built-in ECFP/Tanimoto path which has no external
binary dependency.
"""
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path
import polars as pl

# Make the side-installed datasail importable when invoked from drugclip_env.
DATASAIL_PKGS = Path("/vol/dl-nguyenb5-solar/users/hoangpc/envs/datasail_pkgs")
if DATASAIL_PKGS.exists() and str(DATASAIL_PKGS) not in sys.path:
    sys.path.insert(0, str(DATASAIL_PKGS))

from .common import write_split
from .schemas import hash_manifest_slice


FOLD_NAMES = ["train", "val", "test"]
SPLIT_RATIOS = [0.8, 0.1, 0.1]


def _normalise_assignment(asgn, entity_ids: list[str]) -> dict[str, str]:
    """Convert datasail's per-technique result to {entity_id -> fold_name}."""
    if isinstance(asgn, dict):
        # Map fold-name strings already.
        if all(v in FOLD_NAMES for v in asgn.values()):
            return {str(k): v for k, v in asgn.items()}
        # Maybe int fold ids.
        return {str(k): FOLD_NAMES[int(v)] for k, v in asgn.items()}
    if isinstance(asgn, list):
        # List parallel to entity_ids order.
        if len(asgn) != len(entity_ids):
            raise ValueError(f"unexpected list length {len(asgn)} vs {len(entity_ids)}")
        out = {}
        for k, v in zip(entity_ids, asgn):
            if isinstance(v, str) and v in FOLD_NAMES:
                out[str(k)] = v
            else:
                out[str(k)] = FOLD_NAMES[int(v)]
        return out
    raise TypeError(f"can't normalise assignment of type {type(asgn).__name__}")


def run_s1_ligand(manifest: pl.DataFrame, max_sec: int) -> dict[str, str]:
    """Returns ligand_id -> fold."""
    from datasail.sail import datasail
    ligs = manifest.unique(subset=["ligand_id"]).select(["ligand_id", "smiles"])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ldf = td / "ligands.tsv"
        with ldf.open("w") as f:
            for r in ligs.iter_rows(named=True):
                f.write(f"{r['ligand_id']}\t{r['smiles']}\n")
        ent_ids = ligs["ligand_id"].to_list()
        e_splits, _f_splits, _inter = datasail(
            techniques=["C1e"],
            splits=SPLIT_RATIOS, names=FOLD_NAMES,
            e_type="M", e_data=str(ldf), e_sim="ecfp",
            solver="SCIP", max_sec=max_sec,
        )
        asgn = e_splits["C1e"]
        return _normalise_assignment(asgn, ent_ids)


def run_s1_protein(manifest: pl.DataFrame, protein_meta: Path | None,
                   max_sec: int) -> dict[str, str]:
    """Returns target_id -> fold. Requires sequences in protein_meta.parquet."""
    from datasail.sail import datasail
    if protein_meta is None or not protein_meta.exists():
        raise SystemExit("protein_meta.parquet required for S1-protein; "
                         "run fetch_sequences.py first.")
    meta = pl.read_parquet(protein_meta)
    if meta["sequence"].is_null().any():
        n = meta["sequence"].is_null().sum()
        print(f"  WARN: {n} targets missing sequence — labelled target_split_control",
              file=sys.stderr)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pdf = td / "proteins.tsv"
        with pdf.open("w") as f:
            for r in meta.iter_rows(named=True):
                if r["sequence"]:
                    f.write(f"{r['target_id']}\t{r['sequence']}\n")
        ent_ids = [r["target_id"] for r in meta.iter_rows(named=True) if r["sequence"]]
        e_splits, _f, _i = datasail(
            techniques=["C1e"],
            splits=SPLIT_RATIOS, names=FOLD_NAMES,
            e_type="P", e_data=str(pdf), e_sim="mmseqs",
            solver="SCIP", max_sec=max_sec,
        )
        asgn = e_splits["C1e"]
        return _normalise_assignment(asgn, ent_ids)


def run_s2(manifest: pl.DataFrame, protein_meta: Path | None,
           max_sec: int) -> tuple[dict[str, str], dict[str, str], list[tuple[str, str]]]:
    """Returns (ligand_id->fold, target_id->fold, dropped_interactions)."""
    from datasail.sail import datasail
    if protein_meta is None or not protein_meta.exists():
        raise SystemExit("protein_meta.parquet required for S2.")
    meta = pl.read_parquet(protein_meta).filter(pl.col("sequence").is_not_null())

    ligs = manifest.unique(subset=["ligand_id"]).select(["ligand_id", "smiles"])
    inter = manifest.select(["ligand_id", "target_id"]).unique()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ldf = td / "ligands.tsv"
        pdf = td / "proteins.tsv"
        idf = td / "inter.tsv"
        with ldf.open("w") as f:
            for r in ligs.iter_rows(named=True):
                f.write(f"{r['ligand_id']}\t{r['smiles']}\n")
        with pdf.open("w") as f:
            for r in meta.iter_rows(named=True):
                f.write(f"{r['target_id']}\t{r['sequence']}\n")
        with idf.open("w") as f:
            for r in inter.iter_rows(named=True):
                f.write(f"{r['ligand_id']}\t{r['target_id']}\n")
        e_ids = ligs["ligand_id"].to_list()
        f_ids = meta["target_id"].to_list()
        e_splits, f_splits, inter_splits = datasail(
            techniques=["C2"],
            splits=SPLIT_RATIOS, names=FOLD_NAMES,
            e_type="M", e_data=str(ldf), e_sim="ecfp",
            f_type="P", f_data=str(pdf), f_sim="mmseqs",
            inter=str(idf),
            solver="SCIP", max_sec=max_sec,
        )
        e_map = _normalise_assignment(e_splits["C2"], e_ids)
        f_map = _normalise_assignment(f_splits["C2"], f_ids)
        # interactions DataSAIL dropped: examples whose ligand & protein
        # got different folds.
        dropped = []
        for r in inter.iter_rows(named=True):
            lf = e_map.get(r["ligand_id"]); pf = f_map.get(r["target_id"])
            if lf is None or pf is None or lf != pf:
                dropped.append((r["ligand_id"], r["target_id"]))
        return e_map, f_map, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--protein-meta", required=False, type=Path, default=None)
    ap.add_argument("--datasail-mode", required=True,
                    choices=["s1_ligand", "s1_protein", "s2"])
    ap.add_argument("--mode",     required=True, choices=["A", "B"])
    ap.add_argument("--out",      required=True, type=Path)
    ap.add_argument("--max-sec",  default=600, type=int)
    ap.add_argument("--seed",     default=2025, type=int)
    ap.add_argument("--subset-dir", required=False, type=Path, default=None)
    args = ap.parse_args()
    if args.mode != "B":
        raise SystemExit("DataSAIL splitter is Mode B only.")

    manifest = pl.read_parquet(args.manifest)
    rows: list[dict] = []
    input_hash = hash_manifest_slice(manifest)

    if args.datasail_mode == "s1_ligand":
        lmap = run_s1_ligand(manifest, args.max_sec)
        for r in manifest.iter_rows(named=True):
            fold = lmap.get(r["ligand_id"])
            if fold is None:
                continue
            rows.append({"example_id": r["example_id"], "target_id": r["target_id"],
                         "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
                         "fold": fold, "input_hash": input_hash})
    elif args.datasail_mode == "s1_protein":
        pmap = run_s1_protein(manifest, args.protein_meta, args.max_sec)
        for r in manifest.iter_rows(named=True):
            fold = pmap.get(r["target_id"])
            if fold is None:
                continue
            rows.append({"example_id": r["example_id"], "target_id": r["target_id"],
                         "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
                         "fold": fold, "input_hash": input_hash})
    else:  # s2
        lmap, pmap, dropped = run_s2(manifest, args.protein_meta, args.max_sec)
        dropped_set = set(dropped)
        for r in manifest.iter_rows(named=True):
            key = (r["ligand_id"], r["target_id"])
            if key in dropped_set:
                continue
            lf = lmap.get(r["ligand_id"]); pf = pmap.get(r["target_id"])
            if lf is None or pf is None or lf != pf:
                continue
            rows.append({"example_id": r["example_id"], "target_id": r["target_id"],
                         "ligand_id":  r["ligand_id"],  "label":     int(r["label"]),
                         "fold": lf, "input_hash": input_hash})
        print(f"  S2 dropped {len(dropped)} of {manifest.height} interactions "
              f"({100.0*len(dropped)/max(manifest.height,1):.1f}%)")

    write_split(rows, args.out, input_hash=input_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
