"""Build DrugCLIP-format unimol LMDBs from v2 PDBBind split parquets.

For each example in the split: looks up the PDBBind raw structure, reads the
pre-extracted pocket.pdb (PDBBind ships pocket atoms within ~6 A of the
ligand), generates RDKit 3D conformer(s) for the ligand, and writes a row in
DrugCLIP's expected schema:

    {
      "atoms":              [str, ...]            ligand element symbols
      "coordinates":        [np.ndarray, ...]     N_conf 3D conformers
      "pocket_atoms":       [str, ...]            pocket atom names (e.g. CA, CB)
      "pocket_coordinates": [np.ndarray]          one 3D pocket conformer
      "smi":                str                   canonical SMILES
      "pocket":             str                   pdb id
      "label":              float                 0/1 (v2 binarized pK)
    }

Usage:
    python v2_to_drugclip_lmdb.py \
        --split-parquet outputs/v2/phase1_full/splits/pdbbind/ligand.parquet \
        --pdbbind-root /vol/.../raw/PBDBind/extracted/P-L \
        --out-dir       /vol/.../DrugCLIP/data/v2_pdbbind_ligand
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def build_pdbid_index(pdbbind_root: Path) -> dict[str, Path]:
    """Map pdb_id → directory containing <pdb_id>_pocket.pdb."""
    idx: dict[str, Path] = {}
    for year_dir in pdbbind_root.iterdir():
        if not year_dir.is_dir():
            continue
        for pdb_dir in year_dir.iterdir():
            if not pdb_dir.is_dir():
                continue
            idx[pdb_dir.name.lower()] = pdb_dir
    return idx


def extract_pdb_id(example_id: str) -> str:
    # Example::pdbbind::complex:PDBBind:3f1a  →  3f1a
    return example_id.split(":")[-1].lower()


def read_pocket(pocket_pdb_path: Path) -> tuple[list[str], np.ndarray] | None:
    try:
        df = PandasPdb().read_pdb(str(pocket_pdb_path)).df["ATOM"]
    except Exception:
        return None
    if len(df) == 0:
        return None
    atoms = df["atom_name"].astype(str).tolist()
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
    return atoms, coords


def gen_conformers(smi: str | None, num_conf: int = 3) -> tuple[list[str], list[np.ndarray]] | None:
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol, numConfs=num_conf, pruneRmsThresh=1.0, maxAttempts=200, useRandomCoords=False
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    if mol.GetNumConformers() == 0:
        return None
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [
        np.array(mol.GetConformer(i).GetPositions(), dtype=np.float32)
        for i in range(mol.GetNumConformers())
    ]
    return atoms, coords


def process_one(args) -> tuple[bytes, bytes] | None:
    idx, pdb_id, smi, label, pdbid_dirs_path = args
    try:
        pdb_dir = pdbid_dirs_path.get(pdb_id)
        if pdb_dir is None:
            return None
        pocket_path = pdb_dir / f"{pdb_id}_pocket.pdb"
        if not pocket_path.exists():
            return None
        pocket = read_pocket(pocket_path)
        if pocket is None:
            return None
        pocket_atoms, pocket_coords = pocket

        ligand = gen_conformers(smi, num_conf=3)
        if ligand is None:
            return None
        atoms, coords = ligand
    except Exception:
        return None

    row = {
        "atoms": atoms,
        "coordinates": coords,
        "pocket_atoms": pocket_atoms,
        "pocket_coordinates": pocket_coords,
        "smi": smi,
        "pocket": pdb_id,
        "label": float(label),
    }
    return (str(idx).encode(), pickle.dumps(row))


def write_split_lmdb(
    df: pl.DataFrame,
    pdbid_dirs: dict[str, Path],
    out_path: Path,
    *,
    positives_only: bool = True,
    num_workers: int = 16,
) -> tuple[int, int]:
    if positives_only:
        df = df.filter(pl.col("label") == 1.0)

    work = []
    for i, row in enumerate(df.iter_rows(named=True)):
        pdb_id = extract_pdb_id(row["example_id"])
        work.append((i, pdb_id, row["smiles"], row["label"], pdbid_dirs))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(out_path),
        map_size=20 * 1024**3,
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    n_ok, n_err = 0, 0
    with env.begin(write=True) as txn:
        with mp.Pool(num_workers) as pool:
            for res in tqdm(pool.imap_unordered(process_one, work, chunksize=32), total=len(work)):
                if res is None:
                    n_err += 1
                    continue
                k, v = res
                txn.put(k, v)
                n_ok += 1
    env.sync()
    env.close()
    return n_ok, n_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-parquet", required=True, type=Path)
    p.add_argument("--pdbbind-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--num-workers", type=int, default=16)
    args = p.parse_args()

    print(f"Loading split: {args.split_parquet}")
    df = pl.read_parquet(args.split_parquet)
    print(f"  {df.shape[0]} examples ({df.filter(pl.col('label')==1.0).shape[0]} positive)")

    print(f"Indexing PDBBind at: {args.pdbbind_root}")
    pdbid_dirs = build_pdbid_index(args.pdbbind_root)
    print(f"  {len(pdbid_dirs)} pdb directories")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for part in ("train", "valid", "test"):
        sub = df.filter(pl.col("partition") == ("val" if part == "valid" else part))
        if sub.shape[0] == 0:
            print(f"[skip] empty {part}")
            continue
        out_path = args.out_dir / f"{part}.lmdb"
        print(f"\n→ Writing {part}.lmdb  ({sub.shape[0]} rows, positives_only={part != 'test'})")
        n_ok, n_err = write_split_lmdb(
            sub,
            pdbid_dirs,
            out_path,
            positives_only=(part != "test"),
            num_workers=args.num_workers,
        )
        print(f"  done. {n_ok} written, {n_err} skipped (missing/failed).")


if __name__ == "__main__":
    main()
