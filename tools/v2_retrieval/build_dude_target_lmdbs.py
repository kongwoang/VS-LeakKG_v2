"""Build per-target unimol LMDBs for the DUD-E retrieval audit (Group C).

For each target with an extracted pocket PDB:
  data/dude_retrieval/<target_name>/pocket.lmdb  — 1 entry: pocket_atoms + pocket_coordinates
  data/dude_retrieval/<target_name>/mols.lmdb    — N entries: ligand atoms/coords/smi/label

Per-target subsampling:
  - all actives kept
  - decoys subsampled to --max-decoys (default 1000) for tractable RDKit gen
  - 1 RDKit conformer per molecule (sufficient for retrieval; faster)

This is sized for proof-of-protocol. After validating the protocol we can
scale --max-decoys back up.

This script writes nothing under the v2/ outputs tree — all output is under
the user-provided --out-root.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def find_pocket(pocket_root: Path, pdb_code: str) -> Path | None:
    pdb = pdb_code.lower()
    for year_dir in pocket_root.iterdir():
        if not year_dir.is_dir():
            continue
        cand = year_dir / pdb / f"{pdb}_pocket.pdb"
        if cand.exists():
            return cand
    return None


def read_pocket(path: Path) -> tuple[list[str], np.ndarray] | None:
    try:
        df = PandasPdb().read_pdb(str(path)).df["ATOM"]
    except Exception:
        return None
    if len(df) == 0:
        return None
    atoms = df["atom_name"].astype(str).tolist()
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
    return atoms, coords


def gen_one_conformer(smi: str) -> tuple[list[str], np.ndarray] | None:
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        cid = AllChem.EmbedMolecule(mol, randomSeed=0, useRandomCoords=False, maxAttempts=200)
        if cid < 0:
            cid = AllChem.EmbedMolecule(mol, randomSeed=0, useRandomCoords=True, maxAttempts=200)
        if cid < 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    if mol.GetNumConformers() == 0:
        return None
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array(mol.GetConformer(0).GetPositions(), dtype=np.float32)
    return atoms, coords


def process_one(work: tuple) -> tuple[bytes, bytes] | None:
    idx, smi, label, target_id = work
    res = gen_one_conformer(smi)
    if res is None:
        return None
    atoms, coords = res
    row = {
        "atoms": atoms,
        "coordinates": [coords],   # unimol's AffinityDataset expects a list
        "smi": smi,
        "label": float(label),
        "target_id": target_id,
        "pocket_atoms": [],          # placeholder, ignored (pocket loaded from pocket.lmdb)
        "pocket_coordinates": [],
        "pocket": target_id,
    }
    return (str(idx).encode("ascii"), pickle.dumps(row))


def write_pocket_lmdb(out_dir: Path, pocket_path: Path, target_id: str) -> bool:
    pkt = read_pocket(pocket_path)
    if pkt is None:
        return False
    atoms, coords = pkt
    row = {
        "atoms": [],
        "coordinates": [],
        "pocket_atoms": atoms,
        "pocket_coordinates": coords,   # bare 2D array
        "smi": "",
        "pocket": target_id,
        "label": 0.0,
    }
    out = out_dir / "pocket.lmdb"
    if out.exists():
        out.unlink()
    env = lmdb.open(str(out), map_size=64 * 1024 * 1024, subdir=False)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps(row))
    env.close()
    return True


def write_mols_lmdb(
    out_dir: Path,
    target_name: str,
    target_id: str,
    actives: list[str],
    decoys: list[str],
    num_workers: int,
) -> tuple[int, int]:
    out = out_dir / "mols.lmdb"
    if out.exists():
        out.unlink()

    work = []
    idx = 0
    for smi in actives:
        work.append((idx, smi, 1, target_id))
        idx += 1
    for smi in decoys:
        work.append((idx, smi, 0, target_id))
        idx += 1

    env = lmdb.open(str(out), map_size=10 * 1024**3, subdir=False, map_async=True)
    n_ok, n_err = 0, 0
    with env.begin(write=True) as txn:
        with mp.Pool(num_workers) as pool:
            for res in tqdm(pool.imap_unordered(process_one, work, chunksize=32),
                            total=len(work), desc=target_name, leave=False):
                if res is None:
                    n_err += 1
                    continue
                k, v = res
                txn.put(k, v)
                n_ok += 1
    env.sync()
    env.close()
    return n_ok, n_err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-node", required=True, type=Path)
    p.add_argument("--examples-parquet", required=True, type=Path)
    p.add_argument("--pocket-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--max-decoys", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated target names; build only these (for smoke test)")
    args = p.parse_args()

    targets = pl.read_parquet(args.target_node).filter(pl.col("has_pocket_pdb"))
    examples = pl.read_parquet(args.examples_parquet).filter(pl.col("source") == "DUD-E")
    examples = examples.filter(pl.col("parse_ok"))

    if args.only:
        wanted = {n.strip().lower() for n in args.only.split(",")}
        targets = targets.filter(pl.col("target_name").is_in(list(wanted)))

    print(f"targets to build: {targets.shape[0]}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = []
    for row in targets.iter_rows(named=True):
        name = row["target_name"]
        target_id = row["target_id"]
        pdb_code = row["pdb_code"]

        out_dir = args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pocket
        pocket_path = find_pocket(args.pocket_root, pdb_code)
        if pocket_path is None:
            print(f"  [skip] {name}: no pocket pdb")
            continue
        ok = write_pocket_lmdb(out_dir, pocket_path, target_id)
        if not ok:
            print(f"  [skip] {name}: pocket extraction failed")
            continue

        # Mols
        target_rows = examples.filter(pl.col("target") == name)
        actives = target_rows.filter(pl.col("label") == 1)["smiles_canonical"].to_list()
        decoys = target_rows.filter(pl.col("label") == 0)["smiles_canonical"].to_list()
        if len(decoys) > args.max_decoys:
            idx = rng.choice(len(decoys), size=args.max_decoys, replace=False)
            decoys = [decoys[i] for i in idx]

        n_ok, n_err = write_mols_lmdb(out_dir, name, target_id, actives, decoys, args.num_workers)
        summary.append({
            "target_name": name,
            "target_id": target_id,
            "pdb_code": pdb_code,
            "n_actives_listed": len(actives),
            "n_decoys_listed": len(decoys),
            "n_written": n_ok,
            "n_failed_rdkit": n_err,
        })

    sum_df = pl.DataFrame(summary)
    sum_df.write_csv(args.out_root / "build_summary.csv")
    print(sum_df.head(15))
    print(f"total targets built: {sum_df.shape[0]}")
    print(f"total mol rows written: {int(sum_df['n_written'].sum())}")
    print(f"total RDKit failures: {int(sum_df['n_failed_rdkit'].sum())}")


if __name__ == "__main__":
    main()
