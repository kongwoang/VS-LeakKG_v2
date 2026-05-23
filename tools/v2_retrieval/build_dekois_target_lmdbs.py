"""Build per-target unimol LMDBs for the DEKOIS 2.0 retrieval audit.

Mirrors build_dude_target_lmdbs.py but:
  - Pocket lives at <root>/<target>/protein/<target>_pocket_ligH12A.pdb
  - Uses ALL decoys per target (DEKOIS already has ~1100 per target — tractable)
  - 1 RDKit conformer per molecule

Output: <out-root>/<target>/{pocket,mols}.lmdb
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
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
        "coordinates": [coords],
        "smi": smi,
        "label": float(label),
        "target_id": target_id,
        "pocket_atoms": [],
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
        "atoms": [], "coordinates": [],
        "pocket_atoms": atoms, "pocket_coordinates": coords,
        "smi": "", "pocket": target_id, "label": 0.0,
    }
    out = out_dir / "pocket.lmdb"
    if out.exists():
        out.unlink()
    env = lmdb.open(str(out), map_size=64 * 1024 * 1024, subdir=False)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps(row))
    env.close()
    return True


def write_mols_lmdb(out_dir: Path, target_name: str, target_id: str,
                    actives: list[str], decoys: list[str], num_workers: int) -> tuple[int, int]:
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
    p.add_argument("--dekois-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()

    targets = pl.read_parquet(args.target_node).filter(pl.col("has_pocket_pdb"))
    examples = pl.read_parquet(args.examples_parquet).filter(pl.col("source") == "DEKOIS")
    examples = examples.filter(pl.col("parse_ok"))

    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        targets = targets.filter(pl.col("target_name").is_in(list(wanted)))

    print(f"targets to build: {targets.shape[0]}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for row in targets.iter_rows(named=True):
        name = row["target_name"]
        target_id = row["target_id"]
        out_dir = args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        pocket_path = args.dekois_root / name / "protein" / f"{name}_pocket_ligH12A.pdb"
        if not pocket_path.exists():
            print(f"  [skip] {name}: no pocket pdb at {pocket_path}")
            continue
        ok = write_pocket_lmdb(out_dir, pocket_path, target_id)
        if not ok:
            print(f"  [skip] {name}: pocket extraction failed")
            continue

        target_rows = examples.filter(pl.col("target") == name)
        actives = target_rows.filter(pl.col("label") == 1)["smiles_canonical"].to_list()
        decoys = target_rows.filter(pl.col("label") == 0)["smiles_canonical"].to_list()

        n_ok, n_err = write_mols_lmdb(out_dir, name, target_id, actives, decoys, args.num_workers)
        summary.append({
            "target_name": name, "target_id": target_id,
            "n_actives_listed": len(actives), "n_decoys_listed": len(decoys),
            "n_written": n_ok, "n_failed_rdkit": n_err,
        })

    sum_df = pl.DataFrame(summary)
    sum_df.write_csv(args.out_root / "build_summary.csv")
    print(sum_df.head(15))
    print(f"total targets built: {sum_df.shape[0]}")
    print(f"total mol rows written: {int(sum_df['n_written'].sum())}")
    print(f"total RDKit failures: {int(sum_df['n_failed_rdkit'].sum())}")


if __name__ == "__main__":
    main()
