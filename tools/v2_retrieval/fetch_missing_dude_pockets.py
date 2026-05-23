"""Fetch missing DUD-E reference PDBs from RCSB and extract pockets.

For each DUD-E target whose reference PDB code is NOT already in PDBBind
2020 (~37 of 102), download the PDB from RCSB and extract residues within
6 Å of the crystal ligand atoms. Write a PDBBind-compatible *_pocket.pdb
under --out-root/<pdb_code>/<pdb_code>_pocket.pdb so the existing LMDB
builder can pick it up.

Crystal ligand selection:
  Largest HETATM group by heavy-atom count, excluding a blacklist of
  common buffers/ions/cofactors. Drop-anything below --min-ligand-atoms.

Usage:
  python fetch_missing_dude_pockets.py \
      --target-mapping LigUnity/test_datasets/dude.json \
      --already-have outputs/v2_retrieval/graph_dude/v2_target_node.parquet \
      --out-root <pocket-root-for-missing> \
      --num-workers 4
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from biopandas.pdb import PandasPdb


LIGAND_BLACKLIST = {
    # Water
    "HOH", "WAT", "TIP", "DOD",
    # Common ions
    "NA", "K", "MG", "CA", "ZN", "FE", "FE2", "FE3", "MN", "CU", "NI", "CO",
    "CL", "BR", "I", "F", "IOD",
    # Common buffers / cryoprotectants
    "SO4", "PO4", "HPO", "GOL", "EDO", "PEG", "PG4", "PGE", "P6G",
    "MES", "EPE", "TRS", "BIS", "CIT", "ACT", "ACY",
    "DMS", "FMT", "BCT", "BME", "OXY", "OXE",
    # Capping groups / linker fragments
    "ACE", "NME", "MLY", "MSE",  # selenomethionine sometimes counted
    # Sugars (cofactor-ish) — keep most ligands but drop common modifications
    "NAG", "BMA", "MAN", "FUC", "GAL", "GLC", "BGC",
}


def fetch_pdb(pdb_code: str, timeout: float = 30.0) -> bytes | None:
    """Download and gunzip PDB structure from RCSB. Returns text or None."""
    pdb = pdb_code.lower()
    url = f"https://files.rcsb.org/download/{pdb}.pdb.gz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "v2-audit/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return gzip.decompress(data)
    except Exception as e:
        print(f"  [fetch err {pdb}] {e}", flush=True)
        return None


def select_ligand(hetatm_df) -> tuple[str, int, int, str] | None:
    """Return (resn, resi, chain, group_id_str) of the largest non-blacklist HETATM group."""
    if len(hetatm_df) == 0:
        return None
    # Group by (chain, residue_number, residue_name) and count atoms (heavy = non-H)
    df = hetatm_df.copy()
    df = df[~df["element_symbol"].isin(["H", "D"])]
    df = df[~df["residue_name"].isin(list(LIGAND_BLACKLIST))]
    if len(df) == 0:
        return None
    grouped = df.groupby(["chain_id", "residue_number", "residue_name"]).size().reset_index(name="natoms")
    grouped = grouped.sort_values("natoms", ascending=False)
    if len(grouped) == 0:
        return None
    top = grouped.iloc[0]
    if top["natoms"] < 7:
        return None
    return top["residue_name"], int(top["residue_number"]), top["chain_id"], int(top["natoms"])


def extract_pocket(pdb_text: bytes, pdb_code: str, dist_cutoff: float = 6.0
                   ) -> tuple[bytes, str] | None:
    """Parse PDB; find crystal ligand; return pocket residues as PDB text."""
    try:
        pdb = PandasPdb().read_pdb_from_list(pdb_text.decode("utf-8", errors="ignore").splitlines())
    except Exception as e:
        return None
    atom_df = pdb.df.get("ATOM")
    het_df = pdb.df.get("HETATM")
    if atom_df is None or len(atom_df) == 0:
        return None
    if het_df is None or len(het_df) == 0:
        return None

    pick = select_ligand(het_df)
    if pick is None:
        return None
    resn, resi, chain, natoms = pick

    lig_df = het_df[
        (het_df["chain_id"] == chain)
        & (het_df["residue_number"] == resi)
        & (het_df["residue_name"] == resn)
    ]
    if len(lig_df) == 0:
        return None
    lig_coords = lig_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)

    prot_coords = atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)

    # For each protein atom, distance to nearest ligand atom
    # Use broadcasting; protein atoms may be ~5K, ligand ~30 → tractable
    diff = prot_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    min_dists = dists.min(axis=1)
    near_mask = min_dists <= dist_cutoff
    if near_mask.sum() == 0:
        return None

    # Expand to whole residue: include ALL atoms of any residue with at least one near atom
    near_residues = atom_df[near_mask][["chain_id", "residue_number"]].drop_duplicates()
    pocket_df = atom_df.merge(near_residues, on=["chain_id", "residue_number"], how="inner")

    # Re-render as PDB
    pdb_out = PandasPdb()
    pdb_out.df["ATOM"] = pocket_df
    pdb_out.df["HETATM"] = lig_df  # include the ligand too, for documentation
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdb") as tf:
        tmp = tf.name
    pdb_out.to_pdb(path=tmp, records=["ATOM", "HETATM"])
    with open(tmp, "rb") as f:
        out_bytes = f.read()
    os.unlink(tmp)
    return out_bytes, f"{resn}_{chain}_{resi}_{natoms}atoms"


def process_one(pdb_code: str, out_root: Path) -> dict:
    pdb = pdb_code.lower()
    out_dir = out_root / pdb
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb}_pocket.pdb"
    if out_path.exists():
        return {"pdb_code": pdb, "status": "already_exists", "ligand": "", "pocket_size": 0}

    raw = fetch_pdb(pdb)
    if raw is None:
        return {"pdb_code": pdb, "status": "fetch_failed", "ligand": "", "pocket_size": 0}

    res = extract_pocket(raw, pdb)
    if res is None:
        return {"pdb_code": pdb, "status": "no_pocket_extracted", "ligand": "", "pocket_size": 0}
    pocket_bytes, lig_id = res
    out_path.write_bytes(pocket_bytes)
    # Quick atom count: re-parse pocket file for atom count
    n = sum(1 for line in pocket_bytes.splitlines() if line.startswith(b"ATOM"))
    return {"pdb_code": pdb, "status": "ok", "ligand": lig_id, "pocket_size": n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-mapping", required=True, type=Path)
    p.add_argument("--already-have", required=True, type=Path,
                   help="v2_target_node.parquet (uses has_pocket_pdb to skip already-have)")
    p.add_argument("--out-root", required=True, type=Path,
                   help="Where to write <pdb>/<pdb>_pocket.pdb files")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    mapping = json.loads(args.target_mapping.read_text())
    already = pl.read_parquet(args.already_have)
    have_pdb = set(already.filter(pl.col("has_pocket_pdb"))["pdb_code"].to_list())
    print(f"already have pockets for: {len(have_pdb)}")

    todo = [
        (target_name.lower(), pdb_code.lower())
        for uniprot, pdb_code, target_name in mapping
        if pdb_code.lower() not in have_pdb
    ]
    print(f"to fetch: {len(todo)}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(process_one, pdb, args.out_root): (tname, pdb)
                   for tname, pdb in todo}
        for fut in futures:
            tname, pdb = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"pdb_code": pdb, "status": f"exception: {e}",
                     "ligand": "", "pocket_size": 0}
            r["target_name"] = tname
            print(f"  {tname:8s} {pdb}  {r['status']:25s}  ligand={r['ligand']}  pocket={r['pocket_size']}atoms",
                  flush=True)
            results.append(r)

    df = pl.DataFrame(results).select(["target_name", "pdb_code", "status", "ligand", "pocket_size"])
    out_csv = args.out_root / "fetch_summary.csv"
    df.write_csv(out_csv)
    print()
    print(f"summary → {out_csv}")
    print(f"successful: {df.filter(pl.col('status')=='ok').shape[0]} of {df.shape[0]}")


if __name__ == "__main__":
    main()
