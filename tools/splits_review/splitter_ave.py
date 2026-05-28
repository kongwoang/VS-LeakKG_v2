"""AVE-minimised ligand splitter (Wallach 2018; LIT-PCBA's `remove_AVE_bias.py`).

Per-target genetic algorithm that minimises
    B(Va, Vi, Ta, Ti) = (AA - AI) + (II - IA)
where each term is the mean nearest-neighbour Tanimoto-presence at thresholds
D = {0.0, 0.1, ..., 1.0} between the named class subsets.

Policy (per the approved protocol):
    * iteration cap = 300 (hard)
    * N_MAX = 5000 actives per target; if exceeded, deterministically subsample
      using seed=2025 and **write the subset to subset_<target_id>.parquet** so
      that every other splitter on this target consumes the same subset.
    * report final B, drop %, GA runtime, "subsampled" flag

This is a faithful port of the AVE bias measure; the GA is a simple
swap-based optimiser following the description in Wallach & Heifets 2018.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import polars as pl

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

from .common import write_split, fold_quotas
from .schemas import hash_manifest_slice


THRESHOLDS = np.arange(0.0, 1.01, 0.1)
N_MAX_ACTIVES = 5000
ITER_CAP = 300


def fingerprints(smiles_list: list[str]) -> list[object]:
    if not HAS_RDKIT:
        raise SystemExit("RDKit is required for AVE splitter.")
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if s else None
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
                   if m is not None else None)
    return out


def tanimoto_max(a, B_list) -> float:
    if a is None or not B_list:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(a, [b for b in B_list if b is not None])
    return float(max(sims) if sims else 0.0)


def H_cumulative(query_fps, ref_fps) -> float:
    """Mean over thresholds d of fraction of query points with max-Tanimoto >= 1-d."""
    if not query_fps or not ref_fps:
        return 0.0
    max_sim = np.array([tanimoto_max(q, ref_fps) for q in query_fps])
    accum = 0.0
    for d in THRESHOLDS:
        # AVE uses *distance* threshold; with Tanimoto similarity, d = 1 - sim.
        accum += float(np.mean(max_sim >= (1.0 - d)))
    return accum / len(THRESHOLDS)


def ave_bias(va_fp, vi_fp, ta_fp, ti_fp) -> float:
    AA = H_cumulative(va_fp, ta_fp)
    AI = H_cumulative(va_fp, ti_fp)
    II = H_cumulative(vi_fp, ti_fp)
    IA = H_cumulative(vi_fp, ta_fp)
    return (AA - AI) + (II - IA)


def split_target_ave(slc: pl.DataFrame, seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    actives  = slc.filter(pl.col("label") == 1).to_dict(as_series=False)
    inactives = slc.filter(pl.col("label") == 0).to_dict(as_series=False)

    n_act = len(actives["smiles"])
    sub_flag = False
    if n_act > N_MAX_ACTIVES:
        sub_flag = True
        idx = rng.choice(n_act, size=N_MAX_ACTIVES, replace=False)
        actives = {k: [v[i] for i in idx] for k, v in actives.items()}
        n_act = N_MAX_ACTIVES

    fp_a = fingerprints(actives["smiles"])
    fp_i = fingerprints(inactives["smiles"])
    if not fp_a or not fp_i:
        return [], {"B": float("nan"), "iters": 0, "subsampled": sub_flag,
                    "dropped_pct": 0.0, "runtime_s": 0.0}

    # Initial random split: 80/10/10 of actives and inactives independently.
    def init_partition(n: int):
        n_tr, n_va, _ = fold_quotas(n)
        idx = np.arange(n); rng.shuffle(idx)
        fold = np.array(["test"] * n, dtype=object)
        fold[idx[:n_tr]]              = "train"
        fold[idx[n_tr:n_tr + n_va]]   = "val"
        return fold

    a_fold = init_partition(n_act)
    i_fold = init_partition(len(fp_i))

    def gather(fps, fold_arr, want):
        return [f for f, fl in zip(fps, fold_arr) if fl == want]

    def current_B():
        ta = gather(fp_a, a_fold, "train"); va = gather(fp_a, a_fold, "test")
        ti = gather(fp_i, i_fold, "train"); vi = gather(fp_i, i_fold, "test")
        return ave_bias(va, vi, ta, ti)

    start = time.time()
    B = current_B()
    best_B = B
    for it in range(ITER_CAP):
        # Try a random pairwise swap (one active swap + one inactive swap).
        ai_tr = np.where(a_fold == "train")[0]
        ai_te = np.where(a_fold == "test")[0]
        ii_tr = np.where(i_fold == "train")[0]
        ii_te = np.where(i_fold == "test")[0]
        if not (len(ai_tr) and len(ai_te) and len(ii_tr) and len(ii_te)):
            break
        ka, kb = int(rng.choice(ai_tr)), int(rng.choice(ai_te))
        ja, jb = int(rng.choice(ii_tr)), int(rng.choice(ii_te))
        a_fold[ka], a_fold[kb] = a_fold[kb], a_fold[ka]
        i_fold[ja], i_fold[jb] = i_fold[jb], i_fold[ja]
        new_B = current_B()
        if abs(new_B) < abs(best_B):
            best_B = new_B
        else:
            # Reject
            a_fold[ka], a_fold[kb] = a_fold[kb], a_fold[ka]
            i_fold[ja], i_fold[jb] = i_fold[jb], i_fold[ja]
        if abs(best_B) < 0.01:
            break

    runtime = time.time() - start
    rows: list[dict] = []
    for src, fold_arr in ((actives, a_fold), (inactives, i_fold)):
        for k in range(len(src["example_id"])):
            rows.append({
                "example_id": src["example_id"][k],
                "target_id":  src["target_id"][k],
                "ligand_id":  src["ligand_id"][k],
                "label":      int(src["label"][k]),
                "fold":       str(fold_arr[k]),
                "input_hash": "ave_per_target",
            })
    stats = {
        "B": float(best_B),
        "iters": it + 1 if "it" in locals() else 0,
        "subsampled": sub_flag,
        "dropped_pct": 0.0,
        "runtime_s": runtime,
    }
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--subset-dir", required=True, type=Path,
                    help="Where to write subset_<target_id>.parquet when AVE "
                         "subsamples. Other splitters must read these files.")
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--stats-out", required=True, type=Path,
                    help="CSV summary of per-target AVE GA outcomes.")
    ap.add_argument("--seed",      default=2025, type=int)
    args = ap.parse_args()
    if args.mode != "A":
        raise SystemExit("AVE is per-target (Mode A) only.")

    manifest = pl.read_parquet(args.manifest)
    args.subset_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    stats_rows: list[dict] = []
    for tid in sorted(manifest["target_id"].unique().to_list()):
        slc = manifest.filter(pl.col("target_id") == tid)
        rows, st = split_target_ave(slc, seed=args.seed)
        all_rows.extend(rows)
        if st["subsampled"]:
            # Write the subset manifest so other splitters see the same input.
            subset = pl.DataFrame([
                {"example_id": r["example_id"], "target_id": r["target_id"],
                 "ligand_id":  r["ligand_id"],  "label":     r["label"]}
                for r in rows
            ])
            # Re-attach manifest columns so downstream splitters keep schema.
            subset = subset.join(manifest.drop(["label"]),
                                 on=["example_id", "target_id", "ligand_id"],
                                 how="left")
            subset.write_parquet(args.subset_dir / f"subset_{tid}.parquet")
        stats_rows.append({"target_id": tid, **st})

    write_split(all_rows, args.out, input_hash="ave_per_target")
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(stats_rows).write_csv(args.stats_out)
    print(f"AVE done; stats at {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
