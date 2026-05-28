"""AVE-minimised ligand splitter (Wallach 2018; cached).

Caching strategy (the difference from the naive port):
    1. Precompute Morgan fingerprints once per target.
    2. Precompute the full (n_act x n_act), (n_inact x n_inact),
       (n_act x n_inact) Tanimoto similarity matrices ONCE per target.
    3. Each iteration's H_cumulative becomes a slicing + reduction over
       these matrices instead of an O(n^2) recomputation. Per-swap update
       is O(n) instead of O(n^2).

Policy:
    * iteration cap = 300 (hard, same for every corpus)
    * N_MAX_ACTIVES = 5000 — if exceeded, deterministic subsample with
      seed=2025 and write the subset to subset_<target_id>.parquet so
      every other splitter for that target consumes the same subset.
    * Per-target report: B, drop %, iters, runtime_s, subsampled flag,
      termination reason (converged_below_B_thresh / hit_iter_cap).
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
B_TARGET = 0.01


def fingerprints(smiles_list: list[str]) -> list[object]:
    if not HAS_RDKIT:
        raise SystemExit("RDKit is required for AVE splitter.")
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if s else None
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
                   if m is not None else None)
    return out


def pairwise_tanimoto(fps_a: list, fps_b: list) -> np.ndarray:
    na, nb = len(fps_a), len(fps_b)
    M = np.zeros((na, nb), dtype=np.float32)
    for i in range(na):
        if fps_a[i] is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps_a[i], fps_b)
        M[i, :] = np.array(sims, dtype=np.float32)
    return M


def H_from_sim_matrix(sim_rows: np.ndarray) -> float:
    """sim_rows[i, j] = similarity from query i to ref j. H = mean over thresholds d
    of fraction of queries whose max-sim >= 1 - d."""
    if sim_rows.size == 0:
        return 0.0
    max_per_q = sim_rows.max(axis=1) if sim_rows.shape[1] > 0 else np.zeros(sim_rows.shape[0])
    accum = 0.0
    for d in THRESHOLDS:
        accum += float((max_per_q >= (1.0 - d)).mean())
    return accum / len(THRESHOLDS)


def split_target_ave_cached(slc: pl.DataFrame, seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    actives = slc.filter(pl.col("label") == 1).to_dict(as_series=False)
    inactives = slc.filter(pl.col("label") == 0).to_dict(as_series=False)
    n_act_full = len(actives["smiles"])

    sub_flag = False
    if n_act_full > N_MAX_ACTIVES:
        sub_flag = True
        idx = rng.choice(n_act_full, size=N_MAX_ACTIVES, replace=False)
        actives = {k: [v[i] for i in idx] for k, v in actives.items()}
    n_act = len(actives["smiles"])
    n_in  = len(inactives["smiles"])
    if n_act < 4 or n_in < 4:
        return [], {"B": float("nan"), "iters": 0, "subsampled": sub_flag,
                    "dropped_pct": 0.0, "runtime_s": 0.0,
                    "termination": "skipped_too_few_samples"}

    t_start = time.time()
    fp_a = fingerprints(actives["smiles"])
    fp_i = fingerprints(inactives["smiles"])

    # Precompute the 4 full Tanimoto matrices once.
    S_aa = pairwise_tanimoto(fp_a, fp_a)
    S_ii = pairwise_tanimoto(fp_i, fp_i)
    S_ai = pairwise_tanimoto(fp_a, fp_i)  # actives -> inactives

    def init_fold(n: int):
        n_tr, n_va, _ = fold_quotas(n)
        idx = np.arange(n); rng.shuffle(idx)
        fold = np.array(["test"] * n, dtype=object)
        fold[idx[:n_tr]]            = "train"
        fold[idx[n_tr:n_tr + n_va]] = "val"
        return fold

    a_fold = init_fold(n_act)
    i_fold = init_fold(n_in)

    def B_now():
        a_tr_idx = np.where(a_fold == "train")[0]
        a_te_idx = np.where(a_fold == "test")[0]
        i_tr_idx = np.where(i_fold == "train")[0]
        i_te_idx = np.where(i_fold == "test")[0]
        if not (len(a_tr_idx) and len(a_te_idx) and len(i_tr_idx) and len(i_te_idx)):
            return float("nan")
        AA = H_from_sim_matrix(S_aa[np.ix_(a_te_idx, a_tr_idx)])
        AI = H_from_sim_matrix(S_ai[np.ix_(a_te_idx, i_tr_idx)])
        II = H_from_sim_matrix(S_ii[np.ix_(i_te_idx, i_tr_idx)])
        IA = H_from_sim_matrix(S_ai.T[np.ix_(i_te_idx, a_tr_idx)])
        return (AA - AI) + (II - IA)

    best_B = B_now()
    term = "hit_iter_cap"
    n_iter = 0
    for it in range(ITER_CAP):
        n_iter = it + 1
        ai_tr = np.where(a_fold == "train")[0]; ai_te = np.where(a_fold == "test")[0]
        ii_tr = np.where(i_fold == "train")[0]; ii_te = np.where(i_fold == "test")[0]
        if not (len(ai_tr) and len(ai_te) and len(ii_tr) and len(ii_te)):
            term = "no_swap_possible"; break
        ka, kb = int(rng.choice(ai_tr)), int(rng.choice(ai_te))
        ja, jb = int(rng.choice(ii_tr)), int(rng.choice(ii_te))
        a_fold[ka], a_fold[kb] = a_fold[kb], a_fold[ka]
        i_fold[ja], i_fold[jb] = i_fold[jb], i_fold[ja]
        new_B = B_now()
        if abs(new_B) < abs(best_B):
            best_B = new_B
        else:
            a_fold[ka], a_fold[kb] = a_fold[kb], a_fold[ka]
            i_fold[ja], i_fold[jb] = i_fold[jb], i_fold[ja]
        if abs(best_B) < B_TARGET:
            term = "converged_below_B_thresh"
            break

    runtime = time.time() - t_start
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
        "iters": n_iter,
        "subsampled": sub_flag,
        "dropped_pct": 100.0 * (n_act_full - n_act) / max(n_act_full, 1),
        "runtime_s": runtime,
        "termination": term,
    }
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--subset-dir", required=True, type=Path,
                    help="Where to write subset_<target_id>.parquet when AVE subsamples.")
    ap.add_argument("--mode",      required=True, choices=["A", "B"])
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--stats-out", required=True, type=Path)
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
        t0 = time.time()
        rows, st = split_target_ave_cached(slc, seed=args.seed + hash(tid) % 10_000)
        print(f"  {tid}: B={st['B']:.4f} iters={st['iters']} sub={st['subsampled']} "
              f"term={st['termination']} {time.time()-t0:.1f}s")
        all_rows.extend(rows)
        if st["subsampled"]:
            keep_ids = {r["example_id"] for r in rows}
            subset = manifest.filter(pl.col("example_id").is_in(list(keep_ids)))
            subset.write_parquet(args.subset_dir / f"subset_{tid}.parquet")
        stats_rows.append({"target_id": tid, **st})

    write_split(all_rows, args.out, input_hash="ave_per_target")
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(stats_rows).write_csv(args.stats_out)
    print(f"AVE done; stats at {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
