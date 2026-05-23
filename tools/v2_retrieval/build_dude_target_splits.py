"""Build target-level splits for the DUD-E retrieval audit (Group C).

Five regimes, each writing outputs/v2_retrieval/splits/dude/<regime>.parquet:

  target-random   — uniform-random target partition (control)
  target-clean    — entire Pfam-family (≥40% seq ID cluster) goes to one side
  active-clean    — targets sharing any active ligand stay on the same side
  scaffold-clean  — targets sharing any Bemis-Murcko scaffold among actives
                    stay on the same side
  dual-clean      — target-clean ∧ active-clean (intersect the constraints)

We restrict to targets with extracted pocket PDBs (--require-pocket) so the
retrieval evaluator has structures for every target.

Test fraction is approximate: clean regimes only honor it if the cluster
sizes allow. Reports actual fraction in the output stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def union_find_clusters(n: int, edges: list[tuple[int, int]]) -> dict[int, int]:
    """Single-linkage cluster ids for n items given edges between them."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return {i: find(i) for i in range(n)}


def partition_clusters(
    target_to_cluster: dict[str, int],
    target_to_size: dict[str, int],
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    """Greedy by-cluster partition. Assigns whole clusters to train or test
    so the realized test fraction (weighted by target size = #rows) is
    close to the requested test_frac.
    """
    rng = np.random.default_rng(seed)

    # Aggregate cluster sizes (sum of constituent target row-counts).
    cluster_targets: dict[int, list[str]] = {}
    for tgt, cl in target_to_cluster.items():
        cluster_targets.setdefault(cl, []).append(tgt)
    cluster_size = {cl: sum(target_to_size[t] for t in members) for cl, members in cluster_targets.items()}

    total = sum(cluster_size.values())
    target_test = total * test_frac

    # Shuffle clusters, fit greedily.
    order = list(cluster_size.keys())
    rng.shuffle(order)

    test_clusters: set[int] = set()
    running = 0
    for cl in order:
        if running + cluster_size[cl] <= target_test or running == 0:
            test_clusters.add(cl)
            running += cluster_size[cl]
        if running >= target_test:
            break

    assignment: dict[str, str] = {}
    for cl, members in cluster_targets.items():
        for t in members:
            assignment[t] = "test" if cl in test_clusters else "train"
    return assignment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-node", required=True, type=Path)
    p.add_argument("--active-of-target", required=True, type=Path)
    p.add_argument("--target-in-family", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--test-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-pocket", action="store_true",
                   help="restrict to targets with an extracted pocket PDB")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = pl.read_parquet(args.target_node)
    if args.require_pocket:
        targets = targets.filter(pl.col("has_pocket_pdb"))
    target_ids = targets["target_id"].to_list()
    print(f"targets in scope: {len(target_ids)}")

    # "size" for fitting test fraction = #actives + #decoys
    target_to_size = dict(
        zip(targets["target_id"], (targets["n_actives"] + targets["n_decoys"]).to_list())
    )

    # ---- Regime 1: target-random ----
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(target_ids))
    sizes = np.array([target_to_size[target_ids[i]] for i in perm], dtype=np.int64)
    cum = np.cumsum(sizes)
    cutoff = int(cum[-1] * args.test_frac)
    test_mask = cum <= cutoff
    if not test_mask.any():
        test_mask[0] = True
    assign_random = {target_ids[perm[i]]: ("test" if test_mask[i] else "train") for i in range(len(target_ids))}

    # ---- Regime 2: target-clean (family-disjoint) ----
    fam_df = pl.read_parquet(args.target_in_family).filter(pl.col("target_id").is_in(target_ids))
    name_to_idx = {n: i for i, n in enumerate(target_ids)}
    fam_to_cluster: dict[str, int] = {}
    target_to_cluster_fam: dict[str, int] = {}
    for row in fam_df.iter_rows(named=True):
        fam = row["family_id"]
        if fam not in fam_to_cluster:
            fam_to_cluster[fam] = len(fam_to_cluster)
        target_to_cluster_fam[row["target_id"]] = fam_to_cluster[fam]
    # any target missing a family becomes its own singleton cluster
    next_cl = len(fam_to_cluster)
    for t in target_ids:
        if t not in target_to_cluster_fam:
            target_to_cluster_fam[t] = next_cl
            next_cl += 1
    assign_target_clean = partition_clusters(target_to_cluster_fam, target_to_size, args.test_frac, args.seed)

    # ---- Regimes 3 & 4: active-clean and scaffold-clean ----
    edges_df = pl.read_parquet(args.active_of_target).filter(pl.col("target_id").is_in(target_ids))

    def cluster_by_shared_attr(attr_col: str) -> dict[str, int]:
        """Targets sharing any value of attr_col are linked in one cluster."""
        edges_local = edges_df.select(["target_id", attr_col]).unique()
        attr_to_targets: dict[str, set[str]] = {}
        for row in edges_local.iter_rows(named=True):
            attr_to_targets.setdefault(row[attr_col], set()).add(row["target_id"])
        # Build UF over targets
        n = len(target_ids)
        edges: list[tuple[int, int]] = []
        for _, ts in attr_to_targets.items():
            if len(ts) < 2:
                continue
            ts_list = sorted(ts)
            anchor = name_to_idx.get(ts_list[0])
            if anchor is None:
                continue
            for other in ts_list[1:]:
                idx = name_to_idx.get(other)
                if idx is None:
                    continue
                edges.append((anchor, idx))
        cl = union_find_clusters(n, edges)
        return {target_ids[i]: c for i, c in cl.items()}

    target_to_cluster_active = cluster_by_shared_attr("smiles_canonical")
    target_to_cluster_scaffold = cluster_by_shared_attr("scaffold_smiles")

    assign_active_clean = partition_clusters(target_to_cluster_active, target_to_size, args.test_frac, args.seed)
    assign_scaffold_clean = partition_clusters(target_to_cluster_scaffold, target_to_size, args.test_frac, args.seed)

    # ---- Regime 5: dual-clean (intersect family + active clusters) ----
    # Two targets are linked if they share a family OR share an active ligand.
    n = len(target_ids)
    edges_dual: list[tuple[int, int]] = []
    # family edges
    fam_to_t: dict[int, list[str]] = {}
    for t, c in target_to_cluster_fam.items():
        fam_to_t.setdefault(c, []).append(t)
    for ts in fam_to_t.values():
        if len(ts) < 2:
            continue
        anchor = name_to_idx[ts[0]]
        for other in ts[1:]:
            edges_dual.append((anchor, name_to_idx[other]))
    # active-share edges
    act_to_t: dict[int, list[str]] = {}
    for t, c in target_to_cluster_active.items():
        act_to_t.setdefault(c, []).append(t)
    for ts in act_to_t.values():
        if len(ts) < 2:
            continue
        anchor = name_to_idx[ts[0]]
        for other in ts[1:]:
            edges_dual.append((anchor, name_to_idx[other]))
    cl_dual = union_find_clusters(n, edges_dual)
    target_to_cluster_dual = {target_ids[i]: c for i, c in cl_dual.items()}
    assign_dual_clean = partition_clusters(target_to_cluster_dual, target_to_size, args.test_frac, args.seed)

    # ---- Write all five ----
    regime_assignments = {
        "target_random": assign_random,
        "target_clean": assign_target_clean,
        "active_clean": assign_active_clean,
        "scaffold_clean": assign_scaffold_clean,
        "dual_clean": assign_dual_clean,
    }
    summary = []
    for regime, assign in regime_assignments.items():
        rows = [
            {"target_id": t, "partition": assign[t]}
            for t in target_ids
        ]
        out = args.out_dir / f"{regime}.parquet"
        pl.DataFrame(rows).write_parquet(out)
        n_test = sum(1 for v in assign.values() if v == "test")
        n_train = sum(1 for v in assign.values() if v == "train")
        test_rows = sum(target_to_size[t] for t, v in assign.items() if v == "test")
        train_rows = sum(target_to_size[t] for t, v in assign.items() if v == "train")
        summary.append({
            "regime": regime,
            "n_train_targets": n_train,
            "n_test_targets": n_test,
            "n_train_rows": train_rows,
            "n_test_rows": test_rows,
            "test_target_frac": n_test / len(target_ids),
            "test_row_frac": test_rows / (train_rows + test_rows),
        })

    sum_df = pl.DataFrame(summary)
    print(sum_df)
    sum_df.write_csv(args.out_dir / "summary.csv")
    print(f"\nwrote splits + summary under {args.out_dir}")


if __name__ == "__main__":
    main()
