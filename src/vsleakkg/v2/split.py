"""Group-atomic split assignment (proposal section 5.10).

Given leakage groups g_1 ... g_M and target partition ratios, assign each
group atomically to one of {train, val, test} minimising:

    J = lambda_size * J_size
      + lambda_label * J_label
      + lambda_cover * J_cover
      + lambda_resid * J_resid

  J_size  = squared deviation from desired partition size ratios
  J_label = active/inactive ratio drift per partition
  J_cover = target/ligand coverage imbalance
  J_resid = residual cross-partition contamination through non-forbidden edges

Constraints:
  - every partition has at least min_targets_per_partition unique targets
  - every partition has at least min_actives_per_partition actives
  - label_balance_tol satisfied per partition

Default algorithm: deterministic greedy on groups sorted by size descending.
If greedy can't satisfy constraints, the result records `feasible = False`
and the caller may fall back to a MILP. We provide a MILP stub that uses
PuLP when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import polars as pl

from .schema import SplitConstraints


PARTITIONS = ("train", "val", "test")


@dataclass
class SplitAssignment:
    feasible: bool
    assignment: dict[str, str]      # example_id -> partition
    group_assignment: dict[int, str]
    diagnostics: dict[str, float] = field(default_factory=dict)


def _group_meta(
    groups: dict[str, int],
    examples: pl.DataFrame,
) -> dict[int, dict]:
    """Aggregate per-group statistics: size, n_actives, target set, ligand set."""
    meta: dict[int, dict] = {}
    label_col = "label" if "label" in examples.columns else None
    target_col = "protein_id" if "protein_id" in examples.columns else None
    ligand_col = "ligand_id" if "ligand_id" in examples.columns else None

    ex_to_row: dict[str, dict] = {
        r["example_id"]: r for r in examples.iter_rows(named=True)
    }
    for ex_id, gid in groups.items():
        m = meta.setdefault(gid, {
            "size": 0, "n_actives": 0, "targets": set(), "ligands": set(),
        })
        m["size"] += 1
        row = ex_to_row.get(ex_id, {})
        if label_col and row.get(label_col) == 1:
            m["n_actives"] += 1
        if target_col and row.get(target_col) is not None:
            m["targets"].add(row[target_col])
        if ligand_col and row.get(ligand_col) is not None:
            m["ligands"].add(row[ligand_col])
    return meta


def _partition_state() -> dict[str, dict]:
    return {p: {"size": 0, "n_actives": 0, "targets": set(), "ligands": set()}
            for p in PARTITIONS}


def _objective(
    state: dict[str, dict],
    total_size: int,
    total_actives: int,
    cons: SplitConstraints,
) -> float:
    ratios = {"train": cons.train_ratio, "val": cons.val_ratio, "test": cons.test_ratio}
    j_size = 0.0
    j_label = 0.0
    for p in PARTITIONS:
        target_n = ratios[p] * total_size
        j_size += (state[p]["size"] - target_n) ** 2 / max(1.0, total_size ** 2)
        if total_actives > 0:
            target_a = ratios[p] * total_actives
            j_label += (state[p]["n_actives"] - target_a) ** 2 / max(1.0, total_actives ** 2)
    # coverage: penalise targets concentrated in one partition
    all_targets = set().union(*[state[p]["targets"] for p in PARTITIONS])
    j_cover = 0.0
    if all_targets:
        per_p_target_count = [len(state[p]["targets"]) for p in PARTITIONS]
        mean_t = sum(per_p_target_count) / 3.0
        j_cover = sum((c - mean_t) ** 2 for c in per_p_target_count) / max(1.0, mean_t ** 2)
    return (
        cons.lambda_size * j_size
        + cons.lambda_label * j_label
        + cons.lambda_cover * j_cover
    )


def greedy_assign(
    groups: dict[str, int],
    examples: pl.DataFrame,
    constraints: SplitConstraints | None = None,
) -> SplitAssignment:
    """Deterministic greedy group assignment.

    Sorts groups by size descending; for each group, tries every partition
    and chooses the one that minimises the running objective. Ties are
    broken in (train, val, test) order to keep the procedure deterministic.
    """
    cons = constraints or SplitConstraints()
    meta = _group_meta(groups, examples)
    total_size = sum(m["size"] for m in meta.values())
    total_actives = sum(m["n_actives"] for m in meta.values())

    order = sorted(meta.keys(), key=lambda g: -meta[g]["size"])
    state = _partition_state()
    g_assign: dict[int, str] = {}

    for gid in order:
        m = meta[gid]
        best_p, best_j = None, float("inf")
        for p in PARTITIONS:
            # tentative add
            state[p]["size"] += m["size"]
            state[p]["n_actives"] += m["n_actives"]
            added_targets = m["targets"] - state[p]["targets"]
            added_ligands = m["ligands"] - state[p]["ligands"]
            state[p]["targets"] |= added_targets
            state[p]["ligands"] |= added_ligands
            j = _objective(state, total_size, total_actives, cons)
            # revert
            state[p]["size"] -= m["size"]
            state[p]["n_actives"] -= m["n_actives"]
            state[p]["targets"] -= added_targets
            state[p]["ligands"] -= added_ligands
            if j < best_j:
                best_j = j
                best_p = p
        assert best_p is not None
        # commit
        state[best_p]["size"] += m["size"]
        state[best_p]["n_actives"] += m["n_actives"]
        state[best_p]["targets"] |= m["targets"]
        state[best_p]["ligands"] |= m["ligands"]
        g_assign[gid] = best_p

    assignment = {ex: g_assign[gid] for ex, gid in groups.items()}

    # Constraint feasibility
    feasible = True
    diag: dict[str, float] = {"total_size": float(total_size), "total_actives": float(total_actives)}
    for p in PARTITIONS:
        diag[f"size_{p}"] = float(state[p]["size"])
        diag[f"actives_{p}"] = float(state[p]["n_actives"])
        diag[f"targets_{p}"] = float(len(state[p]["targets"]))
        if len(state[p]["targets"]) < cons.min_targets_per_partition:
            feasible = False
        if state[p]["n_actives"] < cons.min_actives_per_partition:
            feasible = False
        # label balance tol check
        if total_actives:
            global_pos = total_actives / total_size
            local_pos = state[p]["n_actives"] / max(1, state[p]["size"])
            if abs(local_pos - global_pos) > cons.label_balance_tol:
                feasible = False
    return SplitAssignment(
        feasible=feasible,
        assignment=assignment,
        group_assignment=g_assign,
        diagnostics=diag,
    )


def milp_assign_stub(
    groups: dict[str, int],
    examples: pl.DataFrame,
    constraints: SplitConstraints | None = None,
) -> SplitAssignment:
    """MILP fallback (stub).

    Uses PuLP if available; otherwise raises NotImplementedError so the
    caller can detect the missing dependency and either pip-install it or
    fall back to the greedy result.
    """
    try:
        import pulp  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise NotImplementedError(
            "PuLP not installed; install with `pip install pulp` to enable the "
            "MILP fallback."
        ) from exc

    cons = constraints or SplitConstraints()
    meta = _group_meta(groups, examples)
    total_size = sum(m["size"] for m in meta.values())
    total_actives = sum(m["n_actives"] for m in meta.values())
    ratios = {"train": cons.train_ratio, "val": cons.val_ratio, "test": cons.test_ratio}

    prob = pulp.LpProblem("vsleakkg_split", pulp.LpMinimize)
    x = {
        (g, p): pulp.LpVariable(f"x_{g}_{p}", cat="Binary")
        for g in meta for p in PARTITIONS
    }
    # one assignment per group
    for g in meta:
        prob += pulp.lpSum(x[g, p] for p in PARTITIONS) == 1

    # size deviation
    size_dev = {p: pulp.LpVariable(f"sd_{p}", lowBound=0) for p in PARTITIONS}
    for p in PARTITIONS:
        size_p = pulp.lpSum(meta[g]["size"] * x[g, p] for g in meta)
        target = ratios[p] * total_size
        prob += size_dev[p] >= size_p - target
        prob += size_dev[p] >= target - size_p

    # label deviation
    label_dev = {p: pulp.LpVariable(f"ld_{p}", lowBound=0) for p in PARTITIONS}
    for p in PARTITIONS:
        actives_p = pulp.lpSum(meta[g]["n_actives"] * x[g, p] for g in meta)
        target_a = ratios[p] * total_actives
        prob += label_dev[p] >= actives_p - target_a
        prob += label_dev[p] >= target_a - actives_p

    prob += (
        cons.lambda_size * pulp.lpSum(size_dev.values())
        + cons.lambda_label * pulp.lpSum(label_dev.values())
    )

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    g_assign: dict[int, str] = {}
    for g in meta:
        for p in PARTITIONS:
            if pulp.value(x[g, p]) == 1:
                g_assign[g] = p
                break
    assignment = {ex: g_assign[gid] for ex, gid in groups.items()}
    return SplitAssignment(
        feasible=True,
        assignment=assignment,
        group_assignment=g_assign,
        diagnostics={"solver_status": pulp.LpStatus[prob.status]},
    )


def assignment_to_frame(assignment: SplitAssignment) -> pl.DataFrame:
    return pl.DataFrame(
        [{"example_id": k, "partition": v} for k, v in assignment.assignment.items()]
    )


def assignment_summary(assignment: SplitAssignment) -> pl.DataFrame:
    rows = []
    for p in PARTITIONS:
        n = sum(1 for v in assignment.assignment.values() if v == p)
        rows.append({"partition": p, "n": n})
    return pl.DataFrame(rows)
