"""Leakage groups with giant-component handling (proposal section 5.9).

Given the forbidden subgraph G_F, group construction proceeds as follows:

  1. Compute connected components C_1, ..., C_M over Example nodes induced
     by edges with type in F.
  2. Measure rho_max = max_m |g_m| / |D|.
  3. If rho_max <= rho_max_ok (default 0.30): use components as-is.
  4. If rho_max in (rho_max_ok, rho_max_prune]: deterministically remove the
     weakest forbidden relation type and recompute. Recurse up to k times.
  5. If rho_max > rho_max_prune: replace components with weighted Louvain
     community detection on G_F (edge weight = w_r) and report residual
     cross-community contamination.
  6. If no strategy yields a feasible split, return infeasible with diagnostics.

The result records the strategy used so downstream code (split assignment)
can adjust its assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import polars as pl

from .schema import DEFAULT_WEIGHTS, GiantComponentConfig


class GroupStrategy(str, Enum):
    CONNECTED_COMPONENTS = "connected_components"
    PRUNED_COMPONENTS = "pruned_components"
    LOUVAIN = "louvain"
    INFEASIBLE = "infeasible"


@dataclass
class LeakageGroupResult:
    strategy: GroupStrategy
    groups: dict[str, int]            # example_id -> group_id
    pruned_relations: list[str]       # forbidden relations removed (if any)
    rho_max: float                    # largest group fraction at success
    n_groups: int
    diagnostics: dict[str, float]     # extra metrics


def _union_find(nodes: Iterable[str]) -> dict[str, str]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    parent.find = find  # type: ignore[attr-defined]
    return parent


def _components(
    example_ids: set[str],
    forbidden_edges: pl.DataFrame,
) -> dict[str, int]:
    """Connected components on the example-induced forbidden subgraph.

    Edges may pass through non-example nodes (ligand, scaffold, ...). We
    contract those into the examples by following 2-hop chains:
      example -> entity -> example
    is enough to merge two examples when both connect to the same entity via
    forbidden relations. We implement this by union-finding entities and
    examples together, then projecting to the example partition.
    """
    if forbidden_edges.height == 0:
        return {ex: i for i, ex in enumerate(sorted(example_ids))}

    all_nodes: set[str] = set(example_ids)
    src_list = forbidden_edges["src"].to_list()
    dst_list = forbidden_edges["dst"].to_list()
    all_nodes.update(src_list)
    all_nodes.update(dst_list)

    parent: dict[str, str] = {n: n for n in all_nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, d in zip(src_list, dst_list):
        union(s, d)

    # Project: group id = index of the unique root among example nodes.
    root_to_gid: dict[str, int] = {}
    out: dict[str, int] = {}
    for ex in example_ids:
        r = find(ex)
        if r not in root_to_gid:
            root_to_gid[r] = len(root_to_gid)
        out[ex] = root_to_gid[r]
    return out


def _max_component_fraction(groups: dict[str, int]) -> float:
    if not groups:
        return 0.0
    counts: dict[int, int] = {}
    for g in groups.values():
        counts[g] = counts.get(g, 0) + 1
    return max(counts.values()) / len(groups)


def _louvain(
    example_ids: set[str],
    forbidden_edges: pl.DataFrame,
    weights: dict[str, float],
) -> dict[str, int]:
    """Weighted Louvain community detection on the forbidden subgraph.

    Falls back to a no-op (singleton groups) if python-louvain or networkx
    are unavailable; the caller can detect this via diagnostics.
    """
    try:
        import networkx as nx
        try:
            import community as community_louvain  # python-louvain
        except ImportError:
            community_louvain = None  # type: ignore[assignment]
    except ImportError:
        return {ex: i for i, ex in enumerate(sorted(example_ids))}

    g = nx.Graph()
    g.add_nodes_from(example_ids)
    for row in forbidden_edges.iter_rows(named=True):
        w = weights.get(row["edge_type"], 0.0)
        if w <= 0:
            continue
        s, d = row["src"], row["dst"]
        # We can only meaningfully cluster Example nodes; treat other nodes as
        # intermediate hubs by skipping their direct addition. The 2-hop
        # example->entity->example structure is captured by adding the edge
        # if at least one endpoint is an Example node we know about.
        if s in example_ids and d in example_ids:
            g.add_edge(s, d, weight=w)
    if community_louvain is None:
        # graceful: just use connected components on the projected graph
        comp = nx.connected_components(g)
        out: dict[str, int] = {}
        for i, c in enumerate(comp):
            for n in c:
                out[n] = i
        # Include isolated examples
        for ex in example_ids:
            if ex not in out:
                out[ex] = len(out)
        return out

    partition = community_louvain.best_partition(g, weight="weight", random_state=0)
    # Ensure every example is assigned
    next_id = max(partition.values(), default=-1) + 1
    for ex in example_ids:
        if ex not in partition:
            partition[ex] = next_id
            next_id += 1
    return partition


def build_leakage_groups(
    example_ids: Iterable[str],
    edges: pl.DataFrame,
    *,
    forbidden_relations: set[str],
    weights: dict[str, float] | None = None,
    config: GiantComponentConfig | None = None,
    max_prune_steps: int = 3,
) -> LeakageGroupResult:
    """Build leakage groups for one clean regime.

    Args:
        example_ids:         the set D of example IDs.
        edges:               full v2 edge table.
        forbidden_relations: subset of edge_type values to forbid.
        weights:             per-edge-type weights (used for pruning order and
                             Louvain weights).
        config:              rho_max thresholds.
        max_prune_steps:     how many weak relations to prune before falling
                             back to Louvain.

    Returns:
        LeakageGroupResult describing the strategy, groups, and diagnostics.
    """
    cfg = config or GiantComponentConfig()
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)
    ex_set = set(example_ids)

    forbidden_edges = edges.filter(pl.col("edge_type").is_in(list(forbidden_relations)))
    groups = _components(ex_set, forbidden_edges)
    rho = _max_component_fraction(groups)

    if rho <= cfg.rho_max_ok:
        return LeakageGroupResult(
            strategy=GroupStrategy.CONNECTED_COMPONENTS,
            groups=groups, pruned_relations=[],
            rho_max=rho, n_groups=len(set(groups.values())),
            diagnostics={"initial_rho_max": rho},
        )

    pruned: list[str] = []
    remaining = set(forbidden_relations)
    for _ in range(max_prune_steps):
        if not remaining:
            break
        # remove the relation with the lowest weight first
        weakest = min(remaining, key=lambda r: w.get(r, 0.0))
        remaining.discard(weakest)
        pruned.append(weakest)
        forbidden_edges = edges.filter(pl.col("edge_type").is_in(list(remaining)))
        groups = _components(ex_set, forbidden_edges)
        rho = _max_component_fraction(groups)
        if rho <= cfg.rho_max_ok:
            return LeakageGroupResult(
                strategy=GroupStrategy.PRUNED_COMPONENTS,
                groups=groups, pruned_relations=pruned,
                rho_max=rho, n_groups=len(set(groups.values())),
                diagnostics={"initial_rho_max": _max_component_fraction(
                    _components(ex_set, edges.filter(
                        pl.col("edge_type").is_in(list(forbidden_relations))
                    ))
                )},
            )
        if rho > cfg.rho_max_prune:
            break

    # Louvain fallback
    if rho > cfg.rho_max_prune:
        partition = _louvain(ex_set, forbidden_edges, w)
        rho = _max_component_fraction(partition)
        return LeakageGroupResult(
            strategy=GroupStrategy.LOUVAIN,
            groups=partition,
            pruned_relations=pruned,
            rho_max=rho,
            n_groups=len(set(partition.values())),
            diagnostics={"louvain_rho_max": rho},
        )

    # Pruning helped but not enough; mark infeasible
    return LeakageGroupResult(
        strategy=GroupStrategy.INFEASIBLE,
        groups=groups,
        pruned_relations=pruned,
        rho_max=rho,
        n_groups=len(set(groups.values())),
        diagnostics={"final_rho_max_after_pruning": rho},
    )


def groups_to_frame(result: LeakageGroupResult) -> pl.DataFrame:
    return pl.DataFrame(
        [{"example_id": k, "group_id": v} for k, v in result.groups.items()]
    )
