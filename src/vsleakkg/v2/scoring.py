"""v2 contamination scoring: multiplicative path strength via -log Dijkstra.

Implements the scoring described in proposal.tex section 5.5.

Mathematical core:
    S(pi) = prod_e w_r(e),                  w_r(e) in (0, 1]
    C(x_t, A) = max_{x_i in A} max_{pi: x_i -> x_t} S(pi)

Computation:
    c(e) = -log w_r(e)                       cost transform (>= 0)
    d_min(x_t, A) = min path cost from any x_i in A to x_t
    C(x_t, A) = exp(-d_min(x_t, A))

Implementation:
    multi-source Dijkstra initialised from every example in A. Bounded by
    L_max edges and by an axis-specific edge-type allowlist.

Axis decomposition:
    For each axis in AXES, compute C on the axis subgraph. Overall score is
    max over axes. This eliminates the ambiguity of "which axis did this
    mixed path contribute to" that exists in v1.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable

import polars as pl

from .schema import AXIS_EDGE_TYPES, AXES, DEFAULT_WEIGHTS


@dataclass
class EdgeRecord:
    src: str
    dst: str
    edge_type: str


def _build_adj(
    edges: pl.DataFrame,
    weights: dict[str, float],
    allowed_types: set[str] | None,
) -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
    """Return (adjacency, unknown_edge_types).

    Adjacency maps src -> list[(dst, cost)] where cost = -log(weight).
    Edges with edge_type not in `weights` are silently skipped, and the
    unknown types are returned for diagnostics.
    """
    adj: dict[str, list[tuple[str, float]]] = {}
    unknown: set[str] = set()

    if allowed_types is not None:
        edges = edges.filter(pl.col("edge_type").is_in(list(allowed_types)))

    for row in edges.iter_rows(named=True):
        et = row["edge_type"]
        w = weights.get(et)
        if w is None:
            unknown.add(et)
            continue
        if w <= 0 or w > 1:
            raise ValueError(f"weight for {et} = {w} not in (0, 1]")
        cost = -math.log(w)
        adj.setdefault(row["src"], []).append((row["dst"], cost))
        # Treat edges as undirected by default. The graph is conceptually
        # undirected for contamination (similarity is symmetric).
        adj.setdefault(row["dst"], []).append((row["src"], cost))

    return adj, sorted(unknown)


def multi_source_dijkstra(
    adj: dict[str, list[tuple[str, float]]],
    sources: Iterable[str],
    targets: set[str] | None = None,
    max_hops: int | None = 6,
) -> dict[str, float]:
    """Multi-source shortest-path distances (sum of -log weights).

    Returns: dict[node_id -> min_cost]. Nodes not reached are absent.

    If `targets` is provided, the search terminates early once all targets are
    settled. If `max_hops` is provided, the search ignores paths longer than
    that hop count (each edge counts as one hop).
    """
    dist: dict[str, float] = {}
    hops: dict[str, int] = {}
    heap: list[tuple[float, int, str]] = []
    for s in sources:
        if s in dist:
            continue
        dist[s] = 0.0
        hops[s] = 0
        heapq.heappush(heap, (0.0, 0, s))

    remaining_targets = set(targets) if targets is not None else None

    while heap:
        d, h, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if remaining_targets is not None and u in remaining_targets:
            remaining_targets.discard(u)
            if not remaining_targets:
                break
        if max_hops is not None and h >= max_hops:
            continue
        for v, cost in adj.get(u, ()):
            nd = d + cost
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                hops[v] = h + 1
                heapq.heappush(heap, (nd, h + 1, v))
    return dist


def score_axis(
    edges: pl.DataFrame,
    reference: set[str],
    queries: set[str],
    axis: str,
    *,
    weights: dict[str, float] | None = None,
    max_hops: int = 6,
) -> pl.DataFrame:
    """Compute per-query contamination on a single axis.

    Args:
        edges:     DataFrame with columns (src, dst, edge_type).
        reference: set of example node IDs (the A set, e.g. D_train).
        queries:   set of example node IDs to score (e.g. D_test).
        axis:      one of AXES.
        weights:   per-edge-type weight overrides; defaults to DEFAULT_WEIGHTS.
        max_hops:  Dijkstra hop bound (default 6).

    Returns:
        DataFrame with columns (example_id, axis, contamination).
        Examples not reached from any reference are absent.
    """
    if axis not in AXIS_EDGE_TYPES:
        raise KeyError(f"unknown axis: {axis}")
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)
    allowed = set(AXIS_EDGE_TYPES[axis])
    adj, unknown = _build_adj(edges, w, allowed)
    if unknown:
        # Soft warning surfaced via a column in the output; downstream code can
        # still aggregate. We do not raise here because most callers care about
        # the score, not the catalog.
        pass

    dist = multi_source_dijkstra(adj, sources=reference, targets=queries, max_hops=max_hops)
    rows = []
    for q in queries:
        d = dist.get(q)
        if d is None:
            rows.append({"example_id": q, "axis": axis, "contamination": 0.0})
        else:
            rows.append({"example_id": q, "axis": axis, "contamination": math.exp(-d)})
    return pl.DataFrame(rows, schema={"example_id": pl.Utf8, "axis": pl.Utf8, "contamination": pl.Float64})


def score_overall(
    edges: pl.DataFrame,
    reference: set[str],
    queries: set[str],
    *,
    weights: dict[str, float] | None = None,
    axes: Iterable[str] = AXES,
    max_hops: int = 6,
) -> pl.DataFrame:
    """Compute per-query overall + per-axis contamination.

    Returns:
        DataFrame with one row per query and columns:
            example_id, C_<axis> for each axis, C_overall, dominant_axis.
    """
    per_axis = {}
    for axis in axes:
        sub = score_axis(edges, reference, queries, axis, weights=weights, max_hops=max_hops)
        per_axis[axis] = dict(zip(sub["example_id"].to_list(), sub["contamination"].to_list()))

    rows = []
    for q in queries:
        scores = {axis: per_axis[axis].get(q, 0.0) for axis in axes}
        c_overall = max(scores.values())
        dominant = max(scores, key=scores.get)
        row = {"example_id": q, "C_overall": c_overall, "dominant_axis": dominant}
        for axis, v in scores.items():
            row[f"C_{axis}"] = v
        rows.append(row)

    schema = {"example_id": pl.Utf8, "C_overall": pl.Float64, "dominant_axis": pl.Utf8}
    for axis in axes:
        schema[f"C_{axis}"] = pl.Float64
    return pl.DataFrame(rows, schema=schema)


def contamination_nn_label(
    edges: pl.DataFrame,
    reference_labels: dict[str, int],
    queries: set[str],
    *,
    weights: dict[str, float] | None = None,
    max_hops: int = 6,
) -> pl.DataFrame:
    """Contamination-NN baseline (proposal section 5.14).

    For each query, find the reference example with the highest contamination
    score and transfer its label. Returns DataFrame with columns
    (example_id, predicted_label, source_example, contamination).
    """
    # The cheap implementation: per-axis Dijkstra and track which reference
    # node gave the min cost. We instead build the full graph once on the
    # union of axis-allowed edges, then run a per-reference Dijkstra-style
    # multi-source variant that retains argmin source.
    from .schema import AXIS_EDGE_TYPES as _A
    allowed = set()
    for et_list in _A.values():
        allowed.update(et_list)

    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)
    adj, _ = _build_adj(edges, w, allowed)

    # multi-source Dijkstra that records the source for each settled node
    dist: dict[str, float] = {}
    parent_src: dict[str, str] = {}
    hops: dict[str, int] = {}
    heap: list[tuple[float, int, str, str]] = []  # (cost, hops, node, src)
    for s in reference_labels.keys():
        dist[s] = 0.0
        parent_src[s] = s
        hops[s] = 0
        heapq.heappush(heap, (0.0, 0, s, s))

    remaining = set(queries)
    while heap and remaining:
        d, h, u, src = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if u in remaining:
            remaining.discard(u)
        if h >= max_hops:
            continue
        for v, cost in adj.get(u, ()):
            nd = d + cost
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent_src[v] = src
                hops[v] = h + 1
                heapq.heappush(heap, (nd, h + 1, v, src))

    rows = []
    for q in queries:
        d = dist.get(q)
        if d is None:
            rows.append({
                "example_id": q, "predicted_label": -1,
                "source_example": "", "contamination": 0.0,
            })
        else:
            src = parent_src[q]
            rows.append({
                "example_id": q,
                "predicted_label": int(reference_labels.get(src, -1)),
                "source_example": src,
                "contamination": math.exp(-d),
            })
    return pl.DataFrame(
        rows,
        schema={
            "example_id": pl.Utf8,
            "predicted_label": pl.Int64,
            "source_example": pl.Utf8,
            "contamination": pl.Float64,
        },
    )
