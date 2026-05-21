"""Unit tests for leakage-group construction with giant-component fallback."""
from __future__ import annotations

import polars as pl

from vsleakkg.v2.leakage_groups import (
    GroupStrategy,
    build_leakage_groups,
)
from vsleakkg.v2.schema import GiantComponentConfig


def _edges(tuples):
    return pl.DataFrame(
        [{"src": s, "dst": d, "edge_type": t} for s, d, t in tuples],
        schema={"src": pl.Utf8, "dst": pl.Utf8, "edge_type": pl.Utf8},
    )


def test_simple_two_groups_via_ligand_exact():
    # Two pairs that each share a ligand exactly; expect 2 groups.
    # Use relaxed rho_max_ok because in a 4-row test each group is 50%
    # of the corpus (the default 0.30 threshold targets million-row corpora).
    edges = _edges([
        ("ex:1", "lig:A", "ligand_exact"),
        ("ex:2", "lig:A", "ligand_exact"),
        ("ex:3", "lig:B", "ligand_exact"),
        ("ex:4", "lig:B", "ligand_exact"),
    ])
    result = build_leakage_groups(
        example_ids={"ex:1", "ex:2", "ex:3", "ex:4"},
        edges=edges,
        forbidden_relations={"ligand_exact"},
        config=GiantComponentConfig(rho_max_ok=0.6, rho_max_prune=0.9),
    )
    assert result.strategy == GroupStrategy.CONNECTED_COMPONENTS
    # ex:1 and ex:2 should share a group; ex:3 and ex:4 should share a group
    assert result.groups["ex:1"] == result.groups["ex:2"]
    assert result.groups["ex:3"] == result.groups["ex:4"]
    assert result.groups["ex:1"] != result.groups["ex:3"]
    assert result.n_groups == 2


def test_isolated_examples_get_singleton_groups():
    edges = _edges([])
    result = build_leakage_groups(
        example_ids={"ex:1", "ex:2"},
        edges=edges,
        forbidden_relations={"ligand_exact"},
    )
    assert result.n_groups == 2


def test_giant_component_triggers_pruning():
    # All 10 examples linked through a single hub scaffold (very common in
    # real data when an unfiltered scaffold has degree N). Without
    # mitigation, this gives one giant component.
    edges_list = []
    for i in range(10):
        edges_list.append((f"ex:{i}", "lig:X", "ligand_scaffold"))
    # Plus one weak link so we have two relation types
    edges_list.append(("ex:0", "lig:Y", "ligand_similar"))
    edges_list.append(("ex:1", "lig:Y", "ligand_similar"))
    edges = _edges(edges_list)

    result = build_leakage_groups(
        example_ids={f"ex:{i}" for i in range(10)},
        edges=edges,
        forbidden_relations={"ligand_scaffold", "ligand_similar"},
        config=GiantComponentConfig(rho_max_ok=0.30, rho_max_prune=0.60),
    )
    # rho_max with both relations is 1.0; pruning the weaker similarity edge
    # doesn't break the scaffold-driven giant component, so we'd expect
    # either pruned or louvain / infeasible.
    assert result.strategy in {
        GroupStrategy.PRUNED_COMPONENTS,
        GroupStrategy.LOUVAIN,
        GroupStrategy.INFEASIBLE,
    }
    # Must have recorded a pruning attempt
    assert len(result.pruned_relations) >= 1
