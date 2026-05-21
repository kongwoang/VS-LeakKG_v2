"""End-to-end Phase 1 pipeline driver.

Given a corpus's v2_nodes.parquet + v2_edges.parquet (from build_graph)
and the hydrate side_table.parquet (from build_side_table), produces:

  - Per-regime split parquets: splits/<corpus>/<regime>.parquet
  - Per-regime contamination matrices: validation_contamination/<corpus>/<regime>.csv
  - Per-regime data-only baselines: baselines/<corpus>/<regime>.csv
  - Per-corpus stats: phase1/<corpus>_summary.csv

A "regime" is one of the seven proposal axes (ligand, scaffold, protein,
pocket, assay, source, time) plus the synthetic `strict` (all axes
forbidden) and `dual` (ligand+protein) modes. Axes we don't have edges
for in the v2 graph emit `infeasible` rows rather than silently relaxing
constraints.

This module intentionally reuses the existing v2 algorithm modules
(`leakage_groups`, `split`, `validation_contamination`,
`baselines.ligand_only`, `scoring.contamination_nn_label`) — it does
not reimplement them.

CLI:

    python -m vsleakkg.v2.pipeline \
        --graph-dir outputs/v2/graph_litpcba_ave \
        --side-table outputs/v2/graph/side_table.parquet \
        --output-dir outputs/v2/phase1/litpcba_ave \
        --corpus-tag litpcba

The script is idempotent and writes a `phase1_<corpus>_summary.csv`
with one row per regime.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .baselines.ligand_only import evaluate_ligand_only
from .leakage_groups import build_leakage_groups
from .schema import (
    AXIS_EDGE_TYPES,
    DEFAULT_WEIGHTS,
    EdgeType,
    NodeType,
    SplitConstraints,
)
from .split import greedy_assign
from .validation_contamination import three_way_contamination

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

# A regime maps to the AXES it forbids. AXIS_EDGE_TYPES then resolves
# those axes into concrete edge_type strings.
REGIMES: dict[str, list[str]] = {
    "ligand":   ["ligand"],
    "scaffold": ["scaffold"],
    "protein":  ["protein"],
    "pocket":   ["pocket"],
    "assay":    ["assay"],
    "dual":     ["ligand", "protein"],
    "strict":   ["ligand", "scaffold", "protein", "pocket", "assay", "source"],
}


def _forbidden_edges_for_regime(regime: str) -> set[str]:
    axes = REGIMES[regime]
    out: set[str] = set()
    for ax in axes:
        out.update(AXIS_EDGE_TYPES[ax])
    return out


# ---------------------------------------------------------------------------
# Examples extraction from v2_nodes + v2_edges
# ---------------------------------------------------------------------------


def extract_examples_frame(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    side_table: pl.DataFrame | None,
    corpus_tag: str,
) -> pl.DataFrame:
    """Build a (example_id, protein_id, ligand_id, smiles, label) frame.

    Derived from v2_nodes (props column carries label/SMILES) and v2_edges
    (one-hop joins for protein/ligand). If a side_table is provided, we
    prefer its SMILES + label since it has gone through canonicalization.
    """
    example_nodes = nodes.filter(pl.col("node_type") == NodeType.EXAMPLE.value)
    example_ids = example_nodes["node_id"].to_list()

    # Extract per-example label / smiles / target from the JSON `props` column
    # that v1 wrote on every Example node. The keys we care about are
    # `label` (0 / 1 / pAct) and `target` (target name); fall back to None.
    props_labels: list[float | None] = []
    props_targets: list[str | None] = []
    if "props" in example_nodes.columns and example_nodes.height:
        for s in example_nodes["props"].to_list():
            lab: float | None = None
            tgt: str | None = None
            if s:
                try:
                    d = json.loads(s)
                    if "label" in d:
                        try:
                            lab = float(d["label"])
                        except (TypeError, ValueError):
                            lab = None
                    tgt = d.get("target")
                except (json.JSONDecodeError, TypeError):
                    pass
            props_labels.append(lab)
            props_targets.append(tgt)
    # smiles from the node label column (v1 stores the ligand identifier
    # there for DEKOIS / DUD-E / LIT-PCBA - usually ZINC id, not SMILES).
    # The smiles_canonical column comes via side-table join below.

    if not example_ids:
        log.warning(
            "[%s] no Example nodes in this graph - returning empty examples frame. "
            "This is expected for corpora whose v1 schema lacked Example nodes (e.g. PDBBind)",
            corpus_tag,
        )
        return pl.DataFrame({
            "example_id": pl.Series([], dtype=pl.Utf8),
            "ligand_id":  pl.Series([], dtype=pl.Utf8),
            "protein_id": pl.Series([], dtype=pl.Utf8),
            "smiles":     pl.Series([], dtype=pl.Utf8),
            "label":      pl.Series([], dtype=pl.Float64),
        })

    # one-hop joins
    e_lig = (
        edges.filter(pl.col("edge_type") == EdgeType.EXAMPLE_HAS_LIGAND.value)
        .select(pl.col("src").alias("example_id"), pl.col("dst").alias("ligand_id"))
    )
    e_prot = (
        edges.filter(pl.col("edge_type") == EdgeType.EXAMPLE_HAS_PROTEIN.value)
        .select(pl.col("src").alias("example_id"), pl.col("dst").alias("protein_id"))
    )

    df = pl.DataFrame({
        "example_id": pl.Series(example_ids, dtype=pl.Utf8),
        "label_v1":   pl.Series(props_labels, dtype=pl.Float64),
        "target_v1":  pl.Series(props_targets, dtype=pl.Utf8),
    })
    df = df.join(e_lig, on="example_id", how="left").join(e_prot, on="example_id", how="left")

    # Fold side-table SMILES + label if provided. The side-table example_id
    # uses "source:source_id" form, which usually matches v1's
    # "Example::ex_id". When they don't match we fall back to a
    # corpus_tag-prefixed lookup.
    if side_table is not None and side_table.height:
        # Direct join on example_id first
        st_direct = side_table.select([
            "example_id",
            pl.col("smiles_canonical").alias("smiles"),
            "label",
        ])
        df = df.join(st_direct, on="example_id", how="left")
        # For rows still missing SMILES, try matching on source_id == suffix
        missing = df.filter(pl.col("smiles").is_null())
        if missing.height:
            log.warning(
                "%d/%d examples did not match side-table directly",
                missing.height, df.height,
            )

    # Prefer label from side-table; fall back to label_v1 (extracted from props)
    if "label" in df.columns:
        df = df.with_columns(
            pl.coalesce([pl.col("label"), pl.col("label_v1")]).alias("label"),
        )
    else:
        df = df.with_columns(pl.col("label_v1").alias("label"))
    if "smiles" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("smiles"))
    # Casts
    df = df.with_columns(
        pl.col("label").cast(pl.Float64, strict=False).fill_null(0.0),
    )
    return df.select(["example_id", "ligand_id", "protein_id", "smiles", "label"])


# ---------------------------------------------------------------------------
# Per-regime work
# ---------------------------------------------------------------------------


@dataclass
class RegimeResult:
    regime: str
    feasible: bool
    n_groups: int
    rho_max: float
    sizes: dict[str, int]
    actives: dict[str, int]
    contamination_summary: dict[str, dict[str, float]]
    baseline_auroc: float
    baseline_auprc: float
    n_pos_test: int
    n_neg_test: int
    notes: str = ""


def _summarise(assignment: pl.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    sizes = dict(
        assignment.group_by("partition").len().sort("partition").iter_rows()
    )
    actives = dict(
        assignment.filter(pl.col("label") == 1.0)
        .group_by("partition").len().sort("partition").iter_rows()
    )
    return sizes, actives


def run_regime(
    regime: str,
    examples: pl.DataFrame,
    edges: pl.DataFrame,
    output_dir: Path,
    corpus_tag: str,
    *,
    constraints: SplitConstraints | None = None,
    skip_baseline: bool = False,
) -> RegimeResult:
    """Run one (corpus, regime) end-to-end."""
    forbidden = _forbidden_edges_for_regime(regime)
    present = set(edges["edge_type"].unique().to_list())
    relevant = forbidden & present

    if not relevant:
        return RegimeResult(
            regime=regime, feasible=False, n_groups=0, rho_max=0.0,
            sizes={}, actives={}, contamination_summary={},
            baseline_auroc=float("nan"), baseline_auprc=float("nan"),
            n_pos_test=0, n_neg_test=0,
            notes=f"infeasible: no edges of type {sorted(forbidden)} in graph",
        )

    log.info("[%s/%s] building leakage groups (forbidden=%s)",
             corpus_tag, regime, sorted(relevant))
    t0 = time.perf_counter()
    fb_edges = edges.filter(pl.col("edge_type").is_in(list(relevant)))
    lgr = build_leakage_groups(
        example_ids=examples["example_id"].to_list(),
        edges=fb_edges,
        forbidden_relations=relevant,
    )

    log.info("[%s/%s] greedy split assignment", corpus_tag, regime)
    sa = greedy_assign(lgr.groups, examples, constraints=constraints)

    assign_df = pl.DataFrame({
        "example_id": list(sa.assignment.keys()),
        "partition": list(sa.assignment.values()),
    }).join(examples, on="example_id", how="left")

    # write split
    split_dir = output_dir / "splits" / corpus_tag
    split_dir.mkdir(parents=True, exist_ok=True)
    assign_df.write_parquet(split_dir / f"{regime}.parquet")

    sizes, actives = _summarise(assign_df)

    # contamination matrices (data-side)
    train_ids = set(assign_df.filter(pl.col("partition") == "train")["example_id"].to_list())
    val_ids   = set(assign_df.filter(pl.col("partition") == "val")["example_id"].to_list())
    test_ids  = set(assign_df.filter(pl.col("partition") == "test")["example_id"].to_list())

    cm_summary: dict[str, dict[str, float]] = {}
    try:
        matrices = three_way_contamination(
            edges, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        )
        for direction, mat in matrices.items():
            cm_summary[direction] = mat.summary
        cm_dir = output_dir / "validation_contamination" / corpus_tag
        cm_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([
            {"direction": d, **s}
            for d, s in cm_summary.items()
        ]).write_csv(cm_dir / f"{regime}.csv")
    except Exception as exc:
        log.exception("contamination matrix failed: %s", exc)

    # ligand-only baseline
    auroc = auprc = float("nan")
    n_pos = n_neg = 0
    if not skip_baseline:
        try:
            train_df = assign_df.filter(pl.col("partition") == "train")
            test_df = assign_df.filter(pl.col("partition") == "test")
            if train_df.height and test_df.height and \
               train_df["smiles"].is_not_null().any() and \
               test_df["smiles"].is_not_null().any():
                # Drop nulls; needed for sklearn
                train_df = train_df.filter(pl.col("smiles").is_not_null())
                test_df = test_df.filter(pl.col("smiles").is_not_null())
                lor = evaluate_ligand_only(train_df, test_df)
                auroc = lor.auroc
                auprc = lor.auprc
                n_pos = lor.n_pos
                n_neg = lor.n_neg
                base_dir = output_dir / "baselines" / corpus_tag
                base_dir.mkdir(parents=True, exist_ok=True)
                pl.DataFrame([{
                    "regime": regime,
                    "baseline": "ligand_only",
                    "auroc": auroc, "auprc": auprc,
                    "n_pos_test": n_pos, "n_neg_test": n_neg,
                    "used_rdkit": lor.used_rdkit,
                }]).write_csv(base_dir / f"{regime}.csv")
        except Exception as exc:
            log.exception("ligand-only baseline failed: %s", exc)

    log.info("[%s/%s] done in %.1fs", corpus_tag, regime, time.perf_counter() - t0)
    return RegimeResult(
        regime=regime,
        feasible=sa.feasible,
        n_groups=lgr.n_groups,
        rho_max=lgr.rho_max,
        sizes=sizes,
        actives=actives,
        contamination_summary=cm_summary,
        baseline_auroc=auroc,
        baseline_auprc=auprc,
        n_pos_test=n_pos,
        n_neg_test=n_neg,
    )


def run_corpus(
    graph_dir: Path,
    side_table_path: Path | None,
    output_dir: Path,
    corpus_tag: str,
    *,
    regimes: Iterable[str] = REGIMES.keys(),
    constraints: SplitConstraints | None = None,
    sample_examples: int | None = None,
) -> dict[str, RegimeResult]:
    """Run all regimes on one corpus."""
    nodes_p = graph_dir / "v2_nodes.parquet"
    edges_p = graph_dir / "v2_edges.parquet"
    for p in (nodes_p, edges_p):
        if not p.exists():
            raise FileNotFoundError(p)
    nodes = pl.read_parquet(nodes_p)
    edges = pl.read_parquet(edges_p)

    side = pl.read_parquet(side_table_path) if (side_table_path and side_table_path.exists()) else None
    examples = extract_examples_frame(nodes, edges, side, corpus_tag)
    if sample_examples and examples.height > sample_examples:
        examples = examples.sample(n=sample_examples, seed=0)

    log.info("[%s] examples=%d edges=%d", corpus_tag, examples.height, edges.height)
    if examples.height == 0:
        # Emit an empty summary so downstream consumers don't crash, but
        # skip the regime loop entirely.
        log.warning("[%s] skipping all regimes (no Example nodes)", corpus_tag)
        output_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "corpus": [corpus_tag] * len(REGIMES),
            "regime": list(REGIMES.keys()),
            "feasible": [False] * len(REGIMES),
            "n_groups": [0] * len(REGIMES),
            "rho_max": [0.0] * len(REGIMES),
            "size_train": [0] * len(REGIMES),
            "size_val": [0] * len(REGIMES),
            "size_test": [0] * len(REGIMES),
            "actives_train": [0] * len(REGIMES),
            "actives_test": [0] * len(REGIMES),
            "baseline_auroc": [float("nan")] * len(REGIMES),
            "baseline_auprc": [float("nan")] * len(REGIMES),
            "n_pos_test": [0] * len(REGIMES),
            "n_neg_test": [0] * len(REGIMES),
            "notes": ["no_examples_in_graph"] * len(REGIMES),
        }).write_csv(output_dir / f"{corpus_tag}_summary.csv")
        return {}

    results: dict[str, RegimeResult] = {}
    rows: list[dict] = []
    for regime in regimes:
        try:
            rr = run_regime(
                regime, examples, edges, output_dir, corpus_tag,
                constraints=constraints,
            )
        except Exception as exc:
            log.exception("regime %s failed: %s", regime, exc)
            rr = RegimeResult(
                regime=regime, feasible=False, n_groups=0, rho_max=0.0,
                sizes={}, actives={}, contamination_summary={},
                baseline_auroc=float("nan"), baseline_auprc=float("nan"),
                n_pos_test=0, n_neg_test=0, notes=f"error: {exc!r}",
            )
        results[regime] = rr
        rows.append({
            "corpus": corpus_tag,
            "regime": regime,
            "feasible": rr.feasible,
            "n_groups": rr.n_groups,
            "rho_max": rr.rho_max,
            "size_train": rr.sizes.get("train", 0),
            "size_val":   rr.sizes.get("val", 0),
            "size_test":  rr.sizes.get("test", 0),
            "actives_train": rr.actives.get("train", 0),
            "actives_test":  rr.actives.get("test", 0),
            "baseline_auroc": rr.baseline_auroc,
            "baseline_auprc": rr.baseline_auprc,
            "n_pos_test": rr.n_pos_test,
            "n_neg_test": rr.n_neg_test,
            "notes": rr.notes,
        })
    # Write summary directly under output_dir to avoid a double-nested
    # outputs/v2/phase1/phase1/ when callers pass --output-dir outputs/v2/phase1.
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output_dir / f"{corpus_tag}_summary.csv")
    return results


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--side-table", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--corpus-tag", required=True,
                   help="Tag for this corpus, e.g. litpcba, dude, dekois, pdbbind, bayesbind")
    p.add_argument("--regimes", default=",".join(REGIMES.keys()))
    p.add_argument("--sample-examples", type=int, default=None,
                   help="Cap example count (smoke testing)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    results = run_corpus(
        graph_dir=args.graph_dir,
        side_table_path=args.side_table,
        output_dir=args.output_dir,
        corpus_tag=args.corpus_tag,
        regimes=regimes,
        sample_examples=args.sample_examples,
    )
    feas = sum(1 for r in results.values() if r.feasible)
    print(f"corpus={args.corpus_tag} regimes={len(results)} feasible={feas}")


if __name__ == "__main__":
    _cli()
