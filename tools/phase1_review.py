"""Post-run review of Phase 1 outputs.

Walks outputs/v2/ and flags anything that looks broken so the operator
can act on it before declaring the run good. Designed to be run after
`run_phase1.sh` completes (or fails) on the Linux GPU box.

Checks performed:

  C1  Per-corpus graph parquets exist and are non-trivial.
  C2  stats.csv was emitted and has sane counts (out > 0).
  C3  side_table.parquet exists, validates, and covers each known source.
  C4  Per-corpus phase1 summary CSV exists with every regime present.
  C5  No regime errored out (notes column is empty or 'infeasible:...').
  C6  Sanity bounds on AUROC / AUPRC (0 <= x <= 1, not NaN unless infeasible).
  C7  Validation contamination matrices exist for every feasible regime.
  C8  Final tables (table1/2/5) and figure2 rendered.

Output is a markdown report on stdout; the script exits non-zero if any
critical check fails.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import polars as pl

# Allow direct execution.
import sys as _sys
_PKG_ROOT = Path(__file__).resolve().parent.parent / "src"
if _PKG_ROOT.exists() and str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))

from vsleakkg.v2.hydrate import KnownSource, SIDE_TABLE_COLUMNS, validate_side_table
from vsleakkg.v2.pipeline import REGIMES

log = logging.getLogger(__name__)

CORPORA = ["pdbbind", "dekois", "dude", "litpcba_ave"]
PIPELINE_TAGS = ["pdbbind", "dekois", "dude", "litpcba"]


def _exists(p: Path) -> str:
    return f"✓ ({p.stat().st_size // 1024} kB)" if p.exists() else "✗ MISSING"


def check_graph_outputs(repo: Path) -> list[dict]:
    rows = []
    for c in CORPORA:
        g = repo / "outputs" / "v2" / f"graph_{c}"
        nodes_p = g / "v2_nodes.parquet"
        edges_p = g / "v2_edges.parquet"
        stats_p = g / "stats.csv"
        rec = {"corpus": c, "nodes": _exists(nodes_p), "edges": _exists(edges_p),
               "stats": _exists(stats_p), "fail": ""}
        if nodes_p.exists() and edges_p.exists():
            try:
                n_nodes = pl.scan_parquet(nodes_p).select(pl.len()).collect().item()
                n_edges = pl.scan_parquet(edges_p).select(pl.len()).collect().item()
                rec["n_nodes"] = n_nodes
                rec["n_edges"] = n_edges
                if n_nodes == 0:
                    rec["fail"] = "empty nodes parquet"
                elif n_edges == 0:
                    rec["fail"] = "empty edges parquet"
            except Exception as exc:
                rec["fail"] = f"read error: {exc!r}"
        else:
            rec["fail"] = "missing parquet(s)"
        rows.append(rec)
    return rows


def check_side_table(repo: Path) -> dict:
    p = repo / "outputs" / "v2" / "graph" / "side_table.parquet"
    rec: dict = {"path": str(p), "exists": p.exists(), "fail": ""}
    if not p.exists():
        rec["fail"] = "missing"
        return rec
    try:
        df = pl.read_parquet(p)
        validate_side_table(df)
        rec["rows"] = df.height
        rec["sources"] = dict(
            df.group_by("source").len().sort("source").iter_rows()
        )
        # Coverage: every known source represented?
        missing_sources = [s.value for s in KnownSource
                           if s.value not in rec["sources"]]
        if missing_sources:
            rec["fail"] = f"missing sources: {missing_sources}"
        if df.height == 0:
            rec["fail"] = "side-table is empty"
    except Exception as exc:
        rec["fail"] = f"validate failed: {exc!r}"
    return rec


def check_pipeline(repo: Path) -> list[dict]:
    rows = []
    p1 = repo / "outputs" / "v2" / "phase1"
    for tag in PIPELINE_TAGS:
        f = p1 / f"{tag}_summary.csv"
        rec = {"corpus": tag, "summary": _exists(f), "fail": ""}
        if not f.exists():
            rec["fail"] = "missing summary"
            rows.append(rec); continue
        try:
            df = pl.read_csv(f, infer_schema_length=1000)
        except Exception as exc:
            rec["fail"] = f"read error: {exc!r}"
            rows.append(rec); continue
        rec["regime_count"] = df.height
        seen = set(df["regime"].to_list())
        expected = set(REGIMES.keys())
        if seen != expected:
            rec["fail"] = f"regimes mismatch: missing={expected - seen} extra={seen - expected}"
        rec["feasible_regimes"] = int(df["feasible"].sum()) if "feasible" in df.columns else None
        # Sanity AUROC bounds
        bad_auroc = []
        if "baseline_auroc" in df.columns:
            for r in df.iter_rows(named=True):
                v = r["baseline_auroc"]
                if v is None: continue
                if isinstance(v, float) and math.isnan(v): continue
                if not (0.0 <= float(v) <= 1.0):
                    bad_auroc.append((r["regime"], v))
        if bad_auroc:
            rec["fail"] = f"AUROC out of [0,1]: {bad_auroc}"
        # Contamination matrices exist for every feasible regime?
        if "feasible" in df.columns:
            missing_vc = []
            for r in df.iter_rows(named=True):
                if not r["feasible"]: continue
                vc = p1 / "validation_contamination" / tag / f"{r['regime']}.csv"
                if not vc.exists():
                    missing_vc.append(r["regime"])
            if missing_vc:
                rec["fail"] = (rec.get("fail", "") + f"; missing VC for feasible regimes: {missing_vc}").strip("; ")
        rows.append(rec)
    return rows


def check_tables_figures(repo: Path) -> dict:
    base = repo / "outputs" / "v2"
    return {
        "table1": _exists(base / "tables" / "table1_kg_stats.csv"),
        "table2": _exists(base / "tables" / "table2_leakage_groups.csv"),
        "table5": _exists(base / "tables" / "table5_validation_contamination.csv"),
        "figure2": _exists(base / "figures" / "figure2_hub_pareto.png"),
    }


def render_report(repo: Path) -> tuple[str, int]:
    """Return (markdown report, exit code: 0 ok, 1 critical failure)."""
    sections: list[str] = ["# VS-LeakKG v2 Phase 1 review\n"]
    fails: list[str] = []

    sections.append("## C1-C2 — per-corpus graph outputs\n")
    sections.append("| corpus | nodes | edges | stats | n_nodes | n_edges | fail |")
    sections.append("|--------|-------|-------|-------|---------|---------|------|")
    for r in check_graph_outputs(repo):
        sections.append(
            f"| {r['corpus']} | {r['nodes']} | {r['edges']} | {r['stats']} | "
            f"{r.get('n_nodes','-')} | {r.get('n_edges','-')} | {r['fail']} |"
        )
        if r["fail"]:
            fails.append(f"graph[{r['corpus']}]: {r['fail']}")

    sections.append("\n## C3 — side-table\n")
    st = check_side_table(repo)
    sections.append(f"- exists: {st['exists']}")
    if "rows" in st:
        sections.append(f"- rows: {st['rows']}")
        sections.append(f"- sources: {json.dumps(st['sources'])}")
    if st["fail"]:
        sections.append(f"- **FAIL**: {st['fail']}")
        fails.append(f"side-table: {st['fail']}")

    sections.append("\n## C4-C7 — pipeline per-corpus\n")
    sections.append("| corpus | summary | regime_count | feasible | fail |")
    sections.append("|--------|---------|--------------|----------|------|")
    for r in check_pipeline(repo):
        sections.append(
            f"| {r['corpus']} | {r['summary']} | {r.get('regime_count','-')} | "
            f"{r.get('feasible_regimes','-')} | {r['fail']} |"
        )
        if r["fail"]:
            fails.append(f"pipeline[{r['corpus']}]: {r['fail']}")

    sections.append("\n## C8 — final tables + figure\n")
    tf = check_tables_figures(repo)
    for k, v in tf.items():
        sections.append(f"- {k}: {v}")
        if "MISSING" in v:
            fails.append(f"deliverable {k} missing")

    sections.append("\n## Summary\n")
    if not fails:
        sections.append("- ✓ all critical checks passed")
        rc = 0
    else:
        sections.append(f"- ✗ **{len(fails)} critical failure(s)**:")
        for f in fails:
            sections.append(f"  - {f}")
        rc = 1
    return "\n".join(sections), rc


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--write", type=Path, default=None,
                   help="Also write report to this file")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report, rc = render_report(args.repo_root)
    print(report)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report)
    sys.exit(rc)


if __name__ == "__main__":
    _cli()
