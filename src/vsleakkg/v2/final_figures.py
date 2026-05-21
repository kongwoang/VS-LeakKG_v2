"""Phase 1 deliverable: Table 1 / Table 2 / Figure 2 from outputs/v2/phase1.

This is the data-side subset of proposal section 5.14's "final figures"
script. Phase 2 figures (Table 3, Table 4, Figure 1) are model-dependent
and live in a follow-up module.

Inputs:
  outputs/v2/graph_*/stats.csv               (from build_graph)
  outputs/v2/phase1/<corpus>_summary.csv     (from pipeline)
  outputs/v2/phase1/phase1_combined.csv      (consolidated)

Outputs:
  outputs/v2/tables/table1_kg_stats.csv
  outputs/v2/tables/table2_leakage_groups.csv
  outputs/v2/tables/table5_validation_contamination.csv
  outputs/v2/figures/figure2_hub_pareto.png   (if matplotlib available)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def render_table1_kg_stats(graph_dirs: list[Path], out: Path) -> pl.DataFrame:
    """Per-corpus KG statistics from the build_graph stats.csv files."""
    rows: list[dict] = []
    for d in graph_dirs:
        f = d / "stats.csv"
        if not f.exists():
            continue
        s = pl.read_csv(f, infer_schema_length=200)
        rec: dict[str, object] = {"corpus": d.name.replace("graph_", "")}
        for key, val in zip(s["key"].to_list(), s["value"].to_list()):
            rec[key] = val
        rows.append(rec)
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    return df


def render_table2_leakage_groups(combined_csv: Path, out: Path) -> pl.DataFrame:
    if not combined_csv.exists():
        log.warning("missing combined summary at %s", combined_csv)
        return pl.DataFrame()
    df = pl.read_csv(combined_csv, infer_schema_length=1000)
    cols = [
        "corpus", "regime", "feasible", "n_groups", "rho_max",
        "size_train", "size_val", "size_test",
        "actives_train", "actives_test",
        "n_pos_test", "n_neg_test",
    ]
    keep = [c for c in cols if c in df.columns]
    df2 = df.select(keep)
    out.parent.mkdir(parents=True, exist_ok=True)
    df2.write_csv(out)
    return df2


def render_table5_validation_contamination(phase1_dir: Path, out: Path) -> pl.DataFrame:
    """Aggregate per-corpus validation_contamination CSVs into one table."""
    rows: list[dict] = []
    vc_root = phase1_dir / "validation_contamination"
    if not vc_root.exists():
        log.warning("no validation_contamination dir at %s", vc_root)
        return pl.DataFrame()
    for corpus_dir in sorted(vc_root.glob("*")):
        if not corpus_dir.is_dir():
            continue
        for f in sorted(corpus_dir.glob("*.csv")):
            try:
                mat = pl.read_csv(f, infer_schema_length=100)
            except Exception:
                continue
            for r in mat.iter_rows(named=True):
                rows.append({
                    "corpus": corpus_dir.name,
                    "regime": f.stem,
                    **r,
                })
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    return df


def render_figure2_hub_pareto(phase1_dir: Path, out: Path) -> bool:
    """Hub-Pareto curve: regime size vs feasibility / leakage rho_max."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping figure")
        return False
    summary = phase1_dir / "phase1_combined.csv"
    if not summary.exists():
        log.warning("missing %s", summary)
        return False
    df = pl.read_csv(summary, infer_schema_length=1000)
    # x = total size, y = rho_max, colour = corpus
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for corpus, grp in df.group_by("corpus"):
        # tolerate the new polars `group_by` returning a tuple
        cname = corpus[0] if isinstance(corpus, tuple) else corpus
        ax.scatter(
            grp["size_train"] + grp["size_val"] + grp["size_test"],
            grp["rho_max"],
            label=str(cname),
            s=60, alpha=0.75,
        )
    ax.set_xlabel("Total examples in split")
    ax.set_ylabel("rho_max (largest leakage group fraction)")
    ax.set_title("Figure 2: Hub-pareto curve per corpus x regime")
    ax.legend(loc="best")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("wrote figure to %s", out)
    return True


def render_all(repo_root: Path) -> dict[str, Path]:
    outputs_v2 = repo_root / "outputs" / "v2"
    tables_dir = outputs_v2 / "tables"
    figs_dir = outputs_v2 / "figures"
    phase1_dir = outputs_v2 / "phase1"
    graph_dirs = sorted(p for p in outputs_v2.glob("graph_*") if p.is_dir())

    paths: dict[str, Path] = {}
    paths["table1"] = tables_dir / "table1_kg_stats.csv"
    paths["table2"] = tables_dir / "table2_leakage_groups.csv"
    paths["table5"] = tables_dir / "table5_validation_contamination.csv"
    paths["figure2"] = figs_dir / "figure2_hub_pareto.png"

    render_table1_kg_stats(graph_dirs, paths["table1"])
    render_table2_leakage_groups(phase1_dir / "phase1_combined.csv", paths["table2"])
    render_table5_validation_contamination(phase1_dir, paths["table5"])
    render_figure2_hub_pareto(phase1_dir, paths["figure2"])
    return paths


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = render_all(args.repo_root)
    for k, v in paths.items():
        print(f"{k}: {v.relative_to(args.repo_root)}  ({'OK' if v.exists() else 'MISSING'})")


if __name__ == "__main__":
    _cli()
