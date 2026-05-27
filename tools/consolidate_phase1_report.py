"""Consolidate every landed Phase 1 audit table into a single markdown report.

Reads from <repo>/outputs/reports/data/ and writes
<repo>/outputs/reports/phase1_kg_audit_report.md.

Sections:
  1. Split sensitivity (Task 1) — Morgan baseline + C-NN
  2. Contamination-bin curves (Task 2) — AUROC vs C bin
  3. Path attribution (Task 3) — dominant axis share
  4. C-NN shortcut (Task 4) — label-copying baseline
  5. Provenance & shortcut probes (Task 5)
  6. Coverage + residual contamination (Task 6)
  7. Robustness controls (Task 7) — shuffle + threshold sweep
  8. Per-target heterogeneity + cross-benchmark map (Task 8)
  9. Plots index (Stage F)
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def fmt(v, n=4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v) if v not in (None, "") else "—"
    if x != x:  # NaN
        return "—"
    return f"{x:.{n}f}"


def section_split_sens(data: Path) -> str:
    rows = read_csv(data / "table_split_sensitivity.csv")
    if not rows:
        return "## 1 · Split sensitivity\n\n_No table_split_sensitivity.csv._\n"
    # Pivot: corpus × regime, two models stacked
    corpora = sorted({r["corpus"] for r in rows})
    regimes_order = ["random", "strict", "ligand", "scaffold", "protein", "pocket", "dual"]
    regimes = [r for r in regimes_order if any(x["regime"] == r for x in rows)]
    out = ["## 1 · Split sensitivity — AUROC by (corpus × regime × model)\n"]
    out.append("Source: `table_split_sensitivity.csv` "
               "(morgan_rf = Phase 1 ligand-only Morgan baseline; "
               "cnn = label-copying baseline via KG neighbours).\n")
    for model in ("morgan_rf", "cnn"):
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        out.append(f"\n### model = `{model}`\n")
        header = "| corpus | " + " | ".join(regimes) + " |"
        sep    = "|" + "|".join(["---"] * (len(regimes) + 1)) + "|"
        out.extend([header, sep])
        for c in corpora:
            cells = []
            for reg in regimes:
                m = [x for x in sub if x["corpus"] == c and x["regime"] == reg]
                cells.append(fmt(m[0]["auroc"]) if m else "—")
            out.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def section_contam_bins(data: Path) -> str:
    files = sorted(data.glob("table_contam_bins_*.csv"))
    if not files:
        return "## 2 · Contamination-score bins\n\n_No table_contam_bins_*.csv._\n"
    out = ["## 2 · Contamination-score bins — AUROC vs C bin\n",
           f"Source: {len(files)} files `table_contam_bins_<corpus>_<axis>.csv`.\n",
           "Each row aggregates test examples whose contamination score on the "
           "named axis falls in the given bin and reports Morgan-baseline AUROC.\n"]
    by_corpus: dict[tuple[str, str], list[dict]] = {}
    for f in files:
        # filename: table_contam_bins_<corpus>_<axis>.csv
        stem = f.stem.removeprefix("table_contam_bins_")
        try:
            corpus, axis = stem.split("_", 1)
        except ValueError:
            continue
        by_corpus[(corpus, axis)] = read_csv(f)
    for (corpus, axis), rows in sorted(by_corpus.items()):
        if not rows:
            continue
        # Pick the AUROC column (varies by writer; pick first numeric column besides bin/c_min/c_max/n).
        keys = list(rows[0].keys())
        bin_keys = [k for k in keys if k.lower() in {"bin", "c_bin", "c_min", "c_max"}]
        n_key = next((k for k in keys if k.lower() in {"n", "n_test", "n_total"}), None)
        auroc_key = next((k for k in keys if "auroc" in k.lower()), None)
        if not auroc_key:
            continue
        out.append(f"\n### {corpus} / axis = `{axis}`\n")
        bin_label = bin_keys[0] if bin_keys else "row"
        header = f"| {bin_label} | {n_key or 'n'} | {auroc_key} |"
        sep    = "|---|---|---|"
        out.extend([header, sep])
        for r in rows[:12]:  # bins are small; cap defensively
            out.append(f"| {r.get(bin_label, '')} | {r.get(n_key, '')} | {fmt(r.get(auroc_key, ''))} |")
    return "\n".join(out) + "\n"


def section_path_attr(data: Path) -> str:
    rows = read_csv(data / "table_path_attribution.csv")
    if not rows:
        return "## 3 · Path attribution\n\n_No table._\n"
    out = ["## 3 · Path attribution — dominant axis share per (corpus, regime)\n",
           "Source: `table_path_attribution.csv` (scope=`axis_nonzero`: fraction of "
           "test rows with ≥1 nonzero contamination edge of that axis, plus mean C).\n"]
    sub = [r for r in rows if r.get("scope") == "axis_nonzero"]
    by = {}
    for r in sub:
        by.setdefault((r["corpus"], r["regime"]), []).append(r)
    out.append("\n| corpus | regime | ligand share / C̄ | scaffold share / C̄ | protein share / C̄ | source share / C̄ |")
    out.append("|---|---|---|---|---|---|")
    for (c, reg), rs in sorted(by.items()):
        def cell(axis):
            m = [r for r in rs if r["axis"] == axis]
            if not m:
                return "—"
            return f"{fmt(m[0]['share'], 2)} / {fmt(m[0]['c_mean'], 2)}"
        out.append(f"| {c} | {reg} | {cell('ligand')} | {cell('scaffold')} | {cell('protein')} | {cell('source')} |")
    return "\n".join(out) + "\n"


def section_cnn(data: Path) -> str:
    rows = read_csv(data / "table_cnn_baseline.csv")
    if not rows:
        return "## 4 · C-NN baseline\n\n_No table._\n"
    out = ["## 4 · C-NN (label-copying via KG neighbours) baseline\n",
           "Source: `table_cnn_baseline.csv` (variant=`all_axis`: uses every axis "
           "to find nearest contaminated train neighbour; AUROC is the label-copy prediction).\n"]
    sub = [r for r in rows if r["variant"] == "all_axis"]
    corpora = sorted({r["corpus"] for r in sub})
    regimes_order = ["random", "ligand", "scaffold", "protein", "pocket", "dual"]
    regimes = [reg for reg in regimes_order if any(r["regime"] == reg for r in sub)]
    out.append("| corpus | " + " | ".join(regimes) + " |")
    out.append("|" + "|".join(["---"] * (len(regimes) + 1)) + "|")
    for c in corpora:
        cells = []
        for reg in regimes:
            m = [r for r in sub if r["corpus"] == c and r["regime"] == reg]
            cells.append(fmt(m[0]["auroc"]) if m else "—")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("\n_Per-axis breakdown (variant=`per_axis`) is in the raw CSV._\n")
    return "\n".join(out) + "\n"


def section_provenance(data: Path) -> str:
    rows = read_csv(data / "table_provenance.csv")
    if not rows:
        return "## 5 · Provenance probes\n\n_No table._\n"
    out = ["## 5 · Provenance & shortcut probes\n",
           f"Source: `table_provenance.csv` ({len(rows)} rows).\n"]
    keys = list(rows[0].keys())
    out.append("| " + " | ".join(keys) + " |")
    out.append("|" + "|".join(["---"] * len(keys)) + "|")
    for r in rows[:60]:
        out.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
    if len(rows) > 60:
        out.append(f"\n_… {len(rows) - 60} more rows in raw CSV._\n")
    return "\n".join(out) + "\n"


def section_coverage(data: Path) -> str:
    rows = read_csv(data / "table_coverage.csv")
    if not rows:
        return "## 6 · Coverage\n\n_No table._\n"
    out = ["## 6 · Coverage — fraction of test rows reachable by axis\n",
           "Source: `table_coverage.csv`.\n"]
    # Dedup rows
    seen = set(); uniq = []
    for r in rows:
        k = (r["corpus"], r["axis"])
        if k in seen: continue
        seen.add(k); uniq.append(r)
    corpora = sorted({r["corpus"] for r in uniq})
    axes_order = ["ligand", "scaffold", "protein", "pocket", "assay", "source", "time"]
    out.append("| corpus | " + " | ".join(axes_order) + " |")
    out.append("|" + "|".join(["---"] * (len(axes_order) + 1)) + "|")
    for c in corpora:
        cells = []
        for ax in axes_order:
            m = [r for r in uniq if r["corpus"] == c and r["axis"] == ax]
            cells.append(fmt(m[0]["coverage_frac"], 3) if m else "—")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("\n_Zeros indicate the KG carries no edges of that axis — see `absence_reason` in raw CSV._\n")
    return "\n".join(out) + "\n"


def section_robustness(data: Path) -> str:
    shuf = read_csv(data / "table_robustness_shuffle.csv")
    thresh = read_csv(data / "table_robustness_thresholds.csv")
    if not shuf and not thresh:
        return "## 7 · Robustness\n\n_No tables._\n"
    out = ["## 7 · Robustness & negative controls\n"]
    if shuf:
        out.append(f"\n### 7a · Label-shuffle ({len(shuf)} rows from `table_robustness_shuffle.csv`)\n")
        keys = list(shuf[0].keys())
        out.append("| " + " | ".join(keys) + " |")
        out.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in shuf[:80]:
            out.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
    if thresh:
        out.append(f"\n### 7b · Contamination-threshold sweep ({len(thresh)} rows from `table_robustness_thresholds.csv`)\n")
        keys = list(thresh[0].keys())
        out.append("| " + " | ".join(keys) + " |")
        out.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in thresh[:60]:
            out.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
    return "\n".join(out) + "\n"


def section_heterogeneity(data: Path) -> str:
    het = read_csv(data / "table_per_target_heterogeneity.csv")
    cross = read_csv(data / "table_cross_benchmark_map.csv")
    out = ["## 8 · Per-target heterogeneity + cross-benchmark map\n"]
    if het:
        out.append(f"\n### 8a · Per-target heterogeneity ({len(het)} rows from `table_per_target_heterogeneity.csv`)\n")
        keys = list(het[0].keys())
        out.append("| " + " | ".join(keys) + " |")
        out.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in het[:80]:
            out.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
        if len(het) > 80:
            out.append(f"\n_… {len(het) - 80} more rows in raw CSV._\n")
    if cross:
        out.append(f"\n### 8b · Cross-benchmark map ({len(cross)} rows from `table_cross_benchmark_map.csv`)\n")
        keys = list(cross[0].keys())
        out.append("| " + " | ".join(keys) + " |")
        out.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in cross[:80]:
            out.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
        if len(cross) > 80:
            out.append(f"\n_… {len(cross) - 80} more rows in raw CSV._\n")
    if not het and not cross:
        out.append("_No tables._\n")
    return "\n".join(out) + "\n"


def section_plots(plots_dir: Path) -> str:
    if not plots_dir.exists():
        return "## 9 · Plots\n\n_Plot dir not present._\n"
    pngs = sorted(plots_dir.glob("*.png"))
    if not pngs:
        return "## 9 · Plots\n\n_No PNGs in plot dir._\n"
    out = ["## 9 · Plots (Stage F)\n"]
    for p in pngs:
        out.append(f"![{p.stem}](plots/{p.name})")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out",  required=True, type=Path)
    args = ap.parse_args()

    data = args.repo / "outputs/reports/data"
    plots = args.repo / "outputs/reports/plots"

    chunks: list[str] = [
        "# Phase 1 — KG audit report\n",
        "_Generated by `tools/consolidate_phase1_report.py`._  \n",
        "_Sources: every `outputs/reports/data/table_*.csv` produced by `scripts/phase1_audit_chain.sh` "
        "+ `outputs/v2/phase1/phase1_combined.csv` for the Morgan baseline._\n",
        "## Table of contents\n",
        "1. [Split sensitivity](#1--split-sensitivity--auroc-by-corpus--regime--model)\n"
        "2. [Contamination-score bins](#2--contamination-score-bins--auroc-vs-c-bin)\n"
        "3. [Path attribution](#3--path-attribution--dominant-axis-share-per-corpus-regime)\n"
        "4. [C-NN baseline](#4--c-nn-label-copying-via-kg-neighbours-baseline)\n"
        "5. [Provenance probes](#5--provenance--shortcut-probes)\n"
        "6. [Coverage](#6--coverage--fraction-of-test-rows-reachable-by-axis)\n"
        "7. [Robustness controls](#7--robustness--negative-controls)\n"
        "8. [Heterogeneity + cross-benchmark map](#8--per-target-heterogeneity--cross-benchmark-map)\n"
        "9. [Plots](#9--plots-stage-f)\n",
        section_split_sens(data),
        section_contam_bins(data),
        section_path_attr(data),
        section_cnn(data),
        section_provenance(data),
        section_coverage(data),
        section_robustness(data),
        section_heterogeneity(data),
        section_plots(plots),
    ]
    args.out.write_text("\n".join(chunks))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
