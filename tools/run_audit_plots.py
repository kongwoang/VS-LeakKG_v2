"""Task 9: consolidated audit plots.

Reads accumulated CSVs under outputs/reports/data/ and produces PNG
figures for the consolidated report.

  Fig 1  split_sensitivity_heatmap.png    : AUROC by corpus x regime, Morgan-RF
  Fig 2  contam_bin_curves.png            : per-corpus per-regime AUROC vs bin C
  Fig 3  cnn_baseline_grouped_bar.png     : per-axis C-NN AUROC vs Morgan-RF
  Fig 4  dominant_axis_share.png          : path attribution stacked bar
  Fig 5  residual_contamination_box.png   : residual C distributions by corpus x regime
  Fig 6  threshold_sweep.png              : c_mean vs sweep tag (PDBBind protein)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_split_sensitivity(data_dir: Path, out: Path):
    df = pl.read_csv(data_dir / "table_split_sensitivity.csv")
    df = df.filter(pl.col("model") == "morgan_rf")
    df = df.with_columns(pl.col("auroc").cast(pl.Float64, strict=False))
    pivot = df.pivot(values="auroc", index="corpus", on="regime", aggregate_function="first")
    regimes_order = ["random", "ligand", "scaffold", "protein", "pocket", "dual"]
    cols = [c for c in regimes_order if c in pivot.columns]
    mat = np.array([[pivot.row(i, named=True).get(c, None) for c in cols] for i in range(pivot.height)], dtype=object)
    arr = np.array([[float("nan") if v is None else float(v) for v in row] for row in mat])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(pivot.height)); ax.set_yticklabels(pivot["corpus"].to_list())
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.7 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Morgan-RF AUROC")
    ax.set_title("Phase 1 split-sensitivity: Morgan-RF AUROC by (corpus x regime)")
    _save(fig, out)


def fig_contam_bins(data_dir: Path, out: Path):
    files = sorted(data_dir.glob("table_contam_bins_*.csv"))
    if not files:
        print("no contam-bin CSVs"); return
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    axes = axes.flatten()
    used = 0
    for f in files:
        if used >= len(axes): break
        d = pl.read_csv(f)
        body = d.filter(~pl.col("bin").str.starts_with("_"))
        body = body.filter(pl.col("auroc").is_not_null() & pl.col("n").cast(pl.Int64) >= 5)
        if body.height < 2: continue
        ax = axes[used]
        x = body["c_mean"].to_list(); y = body["auroc"].to_list()
        ax.plot(x, y, "o-", lw=1.5, ms=6)
        ax.set_title(f.stem.replace("table_contam_bins_", ""), fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("bin mean C"); ax.set_ylabel("AUROC")
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        used += 1
    for k in range(used, len(axes)): axes[k].axis("off")
    fig.suptitle("Contamination-bin AUROC trends", fontsize=12)
    fig.tight_layout()
    _save(fig, out)


def fig_cnn_bar(data_dir: Path, out: Path):
    f = data_dir / "table_cnn_baseline.csv"
    if not f.exists(): print("no cnn CSV"); return
    d = pl.read_csv(f)
    fig, ax = plt.subplots(figsize=(11, 5))
    keys = sorted({(r["corpus"], r["regime"]) for r in d.iter_rows(named=True)})
    bar_w = 0.10
    variants = ["all_axis", "ligand", "scaffold", "protein", "source"]
    palette = plt.get_cmap("tab10")
    for vi, v in enumerate(variants):
        ys = []
        for c, r in keys:
            if v == "all_axis":
                row = d.filter((pl.col("corpus") == c) & (pl.col("regime") == r) & (pl.col("variant") == "all_axis"))
            else:
                row = d.filter((pl.col("corpus") == c) & (pl.col("regime") == r) & (pl.col("axis") == v))
            ys.append(float(row["auroc"][0]) if row.height else float("nan"))
        x = np.arange(len(keys)) + vi * bar_w
        ax.bar(x, ys, bar_w, label=v, color=palette(vi))
    ax.set_xticks(np.arange(len(keys)) + bar_w * 2)
    ax.set_xticklabels([f"{c}/{r}" for c, r in keys], rotation=45, ha="right")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("C-NN AUROC")
    ax.set_title("Contamination-NN shortcut baseline (all-axis + per-axis)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, out)


def fig_dominant_axis(data_dir: Path, out: Path):
    f = data_dir / "table_path_attribution.csv"
    if not f.exists(): print("no path attribution CSV"); return
    d = pl.read_csv(f).filter(pl.col("scope") == "all_test")
    keys = sorted({(r["corpus"], r["regime"]) for r in d.iter_rows(named=True)})
    axes_order = ["protein", "pocket", "ligand", "scaffold", "assay", "source", "time"]
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(keys))
    palette = plt.get_cmap("Set3")
    for ai, axis in enumerate(axes_order):
        ys = []
        for c, r in keys:
            row = d.filter((pl.col("corpus") == c) & (pl.col("regime") == r) & (pl.col("axis") == axis))
            ys.append(float(row["share"][0]) if row.height else 0.0)
        ax.bar(range(len(keys)), ys, bottom=bottom, label=axis, color=palette(ai))
        bottom += np.array(ys)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{c}/{r}" for c, r in keys], rotation=45, ha="right")
    ax.set_ylabel("share of test rows (dominant axis)")
    ax.set_title("Path attribution: dominant axis share among all test rows")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    fig.tight_layout()
    _save(fig, out)


def fig_residual(data_dir: Path, out: Path):
    f = data_dir / "raw_table5_validation_contamination.csv"
    if not f.exists(): print("no residual CSV"); return
    d = pl.read_csv(f).filter(pl.col("direction") == "train->test")
    fig, ax = plt.subplots(figsize=(11, 4))
    keys = list(zip(d["corpus"].to_list(), d["regime"].to_list()))
    means = d["mean"].to_list()
    p90 = d["p90"].to_list()
    x = range(len(keys))
    ax.bar(x, means, color="steelblue", alpha=0.8, label="mean")
    ax.scatter(x, p90, color="darkred", marker="x", label="p90", s=30)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}/{r}" for c, r in keys], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("residual contamination C (train -> test)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Residual contamination after splits (train -> test)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)


def fig_threshold_sweep(data_dir: Path, out: Path):
    f = data_dir / "table_robustness_thresholds.csv"
    if not f.exists(): print("no threshold sweep CSV"); return
    d = pl.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(d["sweep"].to_list(), d["c_mean"].to_list(), color="slateblue")
    ax.set_ylabel("mean contamination C")
    ax.set_title(f"Threshold sensitivity ({d['corpus'][0]}/{d['regime'][0]})")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("outputs/reports/data"))
    ap.add_argument("--out-dir",  type=Path, default=Path("outputs/reports/plots"))
    args = ap.parse_args()

    fig_split_sensitivity(args.data_dir, args.out_dir / "split_sensitivity_heatmap.png")
    fig_contam_bins(args.data_dir, args.out_dir / "contam_bin_curves.png")
    fig_cnn_bar(args.data_dir, args.out_dir / "cnn_baseline_grouped_bar.png")
    fig_dominant_axis(args.data_dir, args.out_dir / "dominant_axis_share.png")
    fig_residual(args.data_dir, args.out_dir / "residual_contamination_bar.png")
    fig_threshold_sweep(args.data_dir, args.out_dir / "threshold_sweep.png")
    print("done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
