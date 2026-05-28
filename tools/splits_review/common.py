"""Shared utilities for the splits_review pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl

from .schemas import (
    CORPUS_MANIFEST_SCHEMA, SPLIT_SCHEMA,
    hash_manifest_slice,
)


def parse_common_args(description: str) -> argparse.ArgumentParser:
    """Common CLI shape for every splitter."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--subset-dir", required=False, type=Path, default=None,
                    help="Directory holding per-target subset_<target_id>.parquet "
                         "files written by splitter_ave. If present for a target, "
                         "the splitter MUST consume the subset file in place of "
                         "the manifest slice for that target.")
    ap.add_argument("--mode", required=True, choices=["A", "B"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", default=2025, type=int)
    return ap


def load_manifest(manifest_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(manifest_path)
    missing = set(CORPUS_MANIFEST_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return df


def materialise_per_target(
    manifest: pl.DataFrame, subset_dir: Path | None
) -> dict[str, pl.DataFrame]:
    """Per-target manifest slices, honouring the subset-manifest rule.

    Returns {target_id -> slice_df}. If subset_dir/<target>.parquet exists,
    that file replaces the slice from the manifest for that target.
    """
    out: dict[str, pl.DataFrame] = {}
    for tid in sorted(manifest["target_id"].unique().to_list()):
        slc = manifest.filter(pl.col("target_id") == tid)
        if subset_dir is not None:
            subf = subset_dir / f"subset_{tid}.parquet"
            if subf.exists():
                slc = pl.read_parquet(subf)
        out[tid] = slc
    return out


def write_split(rows: list[dict], out_path: Path, input_hash: str) -> None:
    """Write a split parquet matching SPLIT_SCHEMA."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows, schema=SPLIT_SCHEMA)
    df = df.with_columns(pl.lit(input_hash).alias("input_hash"))
    df.write_parquet(out_path)
    print(f"wrote {out_path}  n={df.height}  hash={input_hash}")


def fold_quotas(n: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[int, int, int]:
    n_train = int(round(n * ratios[0]))
    n_val   = int(round(n * ratios[1]))
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test
