"""Thin DataSAIL wrapper (SCIP only).

Loads the corpus manifest, calls datasail.run with one of the three modes
(S1-ligand, S1-protein, S2), and emits a SPLIT_SCHEMA parquet. Drop counts
are recorded (only meaningful for S2).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl

from .common import write_split
from .schemas import hash_manifest_slice


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--datasail-mode", required=True,
                    choices=["s1_ligand", "s1_protein", "s2"])
    ap.add_argument("--mode",     required=True, choices=["A", "B"])
    ap.add_argument("--out",      required=True, type=Path)
    ap.add_argument("--seed",     default=2025, type=int)
    ap.add_argument("--subset-dir", required=False, type=Path, default=None)
    args = ap.parse_args()
    if args.mode != "B":
        raise SystemExit("DataSAIL S1/S2 used in Mode B only.")

    try:
        import datasail  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "datasail not installed. Activate datasail_env "
            "(see kg_split_benchmark_implementation_plan.md §1)."
        ) from e

    manifest = pl.read_parquet(args.manifest)
    # NOTE: this is a skeleton. Full integration requires:
    #   - converting manifest to datasail's expected input (ligand list,
    #     protein list, interaction table)
    #   - choosing solver = "SCIP"
    #   - mapping returned cluster->fold assignments back to example_ids
    # That code lands in the next iteration alongside the SCIP env install.
    raise SystemExit(
        "splitter_datasail: datasail integration not yet wired. "
        "Stage 0c lands the CLI skeleton; the SCIP env + datasail call land "
        "in Stage 1b."
    )


if __name__ == "__main__":
    raise SystemExit(main())
