"""Smoke test for tools/build_protein_seq_lookup.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl


def _load_tool():
    repo = Path(__file__).resolve().parents[2]
    p = repo / "tools" / "build_protein_seq_lookup.py"
    spec = importlib.util.spec_from_file_location("build_protein_seq_lookup", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_pdbbind_proteins(processed_dir: Path) -> None:
    """Synth pdbbind_proteins.parquet with the v1 schema."""
    pl.DataFrame({
        "seq_sha256":      ["aaaa", "bbbb"],
        "sequence_concat": ["MKTAYIAKQRQ", "MGSSHHHHHHGSSDP"],
        "n_chains":        [1, 1],
        "n_residues":      [11, 15],
        "n_atoms":         [88, 120],
        "pdb_ids":         [["1abc", "2def"], ["3ghi"]],
        "n_complexes":     [2, 1],
    }).write_parquet(processed_dir / "pdbbind_proteins.parquet")


def test_pdbbind_lookup(tmp_path):
    tool = _load_tool()
    proc = tmp_path / "processed"
    proc.mkdir()
    _make_pdbbind_proteins(proc)
    out = tmp_path / "lookup.parquet"
    counts = tool.build_lookup(processed=proc, sources=["pdbbind"], output=out)
    # 2 + 1 = 3 unique pdb_ids
    assert counts["pdbbind"] == 3
    assert counts["total"] == 3
    df = pl.read_parquet(out)
    assert set(df["target_id"].to_list()) == {"1abc", "2def", "3ghi"}
    assert df.filter(pl.col("target_id") == "1abc")[0, "target_sequence"] == "MKTAYIAKQRQ"


def test_unknown_source_returns_empty(tmp_path):
    tool = _load_tool()
    proc = tmp_path / "processed"
    proc.mkdir()
    out = tmp_path / "lookup.parquet"
    counts = tool.build_lookup(processed=proc, sources=["nonsense_corpus"], output=out)
    assert counts["nonsense_corpus"] == 0
    assert counts["total"] == 0
