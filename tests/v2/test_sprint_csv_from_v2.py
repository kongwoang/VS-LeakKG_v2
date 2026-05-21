"""Smoke test for tools/sprint_csv_from_v2.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

from vsleakkg.v2.hydrate import (
    Hydrator,
    SIDE_TABLE_COLUMNS,
    SIDE_TABLE_SCHEMA,
)


def _load_sprint_tool():
    """Import tools/sprint_csv_from_v2.py as a module."""
    repo = Path(__file__).resolve().parents[2]
    p = repo / "tools" / "sprint_csv_from_v2.py"
    spec = importlib.util.spec_from_file_location("sprint_csv_from_v2", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _side_table(rows):
    base = {c: None for c in SIDE_TABLE_COLUMNS}
    base["label"] = 0.0
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        out.append(d)
    return pl.DataFrame(out, schema=SIDE_TABLE_SCHEMA)


def test_emits_three_partitions(tmp_path):
    sprint_tool = _load_sprint_tool()
    side = _side_table([
        {"example_id": "litpcba:1", "source": "litpcba", "source_id": "1",
         "smiles_canonical": "CCO", "target_sequence": "MKT", "label": 1.0},
        {"example_id": "litpcba:2", "source": "litpcba", "source_id": "2",
         "smiles_canonical": "CCN", "target_sequence": "MKT", "label": 0.0},
        {"example_id": "litpcba:3", "source": "litpcba", "source_id": "3",
         "smiles_canonical": "CCC", "target_sequence": "MKT", "label": 1.0},
    ])
    h = Hydrator(side)
    split = tmp_path / "split.parquet"
    pl.DataFrame({
        "example_id": ["litpcba:1", "litpcba:2", "litpcba:3"],
        "partition": ["train", "val", "test"],
    }).write_parquet(split)
    out = tmp_path / "data"
    counts = sprint_tool.split_to_sprint_csvs(split, h, out)
    assert counts == {"train": 1, "val": 1, "test": 1}
    train = pl.read_csv(out / "train.csv")
    assert list(train.columns) == ["Drug", "Target", "Y"]
    assert train["Drug"][0] == "CCO"


def test_drops_missing_target(tmp_path):
    sprint_tool = _load_sprint_tool()
    side = _side_table([
        {"example_id": "litpcba:1", "source": "litpcba", "source_id": "1",
         "smiles_canonical": "CCO", "target_sequence": None, "label": 1.0},
        {"example_id": "litpcba:2", "source": "litpcba", "source_id": "2",
         "smiles_canonical": "CCN", "target_sequence": "MKT", "label": 0.0},
    ])
    h = Hydrator(side)
    split = tmp_path / "split.parquet"
    pl.DataFrame({
        "example_id": ["litpcba:1", "litpcba:2"],
        "partition": ["train", "train"],
    }).write_parquet(split)
    out = tmp_path / "data"
    counts = sprint_tool.split_to_sprint_csvs(split, h, out)
    assert counts["train"] == 1  # the null-target one is dropped
    assert counts["val"] == 0
    assert counts["test"] == 0


def test_foldseek_variant(tmp_path):
    sprint_tool = _load_sprint_tool()
    side = _side_table([
        {"example_id": "litpcba:1", "source": "litpcba", "source_id": "1",
         "smiles_canonical": "CCO",
         "target_sequence": "MKT",
         "target_sequence_saprot": "MdKvTl",
         "label": 1.0},
    ])
    h = Hydrator(side)
    split = tmp_path / "split.parquet"
    pl.DataFrame({"example_id": ["litpcba:1"], "partition": ["train"]}).write_parquet(split)
    out = tmp_path / "data"
    sprint_tool.split_to_sprint_csvs(split, h, out, foldseek=True)
    assert (out / "train_foldseek.csv").exists()
    train = pl.read_csv(out / "train_foldseek.csv")
    assert train["Target"][0] == "MdKvTl"
