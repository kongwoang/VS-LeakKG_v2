"""Smoke test for tools/v2_to_sprint_csv.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl


def _load_tool():
    repo = Path(__file__).resolve().parents[2]
    p = repo / "tools" / "v2_to_sprint_csv.py"
    spec = importlib.util.spec_from_file_location("v2_to_sprint_csv", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_split():
    return pl.DataFrame({
        "example_id": ["Example::pdbbind::1abc", "Example::pdbbind::2def",
                       "Example::pdbbind::3ghi", "Example::pdbbind::4jkl",
                       "Example::pdbbind::5mno", "Example::pdbbind::6pqr"],
        "partition":  ["train", "train", "train", "val", "test", "test"],
        "ligand_id":  ["lig:1", "lig:2", "lig:3", "lig:4", "lig:5", "lig:6"],
        "protein_id": ["Protein::1abc", "Protein::2def",
                       "Protein::missing_in_lookup", "Protein::1abc",
                       "Protein::2def", "Protein::1abc"],
        "smiles":     ["CCO", "c1ccccc1", "CCN", "CCC",
                       "CCCC", None],   # last row has null SMILES
        "label":      [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    })


def _make_protein_seq():
    return pl.DataFrame({
        "target_id":       ["1abc", "2def"],
        "target_sequence": ["MKTAYIAKQRQ", "MGSSHHHHHH"],
        "source":          ["pdbbind", "pdbbind"],
    })


def test_emits_three_partitions(tmp_path):
    tool = _load_tool()
    split = _make_split()
    protein_seq = _make_protein_seq()
    out = tmp_path / "data"
    counts = tool.split_to_sprint_csvs(split, protein_seq, out)
    # train: 3 rows in, 1abc/2def found (2), 3rd is missing -> 2 rows out
    assert counts["train"] == 2
    assert counts["dropped_train"] == 1
    # val: 1abc found
    assert counts["val"] == 1
    # test: 2def found (1 row), 1abc was SMILES=None (dropped)
    assert counts["test"] == 1
    # Check the actual content
    train = pl.read_csv(out / "train.csv")
    assert list(train.columns) == ["SMILES", "Target Sequence", "Label"]
    assert train.height == 2
    assert "MKTAYIAKQRQ" in set(train["Target Sequence"].to_list())


def test_rejects_bad_split_schema(tmp_path):
    tool = _load_tool()
    bad_split = pl.DataFrame({"example_id": ["x"], "partition": ["train"]})
    protein_seq = _make_protein_seq()
    out = tmp_path / "data"
    try:
        tool.split_to_sprint_csvs(bad_split, protein_seq, out)
    except ValueError as e:
        assert "missing required columns" in str(e)
        return
    assert False, "should have raised ValueError"
