"""Tests for vsleakkg.v2.build_side_table.

Run against synthetic v1-shaped parquets so the test passes without
needing the unpacked dataset archive.
"""
from __future__ import annotations

import polars as pl
import pytest

from vsleakkg.v2 import build_side_table as bst
from vsleakkg.v2.hydrate import (
    Hydrator,
    KnownSource,
    SIDE_TABLE_COLUMNS,
    SIDE_TABLE_SCHEMA,
    validate_side_table,
)


def _make_v1_processed(tmp_path):
    """Create a tiny v1-shaped processed/ directory."""
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    # ChEMBL ligands
    pl.DataFrame({
        "chembl_id": ["CHEMBL1", "CHEMBL2"],
        "canonical_smiles": ["CCO", "c1ccccc1"],
        "standard_inchi_key": ["AAAA-AAAA-N", "BBBB-BBBB-N"],
    }).write_parquet(proc / "chembl_ligands.parquet")
    # BindingDB records
    pl.DataFrame({
        "bindingdb_id": ["BDB1", "BDB2"],
        "ligand_smiles": ["CC(=O)O", "CCN"],
        "target_uniprot": ["P00001", "P00002"],
        "target_sequence": ["MKTAYIAKQR", "MGSSHHHHHH"],
    }).write_parquet(proc / "bindingdb_records_minimal.parquet")
    # PDBBind
    pl.DataFrame({"pdb_id": ["1abc", "2xyz"],
                  "pk": [6.5, 7.2]}).write_parquet(proc / "pdbbind_index.parquet")
    pl.DataFrame({"pdb_id": ["1abc", "2xyz"],
                  "smiles": ["CCO", "c1ccccc1"]}).write_parquet(proc / "pdbbind_ligands.parquet")
    pl.DataFrame({"pdb_id": ["1abc", "2xyz"],
                  "uniprot": ["P11111", "P22222"],
                  "sequence": ["AAA", "BBB"]}).write_parquet(proc / "pdbbind_proteins.parquet")
    # LIT-PCBA (AVE)
    pl.DataFrame({
        "compound_id": ["lpc_1", "lpc_2"],
        "smiles": ["c1ccccc1", "CCO"],
        "inchikey": ["XXX-XXX-N", "YYY-YYY-N"],
        "uniprot": ["P00100", "P00100"],
        "label": [1.0, 0.0],
        "label_type": ["binary", "binary"],
    }).write_parquet(proc / "litpcba_ave_examples.parquet")
    # DUD-E
    pl.DataFrame({
        "compound_id": ["dude_1"],
        "smiles": ["CCC"],
        "label": [1.0],
        "label_type": ["active"],
    }).write_parquet(proc / "dude_examples.parquet")
    # DEKOIS
    pl.DataFrame({
        "compound_id": ["dek_1"],
        "smiles": ["CN"],
        "label": [0.0],
        "label_type": ["decoy"],
    }).write_parquet(proc / "dekois_examples.parquet")
    # BayesBind
    pl.DataFrame({
        "compound_id": ["bb_1"],
        "smiles_canonical": ["CCO"],
        "label": [5.5],
        "label_type": ["pAct"],
    }).write_parquet(proc / "bayesbind_examples.parquet")
    return proc


def test_build_side_table_all_sources(tmp_path, monkeypatch):
    _make_v1_processed(tmp_path)
    monkeypatch.setenv("VSLEAKKG_V1_ROOT", str(tmp_path))
    out = tmp_path / "side_table.parquet"
    counts = bst.build_side_table(out)
    # Sanity: every source has rows
    for src in (KnownSource.CHEMBL, KnownSource.BINDINGDB, KnownSource.PDBBIND,
                KnownSource.LITPCBA, KnownSource.DUDE, KnownSource.DEKOIS,
                KnownSource.BAYESBIND):
        assert counts.get(src.value, 0) > 0, f"no rows for {src.value}"
    assert counts["total"] >= sum(counts.get(s.value, 0) for s in KnownSource) - 5
    # Schema preserved
    df = pl.read_parquet(out)
    assert list(df.columns) == SIDE_TABLE_COLUMNS
    validate_side_table(df)


def test_build_side_table_subset(tmp_path, monkeypatch):
    _make_v1_processed(tmp_path)
    monkeypatch.setenv("VSLEAKKG_V1_ROOT", str(tmp_path))
    out = tmp_path / "side_table.parquet"
    counts = bst.build_side_table(out, sources=[KnownSource.LITPCBA, KnownSource.DUDE])
    assert counts[KnownSource.LITPCBA.value] == 2
    assert counts[KnownSource.DUDE.value] == 1
    # chembl/bindingdb/pdbbind/etc should be absent
    assert counts.get(KnownSource.CHEMBL.value, 0) == 0
    assert counts.get(KnownSource.BINDINGDB.value, 0) == 0


def test_hydrator_consumes_built_side_table(tmp_path, monkeypatch):
    _make_v1_processed(tmp_path)
    monkeypatch.setenv("VSLEAKKG_V1_ROOT", str(tmp_path))
    out = tmp_path / "side_table.parquet"
    bst.build_side_table(out)
    h = Hydrator.from_parquet(out)
    # Pick a known example_id
    result = h.hydrate(["pdbbind:1abc", "chembl:CHEMBL1", "litpcba:lpc_1"])
    assert result.found == 3
    pdbb_row = result.rows.filter(pl.col("example_id") == "pdbbind:1abc")
    assert pdbb_row[0, "uniprot"] == "P11111"
    assert pdbb_row[0, "label"] == pytest.approx(6.5)


def test_build_side_table_dedups_example_ids(tmp_path, monkeypatch):
    proc = _make_v1_processed(tmp_path)
    # duplicate one chembl row to verify dedup
    pl.DataFrame({
        "chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL_DUP"],
        "canonical_smiles": ["CCO", "CCO", "CCC"],
        "standard_inchi_key": ["A", "A", "B"],
    }).write_parquet(proc / "chembl_ligands.parquet")
    monkeypatch.setenv("VSLEAKKG_V1_ROOT", str(tmp_path))
    out = tmp_path / "side_table.parquet"
    counts = bst.build_side_table(out, sources=[KnownSource.CHEMBL])
    # 3 in, 2 unique by chembl_id
    assert counts[KnownSource.CHEMBL.value] == 2
