"""Tests for vsleakkg.v2.hydrate.

These exercise the *consumer* side of the side-table. The Linux builder
that actually populates the parquet from v1 raw loaders is tested
separately on the GPU box once the loaders are available.
"""
from __future__ import annotations

import polars as pl
import pytest

from vsleakkg.v2.hydrate import (
    SIDE_TABLE_COLUMNS,
    SIDE_TABLE_SCHEMA,
    Hydrator,
    KnownSource,
    canonicalize_smiles,
    make_example_id,
    parse_example_id,
    validate_side_table,
)


def _synth_side_table(extra_rows: list[dict] | None = None) -> pl.DataFrame:
    """A 3-row side-table covering ChEMBL, BindingDB, and PDBBind."""
    rows = [
        {
            "example_id": "chembl:ACT_1",
            "source": "chembl",
            "source_id": "ACT_1",
            "smiles": "CCO",
            "smiles_canonical": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "uniprot": "P12345",
            "target_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLG",
            "target_sequence_saprot": None,
            "pdb_id": None,
            "chembl_id": "CHEMBL12345",
            "bindingdb_id": None,
            "assay_id": "CHEMBL_ASSAY_1",
            "label": 7.5,
            "label_kind": "pact",
        },
        {
            "example_id": "bindingdb:BDB_1",
            "source": "bindingdb",
            "source_id": "BDB_1",
            "smiles": "c1ccccc1",
            "smiles_canonical": "c1ccccc1",
            "inchikey": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
            "uniprot": "P67890",
            "target_sequence": None,
            "target_sequence_saprot": None,
            "pdb_id": None,
            "chembl_id": None,
            "bindingdb_id": "BDB_1",
            "assay_id": "BDB_ASSAY_42",
            "label": 1.0,
            "label_kind": "binary",
        },
        {
            "example_id": "pdbbind:1abc",
            "source": "pdbbind",
            "source_id": "1abc",
            "smiles": "CC(=O)O",
            "smiles_canonical": "CC(=O)O",
            "inchikey": None,
            "uniprot": "P11111",
            "target_sequence": None,
            "target_sequence_saprot": None,
            "pdb_id": "1abc",
            "chembl_id": None,
            "bindingdb_id": None,
            "assay_id": None,
            "label": 6.2,
            "label_kind": "pact",
        },
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return pl.DataFrame(rows, schema=SIDE_TABLE_SCHEMA)


def test_make_example_id_round_trips():
    eid = make_example_id(KnownSource.CHEMBL, "ACT_42")
    assert eid == "chembl:ACT_42"
    src, sid = parse_example_id(eid)
    assert src == "chembl"
    assert sid == "ACT_42"


def test_make_example_id_accepts_string_source():
    eid = make_example_id("dude", "AKT1__decoy_17")
    src, sid = parse_example_id(eid)
    assert src == "dude"
    assert sid == "AKT1__decoy_17"


def test_make_example_id_rejects_bad_inputs():
    with pytest.raises(ValueError):
        make_example_id("bad:source", "x")
    with pytest.raises(ValueError):
        make_example_id("chembl", "")
    with pytest.raises(ValueError):
        make_example_id("chembl", ":leading_colon")


def test_parse_example_id_rejects_malformed():
    for bad in ["no_colon", "::empty", "", "Trailing:"]:
        with pytest.raises(ValueError):
            parse_example_id(bad)


def test_canonicalize_smiles_returns_none_for_garbage_or_no_rdkit():
    # Either no RDKit (returns None for everything) or RDKit installed
    # and the garbage string fails to parse (also None).
    assert canonicalize_smiles(None) is None
    assert canonicalize_smiles("not a smiles!!!") is None


def test_validate_side_table_accepts_well_formed():
    df = _synth_side_table()
    validate_side_table(df)  # no raise


def test_validate_side_table_rejects_missing_column():
    df = _synth_side_table().drop("inchikey")
    with pytest.raises(ValueError, match="missing required columns"):
        validate_side_table(df)


def test_validate_side_table_rejects_extra_column():
    df = _synth_side_table().with_columns(pl.lit("extra").alias("oops"))
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_side_table(df)


def test_validate_side_table_rejects_dtype_mismatch():
    df = _synth_side_table().with_columns(pl.col("label").cast(pl.Utf8))
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_side_table(df)


def test_validate_side_table_rejects_duplicate_example_ids():
    base = _synth_side_table()
    dup = pl.concat([base, base[[0]]])
    with pytest.raises(ValueError, match="duplicate example_id"):
        validate_side_table(dup)


def test_validate_side_table_rejects_unknown_source():
    # Build a row with an unknown source value.
    bad = {col: None for col in SIDE_TABLE_COLUMNS}
    bad["example_id"] = "wikipedia:foo"
    bad["source"] = "wikipedia"
    bad["source_id"] = "foo"
    bad["label"] = 0.0
    df = pl.DataFrame([bad], schema=SIDE_TABLE_SCHEMA)
    with pytest.raises(ValueError, match="unknown source values"):
        validate_side_table(df)


def test_hydrator_hits_and_misses():
    h = Hydrator(_synth_side_table())
    result = h.hydrate(["chembl:ACT_1", "pdbbind:1abc", "chembl:not_present"])
    assert result.found == 2
    assert result.missing == ["chembl:not_present"]
    assert result.coverage == pytest.approx(2 / 3)
    eids = result.rows.get_column("example_id").to_list()
    assert eids == ["chembl:ACT_1", "pdbbind:1abc"]


def test_hydrator_all_missing_returns_empty_frame_with_schema():
    h = Hydrator(_synth_side_table())
    result = h.hydrate(["nope:1", "nada:2"])
    assert result.found == 0
    assert result.missing == ["nope:1", "nada:2"]
    # schema preserved
    assert list(result.rows.columns) == SIDE_TABLE_COLUMNS


def test_hydrator_contains_and_len():
    h = Hydrator(_synth_side_table())
    assert len(h) == 3
    assert "chembl:ACT_1" in h
    assert "bogus:99" not in h


def test_hydrator_from_parquet_roundtrip(tmp_path):
    df = _synth_side_table()
    p = tmp_path / "side_table.parquet"
    df.write_parquet(p)
    h = Hydrator.from_parquet(p)
    assert len(h) == 3
    result = h.hydrate(["bindingdb:BDB_1"])
    assert result.found == 1
    assert result.rows[0, "uniprot"] == "P67890"
