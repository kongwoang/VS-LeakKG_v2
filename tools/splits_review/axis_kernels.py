"""Per-axis residual contamination kernels for the splits-review benchmark.

For each axis a, define a similarity sim_a(e_i, e_j) ∈ [0, 1] between two
examples. The per-test residual is the max over train of sim_a:
    c_a(test_i) = max_{j ∈ train} sim_a(test_i, train_j)
Reported per-axis residual is the mean of c_a(test_i) over the test fold.

Axis catalogue:
    ligand          : Tanimoto on ECFP4 (continuous). Train sample capped at TRAIN_SAMPLE_CAP.
    scaffold        : 1 iff Bemis-Murcko scaffold matches some train scaffold (discrete).
    protein         : 1 iff target_id appears in train (discrete). Per-target-degenerate in Mode A.
    protein_family  : 1 iff protein_family appears in train (discrete).
    pocket          : 1 iff (target_id, pdb_id) appears in train (discrete). Often ≡ protein on VS corpora.
    assay           : 1 iff assay_id appears in train (discrete).
    source          : 1 iff source appears in train (discrete).
    time            : 1 / (1 + min_{j ∈ train} |year_i - year_j|) (continuous).

A column is marked:
    usable      : manifest has at least one non-null value AND not collapsed with another axis.
    degenerate  : column is non-null but collapses (e.g. assay ≡ target on LIT-PCBA; source constant).
    unavailable : column is all-null on this corpus.

The scoring method per axis is returned alongside the value for logging.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import polars as pl

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


TRAIN_SAMPLE_CAP = 5000  # for Tanimoto NN on ligand axis
RNG = np.random.default_rng(2025)


@dataclass
class AxisResult:
    c_mean:  float
    method:  str       # human-readable description of how it was scored
    status:  str       # "usable" | "degenerate" | "unavailable"


def _morgan_fps(smiles: list[str]):
    if not HAS_RDKIT:
        return [None] * len(smiles)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None)
    return out


def _set_membership(train: pl.Series, test: pl.Series) -> float:
    """Mean over test of 1[v in train_set]. Nulls in test are treated as not-in-set."""
    tset = set(v for v in train.to_list() if v is not None and v != "")
    if not tset:
        return 0.0
    vals = test.to_list()
    if not vals:
        return 0.0
    hits = sum(1 for v in vals if v in tset)
    return float(hits) / float(len(vals))


def axis_ligand(train: pl.DataFrame, test: pl.DataFrame) -> AxisResult:
    if not HAS_RDKIT or train.height == 0 or test.height == 0:
        return AxisResult(float("nan"), "ligand: skipped (RDKit missing or empty fold)", "unavailable")
    if train.height > TRAIN_SAMPLE_CAP:
        idx = RNG.choice(train.height, size=TRAIN_SAMPLE_CAP, replace=False)
        train_smi = [train["smiles"][int(i)] for i in idx]
        method_note = f"max Tanimoto ECFP4; train sampled to {TRAIN_SAMPLE_CAP}"
    else:
        train_smi = train["smiles"].to_list()
        method_note = "max Tanimoto ECFP4"
    train_fps = [f for f in _morgan_fps(train_smi) if f is not None]
    test_fps  = _morgan_fps(test["smiles"].to_list())
    if not train_fps:
        return AxisResult(float("nan"), "ligand: no valid train FPs", "unavailable")
    sims = []
    for q in test_fps:
        if q is None:
            sims.append(0.0); continue
        s = DataStructs.BulkTanimotoSimilarity(q, train_fps)
        sims.append(float(max(s) if s else 0.0))
    return AxisResult(float(np.mean(sims)), method_note, "usable")


def axis_scaffold(train: pl.DataFrame, test: pl.DataFrame) -> AxisResult:
    if "scaffold_smiles" not in train.columns:
        return AxisResult(float("nan"), "scaffold: column missing", "unavailable")
    val = _set_membership(train["scaffold_smiles"], test["scaffold_smiles"])
    return AxisResult(val, "fraction of test rows whose Bemis-Murcko scaffold appears in train",
                      "usable")


def axis_protein(train: pl.DataFrame, test: pl.DataFrame, mode: str) -> AxisResult:
    val = _set_membership(train["target_id"], test["target_id"])
    status = "degenerate" if mode == "A" else "usable"
    method = "fraction of test rows whose target_id appears in train" + \
             (" (per-target Mode A → degenerate)" if mode == "A" else "")
    return AxisResult(val, method, status)


def axis_protein_family(train: pl.DataFrame, test: pl.DataFrame, mode: str) -> AxisResult:
    if "protein_family" not in train.columns or train["protein_family"].is_null().all():
        return AxisResult(float("nan"), "protein_family: column unavailable", "unavailable")
    val = _set_membership(train["protein_family"], test["protein_family"])
    status = "degenerate" if mode == "A" else "usable"
    method = "fraction of test rows whose protein_family appears in train" + \
             (" (per-target Mode A → degenerate)" if mode == "A" else "")
    return AxisResult(val, method, status)


def axis_pocket(train: pl.DataFrame, test: pl.DataFrame, mode: str) -> AxisResult:
    if "pdb_id" not in train.columns or train["pdb_id"].is_null().all():
        return AxisResult(float("nan"),
            "pocket: pdb_id null on this corpus; pocket ≡ protein for these VS benchmarks",
            "degenerate")
    val = _set_membership(train["pdb_id"], test["pdb_id"])
    status = "degenerate" if mode == "A" else "usable"
    return AxisResult(val, "fraction of test rows whose pdb_id appears in train", status)


def axis_assay(train: pl.DataFrame, test: pl.DataFrame, corpus: str) -> AxisResult:
    # Hard-coded statuses per the protocol (DUD-E/DEKOIS unavailable; LIT-PCBA degenerate ≡ target).
    if corpus in ("dude", "dekois"):
        return AxisResult(float("nan"), "assay_id: collapsed during corpus construction", "unavailable")
    if corpus == "litpcba":
        return AxisResult(float("nan"), "assay_id: 1 AID per target → axis ≡ protein (degenerate)",
                          "degenerate")
    if "assay_id" not in train.columns or train["assay_id"].is_null().all():
        return AxisResult(float("nan"), "assay_id: column unavailable", "unavailable")
    val = _set_membership(train["assay_id"], test["assay_id"])
    return AxisResult(val, "fraction of test rows whose assay_id appears in train", "usable")


def axis_source(train: pl.DataFrame, test: pl.DataFrame, corpus: str) -> AxisResult:
    # All three VS corpora have a single dominant source per corpus → degenerate.
    return AxisResult(float("nan"),
        f"source: constant per corpus ({corpus}) → degenerate",
        "degenerate")


def axis_time(train: pl.DataFrame, test: pl.DataFrame, corpus: str) -> AxisResult:
    if "timestamp_year" not in train.columns or train["timestamp_year"].is_null().all():
        if corpus in ("dude", "dekois"):
            return AxisResult(float("nan"),
                "timestamp_year: decoys have no measurement date → unavailable",
                "unavailable")
        return AxisResult(float("nan"), "timestamp_year: column unavailable", "unavailable")
    train_years = [y for y in train["timestamp_year"].to_list() if y is not None]
    if not train_years:
        return AxisResult(float("nan"), "timestamp_year: no train years", "unavailable")
    train_arr = np.array(train_years)
    sims = []
    for y in test["timestamp_year"].to_list():
        if y is None:
            sims.append(0.0); continue
        d = int(np.min(np.abs(train_arr - y)))
        sims.append(1.0 / (1.0 + d))
    if not sims:
        return AxisResult(float("nan"), "timestamp_year: no test years", "unavailable")
    return AxisResult(float(np.mean(sims)),
        "mean over test of 1/(1+|year_i - nearest_train_year|)",
        "usable")


def score_all_axes(train: pl.DataFrame, test: pl.DataFrame, *, corpus: str, mode: str
                   ) -> dict[str, AxisResult]:
    return {
        "ligand":         axis_ligand(train, test),
        "scaffold":       axis_scaffold(train, test),
        "protein":        axis_protein(train, test, mode),
        "protein_family": axis_protein_family(train, test, mode),
        "pocket":         axis_pocket(train, test, mode),
        "assay":          axis_assay(train, test, corpus),
        "source":         axis_source(train, test, corpus),
        "time":           axis_time(train, test, corpus),
    }
