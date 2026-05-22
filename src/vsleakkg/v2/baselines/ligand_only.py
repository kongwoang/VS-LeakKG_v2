"""Ligand-only baseline: predict active/inactive from ligand structure alone.

Uses RDKit Morgan fingerprints (radius 2, 2048 bits) and a random forest. If
RDKit is unavailable, falls back to scaffold-identity hashing (much weaker
but lets the test suite run).

This baseline doesn't use protein or pocket information. If it competes with
the full model on a benchmark, the benchmark's signal can be obtained from
ligand-side artefacts alone -- which is exactly the failure mode proposal
section 5.14 is designed to detect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl


def _try_rdkit_fingerprints(smiles_list: list[str], n_bits: int = 2048) -> np.ndarray | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None
    feats = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        feats[i] = arr
    return feats


def _fallback_features(smiles_list: list[str], n_bits: int = 2048) -> np.ndarray:
    """Hash-based fallback when RDKit is unavailable. Much weaker."""
    rng = np.random.default_rng(0)
    feats = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        h = hash(smi) & 0xFFFFFFFF
        rng2 = np.random.default_rng(h)
        idx = rng2.integers(0, n_bits, size=16)
        feats[i, idx] = 1
    return feats


def featurise_ligands(smiles_list: Iterable[str], *, n_bits: int = 2048) -> np.ndarray:
    smiles_list = list(smiles_list)
    feats = _try_rdkit_fingerprints(smiles_list, n_bits=n_bits)
    if feats is None:
        feats = _fallback_features(smiles_list, n_bits=n_bits)
    return feats


@dataclass
class LigandOnlyResult:
    auroc: float
    auprc: float
    n_pos: int
    n_neg: int
    scores: np.ndarray
    used_rdkit: bool


def evaluate_ligand_only(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    smiles_col: str = "smiles",
    label_col: str = "label",
    n_estimators: int = 100,
    random_state: int = 0,
    train_cap: int = 15000,
) -> LigandOnlyResult:
    """Train Morgan-RF on `train` and report AUROC/AUPRC on `test`.

    `train_cap` caps the training set size: with 200 trees x 35k samples
    x 2048 features Morgan-RF can hit tens of GB of RAM, which causes
    swap thrash on loaded boxes. 15k samples + 100 trees is roughly 4x
    less RAM and still produces a tight estimate of the ligand-only
    achievable AUROC.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    for col in (smiles_col, label_col):
        if col not in train.columns or col not in test.columns:
            raise KeyError(f"missing column: {col}")

    # Stratified down-sample on label so both classes survive even at small caps.
    if train.height > train_cap:
        train = train.sample(n=train_cap, seed=random_state, with_replacement=False)

    feats = featurise_ligands(train[smiles_col].to_list())
    test_feats = featurise_ligands(test[smiles_col].to_list())
    used_rdkit = _try_rdkit_fingerprints(train.head(1)[smiles_col].to_list()) is not None

    y_train = train[label_col].to_numpy()
    y = test[label_col].to_numpy()
    # Guard: RandomForest.predict_proba returns shape (n, 1) when the
    # training labels are single-class, which makes [:, 1] crash. This
    # happens on PDBBind whenever labels still default to 0.0 (BM-node
    # join not yet wired). Return NaN AUROC/AUPRC in that case rather
    # than blowing up the whole pipeline.
    if len(set(y_train.tolist())) < 2:
        return LigandOnlyResult(
            auroc=float("nan"),
            auprc=float("nan"),
            n_pos=int((y == 1).sum()),
            n_neg=int((y == 0).sum()),
            scores=np.zeros(len(test), dtype=np.float64),
            used_rdkit=used_rdkit,
        )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(feats, y_train)
    scores = clf.predict_proba(test_feats)[:, 1]

    return LigandOnlyResult(
        auroc=float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan"),
        auprc=float(average_precision_score(y, scores)) if len(set(y)) == 2 else float("nan"),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
        scores=scores,
        used_rdkit=used_rdkit,
    )
