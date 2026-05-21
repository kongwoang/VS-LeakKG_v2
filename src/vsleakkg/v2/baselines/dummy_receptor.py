"""Dummy-receptor baseline (proposal section 5.14).

Replaces the protein / pocket representation with a fixed dummy vector (the
training-set mean by default) while keeping the ligand-side input intact.
Strong performance under this baseline indicates that the model is not
exploiting target-specific signal.

This module is a thin wrapper: it does not know how any given SBVS model
encodes proteins. Callers pass in a function that maps protein_id -> vector
and the wrapper returns a copy of that function with the mean vector
substituted.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def build_dummy_protein_encoder(
    train_protein_ids: list[str],
    protein_encoder: Callable[[str], np.ndarray],
    *,
    mode: str = "mean",
) -> Callable[[str], np.ndarray]:
    """Return a function that ignores its protein_id argument and returns a
    fixed dummy vector.

    Modes:
      'mean' : training-set mean embedding (default; recommended).
      'zero' : zero vector.
      'random': fixed random vector (use seed 0).

    Note: the returned function still accepts a protein_id argument so it can
    drop-in replace the original encoder.
    """
    if mode == "mean":
        vecs = np.stack([protein_encoder(pid) for pid in train_protein_ids])
        dummy = vecs.mean(axis=0)
    elif mode == "zero":
        sample = protein_encoder(train_protein_ids[0])
        dummy = np.zeros_like(sample)
    elif mode == "random":
        sample = protein_encoder(train_protein_ids[0])
        rng = np.random.default_rng(0)
        dummy = rng.standard_normal(sample.shape).astype(sample.dtype)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    def dummy_encoder(_protein_id: str, _cache: dict = {"v": dummy}) -> np.ndarray:
        return _cache["v"]

    return dummy_encoder


def dummy_receptor_report(
    full_metric: float,
    dummy_metric: float,
) -> dict[str, float]:
    """How much of the full-model performance survives a dummy receptor?

    Returns absolute and relative retention. A retention near 1.0 means the
    model is essentially ligand-only.
    """
    drop_abs = full_metric - dummy_metric
    retention = dummy_metric / full_metric if full_metric else 0.0
    return {
        "full_metric": full_metric,
        "dummy_metric": dummy_metric,
        "drop_absolute": drop_abs,
        "retention": retention,
    }
