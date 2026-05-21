"""Shortcut diagnostic baselines (proposal section 5.14).

The four baselines are:
  - ligand_only:    Morgan-RF (Random Forest on Morgan fingerprints)
  - source_only:    logistic regression on dataset-source / decoy-protocol /
                    publication features. The v1 module
                    `vsleakkg.source_only_diagnostics` already implements an
                    equivalent baseline; the v2 wrapper here is a thin
                    adapter that respects v2 splits.
  - contamination_nn: implemented in v2/scoring.py:contamination_nn_label
  - dummy_receptor: replaces protein/pocket embedding with the training-set
                    mean, otherwise reuses the evaluated model.
"""
from __future__ import annotations
