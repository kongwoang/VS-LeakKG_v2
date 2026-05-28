"""Shared schemas for the KG vs split-frameworks benchmark.

Every splitter writes a parquet that conforms to SPLIT_SCHEMA. Every quality
table writes a CSV that conforms to its corresponding TABLE_*_SCHEMA. The
schemas are the contract between stages; build_report.py validates against
them.
"""
from __future__ import annotations
import hashlib
import polars as pl


CORPUS_MANIFEST_SCHEMA = {
    "example_id":     pl.Utf8,
    "target_id":      pl.Utf8,
    "ligand_id":      pl.Utf8,
    "smiles":         pl.Utf8,
    "label":          pl.Int64,   # 0 = inactive/decoy, 1 = active
    "scaffold_smiles": pl.Utf8,   # may be null when RDKit fails
    "uniprot":        pl.Utf8,    # null if unmapped
    "protein_family": pl.Utf8,    # null if unmapped
    "pdb_id":         pl.Utf8,    # null if unmapped
    "assay_id":       pl.Utf8,    # null if unavailable
    "source":         pl.Utf8,    # null if unavailable
    "timestamp_year": pl.Int64,   # null if unavailable
}

SPLIT_SCHEMA = {
    "example_id":  pl.Utf8,
    "target_id":   pl.Utf8,
    "ligand_id":   pl.Utf8,
    "label":       pl.Int64,
    "fold":        pl.Utf8,        # "train" | "val" | "test"
    "input_hash":  pl.Utf8,        # hash of the manifest slice the splitter consumed
}

# Output CSV column lists (order matters for the report builder)

TABLE_SPLIT_QUALITY_COLUMNS = [
    "corpus", "mode", "splitter", "target_id",
    "n_train", "n_val", "n_test", "n_dropped",
    "n_train_pos", "n_train_neg", "n_test_pos", "n_test_neg",
    "class_balance_train", "class_balance_test",
    "c_total_mean",
    # per-axis residuals
    "c_ligand", "c_scaffold", "c_protein", "c_protein_family",
    "c_pocket", "c_assay", "c_source", "c_time",
    # framework-native metrics
    "ave_bias_B", "max_lig_tanimoto", "max_prot_identity",
    "datasail_L_pi",
    # provenance
    "input_hash", "seed", "runtime_s",
]

TABLE_MODELMETRICS_COLUMNS = [
    "corpus", "mode", "splitter", "target_id",
    "model",                # "morgan_rf" | "knn1_ligand" | "kg_cnn"
    "auroc", "ef1pct", "bedroc",
    "n_test_pos", "n_test_neg",
    "input_hash", "seed",
]

TABLE_PER_AXIS_RESIDUAL_COLUMNS = [
    "corpus", "mode", "splitter", "axis",
    "c_mean", "c_p50", "c_p90", "c_p99",
    "axis_status",  # "usable" | "degenerate" | "unavailable"
]

TABLE_PATH_ATTRIBUTION_COLUMNS = [
    "corpus", "mode", "splitter",
    "axis", "share",  # axis = dominant axis of test rows; share in [0, 1]
    "n_test",
]

TABLE_STAT_TESTS_COLUMNS = [
    "corpus", "mode", "metric", "splitter_a", "splitter_b",
    "n", "mean_diff",
    "wilcoxon_stat", "p_value", "p_holm",
    "ci95_lo", "ci95_hi",
]


def hash_manifest_slice(df: pl.DataFrame) -> str:
    """Stable hash of the (example_id, label) pairs in a manifest slice.

    Used to enforce the subset-manifest rule (Section 5 of the protocol):
    if AVE subsampled a target, every other splitter for that target must
    read the subset file and so produce the same hash.
    """
    rows = sorted(zip(df["example_id"].to_list(), df["label"].to_list()))
    h = hashlib.sha256()
    for ex, lab in rows:
        h.update(ex.encode()); h.update(b"|"); h.update(str(lab).encode()); h.update(b"\n")
    return h.hexdigest()[:16]
