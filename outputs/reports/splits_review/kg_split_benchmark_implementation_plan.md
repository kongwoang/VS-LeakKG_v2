# KG vs split-framework benchmark — implementation plan

Concrete file / tool / environment plan that turns the protocol (`kg_split_benchmark_protocol.md`) into runnable code. Reuses Phase 1 machinery where possible.

---

## 1. Environment

Two Python envs are needed:

- `drugclip_env` (existing on VUW at `/vol/dl-nguyenb5-solar/users/hoangpc/envs/drugclip_env/`). Used for Morgan-RF, KG axis kernels, AVE, KG splitters. Already has `scikit-learn`, `rdkit`, `polars`, `numpy`, `scipy`.
- `datasail_env` (new, minimal). Required because DataSAIL's ILP path is heavy. Channel: `bioconda`. Install:
  ```bash
  conda create -n datasail_env -c bioconda -c conda-forge python=3.11 datasail pyscipopt mmseqs2 -y
  ```
  Confirm `pyscipopt` brings the SCIP solver. Gurobi is **not** required and is not installed.

External binaries already on VUW: `mmseqs2`. Confirm with `which mmseqs`.

---

## 2. Existing tools reused (do not reimplement)

| Capability | Existing tool | Reuse as-is |
|---|---|---|
| KG per-axis contamination | `tools/run_contam_bins.py` | yes |
| KG path attribution | `tools/run_path_attribution.py` | yes |
| KG C-NN label-copying baseline | `tools/run_cnn_baseline.py` | yes |
| KG provenance probes | `tools/run_provenance_probes.py` | yes (for Mode B sanity check on assay/source/time) |
| KG coverage summary | `tools/run_coverage_summary.py` | yes (for Table 2 / metadata audit) |
| Phase 1 combined CSV → audit-report builder | `tools/build_split_sensitivity.py`, `tools/consolidate_phase1_report.py` | adapt for the splits-review report |

---

## 3. New tools to write (under `tools/splits_review/`)

| Tool | Purpose |
|---|---|
| `make_corpus_manifest.py` | Build one `corpus_manifest.parquet` per corpus listing `(target_id, ligand_id, smiles, label, pdb_id, uniprot, family, assay_id_or_na, source_or_na, timestamp_or_na)`. Three modes: `dude`, `dekois`, `litpcba`. |
| `splitter_random.py` | Trivial. |
| `splitter_scaffold.py` | Bemis-Murcko (RDKit), naive grouping. |
| `splitter_protein.py` | MMseqs2 cluster on UniProt sequences. |
| `splitter_ave.py` | Port of Wallach `remove_AVE_bias.py`; iteration cap 300; deterministic subsample at `N_MAX=5000` actives/target; logs `B`, drop %, runtime. |
| `splitter_drugood.py` | One file with subcommands `scaffold`, `size`, `protein`, `family`. DrugOOD's domain-sort + disjoint-assign procedure. (`assay` is intentionally absent.) |
| `splitter_datasail.py` | Thin wrapper around `datasail.run` with the three modes (`s1_ligand`, `s1_protein`, `s2`); enforces SCIP solver; logs `L(pi)`, drop count. |
| `splitter_kg.py` | Adapter onto existing KG split builder; produces `ligand_clean`, `scaffold_clean`, `protein_clean`, `dual_clean` from the corpus manifest, using only axes flagged usable for that corpus. |
| `compute_split_quality.py` | Reads a split file + corpus manifest, emits the split-quality row (sizes, max-similarities, `B`, `C_total`, per-axis `c_a`, `L(pi)`). |
| `compute_model_metrics.py` | Trains Morgan-RF, 1-NN, KG C-NN per split; emits per-target AUROC/EF1%/BEDROC + pooled. |
| `compute_stats.py` | Wilcoxon + Holm + paired bootstrap; emits `table_stat_tests.csv`. |
| `build_report.py` | Stitches the seven CSVs into `kg_split_benchmark_report.md`. |

Each splitter accepts the same CLI shape:
```
python -m tools.splits_review.splitter_<name>  \
    --manifest outputs/splits_review/<corpus>/corpus_manifest.parquet  \
    --subset-dir outputs/splits_review/<corpus>/manifests/  \
    --mode {A,B}  \
    --out outputs/splits_review/<corpus>/splits/<splitter>_mode<mode>.parquet  \
    --seed 2025
```
**Subset-manifest rule (binds every splitter).** If `<subset-dir>/subset_<target_id>.parquet` exists for a target, the splitter MUST load that file in place of the slice of `<manifest>` for that target. This enforces the AVE-policy contract (Section 5 of the protocol): when AVE subsampling fires on a target, every other splitter sees the same manifest subset. `splitter_ave.py` writes the subset file; `splitter_random.py`, `splitter_scaffold.py`, etc. read it; `compute_split_quality.py` fails any split whose target-slice hash disagrees with the subset-file hash.

Output schema is fixed (one row per (target_id, ligand_id) with `fold ∈ {train, val, test}`, plus an `input_hash` column for the subset-rule check).

---

## 4. Run order

Phase 1 of the run — manifest + metadata audit (one-time, cheap):
1. `make_corpus_manifest.py` × 3 corpora.
2. `run_coverage_summary.py` × 3 corpora → confirms the metadata table in `kg_vs_split_frameworks_table.md`.

Phase 2 — split production (per corpus, per mode):
3. Run every applicable splitter from the runnability matrix (Table 3). One split file per (corpus × splitter × mode).

Phase 3 — split-quality scoring (no models trained):
4. `compute_split_quality.py` over every split → fills `table_split_quality_mode{A,B}.csv` and `table_per_axis_residual.csv`.
5. `run_path_attribution.py` over every KG split → fills `table_path_attribution.csv`.

Phase 4 — model-metric scoring:
6. `compute_model_metrics.py` over every split → fills `table_split_modelmetrics_mode{A,B}.csv`. Order corpora by cost: LIT-PCBA (small) → DEKOIS → DUD-E.

Phase 5 — statistics:
7. `compute_stats.py` → `table_stat_tests.csv`.

Phase 6 — report:
8. `build_report.py` → `kg_split_benchmark_report.md`.

Optional appendix:
9. Reuse Phase 1 LP-PDBBind artefacts in `outputs/v2/phase1/phase1_combined.csv` to fill a short "complex-level extension" subsection. No new training, no new splitters run. If the artefacts are missing, the appendix is dropped silently — it must not block the main report.

---

## 5. Compute budget and ordering

Rough wall-clock on VUW, single CPU thread per splitter (most are CPU-bound), GPU only for KG C-NN scoring of large splits:

| Step | LIT-PCBA | DEKOIS | DUD-E |
|---|---|---|---|
| manifest | <1 min | <1 min | ~5 min (large) |
| AVE GA (300 iter, ≤5k actives/target) | <1 h | ~2 h | ~12 h (102 targets) |
| DataSAIL S2 (SCIP) | <30 min | ~1 h | ~6 h |
| Morgan-RF (all splits) | ~30 min | ~2 h | ~10 h |
| C-NN scoring | minutes | <1 h | ~3 h |

Run LIT-PCBA end-to-end first to shake out bugs before pointing the pipeline at DEKOIS / DUD-E.

---

## 6. Output layout (mirrors the protocol's deliverables)

```
outputs/splits_review/
├── litpcba/
│   ├── corpus_manifest.parquet
│   ├── splits/
│   │   ├── random_modeA.parquet
│   │   ├── ave_ligand_modeA.parquet
│   │   ├── kg_ligand_clean_modeA.parquet
│   │   ├── datasail_s1_ligand_modeA.parquet
│   │   ├── ...
│   │   └── datasail_s2_modeB.parquet
│   └── data/
│       ├── table_split_quality_modeA.csv
│       ├── table_split_quality_modeB.csv
│       ├── table_split_modelmetrics_modeA.csv
│       ├── table_split_modelmetrics_modeB.csv
│       ├── table_per_axis_residual.csv
│       ├── table_path_attribution.csv
│       └── table_stat_tests.csv
├── dekois/  (same shape)
├── dude/    (same shape)
└── _appendix_pdbbind/
    └── pdbbind_split_quality_appendix.csv   (only if Phase 1 v2 artefacts exist)

outputs/reports/splits_review/
├── split_frameworks_summary.md              (this doc set, already authored)
├── kg_vs_split_frameworks_table.md
├── kg_split_benchmark_protocol.md
├── kg_split_benchmark_implementation_plan.md
└── kg_split_benchmark_report.md             (generated by build_report.py post-run)
```

---

## 7. Reproducibility hooks

- All seeds fixed (`seed = 2025`) and logged per row in every output table.
- DataSAIL solver name + version logged.
- AVE GA termination reason logged (`converged_below_B_thresh`, `hit_iter_cap`).
- MMseqs2 version + parameters logged.
- All tool versions captured by `pip freeze` saved as `outputs/splits_review/_env/<env>_freeze.txt`.

---

## 8. Risks and contingencies

- **DataSAIL S2 OOM on DUD-E.** Mitigation: pre-cluster ligands with `mmseqs easy-cluster` at 0.7 identity to shrink the ILP variable count. If still infeasible, report S2 on DUD-E as solver-limited and run S2 on DEKOIS + LIT-PCBA only.
- **AVE GA wall-time on DUD-E.** Mitigation: the 5 k cap + 300 iter cap is the contract; record `hit_iter_cap` and the residual `B` honestly. We do not extend the cap to chase a smaller `B`.
- **Protein-family axis missing on DEKOIS for some targets.** Mitigation: UniProt batch lookup with a written fallback (BindingDB target-class field); list any unmapped targets in a footnote.
- **Mode B pooled AUROC misleading.** Mitigation: the protocol obligates the per-target distribution alongside any pooled AUROC; `build_report.py` enforces this by failing if a Mode B row is missing the per-target distribution column.
- **LP-PDBBind appendix expansion.** Mitigation: the appendix is reuse-only. If a new artefact would be needed, the appendix is cut.

---

## 9. Not in scope (explicit)

- Retraining LigUnity, DrugCLIP, or any other VS model on these splits. That is the Phase 3 question and belongs in `phase3_model_audit_report.md`.
- New KG construction. We reuse the existing v2 KG.
- New corpora beyond the three anchors plus the PDBBind appendix.
- DrugOOD's `assay` axis on DUD-E / DEKOIS (N/A) or LIT-PCBA (degenerate).
- Per-pair contamination scoring on the synthetic decoy side for `source` / `time` axes — those axes are uninformative by construction on DUD-E / DEKOIS and we say so rather than fabricate a number.
