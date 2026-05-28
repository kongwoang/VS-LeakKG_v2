# KG vs split-framework benchmark — protocol

Pre-registered protocol for the comparison of the KG multi-axis contamination-aware split against AVE, DrugOOD, and DataSAIL on active/decoy VS benchmarks. Implementation steps are in `kg_split_benchmark_implementation_plan.md`.

---

## 1. Corpora

Anchor (headline benchmark):
- **DUD-E** — 102 targets, 22,886 actives, ~1.4 M property-matched decoys (ZINC).
- **DEKOIS 2.0** — 81 targets, 40 actives + 1200 decoys per target (~98 k total).
- **LIT-PCBA** — 15 targets, 7,844 actives + 407,381 experimental inactives (PubChem).

Optional appendix (does not expand main scope):
- **LP-PDBBind** — complex-level extension. Only reuses existing Phase 1 v2 artefacts; no new framework runs; no new training; one summary table.

---

## 2. Splits produced per corpus

Twelve splits per corpus (where applicable per Table 3 of `kg_vs_split_frameworks_table.md`):

1. `random`
2. `scaffold` (Bemis-Murcko, naive grouping)
3. `protein` (MMseqs2 cluster, naive)
4. `ave_ligand` (AVE genetic-algorithm debias, per-target)
5. `drugood_scaffold`
6. `drugood_size`
7. `drugood_protein`
8. `drugood_family`
9. `datasail_s1_ligand`
10. `datasail_s1_protein`
11. `datasail_s2`
12. `kg_ligand_clean`, `kg_scaffold_clean`, `kg_protein_clean`, `kg_dual_clean` (the four KG regimes)

The DrugOOD `assay` axis is **N/A** on DUD-E and DEKOIS, **degenerate** on LIT-PCBA, and is not produced. The DrugOOD `size` split is treated as a control: it represents a weak split that any decent debias should beat.

Split ratios: train / val / test = 80 / 10 / 10, deterministic seed `2025`, applied identically across frameworks. Where a framework requires its own seed (DataSAIL ILP solver, AVE GA), the seed is logged.

---

## 3. Evaluation modes

The comparison is reported in **two clearly-labelled tables per corpus**. Cross-mode mixing is forbidden.

### Mode A — per-target (VS native)
- Unit of evaluation: a single target's actives + decoys.
- Splitter applies inside the target.
- Eligible splits: `random`, `scaffold`, `ave_ligand`, `drugood_scaffold`, `drugood_size`, `datasail_s1_ligand`, `kg_ligand_clean`, `kg_scaffold_clean`.
- Metric headline: per-target AUROC, EF1%, BEDROC, then mean ± Wilcoxon over the target set.
- Primary table in the paper.

### Mode B — pooled cross-target
- Unit of evaluation: corpus-wide pool of (target, ligand) pairs.
- Splitter applies across targets.
- Eligible splits: `random`, `scaffold`, `protein`, `drugood_*` (axis-permitting), `datasail_s1_*`, `datasail_s2`, `kg_protein_clean`, `kg_dual_clean`.
- Metric headline: pooled AUROC + per-target AUROC distribution + target / class composition tables.
- Pooled AUROC is **always** accompanied by the per-target distribution to prevent misleading pooled numbers.

---

## 4. Metrics

Reported for every (corpus × split × mode) cell.

### Split-quality metrics (do not require a trained model)
- Train / val / test size; per-class size; class-balance ratio.
- Dropped sample count and percentage.
- KG residual `C_total` (mean over test pairs against their nearest train neighbour).
- KG residual `c_a` for each axis `a ∈ {ligand, scaffold, protein, pocket, assay, source, time}`; axes flagged as **unavailable**, **degenerate**, or **usable** per Table 2 of `kg_vs_split_frameworks_table.md`.
- AVE bias `B` (the AVE paper's own metric, computed on every split, not only on `ave_ligand`).
- Max train→test ligand Tanimoto similarity (ECFP4).
- Max train→test protein sequence identity (MMseqs2).
- DataSAIL `L(π)` where definable.

### Model-based metrics (single shared model spec)
- Morgan-RF (radius 2, 2048 bits, scikit-learn RF n_estimators=500, max_depth=None, class_weight="balanced"): AUROC / EF1% / BEDROC, per-target distribution + corpus aggregate.
- 1-NN ligand baseline (per AVE paper: `KNN(k=1, metric="jaccard")` on ECFP4): AUROC / EF1%.
- KG C-NN label-copying baseline (existing `tools/run_cnn_baseline.py`): AUROC / EF1%.
- Per-target reporting is mandatory for DUD-E, DEKOIS, LIT-PCBA. Pooled metrics are secondary.

LigUnity, DrugCLIP, and any other trained VS model are explicitly out of scope here — they live in Phase 3.

---

## 5. AVE policy (per approval)

- Fixed iteration cap: **300 GA iterations** (LIT-PCBA default).
- No silent subsampling. If a target's active count > `N_MAX = 5000`, apply a deterministic subsample (`seed=2025`, `np.random.default_rng(seed).choice`) **before** the GA, log the original / subsampled counts.
- **Shared deterministic subset (binding rule).** When subsampling triggers on a target, the resulting manifest subset is written to `outputs/splits_review/<corpus>/manifests/subset_<target_id>.parquet` and is the **mandatory input manifest** for *every other splitter on that target* (random, scaffold, protein, DrugOOD-*, DataSAIL-*, KG-*). No splitter is allowed to consume the original manifest for a target that was subsampled. This is enforced by `compute_split_quality.py`, which fails the row if the splitter's input hash does not match the subset hash for a subsampled target.
- Per-target reporting: final `B`, dropped sample %, GA runtime in seconds, "subsampled" flag.
- A target whose final `B > 0.10` after 300 iterations is flagged in the result table (not dropped).

---

## 6. DataSAIL policy

- Solver: **SCIP** (open). Gurobi only if SCIP fails on a corpus; record solver used per run.
- S2 drop rate is reported per corpus. If S2 drops > 30 % of interactions, flag the run.
- Similarity: Tanimoto/ECFP4 for ligands, MMseqs2 identity for proteins. Same similarity inputs used for KG residual scoring to keep `L(π)` and `C_total` on comparable axes.

---

## 7. KG policy

- KG axes used (when populated): `ligand, scaffold, protein, protein_family, pocket, assay, source, time`. `protein_family` is a **distinct axis** from `protein`; the family axis groups proteins by UniProt class and is excluded from the weight mask whenever an unrelated protein is also a near neighbour on the `protein` axis (avoids double-counting).
- Per-corpus axis availability is the one in Table 2 of `kg_vs_split_frameworks_table.md`; unavailable / degenerate axes are excluded from the weight mask for that corpus. On LIT-PCBA the `assay` axis is **degenerate** with `protein` (one PubChem AID per target) and is therefore excluded from the LIT-PCBA weight mask — it is **not** treated as an independent assay axis.
- Greedy assignment with per-class quota; default `drop=False`.
- Reuse the existing Phase 1 axis kernels (`tools/run_contam_bins.py`, `tools/run_path_attribution.py`, `tools/run_cnn_baseline.py`); no new KG construction.

---

## 8. Statistical tests and CIs

- Per-target Wilcoxon signed-rank, paired across targets in Mode A. Test compared to `random` and to each non-KG framework on each metric (AUROC, EF1%, BEDROC).
- Holm-Bonferroni correction across the framework set, per metric and per corpus.
- 95 % CIs by 1000-iteration paired bootstrap over targets in Mode A; 1000-iteration bootstrap over (target, ligand) pairs in Mode B.
- Report all `p` and CIs alongside point estimates; do not gate publication of a number on its `p`.

---

## 9. Pre-registered hypotheses (and their honest negations)

**H1 — residual contamination.** KG splits yield strictly lower `C_total` than `random`, `scaffold`, `protein`, `ave_ligand`, `drugood_*`, `datasail_s1_*`, `datasail_s2` on every corpus.
- *If `C_total` for KG > DataSAIL S2 or > AVE on any corpus*, H1 is rejected for that corpus; the report states KG does **not** beat that framework on residual contamination there.

**H2 — shortcut performance.** KG splits yield lower Morgan-RF AUROC AND lower 1-NN AUROC AND lower C-NN AUROC than `random` and than `ave_ligand` / `datasail_s2` on each corpus (Mode A for `ave_ligand`, Mode B for `datasail_s2`).
- *If KG ties or loses on any of the three baselines*, H2 is rejected for that corpus.

**H3 — per-axis residual decomposition.** KG produces a per-axis residual `c_a` profile that no other framework reproduces, in particular on family/source/time/pocket where they have no axis at all.
- This is descriptive, not comparative. Always provable on LIT-PCBA on the `time` axis (PubChem AID dates) and the `protein_family` axis (after a 15-target UniProt lookup). The `assay` axis is **not** an independent dimension on LIT-PCBA — there is one PubChem AID per target, so `assay ≡ protein` and we report it as **degenerate** rather than claim a separate axis. On DUD-E/DEKOIS, the assay/source/decoy-time axes contribute ≈ 0 by construction.

**H4 — path attribution.** KG can name the dominant axis that explains each test pair's residual contamination. No external framework offers this.
- Always provable when KG runs. Reported as a property, not a metric.

**Headline framing rule.** The published headline is the conjunction of H1 and H2 with the largest framework comparator that holds. If H1 holds versus AVE and DrugOOD but not DataSAIL S2, the headline is "KG beats AVE and DrugOOD on residual contamination on DUD-E and DEKOIS, ties with DataSAIL S2." If neither H1 nor H2 hold for any non-trivial comparator, the headline is "KG matches the strongest framework on numeric leakage control while extending the axis coverage to assay/source/time/pocket and providing per-pair path attribution," and we ship that without dressing.

---

## 10. Deliverables tied to this protocol

The implementation file (`kg_split_benchmark_implementation_plan.md`) is the only place that lists tools / file paths / run order. This protocol commits to **what** is measured and **how** it is judged; the implementation file commits to **how** it is run.

Tables that this protocol obligates the run to produce, per corpus:
- `table_split_quality_modeA.csv`
- `table_split_quality_modeB.csv`
- `table_split_modelmetrics_modeA.csv`
- `table_split_modelmetrics_modeB.csv`
- `table_per_axis_residual.csv`
- `table_path_attribution.csv` (KG only)
- `table_stat_tests.csv` (Wilcoxon, Holm-corrected `p`, bootstrap CIs)

Plus one combined human-readable report at:
`outputs/reports/splits_review/kg_split_benchmark_report.md` (generated post-run).
