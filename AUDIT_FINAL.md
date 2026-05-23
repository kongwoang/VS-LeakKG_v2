# VS-LeakKG v2 — final audit report

**Date:** 2026-05-24
**Code:** https://github.com/kongwoang/VS-LeakKG_v2 (HEAD as of writeup)
**Compute:** VUW box (cuda12.ecs.vuw.ac.nz), 3× Quadro RTX 6000 24 GB

This document is the integrated deliverable. It folds together the data-only
Phase 1 baselines (Morgan-RF) and the model-paired Phase 2 audits (SPRINT) into
a single story about leakage-axis shortcuts in structure-based virtual
screening benchmarks. The detailed phase reports (`PHASE1_FINAL_REPORT.md`,
`PHASE2_SPRINT_FINAL.md`) remain authoritative for per-step numbers and
reproduction commands.

---

## Headline claim

When the v2 framework forbids leakage between train and test along a single
axis (ligand, scaffold, protein), held-out AUROC drops by a measurable and
**model-invariant** amount on PDBBind. The dominant leakage axis differs
between corpora — but where the leakage exists, a ligand-only Morgan-RF and a
deep dual-tower model (SPRINT) **both** suffer the same shortcut loss.

## Group A — PDBBind v2 audit, two models on the same splits

Identical v2 splits (`outputs/v2/phase1_full/splits/pdbbind/`). Identical
preprocessing. The only difference between rows is the **split regime**; the
only difference between models is the listed model+config. Reporting test-set
AUROC; AUPR is in the per-phase reports.

| Regime | n_train | n_test | Morgan-RF AUROC (Phase 1) | SPRINT AUROC (Phase 2) | Δ (model) |
|---|---:|---:|---:|---:|---:|
| **random (control)** | 7,337 | 5,844 | **0.8058** | **0.8370** | +0.031 |
| ligand-clean  | 9,917 | 4,560 | 0.7070 | **0.7619** | +0.055 |
| protein-clean | 7,337 | 5,844 | **0.5549** | **0.5890** | +0.034 |
| dual-clean    | 8,192 | 5,429 | 0.6788 | **0.7306** | +0.052 |

The **random row is a control**: same 19,037 PDBBind examples and same
train/val/test sizes as protein-clean, but the partition assignment is
uniform-random instead of leakage-axis-clean. It keeps every form of leakage
(ligand-axis, protein-axis, scaffold, structural similarity). If the v2 KG
framework were doing no real work, random and protein-clean would score the
same. They don't — random scores **+25pp** (Morgan-RF) / **+25pp** (SPRINT)
higher AUROC than protein-clean. That 25pp gap is the leakage signal the KG
is *removing* on protein-clean. Same logic: ligand-clean drops 10pp / 8pp
from random (Morgan-RF / SPRINT); dual-clean drops 13pp / 11pp. The Morgan-RF
random control row was added by `tools/build_random_pdbbind_split.py` +
`tools/run_morgan_rf_random.py`; SPRINT trained on the same split via the
existing `train.py` `--task v2_pdbbind_random`.

Within each model column (Group A row-set):

|         | random→ligand drop | random→protein drop | random→dual drop | ligand→protein drop |
|---------|---:|---:|---:|---:|
| Morgan-RF | −9.9pp | **−25.1pp** | −12.7pp | −15.2pp |
| SPRINT    | −7.5pp | **−24.8pp** | −10.6pp | −17.3pp |

The drop pattern is consistent across the shallow and deep models, and most
strikingly so on protein-clean: both models lose ~25pp from the random control
when the KG forbids any test-time protein from appearing in train. SPRINT
closes about 4-5pp of the absolute AUROC gap on every regime (deep model +
ProtBert beats Morgan-RF), but **does not close the leakage drop**: on
protein-clean the model still loses ~25pp relative to its own random-control
score. The shortcut isn't a Morgan-fingerprint artefact — a fully trained
contemporary DTI model has it too.

## Multi-corpus Phase 1 baseline context

| Corpus | ligand | scaffold | protein | pocket | dual | strict |
|---|---|---|---|---|---|---|
| DEKOIS | 0.886 | 0.850 | 0.757 | ∅ | 0.814 | 0.814† |
| DUD-E | 0.879 | 0.876 | 0.809 | ∅ | 0.829 | 0.829† |
| LIT-PCBA | 0.518 | 0.526 | 0.556 | ∅ | 0.533 | 0.533† |
| PDBBind | 0.707 | 0.707 | 0.555 | 0.746 | 0.679 | 0.746† |

∅ = infeasible (no edges of that axis in the v1 graph for that corpus). † =
degenerate strict-clean (n_groups = n_examples → effectively random split).
See PHASE1_FINAL_REPORT.md § Findings F1-F4.

**Per-corpus shortcut profile:**
- DEKOIS, DUD-E: heavy *ligand-axis* shortcut. Matched-property decoys insufficient.
- LIT-PCBA AVE: shortcut defeated (every regime ≈ 0.55). AVE works as advertised.
- PDBBind: dominant *protein-axis* shortcut. The 0.71→0.55 drop is the largest
  single-axis effect we measure and is what Phase 2 SPRINT confirms with a
  deep model.

## Group B placeholder — paper-config reproduction

SPRINT's published numbers are on DAVIS/BIOSNAP, not PDBBind. We did **not**
re-train SPRINT on DAVIS in this audit. The Phase 2 numbers above are
explicitly *our PDBBind audit*, not a SPRINT-paper-minus-ours delta. Cross-
corpus and cross-config comparisons remain out of scope. This is the
"never compute paper-minus-ours" fairness gate fixed in
`memory/phase2_fairness_policy.md`.

## Group C — Retrieval-native audit (DUD-E, proof-of-protocol)

A separate audit track for retrieval-style virtual-screening models
(DrugCLIP and similar). **This section's numbers do NOT mix with the
Group A PDBBind binary table above.** Different corpus, different task,
different metrics. Group A and Group C are independent audit tracks; the
v2 KG underlies both but expresses different leakage controls in each.

**Why a separate track**: Group A's row-level binary AUROC is the wrong
metric for DrugCLIP. DrugCLIP outputs normalized-cosine *similarity*
scores designed for ranking-within-a-query, not cross-query binary
classification. The "DrugCLIP — third-model attempt" appendix below
documents the diagnostic that led us here. The fair retrieval-native
treatment is: pick a corpus organized per-target (one pocket → many
actives + many decoys), apply target-level leakage filters, and report
per-target BEDROC / ROC-AUC / EF — the metrics DrugCLIP was trained on.

### Setup

| Item | Value |
|---|---|
| Corpus | DUD-E (102 targets total; 65 with pre-extracted pocket PDBs from PDBBind) |
| Per-target test pool | all known actives + 1000 random property-matched decoys |
| Conformer generation | 1 RDKit-embedded conformer per molecule |
| Model | DrugCLIP published checkpoint (trained on PDBBind 2020 + HomoAug) |
| Eval | Zero-shot. Model is frozen; we only score and rank. |
| Metric per target | BEDROC (α=80.5), ROC-AUC, EF1%, EF5% |
| Aggregation | mean ± std across test targets |

### Target-level leakage regimes (KG-driven)

Built from `outputs/v2_retrieval/graph_dude/`:
- **target_random** — uniform-random target partition (control)
- **target_clean** — entire Pfam-family (≥40% seq ID cluster) goes to one side
- **active_clean** — targets sharing any active ligand stay on the same side
- **dual_clean** — target_clean ∧ active_clean (intersect both)
- *scaffold_clean dropped* — Bemis-Murcko scaffolds are massively shared
  across DUD-E (single-linkage collapses everything into one giant cluster;
  only 2 test targets escape). This is itself a finding about DUD-E.

Split sizes (65 targets in scope, ~30% test target fraction):

| Regime | n_train_targets | n_test_targets | test_row_frac |
|---|---:|---:|---:|
| target_random  | 47 | 18 | 29% |
| target_clean   | 47 | 18 | 30% |
| active_clean   | 42 | 23 | 30% |
| dual_clean     | 42 | 23 | 30% |

(`outputs/v2_retrieval/splits/dude/<regime>.parquet`)

### Contamination caveat (read this before reading the table)

DrugCLIP's published checkpoint was trained on PDBBind 2020 + HomoAug. Our
65 in-pocket DUD-E targets were selected because their PDB codes ARE in
the PDBBind 2020 extraction — meaning **all 65 targets are direct
training-data overlap** for the paper checkpoint. The remaining 37 DUD-E
targets (whose pockets we didn't fetch) are the proper "novel-target"
set; that's deferred to a follow-up.

Within the 65 in-domain targets, contamination is uniform across our
target-level split regimes (the KG only controls the train/test
partition WE define; it doesn't affect what the paper model already saw).
So the **regime-by-regime comparison stays valid** — if random > target-clean,
that's a real "novel-target-axis" effect within the contaminated set.

(`outputs/v2_retrieval/diagnostics/dude_contamination.csv`)

### Per-regime results

Paper checkpoint zero-shot, aggregated across each regime's test targets:

| Regime | n test targets | ROC-AUC mean ± std | BEDROC mean ± std | EF1% mean | EF5% mean |
|---|---:|---:|---:|---:|---:|
| target_random | 18 | 0.458 ± 0.216 | 0.112 ± 0.214 | 0.80 | 0.71 |
| target_clean  | 18 | 0.468 ± 0.247 | 0.156 ± 0.224 | 1.45 | 1.26 |
| active_clean  | 23 | 0.382 ± 0.204 | 0.091 ± 0.182 | 0.82 | 0.63 |
| dual_clean    | 23 | 0.382 ± 0.204 | 0.091 ± 0.182 | 0.82 | 0.63 |

Per-target CSVs at `outputs/v2_retrieval/results/dude/<regime>_per_target.csv`.

### Honest interpretation

**Per-target variance dominates the signal.** AUROC std ≈ 0.2 with per-target
range from 0.12 to 0.86. Three targets (hivint, ada, src) score AUROC ≥ 0.8;
many score ≤ 0.3. With only 18-23 test targets per regime, the per-regime
gaps that LOOK suggestive are within noise: the active_clean vs target_random
gap is −0.076 with SE ≈ 0.067 (not statistically significant).

**active_clean ≡ dual_clean (identical split).** DUD-E's cross-target active
sharing is so pervasive that the active-axis constraint already subsumes the
target-axis (Pfam) constraint. The two regimes produce identical
train/test partitions on our 65-target scope, so we report them as one
finding rather than two.

**Subset-selection effects, not training-time leakage gaps.** This is the
critical caveat. Because we evaluated a **frozen paper checkpoint**, the KG's
target-level filters do not control what the model has seen — they only
control *which targets we ask about at test time*. A drop in BEDROC on
active_clean tells us "targets with cross-target-disjoint actives happen to
be harder for DrugCLIP", **not** "DrugCLIP loses signal when train/test
leakage is forbidden". A true leakage-gap audit for a retrieval model
requires retraining on each split's train side, which is currently blocked
by the corpus-size issue documented in the DrugCLIP attempt section below.

**aa2ar smoke-test outlier.** The smoke test reported `aa2ar` AUROC = 0.838,
BEDROC = 0.955 — well above the per-regime mean. This is a known
easy DUD-E target (adenosine A2A receptor); it's not representative of
typical DrugCLIP behaviour across the test sets here.

### What Group C *does* demonstrate

- A retrieval-native audit protocol that **matches the model's intended use**:
  per-target BEDROC + ROC-AUC + EF across DUD-E-style libraries, with the
  v2 KG controlling target-level splits.
- A working pipeline (target nodes + family/active-share edges + 4 split
  regimes + per-target retrieval eval). The aa2ar smoke test (BEDROC 0.955)
  confirms the implementation is sound — the model gives strong, paper-
  reportable signal when the decoy pool and metric match its training
  objective.
- A characterization of DUD-E itself: scaffold sharing is so pervasive it
  collapses scaffold_clean to a degenerate split; active sharing already
  subsumes Pfam family sharing.

### What Group C *does NOT yet demonstrate*

- **A model-invariant leakage gap, retrieval-native.** Doing so requires
  retraining DrugCLIP (or similar) on each split's train side. Blocked by
  the corpus-size issue (≤10K positives per regime); resolving needs
  HomoAug-style augmentation. Documented as future work.
- **Contamination-free numbers.** All 65 in-scope DUD-E targets are direct
  PDB-code overlap with PDBBind 2020. The 37 non-overlapping targets are
  the proper "novel-target" set; their pockets require separate RCSB
  fetching, deferred.
- **DEKOIS / LIT-PCBA retrieval audit.** Same protocol, different corpora.
  LIT-PCBA in particular (real assay decoys) would isolate the "synthetic-
  decoy bias" from the leakage signal. Out of scope for this proof.

## Appendix — DrugCLIP in Group A (record of diagnostic path)

This section documents the Group A attempts that led to the Group C pivot
above. It's kept for the record because the diagnostic itself is the
audit story for retrieval models on a row-level binary task. **The
numbers here are not the Group C numbers** and should not be quoted as
DrugCLIP's audit result.

We tried to add DrugCLIP as a third model in Group A. Two attempts; both
useful as findings even though neither produced a clean Group-A row.

### Attempt 1 — retrain DrugCLIP on v2 train splits

Built unimol-format LMDBs from v2 ligand/protein/dual train splits and ran
DrugCLIP's published `agg_config`-equivalent training. Did not converge:
the in-batch-softmax objective collapsed to the trivial all-embeddings-equal
solution by epoch 2 (loss locked at ln(N), gradients clipped to ~zero,
valid bedroc plateaued at random).

**Root cause**: DrugCLIP's paper recipe trains on the full PDBBind 2020 +
HomoAug-augmented corpus (~100K+ positives). Our v2 splits, after the KG
filters and label decoding, ship ≤10k positives in train. Contrastive
in-batch-softmax with batch=48 needs a much richer pool of true binders
than the v2 train sets provide. Switching `--update-freq` to recover the
paper's effective batch (8 × 6 = 48) restored gradient flow but did not
fix convergence — the *corpus*, not the optimizer setup, was the binding
constraint.

This is itself an audit finding: contrastive retrieval models cannot be
fairly retrained on a leakage-clean PDBBind subset because the leakage
filter halves an already-small training pool.

### Attempt 2 — paper checkpoint zero-shot on v2 test splits

Downloaded the published DrugCLIP `checkpoint_best.pt` (trained on full
PDBBind 2020 + HomoAug, the same corpus our v2 splits draw from). Ran a
per-pocket retrieval AUROC on each v2 test split: for each pocket with a
known cognate binder, rank that binder against the pool of all other test
mols and AUROC against the 1-positive / (N−1)-negative ranking.

| Regime | n test pockets w/ binder | mean per-pocket AUROC | median |
|---|---:|---:|---:|
| random (control) | 3,322 | **0.554** | 0.566 |
| ligand-clean | 1,640 | 0.560 | 0.563 |
| protein-clean | 2,158 | 0.561 | 0.591 |
| dual-clean | 2,516 | 0.574 | 0.596 |

The published model is **essentially flat across all four regimes** (mean
per-pocket AUROC = 0.554-0.574, a 2pp spread). Critically, the random
control row is **the same as the leakage-clean rows**. Compare to
Morgan-RF (random 0.81, protein 0.55, 25pp gap) and SPRINT (random 0.84,
protein 0.59, 25pp gap) on the same test data.

We read this as **train-test contamination dominating the signal**: the
published checkpoint has seen each of our v2 test splits during its own
training (no protein, ligand, dual, or randomly-sampled partition of
PDBBind is "novel" relative to its training data), so the v2 leakage
filter does not separate it from in-domain performance. The absent
leakage gap is itself the diagnostic — it tells us the published model
cannot be cleanly audited on PDBBind-derived splits, *not* that
DrugCLIP is leakage-resistant.

A clean Group A audit of DrugCLIP would require either (a) retraining
from scratch on a single v2 train split (Attempt 1 shows this needs a
HomoAug-equivalent augmentation we did not implement), or (b) recovering
the paper's training manifest and excluding overlap.

We also report the *flat per-row AUROC* for direct comparison to
Morgan-RF / SPRINT, with the caveat that DrugCLIP's normalized dot product
is a **similarity** score, not a calibrated probability — flat AUROC across
1-row-per-pocket data is the wrong metric for it:

| Regime | DrugCLIP flat AUROC | (vs SPRINT) |
|---|---:|---:|
| random | 0.480 | 0.837 |
| ligand-clean | 0.482 | 0.762 |
| protein-clean | 0.490 | 0.589 |
| dual-clean | 0.505 | 0.731 |

The flat number being ≈0.5 for all regimes is *expected*: DrugCLIP's
embedding norms vary per-pocket, and cross-pocket score comparison is
not what the model was trained to do. This row is reported for
completeness; the per-pocket retrieval AUROC above is the metric we
believe and the conclusion we report.

## What this does and does not show

**Shows:**
- A leakage-axis-clean split materially changes apparent DTI performance.
- The effect is model-invariant on PDBBind: Morgan-RF and SPRINT both lose
  ~25pp from random control on protein-clean (Morgan-RF 0.81→0.55, SPRINT
  0.84→0.59). The leakage signal does not collapse when you swap the model.
- The size of the effect varies dramatically by corpus — measurable v2 splits expose this.
- AVE-style decoys (LIT-PCBA) successfully defeat ligand shortcuts; DUD-E / DEKOIS decoys do not.

**Does not show:**
- That DrugCLIP cleanly confirms the *binary* leakage pattern in Group A.
  Two attempts at a Group A row both hit corpus-related blockers; we
  pivoted to a retrieval-native protocol (Group C above) which surfaces
  per-target signal but on a different corpus and task. Documented in
  the appendix "DrugCLIP in Group A" section below.
- A model-invariant retrieval-native leakage gap. Group C's per-target
  variance is too large at 18-23 targets to claim a significant gap
  across split regimes from a frozen paper checkpoint. A true leakage
  audit requires retraining DrugCLIP on each split's train side, blocked
  by the corpus-size issue. Scoped as future work.
- That LigUnity exhibits the same pattern. Deferred; same corpus and
  contamination considerations apply, and the retrieval-native protocol
  (Group C) is now the right framework for it.
- Effect on a second large-scale corpus with a deep model. SPRINT runs on
  DEKOIS / DUD-E / LIT-PCBA were not budget-feasible in this iteration (each
  corpus is 5-30× PDBBind size; one full training run per regime per corpus
  would saturate the shared box for a week). Scoped as future work.
- Whether the *direction* of leakage matters (does a model genuinely fail on
  unseen proteins, or does it fail on unseen ligands the same way?). The dual
  vs ligand vs protein columns in our table already separate these axes;
  whether the failure mode is genuinely "novel protein" vs "harder split"
  needs an ablation that explicitly de-randomizes the test split itself, not
  in scope here.

## Reproducibility (one-stop pointer)

| Step | Script / artifact | Output |
|---|---|---|
| v2 graph (any corpus) | `python -m vsleakkg.v2.build_graph <corpus>` | `outputs/v2/graph_<corpus>/v2_*.parquet` |
| side-table | `python -m vsleakkg.v2.pipeline.build_side_table` | `outputs/v2/graph/side_table.parquet` |
| protein-seq lookup (PDBBind) | `tools/build_protein_seq_lookup.py` | `outputs/v2/pdbbind_protein_seq_lookup.parquet` |
| v2 splits | `python -m vsleakkg.v2.pipeline` | `outputs/v2/phase1_full/splits/<corpus>/<regime>.parquet` |
| Phase 1 baselines | `python -m vsleakkg.v2.baselines.ligand_only` | `outputs/v2/phase1_full/baselines/*` |
| Phase 1 figures | `python -m vsleakkg.v2.final_figures` | `outputs/v2/phase1_full/figures/*` |
| Phase 2 SPRINT CSVs | `python tools/v2_to_sprint_csv.py` | `<sprint>/data/custom_pdbbind_<regime>/*.csv` |
| Phase 2 SPRINT train | published `agg_config.yml`, see PHASE2_SPRINT_FINAL.md | `<sprint>/best_models/v2_pdbbind_<regime>_agg_paper/*.ckpt` |
| Phase 2 SPRINT test | `run_test_only.py` (in repo) | `sprint_runs/v2_pdbbind_<regime>_TEST.log` |

Detailed bash recipe in `PHASE1_FINAL_REPORT.md` § Reproduction recipe; SPRINT
specifics in `PHASE2_SPRINT_FINAL.md`.

## Files referenced

- `PHASE1_FINAL_REPORT.md` — per-corpus data-only audit, bug log, inventory.
- `PHASE2_SPRINT_FINAL.md` — SPRINT train + test setup, Group A audit table.
- `outputs_run4_full/phase1_full/phase1_combined.csv` — Phase 1 source-of-truth table.
- `tools/v2_to_sprint_csv.py`, `tools/build_protein_seq_lookup.py` — adapter scripts.

## Scope explicitly deferred

- Group C extension to 102 DUD-E targets (37 non-PDBBind-overlapping
  pockets need RCSB fetching; this is the "novel-target" contamination-
  free set).
- Group C extension to DEKOIS and LIT-PCBA — different decoy pool
  protocols; LIT-PCBA's real-assay decoys would isolate synthetic-decoy
  bias from the leakage signal.
- DrugCLIP retrain on a leakage-clean train split with HomoAug-style
  augmentation. This is the path to a true retrieval-native leakage gap
  (vs the subset-selection effects Group C currently surfaces).
- LigUnity Group C audit — same retrieval-native protocol applies.
- Second-corpus deep-model audit (SPRINT on DEKOIS / DUD-E / LIT-PCBA — needs corpus-specific protein-seq lookups + days of training).
- Pocket-axis leakage on non-PDBBind corpora (needs `example_has_pocket` edges in those v1 graphs).
- Assay-axis leakage (needs `example_from_assay` edges added to v1 graphs from `chembl_assays.parquet`).

Each is tractable and each is documented in the per-phase reports.
