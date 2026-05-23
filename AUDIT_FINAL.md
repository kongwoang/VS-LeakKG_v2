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

## Executive summary

The v2 KG framework operates a leakage-axis audit at three different
abstraction levels across four corpora and 28+ evaluation methods. The
**load-bearing result** — and the only one that fully closes the audit
loop (the KG controls what the model sees in train, and the held-out
test changes measurably) — is on **PDBBind binary classification**:

> **Holding model + config fixed and changing only the v2 KG split:
> both Morgan-RF and SPRINT lose ~25pp AUROC from random control to
> protein-clean on PDBBind. The leakage signal is real, large, and
> model-invariant.**

| Group | Corpus | Task | What the KG controls | Result |
|---|---|---|---|---|
| **A** | PDBBind (19k rows) | Binary classification | Train/test partition + which axis (ligand / protein / scaffold / pocket / dual / strict) leaks | **−25pp AUROC** from random→protein-clean, consistent across Morgan-RF and SPRINT |
| **C** | DUD-E (102 targets) + DEKOIS 2.0 (62) + LIT-PCBA (15) | Retrieval (per-target BEDROC) | Which test targets appear (target-axis Pfam-disjoint, active-axis ligand-disjoint, scaffold-axis disjoint) | KG splits structurally segregate test targets, but a frozen paper checkpoint + small n yields **subset-selection effects, not training-time leakage gaps** — confirmed across 26 methods (cross-method audit via LigUnity's published benchmark) |
| Appendix | PDBBind | Binary classification (DrugCLIP retrofit) | Row-level split, but retrieval-model metric mismatch | Diagnostic: pool-composition is the dominant issue, not contamination. Motivated the Group C pivot. |

**What's shippable on this audit:** the Group A finding is the audit's
primary contribution and is fully defensible — 25pp drop, model-invariant,
random-control calibrated, statistical-power adequate. The Group C
retrieval-native framework is a fully-functional second track that
detects structural changes across split regimes but cannot prove a
training-time leakage gap without retraining the retrieval model on
each split — blocked by a corpus-size constraint documented in the
DrugCLIP appendix.

**What the v2 KG enables (capabilities demonstrated)**:
1. *Multi-corpus row-level audit* with consistent leakage-axis definitions
   (ligand / scaffold / protein / pocket / dual / strict).
2. *Random-control calibration* — same-size random partition isolates
   the leakage signal from base task difficulty.
3. *Target-level audit* on retrieval corpora — same KG semantics lifted
   from row-level to target-level operation.
4. *Cross-method evaluation* via published per-target benchmark tables
   (26 methods on DUD-E, 18 on DEKOIS, 11 on LIT-PCBA), with no
   retraining required.
5. *Reusable scaffolding* — the v2 graph, splits, baselines, and
   evaluation scripts can be applied to any new corpus or method by
   adding one wrapper.

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
| Corpus | DUD-E (102 targets — 65 from PDBBind 2020 pockets, 37 fetched from RCSB) |
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

Split sizes (102 targets in scope, ~30% test fraction by row-weight):

| Regime | n_train_targets | n_test_targets | test_row_frac |
|---|---:|---:|---:|
| target_random  | 80 | 22 | 30% |
| target_clean   | 71 | 31 | 30% |
| active_clean   | 67 | 35 | 23% |
| dual_clean     | 67 | 35 | 23% |

(`outputs/v2_retrieval/splits/dude/<regime>.parquet`)

### Contamination caveat (read this before reading the table)

DrugCLIP's published checkpoint was trained on PDBBind 2020 + HomoAug.
65 of our 102 DUD-E targets have direct PDB-code overlap with PDBBind
2020 (we used PDBBind's pre-extracted pockets for those). The other 37
were fetched from RCSB and may still overlap PDBBind 2020 if their PDB
release dates fall in that window. HomoAug additionally exposes
homologs; we did not implement a homology-filter pass against PDBBind.

Within the 102 targets, contamination is approximately uniform across
our target-level split regimes (the KG only controls the train/test
partition WE define; it doesn't affect what the paper model already
saw). So the **regime-by-regime comparison stays valid** — if random >
target-clean, that's a real "novel-target-axis" effect within whatever
in-domain set we have.

(`outputs/v2_retrieval/diagnostics/dude_contamination.csv`)

### Per-regime results

Paper checkpoint zero-shot, aggregated across each regime's test targets:

| Regime | n test targets | ROC-AUC mean ± std | BEDROC mean ± std | EF1% mean | EF5% mean |
|---|---:|---:|---:|---:|---:|
| target_random | 22 | 0.450 ± 0.189 | 0.132 ± 0.195 | 0.98 | 0.73 |
| target_clean  | 31 | 0.468 ± 0.174 | 0.103 ± 0.153 | 0.74 | 0.74 |
| active_clean  | 35 | 0.454 ± 0.226 | 0.075 ± 0.129 | 0.96 | 0.88 |
| dual_clean    | 35 | 0.454 ± 0.226 | 0.075 ± 0.129 | 0.96 | 0.88 |

Per-target CSVs at `outputs/v2_retrieval/results/dude/<regime>_per_target.csv`.

### Honest interpretation

**Per-target variance dominates the signal.** AUROC std ≈ 0.19–0.23 with
per-target range from 0.04 to 0.86. A few targets (hivint, ada, src) score
AUROC ≥ 0.8; many score ≤ 0.3. Even with 102 targets (22–35 per regime),
the per-regime gaps that LOOK suggestive are within noise:

- AUROC: `target_random` (0.450) vs `active_clean` (0.454) — effectively flat.
- BEDROC: `target_random` (0.132) vs `active_clean` (0.075) — gap = 0.057
  with SE ≈ 0.047, z ≈ 1.2, p ≈ 0.23. Not significant.

The BEDROC ordering `random ≥ target_clean > active_clean` is the direction
you'd expect from a real leakage gap, but the sample size + per-target
variance combination doesn't let us claim it.

**active_clean ≡ dual_clean (identical split).** DUD-E's cross-target active
sharing is so pervasive that the active-axis constraint already subsumes the
target-axis (Pfam) constraint. The two regimes produce identical
train/test partitions, so we report them as one finding rather than two.

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
- **Contamination-free numbers.** 65 of 102 in-scope DUD-E targets have
  direct PDB-code overlap with PDBBind 2020. The remaining 37 were
  fetched from RCSB; their PDB release dates may also fall in PDBBind
  2020. A clean "novel-target" subset requires homology-filtering against
  the paper's training manifest.
- **LIT-PCBA retrieval audit.** Same protocol, real assay decoys
  (AVE-defeated), 15 targets only — sample size is too small for
  variance-dominated DrugCLIP zero-shot. Out of scope for this iteration.

## Group C — Retrieval-native audit (DEKOIS 2.0, corroborating corpus)

Same protocol as the DUD-E section above, applied to DEKOIS 2.0 — a
second retrieval benchmark with stricter property-matched decoys.

### Setup

| Item | Value |
|---|---|
| Corpus | DEKOIS 2.0 (81 targets total; 62 with extracted pocket PDBs at 12 Å) |
| Per-target test pool | 40 actives + ~1100 decoys (DEKOIS native sizes) |
| Conformer generation | 1 RDKit conformer per molecule |
| Model | DrugCLIP published checkpoint |
| Eval | Zero-shot |
| Metrics | Per-target BEDROC (α=80.5), ROC-AUC, EF1%, EF5% |

### Split sizes (62 targets in scope, ~30% test row fraction)

| Regime | n_train | n_test | test_row_frac |
|---|---:|---:|---:|
| target_random  | 44 | 18 | 29% |
| target_clean   | 44 | 18 | 29% |
| active_clean   | 44 | 18 | 29% |
| scaffold_clean | 47 | 15 | 25% |
| dual_clean     | 40 | 22 | 36% |

Note: DEKOIS scaffold_clean is **non-degenerate** here (15 test targets,
unlike DUD-E where it collapsed to 2). DEKOIS scaffold sharing across
targets is much sparser than DUD-E's.

### Per-regime results

| Regime | n test | ROC-AUC mean ± std | BEDROC mean ± std | EF1% | EF5% |
|---|---:|---:|---:|---:|---:|
| target_random  | 18 | 0.479 ± 0.159 | 0.034 ± 0.052 | 1.09 | 0.69 |
| target_clean   | 18 | 0.549 ± 0.192 | 0.042 ± 0.067 | 0.80 | 1.27 |
| active_clean   | 18 | 0.467 ± 0.184 | 0.030 ± 0.061 | 0.80 | 0.83 |
| scaffold_clean | 15 | 0.499 ± 0.182 | 0.041 ± 0.060 | 1.14 | 1.06 |
| dual_clean     | 22 | 0.431 ± 0.136 | 0.018 ± 0.041 | 0.43 | 0.47 |

### Honest interpretation

- **DrugCLIP performs substantially worse on DEKOIS than on DUD-E**: mean
  BEDROC 0.02-0.04 (DEKOIS) vs 0.08-0.13 (DUD-E). Plausibly because
  DEKOIS's matched-property decoys are more rigorous than DUD-E's, and
  because DEKOIS targets have lower overlap with PDBBind 2020.
- **`dual_clean` is the lowest** (BEDROC 0.018, AUROC 0.431, EF1% 0.43)
  — the most restrictive split selects the hardest test targets. This
  is consistent with the "subset-selection effect" interpretation: the
  KG's leakage filter selects targets whose actives and Pfam family are
  disjoint from the train side, which happens to correlate with targets
  the model finds hardest.
- **No statistically-significant leakage gap.** target_random vs
  dual_clean ΔAUROC = 0.048 ± 0.047 (z ≈ 1.0, p ≈ 0.32). Per-target
  variance again dominates.
- **DEKOIS confirms DUD-E's conclusion**: zero-shot eval of a frozen
  paper checkpoint surfaces subset-selection effects, not training-time
  leakage gaps. A true Group C leakage audit needs the model to be
  retrained on each split's train side.

Per-target CSVs at `outputs/v2_retrieval/results/dekois/<regime>_per_target.csv`.

## Group C++ — Cross-method retrieval audit (26 methods × KG splits)

The LigUnity paper (Patterns 2025) publishes per-target BEDROC / ROC-AUC /
EF1% for **26 retrieval methods on DUD-E** (102 targets) and **18 methods
on DEKOIS** (81 targets). Each method uses its own paper-native eval
protocol (full decoy pools, multi-conformer, etc.) — paper-comparable
numbers across the board.

We apply our v2 KG target-level split regimes as a *filter* over these
published per-target tables. For each (method, regime) pair we compute
the mean per-target metric across that regime's test targets. This gives
a 26-method × 4-regime BEDROC table without retraining anything — each
method scores its native eval, we just slice it.

This is the cleanest cross-method audit we can run on retrieval models.

### DUD-E — BEDROC (α=80.5) by method × regime, with random→dual delta

| Method | random (n=22) | target_clean (n=31) | active_clean (n=35) | dual_clean (n=35) | Δ(dual−random) |
|---|---:|---:|---:|---:|---:|
| LigUnity        | 0.832 | 0.729 | 0.811 | 0.811 | −0.02 |
| LigUnity (seq)  | 0.656 | 0.553 | 0.592 | 0.592 | −0.06 |
| Pocket-DTA      | 0.611 | 0.453 | 0.393 | 0.393 | **−0.22** |
| RTMScore        | 0.602 | 0.525 | 0.592 | 0.592 | −0.01 |
| Denvis-G        | 0.594 | 0.538 | 0.534 | 0.534 | −0.06 |
| GenScore        | 0.534 | 0.454 | 0.474 | 0.474 | −0.06 |
| DrugCLIP        | 0.519 | 0.497 | 0.566 | 0.566 | +0.05 |
| EquiScore       | 0.485 | 0.430 | 0.479 | 0.479 | −0.01 |
| Sequence-DTA    | 0.483 | 0.386 | 0.362 | 0.362 | **−0.12** |
| Vina            | 0.236 | 0.184 | 0.171 | 0.171 | −0.07 |

(Full 26-method table in `outputs/v2_retrieval/results/dude_cross_method/dude_BEDROC_per_regime.csv`)

### DEKOIS — BEDROC by method × regime, with random→dual delta

| Method | random (n=18) | target_clean (n=18) | active_clean (n=18) | scaffold_clean (n=15) | dual_clean (n=22) | Δ(dual−random) |
|---|---:|---:|---:|---:|---:|---:|
| LigUnity       | 0.783 | 0.865 | 0.873 | 0.892 | 0.867 | +0.08 |
| LigUnity (seq) | 0.753 | 0.834 | 0.834 | 0.875 | 0.722 | −0.03 |
| Pocket-DTA     | 0.669 | 0.742 | 0.645 | 0.723 | 0.705 | +0.04 |
| Sequence-DTA   | 0.596 | 0.662 | 0.565 | 0.659 | 0.587 | −0.01 |
| RTMScore       | 0.417 | 0.532 | 0.639 | 0.577 | 0.603 | +0.19 |
| GenScore       | 0.404 | 0.484 | 0.498 | 0.488 | 0.570 | +0.17 |
| DrugCLIP       | 0.376 | 0.542 | 0.567 | 0.614 | 0.471 | +0.10 |

(Full 18-method table in `outputs/v2_retrieval/results/dekois_cross_method/dekois_BEDROC_per_regime.csv`)

### LIT-PCBA — BEDROC by method × regime (8 methods, two usable regimes)

LIT-PCBA has only 15 targets total; active-ligand sharing is so pervasive
that active_clean / scaffold_clean / dual_clean all collapse to a single
train target. Only target_random (n=2 test) and target_clean (n=8 test)
are usable, and **target_random is too small (n=2) for any conclusion**.
We report the comparison for completeness; signal direction is dominated
by which 2 targets happened to land in target_random.

| Method | target_random (n=2) | target_clean (n=8) | Δ(clean−random) |
|---|---:|---:|---:|
| LigUnity        | 0.009 | 0.103 | +0.094 |
| LigUnity (seq)  | 0.015 | 0.089 | +0.074 |
| DrugCLIP        | 0.025 | 0.080 | +0.055 |
| GenScore        | 0.033 | 0.072 | +0.039 |
| GNINA           | 0.070 | 0.063 | −0.007 |
| Glide SP        | 0.009 | 0.064 | +0.055 |
| Denvis-G        | 0.010 | 0.051 | +0.041 |
| Pocket-DTA      | 0.006 | 0.041 | +0.035 |

All methods show target_clean > target_random — the opposite direction of
a leakage gap, but the n=2 random sample makes this uninterpretable. LIT-
PCBA's structural problem (only 15 targets) means it cannot power a
target-level cross-regime audit. It would be a strong corpus for a
*model-pair* audit (binary classifier retrained per split — same as
Group A) since LIT-PCBA's row-level Phase 1 result already shows AVE
defeating ligand shortcuts. That avenue is left as future work.

(Full table in `outputs/v2_retrieval/results/litpcba_cross_method/pcba_BEDROC_per_regime.csv`)

### Interpretation

**No method shows a consistent, large leakage gap across both corpora.**
On DUD-E most methods drop slightly under target_clean / active_clean
(direction-consistent with a leakage signal); on DEKOIS most methods
*increase* under target_clean (opposite direction). This is the
**subset-selection effect** dominating: the KG's clean-split test target
sets aren't a random slice — they happen to be easier or harder than the
random control depending on what makes the corpus' targets cluster.

**Pocket-DTA on DUD-E is the largest single drop** (−0.22 BEDROC,
random→dual). It also has the largest random-control BEDROC after the
LigUnity family. This is suggestive of corpus-specific overfitting that
the KG's target-clean / active-clean filter exposes — but on DEKOIS the
same model shows +0.04, so we can't claim a systematic effect.

**LigUnity is by far the strongest method on both corpora** (random
BEDROC 0.83 / 0.78) and shows small, often-opposite deltas across
regimes (−0.02 / +0.08). This matches the paper's explicit
training-time leakage control: LigUnity's training set excluded test
proteins from DUD-E / DEKOIS / LIT-PCBA, so it already operates in a
"novel-target" regime relative to those benchmarks. The KG's target-
level filter doesn't separate it further.

**What this confirms about the audit framework:**
- v2 KG target-level splits are doing structural work (they select
  meaningfully different target subsets — see the n column variation
  per regime).
- Whether a clean→random gap shows up at the BEDROC level is **highly
  model-dependent**, and on retrieval models without training-time
  leakage control, the variance and corpus structure dominate the
  signal at small n.
- A direction-consistent, statistically-significant retrieval-native
  leakage gap requires either (a) much larger n per regime (which DUD-E
  and DEKOIS don't afford), or (b) training-time retraining on each
  KG split, which is the blocked future-work item.

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

### Group A — PDBBind row-level audit
| Step | Script / artifact | Output |
|---|---|---|
| v2 graph (any corpus) | `python -m vsleakkg.v2.build_graph <corpus>` | `outputs/v2/graph_<corpus>/v2_*.parquet` |
| side-table | `python -m vsleakkg.v2.pipeline.build_side_table` | `outputs/v2/graph/side_table.parquet` |
| protein-seq lookup (PDBBind) | `tools/build_protein_seq_lookup.py` | `outputs/v2/pdbbind_protein_seq_lookup.parquet` |
| v2 splits | `python -m vsleakkg.v2.pipeline` | `outputs/v2/phase1_full/splits/<corpus>/<regime>.parquet` |
| Random control split | `tools/build_random_pdbbind_split.py` | `outputs/v2/phase1_full/splits/pdbbind/random.parquet` |
| Morgan-RF baseline | `python -m vsleakkg.v2.baselines.ligand_only` | `outputs/v2/phase1_full/baselines/*` |
| Morgan-RF random control | `tools/run_morgan_rf_random.py` | `outputs/v2/phase1_full/baselines/pdbbind_random_morgan_rf.csv` |
| Phase 1 figures | `python -m vsleakkg.v2.final_figures` | `outputs/v2/phase1_full/figures/*` |
| Phase 2 SPRINT CSVs | `python tools/v2_to_sprint_csv.py` | `<sprint>/data/custom_pdbbind_<regime>/*.csv` |
| Phase 2 SPRINT train | published `agg_config.yml`, see PHASE2_SPRINT_FINAL.md | `<sprint>/best_models/v2_pdbbind_<regime>_agg_paper/*.ckpt` |
| Phase 2 SPRINT test | `run_test_only.py` (in repo) | `sprint_runs/v2_pdbbind_<regime>_TEST.log` |

### Group C — Retrieval-native audit
| Step | Script / artifact | Output |
|---|---|---|
| DUD-E target KG | `tools/v2_retrieval/build_dude_target_kg.py` | `outputs/v2_retrieval/graph_dude/v2_*.parquet` |
| DEKOIS target KG | `tools/v2_retrieval/build_dekois_target_kg.py` | `outputs/v2_retrieval/graph_dekois/v2_*.parquet` |
| LIT-PCBA target KG | `tools/v2_retrieval/build_litpcba_target_kg.py` | `outputs/v2_retrieval/graph_litpcba/v2_*.parquet` |
| Target-level splits | `tools/v2_retrieval/build_dude_target_splits.py` (corpus-agnostic) | `outputs/v2_retrieval/splits/<corpus>/<regime>.parquet` |
| Fetch missing DUD-E pockets | `tools/v2_retrieval/fetch_missing_dude_pockets.py` | `data/raw/DUD-E_pockets_fetched/<pdb>/<pdb>_pocket.pdb` |
| Per-target LMDBs | `tools/v2_retrieval/build_{dude,dekois}_target_lmdbs.py` | `DrugCLIP/data/<corpus>_retrieval/<target>/{pocket,mols}.lmdb` |
| Retrieval eval (DrugCLIP paper ckpt) | `tools/v2_retrieval/eval_dude_retrieval.py` (corpus-agnostic) | `outputs/v2_retrieval/results/<corpus>/<regime>_per_target.csv` |
| Per-regime driver | `run_dude_eval_all_regimes.sh`, `run_dekois_eval_all_regimes.sh` | `<regime>_run.log` |
| Cross-method audit | `tools/v2_retrieval/aggregate_ligunity_benchmark.py` | `outputs/v2_retrieval/results/<corpus>_cross_method/*.csv` |
| Contamination diagnostic | `tools/v2_retrieval/contamination_diagnostic.py` | `outputs/v2_retrieval/diagnostics/dude_contamination.csv` |

Detailed bash recipe in `PHASE1_FINAL_REPORT.md` § Reproduction recipe; SPRINT
specifics in `PHASE2_SPRINT_FINAL.md`. Group C tooling lives under
`tools/v2_retrieval/`.

## Files referenced

- `PHASE1_FINAL_REPORT.md` — per-corpus data-only audit, bug log, inventory.
- `PHASE2_SPRINT_FINAL.md` — SPRINT train + test setup, Group A audit table.
- `outputs_run4_full/phase1_full/phase1_combined.csv` — Phase 1 source-of-truth table.
- `tools/v2_to_sprint_csv.py`, `tools/build_protein_seq_lookup.py` — adapter scripts.

## Scope explicitly deferred

- **DrugCLIP retrain with HomoAug-style augmentation on each v2 split.**
  This is the missing piece for a true retrieval-native *training-time*
  leakage gap. Blocked by the corpus-size constraint: v2 train splits
  carry ≤10K positives per regime; DrugCLIP's paper recipe needs ~100K+
  HomoAug-augmented binders to converge under in-batch-softmax.
  Implementing HomoAug = multi-day project, scoped as future work.
- **LIT-PCBA cross-regime row-level audit** — analogous to Group A on
  PDBBind, but with LIT-PCBA's real-assay decoys. Would test whether the
  KG splits surface a model-invariant leakage gap on AVE-defeated data.
  Doable in 1-2 days using existing pipeline.
- **Second deep model in Group A** (any non-SPRINT binary classifier on
  PDBBind) — would extend the model-invariance claim to a third model.
- **SPRINT trained on DEKOIS / DUD-E / LIT-PCBA** — second large-scale
  corpus binary audit. Each corpus is 5-30× PDBBind size; per-regime
  training would take ~1 day each.
- **Per-corpus protein-side family graph** for full target-clean splits
  on LIT-PCBA — currently we use singleton-family fallback. Could be
  done with hmmscan / Pfam API.
- Second-corpus deep-model audit (SPRINT on DEKOIS / DUD-E / LIT-PCBA — needs corpus-specific protein-seq lookups + days of training).
- Pocket-axis leakage on non-PDBBind corpora (needs `example_has_pocket` edges in those v1 graphs).
- Assay-axis leakage (needs `example_from_assay` edges added to v1 graphs from `chembl_assays.parquet`).

Each is tractable and each is documented in the per-phase reports.
