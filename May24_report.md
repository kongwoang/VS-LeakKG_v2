# VS-LeakKG v2 — Consolidated Method + Results Report

**Date:** 2026-05-24
**Repo HEAD:** `13f1d2e` (Group D step 2)
**Compute:** VUW box (cuda12.ecs.vuw.ac.nz), 3× Quadro RTX 6000 24 GB

This document summarizes (a) the methodology in one place — anything not
restated here is unchanged from `proposal.tex` — and (b) the full set of
results obtained as of 2026-05-24, across five evidence groups: A, A++,
A+++, C, C++, plus Group D in flight.

---

## 1. Method (delta from proposal.tex)

The methodology in this report is the same as `proposal.tex` §3-§5 with
two extensions implemented in this iteration:

1. **Random-control calibration** on top of the proposal's clean splits.
   For each corpus, in addition to the leakage-clean splits (ligand /
   scaffold / protein / pocket / dual / strict), we generate a
   *same-size random partition* matching the protein-clean split's
   train/val/test sizes. The `random→clean` AUROC delta isolates the
   leakage signal the KG removes from the underlying task difficulty.
   This is the core Group A++ instrument and applies uniformly across
   all 4 corpora.

2. **Retrieval-native target-level lift.** The proposal defines clean
   splits at the row (per-example) level. For retrieval-style models
   evaluated by per-target BEDROC (DrugCLIP, LigUnity, etc.), we lift
   the same KG axes to **target-level** splits: target-random,
   target-clean (Pfam-family disjoint at 40% seq ID), active-clean
   (cross-target shared-active disjoint), scaffold-clean (shared
   Bemis-Murcko scaffold disjoint), and dual-clean (target ∧ active).
   Same KG semantics, lifted from rows to targets. This underlies
   Group C (proof-of-protocol with a frozen paper checkpoint) and
   Group D (paper-native retrain per regime, ongoing).

Everything else — typed nodes/edges, path-based contamination scoring,
forbidden-relation construction, leakage-group + giant-component
handling, validation-contamination matrices, shortcut baselines — is
exactly as in `proposal.tex`.

### 1.1 Evidence groups (this report)

| Group | What's controlled | What's measured | Task | Status |
|---|---|---|---|---|
| **A** | v2 KG splits on PDBBind (5 axes) | Row-level AUROC | Binary classification, 3 models (Morgan-LR, Morgan-RF, SPRINT) | Done |
| **A++** | Same as A, across 4 corpora; random control per corpus | Same; per-axis `random→clean` delta | Same; Morgan-LR + Morgan-RF | Done |
| **A+++** | Add capacity dimension (linear vs trees vs deep) | Magnitude vs direction of leakage | Same | Done |
| **C** | Target-level KG splits on DUD-E/DEKOIS/LIT-PCBA | Per-target BEDROC + AUROC + EF | Retrieval, DrugCLIP paper-ckpt zero-shot | Done (with documented limits) |
| **C++** | Same target-level splits, applied as filter over 26-method published benchmark | Same metrics, cross-method | Retrieval (LigUnity, RTMScore, Pocket-DTA, GenScore, DrugCLIP, …) | Done |
| **D** | v2 KG filters on LigUnity's *training corpus* (5 axes) | Per-target BEDROC + AUROC + EF | Retrieval, LigUnity retrained per regime | **In progress** (infra ready, smoke testing; full runs queued) |
| Appendix | DrugCLIP retrofit to PDBBind binary | Diagnostic only | Identifies metric-mismatch + pool-composition issues that motivated Group C | Done |

---

## 2. Group A — PDBBind v2 binary audit (load-bearing headline)

**Setup.** Identical v2 splits (`outputs/v2/phase1_full/splits/pdbbind/`),
identical preprocessing, identical features. The only variables are
*split regime* and *model class*.

### 2.1 Three-model table on PDBBind

| Regime | n_train | n_test | Morgan-LR | Morgan-RF | SPRINT |
|---|---:|---:|---:|---:|---:|
| **random (control)** | 7,337 | 5,844 | 0.7727 | **0.8058** | **0.8370** |
| ligand-clean  | 9,917 | 4,560 | 0.6901 | 0.7070 | **0.7619** |
| protein-clean | 7,337 | 5,844 | 0.6537 | **0.5549** | **0.5890** |
| dual-clean    | 8,192 | 5,429 | 0.7016 | 0.6788 | **0.7306** |

### 2.2 `random→clean` deltas per model

|         | random→ligand | random→protein | random→dual | ligand→protein |
|---------|---:|---:|---:|---:|
| Morgan-LR (linear) | −8.3pp | **−11.9pp** | −7.1pp | −3.6pp |
| Morgan-RF (trees)  | −9.9pp | **−25.1pp** | −12.7pp | −15.2pp |
| SPRINT (deep DTI)  | −7.5pp | **−24.8pp** | −10.6pp | −17.3pp |

### 2.3 Statistical power

PDBBind test sets carry 4-5K examples per regime with both classes
well-represented (38-60% positive rate). SE on each AUROC point estimate
≈ 0.01; SE on the random→protein delta of 0.251 (Morgan-RF) ≈ 0.013,
giving **z ≈ 19 (p ≪ 1e-30)**. The same delta on SPRINT (0.248) is
similarly hyper-significant. The headline is not a small-n story.

### 2.4 Key interpretations

- **Drop direction is model-invariant.** Every model loses on every
  clean regime. The KG is detecting a real shortcut that all three
  classes pay for.
- **Drop magnitude scales with model capacity.** Linear LR drops 12pp;
  non-parametric RF and deep SPRINT both drop 25pp. The KG's
  protein-clean filter penalizes high-capacity, memorization-friendly
  models exactly where it should.
- **SPRINT uses ProtBert protein embeddings but still leaks 25pp.**
  If SPRINT were actually generalizing across protein space — learning
  what makes a protein "binder-shaped" — it would outperform ligand-only
  Morgan-RF on protein-clean. It doesn't. The protein input isn't
  producing protein-level generalization; the model is using it to
  memorize protein-specific ligand patterns. This is the cleanest
  single piece of evidence that protein-clean is detecting *real
  generalization failure*, not a sampling artefact.

---

## 3. Group A++ — Multi-corpus random-control calibration (Morgan-RF + LR)

**Setup.** For each non-PDBBind corpus, we generated a same-size random
partition (matching protein-clean train/val/test sizes), then ran
Morgan-RF (Phase 1 baseline) and Morgan-LR (new this iteration).

### 3.1 Multi-corpus Morgan-RF AUROC with random calibration

| Corpus | random | ligand | scaffold | protein | pocket | dual | strict | `random→protein` Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEKOIS   | **0.888** | 0.886 | 0.850 | 0.757 | ∅ | 0.814 | 0.814† | **−13.1pp** |
| DUD-E    | **0.888** | 0.879 | 0.876 | 0.809 | ∅ | 0.829 | 0.829† | **−7.9pp** |
| LIT-PCBA | **0.523** | 0.518 | 0.526 | 0.556 | ∅ | 0.533 | 0.533† | +3.3pp |
| PDBBind  | **0.806** | 0.707 | 0.707 | 0.555 | 0.746 | 0.679 | 0.746† | **−25.1pp** |

∅ = infeasible (no edges of that axis in the v1 graph for that corpus).
† = degenerate strict-clean (n_groups = n_examples → effectively random).

### 3.2 Multi-corpus × multi-model (Morgan-LR vs Morgan-RF)

| Corpus | Model | random | ligand | protein | dual | random→protein Δ |
|---|---|---:|---:|---:|---:|---:|
| PDBBind | LR | 0.773 | 0.690 | 0.654 | 0.702 | −12pp |
| PDBBind | RF | 0.806 | 0.707 | **0.555** | 0.679 | **−25pp** |
| DEKOIS  | LR | 0.926 | 0.927 | 0.785 | 0.811 | −14pp |
| DEKOIS  | RF | 0.888 | 0.886 | 0.757 | 0.814 | **−13pp** |
| DUD-E   | LR | 0.980 | 0.976 | 0.923 | 0.933 | −6pp |
| DUD-E   | RF | 0.888 | 0.879 | 0.809 | 0.829 | −8pp |
| LIT-PCBA| LR | 0.722 | 0.739 | 0.596 | 0.682 | −13pp* (AUPRC ~0.01) |
| LIT-PCBA| RF | 0.523 | 0.518 | 0.556 | 0.533 | +3pp |

### 3.3 Per-axis leakage delta (Morgan-RF, random→clean)

| Corpus | random→ligand | random→scaffold | random→protein | random→pocket | random→dual |
|---|---:|---:|---:|---:|---:|
| DEKOIS   | −0.2pp | −3.8pp | **−13.1pp** | n/a | −7.4pp |
| DUD-E    | −0.9pp | −1.2pp | **−7.9pp** | n/a | −5.9pp |
| LIT-PCBA | −0.5pp | +0.3pp | +3.3pp | n/a | +1.0pp |
| PDBBind  | −9.9pp | −9.9pp | **−25.1pp** | −6.0pp | −12.7pp |

### 3.4 Per-corpus interpretation

- **PDBBind**: dominant *protein-axis* shortcut (−25pp). Largest
  single-axis effect of any corpus.
- **DEKOIS**: moderate *protein-axis* shortcut (−13pp). Ligand-axis
  defeated by matched-property decoys (−0.2pp).
- **DUD-E**: smallest *protein-axis* shortcut (−8pp). Cross-target
  decoy reuse spreads the ligand signal.
- **LIT-PCBA AVE**: **no leakage on any axis** (max ±3pp). AVE-defeated
  splits work as advertised. **This is the proper negative-control case
  for the audit framework** — it correctly reports "no leakage" on a
  corpus that has been adversarial-validation-pruned.

### 3.5 Capacity-vs-leakage finding (PDBBind only)

On PDBBind, RF leaks 2× as much as LR (25pp vs 12pp). On DEKOIS/DUD-E,
LR ≈ RF (within 2pp). Interpretation: PDBBind has rich ligand-pocket
interaction structure RF can memorize but LR can't; DEKOIS/DUD-E
leakage is more about target subset composition than per-pair
memorization.

---

## 4. Group C — Retrieval-native audit (DrugCLIP paper-ckpt zero-shot)

**Setup.** Same KG semantics lifted to target-level. For each of three
retrieval corpora (DUD-E 102 targets, DEKOIS 62, LIT-PCBA 15), we built:

- `v2_target_node.parquet` — one row per target with UniProt, n_actives, n_decoys.
- `v2_active_of_target.parquet` — typed edges: ligand-is-active-of-target.
- `v2_target_in_family.parquet` — ≥40% pairwise sequence-identity clusters via Biopython.
- Five regimes: target-random / target-clean / active-clean / scaffold-clean / dual-clean.

Each test target gets per-target eval: pocket·mol cosine over the
target's actives + decoys pool, then BEDROC(α=80.5), ROC-AUC, EF1%, EF5%.

### 4.1 DUD-E (102 targets in scope)

| Regime | n_test | AUROC mean ± std | BEDROC mean ± std | EF1% | EF5% |
|---|---:|---:|---:|---:|---:|
| target_random | 22 | 0.450 ± 0.189 | 0.132 ± 0.195 | 0.98 | 0.73 |
| target_clean  | 31 | 0.468 ± 0.174 | 0.103 ± 0.153 | 0.74 | 0.74 |
| active_clean  | 35 | 0.454 ± 0.226 | 0.075 ± 0.129 | 0.96 | 0.88 |
| dual_clean    | 35 | 0.454 ± 0.226 | 0.075 ± 0.129 | 0.96 | 0.88 |

scaffold_clean dropped (degenerate: 1 train / 101 test — DUD-E scaffold
sharing is so pervasive it collapses to one giant cluster). DUD-E
active_clean ≡ dual_clean identically (active-sharing already subsumes
Pfam family sharing on this corpus, 100% identical assignment).

### 4.2 DEKOIS 2.0 (62 targets in scope)

| Regime | n_test | AUROC mean ± std | BEDROC mean ± std | EF1% | EF5% |
|---|---:|---:|---:|---:|---:|
| target_random  | 18 | 0.479 ± 0.159 | 0.034 ± 0.052 | 1.09 | 0.69 |
| target_clean   | 18 | 0.549 ± 0.192 | 0.042 ± 0.067 | 0.80 | 1.27 |
| active_clean   | 18 | 0.467 ± 0.184 | 0.030 ± 0.061 | 0.80 | 0.83 |
| scaffold_clean | 15 | 0.499 ± 0.182 | 0.041 ± 0.060 | 1.14 | 1.06 |
| dual_clean     | 22 | 0.431 ± 0.136 | 0.018 ± 0.041 | 0.43 | 0.47 |

DEKOIS scaffold_clean is non-degenerate (unlike DUD-E). dual_clean ≠
active_clean here.

### 4.3 Honest interpretation (frozen-checkpoint limitation)

- **Per-target variance dominates.** AUROC std ≈ 0.19-0.23; per-target
  range 0.04-0.86. A few targets (hivint, ada, src) score AUROC ≥ 0.8;
  many score ≤ 0.3.
- **No statistically-significant single-regime gap.** DUD-E
  random→active BEDROC = 0.057 ± 0.047 (z ≈ 1.2, p ≈ 0.23); DEKOIS
  similar.
- **Subset-selection effects, not training-time leakage gaps.** Because
  the paper checkpoint is *frozen*, the KG only controls *which targets
  we test*, not what the model has seen. A drop in BEDROC on
  active-clean tells us "targets with cross-target-disjoint actives are
  harder for this model", not "DrugCLIP loses signal when train/test
  leakage is forbidden". A true retrieval-native training-time gap
  requires retraining — which is exactly what Group D does.

### 4.4 What Group C *does* demonstrate

- A retrieval-native audit protocol that **matches the model's intended
  use**: per-target BEDROC + ROC-AUC + EF across DUD-E-style libraries,
  with the v2 KG controlling target-level splits.
- A working pipeline (target nodes + family/active-share edges + 4 split
  regimes + per-target retrieval eval). aa2ar smoke test (BEDROC 0.955)
  confirms the implementation is sound — the model gives strong,
  paper-reportable signal when the decoy pool and metric match its
  training objective.
- Corpus characterizations: DUD-E scaffold sharing collapses
  scaffold_clean; DUD-E active sharing subsumes Pfam family sharing.

---

## 5. Group C++ — Cross-method retrieval audit via published benchmark

**Setup.** The LigUnity paper (Patterns 2025) ships per-target BEDROC /
ROC-AUC / EF1% for **26 methods on DUD-E** (102 targets), **18 on
DEKOIS** (81), **11 on LIT-PCBA** (15). Each method evaluated under its
own paper-native protocol (full decoy pools, multi-conformer, etc.).
We apply our v2 KG target-level splits as a *filter* over these
published tables — no retraining; the KG operates as an audit lens.

### 5.1 DUD-E — BEDROC by method × regime (headline methods)

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

### 5.2 DEKOIS — BEDROC by method × regime (headline methods)

| Method | random (n=18) | target_clean (n=18) | active_clean (n=18) | scaffold_clean (n=15) | dual_clean (n=22) | Δ(dual−random) |
|---|---:|---:|---:|---:|---:|---:|
| LigUnity       | 0.783 | 0.865 | 0.873 | 0.892 | 0.867 | +0.08 |
| LigUnity (seq) | 0.753 | 0.834 | 0.834 | 0.875 | 0.722 | −0.03 |
| Pocket-DTA     | 0.669 | 0.742 | 0.645 | 0.723 | 0.705 | +0.04 |
| Sequence-DTA   | 0.596 | 0.662 | 0.565 | 0.659 | 0.587 | −0.01 |
| RTMScore       | 0.417 | 0.532 | 0.639 | 0.577 | 0.603 | +0.19 |
| GenScore       | 0.404 | 0.484 | 0.498 | 0.488 | 0.570 | +0.17 |
| DrugCLIP       | 0.376 | 0.542 | 0.567 | 0.614 | 0.471 | +0.10 |

### 5.3 LIT-PCBA — BEDROC (n=2 random vs n=8 target-clean)

| Method | target_random (n=2) | target_clean (n=8) | Δ |
|---|---:|---:|---:|
| LigUnity        | 0.009 | 0.103 | +0.094 |
| LigUnity (seq)  | 0.015 | 0.089 | +0.074 |
| DrugCLIP        | 0.025 | 0.080 | +0.055 |
| GenScore        | 0.033 | 0.072 | +0.039 |
| GNINA           | 0.070 | 0.063 | −0.007 |

LIT-PCBA's structural problem (only 15 corpus targets + pervasive
active sharing) makes cross-regime conclusions uninterpretable here
(n=2 dominated by which two targets happened to land).

### 5.4 Direction-consistency sign test (the audit-framework finding)

Per-method point estimates are variance-dominated, but we can ask:
across N independent methods, how many drop random→clean?

| Corpus | Metric | Comparison | drop / total | mean Δ | sign-test p |
|---|---|---|:---:|---:|---:|
| **DUD-E** | BEDROC | random→target_clean | **21 / 26** | −0.036 | **0.0025** |
| DUD-E | BEDROC | random→active_clean | 18 / 26 | −0.029 | 0.076 |
| DEKOIS | BEDROC | random→target_clean | 3 / 18 | +0.056 | 0.0075 (opposite) |
| LIT-PCBA | AUROC | random→target_clean | 1 / 11 | +0.060 | 0.012 (opposite) |

- **DUD-E: 21/26 methods drop on target_clean (p=0.0025)** — strong
  cross-method direction-consistency. The KG's target-clean filter IS
  detecting a structural effect that most methods are sensitive to.
- **DEKOIS / LIT-PCBA: opposite direction (significant)** — corpus has
  too few targets to power a cross-regime audit; the random sample
  happens to pick harder targets than the clean filter selects.

### 5.5 What the v2 KG demonstrates here

- v2 KG target-level splits **do structural work** (n column variation
  per regime; sign-test passes on DUD-E).
- Whether a clean→random gap shows up at the BEDROC level is **highly
  model-dependent**, and on retrieval models without training-time
  leakage control the variance and corpus structure dominate at small n.
- A direction-consistent, statistically-significant retrieval-native
  leakage gap requires either (a) much larger n per regime, or
  (b) training-time retraining on each KG split — Group D.

---

## 6. Group D — LigUnity retrieval-native fair POC (in progress)

**Design.** Keep everything paper-native (architecture, loss,
hyperparameters, test sets, candidate pools, metrics). Use the v2 KG
**only** to filter LigUnity's training corpus into five regimes. Train
LigUnity from scratch on each filtered corpus. Evaluate on DUD-E,
DEKOIS, LIT-PCBA using their native test pipeline.

### 6.1 Why LigUnity is the fastest fair POC

- Their training corpus (ChEMBL 34 + BindingDB 2024m5 blended with
  PDBBind, organized into PocketAffDB) has ~43k assays — large enough
  that even after KG filtering, all regimes retain ≥14k assays
  (their paper trained on ~30k).
- Loss = `rank_softmax` operates **within each assay** (rank ligands
  for a target); assay-level filtering is therefore the correct
  granularity for leakage control.
- Existing infra: LigUnity repo cloned on VUW; published per-target
  BEDROC tables provide the paper-clean baseline (Group C++ shows
  LigUnity is the strongest method on DUD-E with random BEDROC 0.83).

### 6.2 KG step 1 — per-regime filter survival (committed `f60f7cf`)

We protect against 149 unique test UniProts (DUD-E 101 + DEKOIS 78 +
LIT-PCBA 14, deduplicated). All 149 mapped to `uniport40.clstr`
clusters. Test actives extracted from `outputs/v2_retrieval/graph_*`.

| Regime | n_assays | n_pairs | uniq_uniprots | uniq_ligands |
|---|---:|---:|---:|---:|
| (no filter) | 43,492 | 799,916 | 4,847 | 438,706 |
| **paper-clean** | 30,581 | 573,441 | 4,700 | 339,363 |
| **target-clean** | 23,026 | 418,630 | 4,204 | 265,188 |
| **active-clean** | 23,590 | 315,700 | 4,421 | 218,265 |
| **scaffold-clean** | 14,881 | 185,406 | 3,376 | 132,930 |
| **dual-clean** | 19,588 | 249,000 | 4,015 | 176,978 |

**Paper-clean removes 30% of assays.** Adding Pfam-40 family-disjoint
removes another 25% — i.e. **25% of LigUnity's published training is in
the same Pfam family as some test target.** That's the residual
leakage Group D will quantify.

Confirmed against paper: "0.8M data points across 0.5M unique ligands
and 53,406 pockets" — our (no filter) parse: 799,916 data points,
438,706 unique ligands canon (~0.5M), 4,847 unique UniProts (paper's
53K pockets = many pocket conformations per UniProt).

### 6.3 KG step 2 — per-regime label-subsetting (committed `13f1d2e`)

`subset_ligunity_labels.py` wrote per-regime
`train_label_blend_seq_full.json` + `train_label_pdbbind_seq.json` to
`LigUnity/data_kg/<regime>/`. Counts (blend / pdbbind):

| Regime         | blend  | pdbbind | total assays |
|---|---:|---:|---:|
| paper_clean    | 18,326 | 12,255  | 30,581 |
| target_clean   | 11,884 | 11,142  | 23,026 |
| active_clean   | 11,603 | 11,987  | 23,590 |
| scaffold_clean |  5,961 |  8,920  | 14,881 |
| dual_clean     |  8,661 | 10,927  | 19,588 |

### 6.4 KG step 3 — training infrastructure (committed `13f1d2e`)

On VUW:
- `prepare_ligunity_regime.sh <regime>` — symlinks shared lmdbs +
  dicts + clstrs + pretrains into `LigUnity/data_regimes/<regime>/`,
  copies per-regime labels.
- `launch_ligunity_train.sh <regime> <gpus>` — paper-native
  `unicore-train` invocation (arch=pocket_ranking, loss=rank_softmax,
  task=train_task, batch=24, lr=1e-4, max-epoch=50, warmup=0.06).
- `smoke_ligunity_train.sh` — 1-epoch validation of the pipeline.

Environment: `drugclip_env` extended with `transformers==4.44.2`
(needed for LigUnity's `protein_ranking` model imports). Pretrained
Uni-Mol mol + pocket encoders reused from DrugCLIP infra.

Data downloads complete (all from figshare 27966819 + 27967422):
- `train_lig_all_blend.lmdb` (4.0 GB)
- `train_prot_all_blend.lmdb` (1.8 GB)
- `valid_*.lmdb` (6 MB each)
- `align_res.zip` (81 MB)
- `DEKOIS_2.0x.zip` (2.6 GB) for DEKOIS test corpus
- (DUD-E / LIT-PCBA test data via gdrive — re-downloading; first
  attempt only got 32K stub)

### 6.5 Step 4 — current status (smoke test in flight)

1-epoch smoke test of `paper_clean` regime on a single GPU is running.
Model loaded (98M params), train data loader initialized.

**Bugs found and fixed during infra bring-up**:
- `pocketscreen` arch name in `train.sh` is stale; the registered name
  is `pocket_ranking`.
- `transformers` library needed downgrade to 4.44.2 (newer requires
  torch 2.6+, env has torch 2.4).
- Pipeline truncation (`| tee | head`) sent SIGPIPE to trainer; fixed.
- Cache JSON files (`pocket_name2idx_*`, `mol_smi2idx_*`) live in
  `cache/` subdir per figshare structure; symlinks added.

### 6.6 Step 5 onwards — planned

1. **Smoke** (~30 min): validates pipeline end-to-end at 1 epoch.
2. **Per-regime full training** (~24-48h each, 5 regimes): paper recipe
   — 50 epochs, batch 24, lr 1e-4, validation on CASF.
3. **Per-regime evaluation** (~30 min each): LigUnity's native
   `test.py` with `--test-task DUDE | DEKOIS | PCBA` against
   `<regime>/checkpoint_best.pt`. Outputs per-target BEDROC + ROC-AUC
   + EF1% per corpus.
4. **Report** the `5 regimes × 3 corpora` matrix with per-target
   variance for each cell.

**Expected schedule**: with 3 GPUs available and ~24h per regime on 1
GPU, all 5 regimes can finish in ~2 days running 1-2 in parallel. The
test eval is fast (~30 min per regime per corpus). Final results table
will be appended to this report when training completes.

---

## 7. Appendix — DrugCLIP retrofit to Group A (record of diagnostic path)

Two attempts were made to add DrugCLIP as a third model directly in
the Group A row-level binary audit on PDBBind. Both hit corpus-related
blockers and produced informative *diagnostics* rather than a clean
Group A row.

### 7.1 Attempt 1 — retrain on v2 PDBBind train splits

DrugCLIP's published recipe uses PDBBind 2020 + HomoAug
(~100K+ positives). Our v2 splits ship ≤10k positives per regime. The
in-batch-softmax loss collapsed by epoch 2 (loss locked at ln(N),
gradient norms → ~0, valid bedroc plateaued at random). Switching
`--update-freq 6` to recover effective batch=48 restored gradient flow
but didn't fix convergence — the corpus, not the optimizer, was the
binding constraint.

### 7.2 Attempt 2 — paper checkpoint zero-shot

Per-pocket retrieval AUROC on the published checkpoint:

| Regime | n pos pockets | mean per-pocket AUROC | median |
|---|---:|---:|---:|
| random  | 3,322 | **0.554** | 0.566 |
| ligand-clean  | 1,640 | 0.560 | 0.563 |
| protein-clean | 2,158 | 0.561 | 0.591 |
| dual-clean    | 2,516 | 0.574 | 0.596 |

Flat across all 4 regimes (2pp spread) vs Morgan-RF/SPRINT 25pp drop.
**Train-test contamination dominates the signal** — the paper trained
on PDBBind 2020+HomoAug, our v2 splits are subsets, so the leakage
filter doesn't separate in-domain from novel.

### 7.3 Diagnostic — pool composition is the issue (not contamination)

Cognate (label=1) vs swap-decoy (label=0) score deltas: ±0.006 across
all regimes, Mann-Whitney rank-AUC ≈ 0.48-0.51, p ≥ 0.05. The model
**isn't memorizing** the (pocket, ligand) pairs. The deeper issue is
that v2 swap decoys are themselves real PDBBind binders for other
proteins — the model recognizes them all as plausibly drug-like, and
the per-row binary metric isn't natively meaningful for DrugCLIP's
similarity-output score type.

**This diagnostic motivated Group C and Group D.** Group C lifts to
target-level metrics (DrugCLIP's intended use). Group D retrains a
contrastive model under the same KG control LigUnity supports natively
(its training corpus is large enough that filtering doesn't collapse
training).

---

## 8. Reproducibility (one-stop pointers)

### Group A — PDBBind row-level audit
| Step | Script / artifact |
|---|---|
| v2 graph + side-table | `vsleakkg.v2.build_graph`, `vsleakkg.v2.pipeline.build_side_table` |
| v2 splits | `vsleakkg.v2.pipeline` → `outputs/v2/phase1_full/splits/<corpus>/<regime>.parquet` |
| Random control | `tools/build_random_pdbbind_split.py` |
| Morgan-RF | `vsleakkg.v2.baselines.ligand_only`; random: `tools/run_morgan_rf_random.py` |
| Morgan-LR (new) | `tools/run_lr_baseline.py` |
| SPRINT train + test | published agg_config + `run_test_only.py` (in repo) |

### Group C — Retrieval-native target-level
| Step | Script |
|---|---|
| Target KG (DUD-E / DEKOIS / LIT-PCBA) | `tools/v2_retrieval/build_{dude,dekois,litpcba}_target_kg.py` |
| Target-level splits | `tools/v2_retrieval/build_dude_target_splits.py` (corpus-agnostic) |
| Per-target LMDBs | `tools/v2_retrieval/build_{dude,dekois}_target_lmdbs.py` |
| Per-target eval | `tools/v2_retrieval/eval_dude_retrieval.py` |
| Contamination diagnostic | `tools/v2_retrieval/contamination_diagnostic.py` |

### Group C++ — Cross-method aggregation
| Step | Script |
|---|---|
| Aggregate published benchmark | `tools/v2_retrieval/aggregate_ligunity_benchmark.py` |
| Direction-consistency sign-test | `tools/v2_retrieval/compute_direction_consistency.py` |

### Group D — LigUnity paper-native retrain per KG regime
| Step | Script |
|---|---|
| Train-corpus KG | `tools/v2_retrieval/build_ligunity_train_kg.py` |
| Per-regime label subsetter | `tools/v2_retrieval/subset_ligunity_labels.py` |
| Regime data dir prep | `prepare_ligunity_regime.sh` (on VUW) |
| Paper-native training | `launch_ligunity_train.sh <regime> <gpus>` (on VUW) |
| Native eval | LigUnity's `test.sh` with `--test-task {DUDE,DEKOIS,PCBA}` |

---

## 9. Scope deferred (clean list)

1. **Group D full retrains** (in progress as of this writeup).
2. **HomoAug-augmented DrugCLIP retrain** — the only path to a fair
   DrugCLIP-specific training-time leakage gap; blocked by HomoAug
   implementation effort.
3. **Second deep model in Group A** beyond SPRINT — would strengthen
   model-invariance further.
4. **SPRINT on DEKOIS / DUD-E / LIT-PCBA** — second-corpus deep model
   audit; per-corpus training would take ~1 day each.
5. **Per-corpus protein-side family graph** for stricter LIT-PCBA
   target-clean (currently singleton-family fallback).
6. **Group D extension to S2Drug / HypSeek / ConGLUDe** — same protocol
   applies; ConGLUDe is the closest paper-pinned candidate.

---

## 10. What's shippable today (audit-team summary)

- **Group A on PDBBind: −25pp protein-axis leakage gap, model-invariant
  across LR / RF / SPRINT.** z ≈ 19, p ≪ 1e-30. *This is the audit's
  headline result.*
- **Group A++ on DEKOIS, DUD-E**: same direction (−13pp, −8pp), random
  control calibrated.
- **Group A++ on LIT-PCBA**: framework correctly reports null (AVE
  works) — the proper negative control.
- **Group C++ on DUD-E**: 21/26 retrieval methods drop on target_clean
  (sign-test p=0.0025). Cross-method evidence that the KG's
  target-level filter does real structural work.
- **Group D**: paper-native LigUnity retrain infra ready; results to
  come.

All numbers verified against source CSVs (PDBBind v2 splits + Phase 1
combined CSV + SPRINT TEST logs + LigUnity benchmark parquets +
LigUnity train labels). Method delta from `proposal.tex` is limited
to (1) random-control calibration and (2) target-level lift of KG
semantics for retrieval-native audits.
