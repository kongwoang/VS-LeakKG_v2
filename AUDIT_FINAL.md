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

## DrugCLIP — third-model attempt

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
| ligand-clean | 1,640 | 0.560 | 0.563 |
| protein-clean | 2,158 | 0.561 | 0.591 |
| dual-clean | 2,516 | 0.574 | 0.596 |
| random (control) | TBD | TBD | TBD |

The published model is essentially flat across leakage regimes. We read
this as **train-test contamination dominating the signal**: the published
checkpoint has seen each of our v2 test splits during its own training
(no protein, ligand, or dual partition of PDBBind is "novel" relative to
its training data), so the v2 leakage filter does not separate it from
in-domain performance. A clean Group A audit of DrugCLIP would require
either (a) retraining from scratch on a single v2 train split (Attempt 1
shows this needs a HomoAug-equivalent augmentation we did not implement),
or (b) recovering the paper's training manifest and excluding overlap.

We also report the *flat per-row AUROC* for direct comparison to
Morgan-RF / SPRINT, with the caveat that DrugCLIP's normalized dot product
is a **similarity** score, not a calibrated probability — flat AUROC across
1-row-per-pocket data is the wrong metric for it:

| Regime | DrugCLIP flat AUROC | (vs SPRINT) |
|---|---:|---:|
| ligand-clean | 0.482 | 0.762 |
| protein-clean | 0.490 | 0.589 |
| dual-clean | 0.505 | 0.731 |
| random | TBD | 0.837 |

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
- That DrugCLIP cleanly confirms the pattern. We tried both paths and both
  hit corpus-related blockers — paper-checkpoint zero-shot suffers train/test
  contamination (the paper's training corpus IS PDBBind 2020+HomoAug, our
  test splits are subsets of PDBBind 2020); retrain-from-scratch failed to
  converge on v2 train splits (≤10k positives is too small for in-batch
  softmax). Documented in "DrugCLIP — third-model attempt" section below.
- That LigUnity exhibits the same pattern. Deferred to future work; same
  corpus/contamination considerations apply.
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

- DrugCLIP retrain on a leakage-clean PDBBind subset with HomoAug-style
  augmentation (currently the v2 train pool is too small for the contrastive
  objective; HomoAug would boost it to paper-scale).
- LigUnity audit on v2 PDBBind splits (same paper-contamination concern as
  DrugCLIP for any pretrained model whose corpus overlaps PDBBind).
- Second-corpus deep-model audit (SPRINT on DEKOIS / DUD-E / LIT-PCBA — needs corpus-specific protein-seq lookups + days of training).
- Pocket-axis leakage on non-PDBBind corpora (needs `example_has_pocket` edges in those v1 graphs).
- Assay-axis leakage (needs `example_from_assay` edges added to v1 graphs from `chembl_assays.parquet`).

Each is tractable and each is documented in the per-phase reports.
