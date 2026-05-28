# Split-framework methodology summary

Concise one-page-per-framework summary of the three external split frameworks (AVE, DrugOOD, DataSAIL) compared against the KG multi-axis contamination-aware split. Source PDFs in `D:\hoangpc\VS_paper\`.

---

## 1. AVE — Asymmetric Validation Embedding
Wallach & Heifets, 2018 (arXiv 1706.06619v2); operationalised for VS by Tran-Nguyen, Jacquemard & Rognan (LIT-PCBA, JCIM 2020).

**Scope**: ligand-only, active/decoy classification, per-target.

**Bias measure**.
Let `Va, Vi, Ta, Ti` be validation actives, validation inactives, training actives, training inactives. Let `H(X, Y, d)` = fraction of points in `X` whose nearest neighbour in `Y` is within Tanimoto distance `d` (ECFP4 by default). With `D` a set of thresholds in `[0, 1]`:

```
B = (AA - AI) + (II - IA)
   AA = H(Va, Ta),  AI = H(Va, Ti)
   II = H(Vi, Ti),  IA = H(Vi, Ta)
```
A benchmark is unbiased when `B → 0`. AVE strictly extends MUV's nearest-neighbour analysis by also accounting for inactive-class clumping.

**Debiasing procedure**.
Genetic algorithm `remove_AVE_bias.py` swaps molecules between train and validation until `B < threshold` or iteration cap reached. Can drop samples (LIT-PCBA dropped 25% of two targets to converge).

**Axes covered**: ligand only.
**Sample loss**: yes (variable).
**Stratification**: implicit — active/inactive ratio preserved by swap policy.
**Output**: train/val partition per target; bias `B` reported.

---

## 2. DrugOOD — Single-axis domain split
Ji et al., AAAI-23 (`DrugOOD_2.pdf`, also arXiv `DrugOOD_1.pdf`).

**Scope**: drug-target binding affinity classification curated from ChEMBL (SBAP). Cross-target by construction.

**Domain axes** (five, one per dataset):
1. molecular scaffold (Bemis-Murcko)
2. molecular size (MolWt buckets)
3. protein (single protein per domain)
4. protein family (UniProt class)
5. assay (ChEMBL `assay_id`)

**Splitting procedure (explicit).**
For each chosen axis: (i) compute the per-domain frequency table, (ii) sort domains by frequency in descending order, (iii) greedy bin-packing — walk down the sorted domain list and assign each whole domain to the fold (train / val / test) whose current cumulative count is furthest below its 80 / 10 / 10 target. A domain is **never split across folds** (this is what makes it a domain split). The greedy step terminates when every domain is assigned. Class balance is not enforced inside this loop; it is reported but not used as a constraint.

**Noise tiers**: core / refined / general (ChEMBL confidence filters). Orthogonal to splitting.

**Sample loss**: none from splitting; samples assigned to whichever domain they belong to.
**Stratification**: not enforced; class balance varies by domain.
**Output**: 45 prebuilt datasets (5 axes × 3 noise × 3 measurement types).

**Limit**: one axis at a time. No joint multi-axis cleaning; no continuous leakage score; no enforcement of inactive-class debiasing.

---

## 3. DataSAIL — Similarity-aware ILP split
Joeres, Blumenthal & Kalinina, Nat Comm 2025 (`DataSAIL.pdf`).

**Scope**: generic. Two modes:
- **S1** — one-dimensional, single entity type (e.g. ligand only, or protein only).
- **S2** — two-dimensional, joint cold split over (ligand, protein) pairs.

**Optimisation**.
Cluster the data using a chosen similarity (Tanimoto/ECFP for ligands, MMseqs2 for proteins, FoldSeek for structures). Solve an ILP that assigns clusters to folds to minimise total inter-fold similarity:
```
L(π) = Σ_{x,y : π(x) ≠ π(y)}  sim(x, y) · |x| · |y|
```
subject to fold-size constraints (`|Σ |x|, x∈fold_i| ≈ s_i · N`) and class-stratification constraints (`|Σ |x| of class c in fold_i| ≈ s_i · N_c`). Default solver: SCIP (open) or Gurobi (commercial).

**Sample loss**.
- S1 (S1-ligand or S1-protein): **no interactions are dropped.** S1 clusters and partitions a single entity type; every (ligand, protein) pair simply inherits the fold assignment of its entity, so the join is exhaustive. Reported drop count is structurally zero.
- S2: drops interactions where the two entities are assigned to different folds (i.e. a ligand goes to train but its partner protein goes to test). Drop rate is reported. This is the only DataSAIL mode that can lose data.

**Output**: train/val/test fold IDs; `L(π)` per split (lower = harder).

**Limit**: ignores assay/source/time/pocket; only similarity-based and only over the entity types provided. No notion of multiple semantic leakage paths.

---

## 4. KG multi-axis contamination-aware split (this work)
Reference: Phase 1 audit machinery in this repo (`tools/run_contam_bins.py`, `tools/run_path_attribution.py`, `outputs/reports/phase1_kg_audit_report.md`).

**Scope**: any benchmark with KG-linkable identifiers (DUD-E, DEKOIS, LIT-PCBA, PDBBind have been wired up).

**Per-pair contamination score**.
For a candidate (train_i, test_j) pair, define an axis-decomposed contamination
```
C(i, j) = Σ_a w_a · c_a(i, j)
```
with axes `a ∈ {ligand, scaffold, protein, protein_family, pocket, assay, source, time}` and per-axis component `c_a` from the KG-derived similarity / shared-neighbour kernel (Dijkstra distance on the metadata graph, capped at 1). `protein_family` is treated as a separate axis from `protein` so that family-level leakage is scored even when sequence identity is low. Weights `w_a` are configurable and default to uniform across populated axes; degenerate axes (e.g. `assay` on LIT-PCBA, where AID ≡ target) are excluded from the mask, not given low weight.

**Splitting procedure**.
Iterative greedy selection: assign test items to maximise the sum `Σ_j min_i C(i, j)`-margin, gated by a per-axis cap and a global `C_total` budget. The implementation supports regime aliases `ligand_clean`, `scaffold_clean`, `protein_clean`, `pocket_clean`, `dual_clean` (ligand + protein), and `paper_clean` (reproduce paper splits) as preset weight masks.

**Auxiliary outputs** (this is the contribution beyond an opaque ILP):
- per-axis residual `c_a` distribution after splitting,
- dominant-axis attribution per test pair (which axis carried the leakage),
- contamination-bin AUROC curves (does AUROC fall as `C` rises within the test set?),
- C-NN label-copying baseline (does a 1-NN that uses KG-distance alone beat Morgan-RF?).

**Sample loss**: optional; default off (greedy reassignment). When enabled, drops are reported.
**Stratification**: enforced via per-class quota in the greedy step.

**Limit**: depends on KG coverage. Axes with sparse population (e.g. `assay` for DUD-E) collapse to ~0 weight and effectively drop out.

---

## Side-by-side framing for the benchmark

| Property | AVE | DrugOOD | DataSAIL S1 | DataSAIL S2 | KG multi-axis |
|---|---|---|---|---|---|
| Active/decoy native | yes | partial (affinity) | yes | yes | yes |
| Per-target Mode A | yes | scaffold/size only | S1-ligand only | no | ligand_clean, scaffold_clean |
| Pooled Mode B | n/a | yes (protein, family, assay) | S1-protein | yes | protein_clean, dual_clean |
| Continuous leakage score | yes (`B`) | none | yes (`L(π)`) | yes (`L(π)`) | yes (`C_total`, per-axis `c_a`) |
| Inactive-class control | yes (II−IA term) | no | yes (via class stratification) | yes | yes (per-class quota) |
| Assay / source / time aware | no | assay only | no | no | yes (when populated) |
| Path attribution | no | no | no | no | yes |
| Sample loss | variable | none | none | yes (interactions) | optional |

The headline benchmark question: **does KG multi-axis splitting reduce residual contamination and shortcut performance below what these three frameworks achieve?** With explicit honesty clauses (see `kg_split_benchmark_protocol.md`).
