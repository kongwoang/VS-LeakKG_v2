# KG vs split frameworks — capability matrix

Side-by-side capability and applicability matrix for the benchmark. Tables are organised so reviewers can see at a glance (a) what each framework controls, (b) what metadata each corpus carries, and (c) where each framework can / cannot be run on each corpus.

---

## Table 1 — Framework capability

| Capability | Random | Scaffold (Bemis-Murcko) | Protein (MMseqs2) | AVE | DrugOOD scaffold | DrugOOD size | DrugOOD protein | DrugOOD family | DrugOOD assay | DataSAIL S1-lig | DataSAIL S1-prot | DataSAIL S2 | KG multi-axis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Controls ligand-axis leakage | no | partial (scaffold only) | no | yes (Tanimoto) | partial (scaffold) | very partial (size) | no | no | no | yes (Tanimoto / cluster) | no | yes | yes |
| Controls protein-axis leakage | no | no | yes (sequence) | no | no | no | yes | no | no | no | yes | yes | yes |
| Controls protein-family-axis leakage | no | no | no | no | no | no | no | yes | no | no | no | no | **yes (separate KG axis)** |
| Controls assay/source/time | no | no | no | no | no | no | no | no | yes (ChEMBL `assay_id`) | no | no | no | yes (when populated) |
| Inactive-class debiasing | no | no | no | **yes** ((II−IA) term) | no | no | no | no | no | implicit (class quota) | implicit | implicit | yes (per-class quota) |
| Continuous leakage score reported | no | no | no | yes (`B`) | no | no | no | no | no | yes (`L(π)`) | yes | yes | yes (`C_total`, per-axis `c_a`) |
| Per-axis decomposition / attribution | no | no | no | no | no | no | no | no | no | no | no | no | **yes** |
| Joint multi-axis cleaning | no | no | no | no | no | no | no | no | no | no | no | partial (2 axes) | yes (≤8 axes: ligand, scaffold, protein, protein_family, pocket, assay, source, time) |
| Sample loss | none | none | none | variable (cap+GA) | none | none | none | none | none | none | none | yes (lost interactions) | optional |
| Per-target (Mode A) capable | yes | yes | n/a (cross-target by nature) | yes | yes | yes | n/a | n/a | n/a | yes | n/a | n/a | yes (`ligand_clean`, `scaffold_clean`) |
| Pooled (Mode B) capable | yes | yes | yes | n/a (per-target) | yes | yes | yes | yes | yes | yes | yes | yes | yes (`protein_clean`, `dual_clean`) |

---

## Table 2 — Corpus metadata availability

Legend: ✓ = usable; **degenerate** = present but uninformative as an independent split axis; **unavailable** = absent and not recoverable; ~ = partially recoverable.

| Field | DUD-E (102 targets) | DEKOIS 2.0 (81 targets) | LIT-PCBA (15 targets) |
|---|---|---|---|
| Ligand SMILES | ✓ | ✓ | ✓ |
| Active / inactive labels | ✓ (synthetic decoys) | ✓ (synthetic decoys) | ✓ (**experimental** inactives) |
| Bemis-Murcko scaffold | ✓ (derive, RDKit) | ✓ (derive) | ✓ (derive) |
| Protein PDB | ✓ | ✓ | ✓ (template per target) |
| Protein sequence | ✓ (PDB → seq) | ✓ | ✓ |
| Protein family | ✓ (DUD-E ships 8 groups) | ✓ (BindingDB / UniProt class lookup, ~81 IDs) | ✓ (UniProt lookup, 15 IDs) |
| Pocket / binding-site definition | ✓ (DUD-E co-crystal pocket) | ✓ (DEKOIS site) | ✓ (LIT-PCBA template) |
| Assay ID | **unavailable** (decoys are ZINC; actives' assay collapsed during DUD-E construction) | **unavailable** (BindingDB collapsed) | **degenerate** (1 PubChem AID per target → axis ≡ protein axis; do not treat as an independent split axis) |
| Source | constant per class (actives = ChEMBL/BindingDB, decoys = ZINC) — uninformative as a split axis | same | constant per corpus (PubChem) — uninformative |
| Timestamp (actives) | ~ (ChEMBL year recoverable for many actives) | ~ (BindingDB year for actives) | ✓ (PubChem AID deposit/modify date) |
| Timestamp (decoys/inactives) | **unavailable** (ZINC has no measurement date) | **unavailable** | ✓ (same AID as actives) |

Headline implications:
- `assay`, `source`, and decoy-side `time` axes are not independent split dimensions on DUD-E and DEKOIS — they collapse with the class label. They will be carried in the KG `C` score for completeness but will not be used as standalone splitting axes on those corpora.
- LIT-PCBA's experimental inactives are a strict upside: assay/source/time are uniform across labels, so the framework comparison there is the cleanest.

---

## Table 3 — Framework × corpus runnability (Mode A = per-target, Mode B = pooled)

| Splitter | DUD-E Mode A | DUD-E Mode B | DEKOIS Mode A | DEKOIS Mode B | LIT-PCBA Mode A | LIT-PCBA Mode B |
|---|---|---|---|---|---|---|
| Random | run | run | run | run | run | run |
| Scaffold (Bemis-Murcko) | run | run | run | run | run | run |
| Protein (MMseqs2) | n/a | run (102 prot) | n/a | run (81 prot) | n/a | run (15 prot, low statistical power — report with caveat) |
| AVE (GA, iter cap) | run | n/a | run | n/a | run (re-debias on top of LIT-PCBA's existing AVE pass; report residual `B`) | n/a |
| DrugOOD scaffold | run | run | run | run | run | run |
| DrugOOD size | run | run | run | run | run | run |
| DrugOOD protein | n/a | run | n/a | run | n/a | run (15 targets → small folds) |
| DrugOOD protein family | n/a | run | n/a | run | n/a | run |
| DrugOOD assay | **N/A — unavailable** | **N/A — unavailable** | **N/A — unavailable** | **N/A — unavailable** | **N/A — degenerate (AID ≡ target)** | **N/A — degenerate** |
| DataSAIL S1-ligand | run | run | run | run | run | run |
| DataSAIL S1-protein | n/a | run | n/a | run | n/a | run (small; no interaction drop — S1 partitions by entity, only S2 can drop) |
| DataSAIL S2 | n/a | run (report drop %) | n/a | run | n/a | run |
| KG `ligand_clean` | run | run | run | run | run | run |
| KG `scaffold_clean` | run | run | run | run | run | run |
| KG `protein_clean` | n/a | run | n/a | run | n/a | run |
| KG `dual_clean` (ligand+protein) | n/a | run | n/a | run | n/a | run |

Note: `n/a` means the splitter is conceptually undefined in that mode (e.g. a protein-axis split inside a single target). It is **not** a missed run — it is an honest absence.

---

## Table 4 — Headline framework comparison the benchmark will publish

For the executive table that goes into the paper, only the four headline contenders are juxtaposed against KG. Per the approved constraints, the focus is:

| Framework | Best regime to compare against KG | Mode | KG counterpart |
|---|---|---|---|
| AVE (LIT-PCBA-style) | AVE-debias ligand split (per-target) | A | `ligand_clean` |
| DrugOOD scaffold | scaffold-domain split | A & B | `scaffold_clean` |
| DrugOOD protein | protein-domain split | B | `protein_clean` |
| DrugOOD protein family | family-domain split | B | `protein_clean` (family-aware variant) |
| DataSAIL S1-ligand | similarity-aware ligand cold split | A & B | `ligand_clean` |
| DataSAIL S2 | joint ligand+protein cold split | B | `dual_clean` |

Everything else (size, S1-protein, etc.) is reported in the appendix for completeness but is not in the headline comparison.

---

## Honest-claim guardrails (carried from approval)

1. KG cannot claim a residual-contamination win unless `C_total` and at least one shortcut baseline (C-NN, 1-NN, AVE `B`) both fall below the corresponding values produced by DataSAIL S2 / AVE.
2. KG cannot claim a model-AUROC win unless the per-target Wilcoxon test (Mode A) over the 81/102/15 targets reaches `p < 0.05` after Holm correction across the framework set.
3. If KG matches DataSAIL S2 on max-similarity / `L(π)` but extends the leakage account to assay/source/time/pocket axes that DataSAIL ignores, the claim is **breadth of leakage control**, not strict tightness.
4. If KG ties or loses on every numeric column, the headline contribution is **interpretability + path attribution + per-axis residual scoring** and we say so in plain text without dressing it up.
