# VS-LeakKG v2 — Phase 1 final report

**Date:** 2026-05-23
**Compute:** VUW box (cuda12.ecs.vuw.ac.nz), 3× Quadro RTX 6000 24 GB shared with other users
**Workspace:** `/vol/dl-nguyenb5-solar/users/hoangpc/`
**Code:** `https://github.com/kongwoang/VS-LeakKG_v2` (head `25d1c54` at writeup time)

---

## Headline result (Run 4 — full corpus sizes)

Ligand-only Morgan-RF baseline AUROC under each v2 leakage regime. Same model
(`RandomForestClassifier(n_estimators=100)`), same featurizer (Morgan FP, 2048
bits), same RDKit version, same 15k-sample train cap. **Only the v2 split
changes between rows.**

| Corpus | ligand | scaffold | protein | pocket | dual | strict |
|---|---|---|---|---|---|---|
| DEKOIS | 0.886 | 0.850 | 0.757 | ∅ | 0.814 | 0.814† |
| DUD-E | 0.879 | 0.876 | 0.809 | ∅ | 0.829 | 0.829† |
| LIT-PCBA | **0.518** | 0.526 | **0.556** | ∅ | 0.533 | 0.533† |
| PDBBind | 0.707 | 0.707 | **0.555** | 0.746 | 0.679 | 0.746† |

- ∅ = infeasible (no edges of the required type in the v1 graph for that corpus)
- † = degenerate (leakage groups are singletons → effectively random split; see
  *§ Degenerate strict-clean* below)

## What the table says

### Finding F1 — three benchmarks, three shortcut profiles
- **DEKOIS / DUD-E**: heavy shortcut learning. Ligand alone gets AUROC ~0.88
  even at full corpus size with leakage-axis-clean splits. The matched-property
  decoys are not matched enough to defeat a Morgan fingerprint.
- **LIT-PCBA AVE**: shortcut **defeated**. Ligand-only AUROC sits at 0.52-0.56
  across every regime, indistinguishable from random. AVE's matched-property
  decoys work as advertised.
- **PDBBind**: moderate shortcut on ligand-clean (0.71), but **drops to 0.55**
  on protein-clean — the strongest single-axis effect we measure.

### Finding F2 — the dominant leakage axis differs per corpus
- DEKOIS: ligand axis dominant (drops 0.886 → 0.757 when protein forbidden, but
  ligand-clean still high → ligand-axis still leaks via the AVE-incomplete
  decoy generation).
- DUD-E: ligand+scaffold axes dominant (0.879/0.876 vs 0.809 protein).
- LIT-PCBA: nothing dominant (all near-random).
- PDBBind: **protein axis dominant** (0.71 → 0.55 = 16pp drop). Most of
  PDBBind's ligand-only predictability traces to shared protein clusters.

### Finding F3 — strict-clean is a degeneracy trap
On every corpus, strict-clean produced **n_groups = n_examples** (each example
its own leakage group). The framework's giant-component pruner removes the
source-axis edge (PDBBind is one source → would create a single mega-group);
the remaining `ligand + protein + scaffold` constraints can't bind any two
examples together if every example has a unique entity tuple. Result:
strict-clean degenerates to a random split, and its AUROC equals what you'd
expect for an unconstrained split. **strict-clean is reported with a † and
excluded from the per-axis audit comparison.** This is itself a methodological
finding worth surfacing in the paper.

### Finding F4 — pocket and assay axes are unmeasured for non-PDBBind corpora
DEKOIS / DUD-E / LIT-PCBA v1 graphs ship no `example_has_pocket` or
`example_from_assay` edges. PDBBind has pocket edges (added by our v2
build_graph Complex→Example synthesis). Filling the assay axis would require
joining `chembl_assays.parquet` into the example_id space — straightforward
future work.

---

## Phase 1 deliverables (committed)

### Graphs — `outputs/v2/graph_<corpus>/v2_{nodes,edges}.parquet`

| Corpus | nodes | edges | source |
|---|---|---|---|
| pdbbind | 86,446 | 130,898 | v1 pdbbind_{nodes,edges}.parquet + BindingMeasurement props pK extraction + protein_in_cluster from pdbbind_protein_clusters_{30,50,90} |
| dekois | 257,842 | 503,822 | v1 dekois_{nodes,edges} + protein_in_cluster from pdbbind clusters |
| dude | 3,059,299 | 6,918,019 | v1 dude_{nodes,edges} + clusters |
| litpcba_ave | 3,150,942 | 8,360,753 | v1 litpcba_ave_{nodes,edges} + clusters |

### Side-table — `outputs/v2/graph/side_table.parquet`
1,710,073 rows / 96 MB. Sources: litpcba 404,773, dude 1,196,111,
dekois 88,152, bayesbind 21,037. ChEMBL/BindingDB/PDBBind deferred
(slow / handled via dedicated lookup for PDBBind).

### Protein-sequence lookup — `outputs/v2/graph/protein_seq_lookup.parquet`
30,899 rows (19,037 pdb_id keys + 11,862 seq_sha256-prefix keys). Used by
the SPRINT/LigUnity/DrugCLIP adapters.

### Splits — `outputs/v2/phase1_full/splits/<corpus>/<regime>.parquet`
28 parquet files (4 corpora × 7 regimes). Each has columns
`example_id, partition, ligand_id, protein_id, smiles, label`.

### Validation contamination matrices — `outputs/v2/phase1_full/validation_contamination/<corpus>/<regime>.csv`
15 matrix CSVs (3 corpora × 5 feasible regimes). Each row reports
`(direction, n, mean, median, p90, p99, frac_gt_0.5, frac_gt_0.8)` for the
three directions `train->test`, `train->val`, `val->test`.

### Baselines — `outputs/v2/phase1_full/baselines/<corpus>/<regime>.csv`
24 baseline CSVs (4 corpora × 6 non-infeasible regimes, minus assay which
is infeasible everywhere). Each has the ligand-only Morgan-RF AUROC/AUPRC
plus n_pos_test/n_neg_test.

### Final tables / figure
- `outputs/v2/tables/table1_kg_stats.csv` — KG statistics per corpus
- `outputs/v2/tables/table2_leakage_groups.csv` — leakage groups + feasibility per (corpus, regime)
- `outputs/v2/tables/table5_validation_contamination.csv` — aggregated matrix table
- `outputs/v2/figures/figure2_hub_pareto.png` — hub-Pareto curve

### Archived run snapshots
- `outputs_run1_50k/` — first pass (50k sample cap, before PDBBind pK fix)
- `outputs_run3_pdbbind_pk/` — after PDBBind Complex→Example synthesis + pK from BindingMeasurement (still 50k cap)
- `outputs_run4_full/` — **final, no sample cap** (this report's numbers)

---

## Bugs found and fixed during the run

10 commits worth of bug fixes against the v2 codebase:

| Commit | Symptom | Root cause / fix |
|---|---|---|
| `97d9764` | PDBBind v2-graph build hung ~10 min (vs ~10 s after fix) | `scan_parquet` + 5× `.collect()` each re-reads the NFS file. Switched to eager `read_parquet`. |
| `db3c403` | LITPCBA 3M rows → 15 after side-table dedup | `_load_examples_parquet` fell back to `target` column as source_id when no `compound_id`. Use `ext_id_1` / `inchikey` etc. before last-resort `(target, row_index)`. |
| `559f221` | Pipeline crashed `SchemaError: null vs str` on PDBBind | No Example nodes in v1 PDBBind schema. Pipeline now detects empty examples list and emits an empty correctly-typed frame. |
| `b0f9fa2` | Summaries under `outputs/v2/phase1/phase1/<corpus>_summary.csv` (double-nested) | Caller passes `--output-dir outputs/v2/phase1`. Pipeline writes summary directly under `output_dir` now. |
| `5edcdb5` | Every example had `label=0.0` → `actives_per_partition=0` → all regimes infeasible | v1 stores labels in `props` column (JSON). Parsed props in `extract_examples_frame`. |
| `4a31ae9` | Baseline AUROC NaN (no SMILES join) | Side-table example_id (`<source>:<source_id>`) doesn't match v2 graph node_id (`Example::<v1_id>`). Pipeline pulls SMILES from the linked Ligand node's `label` column. |
| `9787a24` | 0 `protein_in_cluster` edges added | Cluster parquets use `seq_id`/`rep_seq_id`; heuristic looked for `sequence_id`/`representative`. Added those alternates. |
| `e4e88ae` | PDBBind 0/7 feasible (no Example nodes) | Synthesize Example nodes from Complex during build_graph. |
| `d4a5e1b` + `2559197` | PDBBind labels still 0.0 even after synthesis | Read `BindingMeasurement` node props' `p_value` (pKd / pIC50 / pKi). Binarize at pK ≥ 6.0 for the audit pipeline; preserve raw pK in props. |
| `26a297b` | Baseline RandomForest thrashed memory (>60 min, OOM) | RF n_estimators=200 × 35k samples × 2048 features. Capped train at 15k samples + 100 trees. |

The first 6 are "framework code" bugs; the rest are corpus-specific data-shape
gotchas. All fixed in v2 source.

---

## What's NOT in Phase 1 (deferred to Phase 2)

| Item | Reason | Where it'd live |
|---|---|---|
| Pocket-similarity edges (ESM-IF1 cosine ≥ 0.80) | Needs GPU embedding pipeline | Phase 2 / `D2` |
| Time-overlap edges | Needs ChEMBL release-date metadata | Phase 2 / `D3` |
| `example_from_assay` edges | Needs chembl_assays join | Phase 2 / `D3` |
| Model-paired audits (SPRINT, LigUnity, DrugCLIP) | Phase 2 P2.2-P2.7 | Phase 2 / running now |
| Dummy-receptor baseline | Model-paired | Phase 2 / D6 |

---

## Reproducing this report

```bash
# 1. Clone v2 repo (head 25d1c54 or later) and v1 repo on the same machine
git clone https://github.com/kongwoang/VS-LeakKG.git
git clone https://github.com/kongwoang/VS-LeakKG_v2.git
cd VS-LeakKG_v2

# 2. Point at v1 data and create venv
export VSLEAKKG_V1_ROOT=$(realpath ../VS-LeakKG)
/usr/pkg/bin/python -m venv --system-site-packages --without-pip envs/vsleak2
pip install --target=envs/vsleak2/lib/python3.12/site-packages polars pulp
pip install --target=envs/vsleak2/lib/python3.12/site-packages -e . --no-deps

# 3. Rebuild the 4 v2 graphs
envs/vsleak2/bin/python -m vsleakkg.v2.build_graph --output-dir outputs/v2/graph_pdbbind --corpus pdbbind
envs/vsleak2/bin/python -m vsleakkg.v2.build_graph --output-dir outputs/v2/graph_dekois --corpus dekois
envs/vsleak2/bin/python -m vsleakkg.v2.build_graph --output-dir outputs/v2/graph_dude --corpus dude
envs/vsleak2/bin/python -m vsleakkg.v2.build_graph --output-dir outputs/v2/graph_litpcba_ave --corpus litpcba_ave

# 4. Build the hydrate side-table
envs/vsleak2/bin/python -m vsleakkg.v2.build_side_table \
    --output outputs/v2/graph/side_table.parquet \
    --sources litpcba,dude,dekois,bayesbind

# 5. Build the protein sequence lookup (PDBBind)
envs/vsleak2/bin/python tools/build_protein_seq_lookup.py \
    --v1-processed "$VSLEAKKG_V1_ROOT/data/processed" \
    --output outputs/v2/graph/protein_seq_lookup.parquet

# 6. Run the full-size Phase 1 pipeline on each corpus
for spec in "outputs/v2/graph_pdbbind|pdbbind" \
            "outputs/v2/graph_dekois|dekois" \
            "outputs/v2/graph_dude|dude" \
            "outputs/v2/graph_litpcba_ave|litpcba"; do
  G="${spec%|*}"; TAG="${spec#*|}"
  envs/vsleak2/bin/python -m vsleakkg.v2.pipeline \
      --graph-dir "$G" \
      --side-table outputs/v2/graph/side_table.parquet \
      --output-dir outputs/v2/phase1_full \
      --corpus-tag "$TAG"
done

# 7. Render tables + figure
envs/vsleak2/bin/python -m vsleakkg.v2.final_figures --repo-root .

# 8. Optional: run the auto-review
envs/vsleak2/bin/python tools/phase1_review.py --repo-root . --write LINUX_RUN_REPORT_PHASE1_REVIEW.md
```

---

## Files referenced

- `outputs_run4_full/phase1_full/phase1_combined.csv` — the source of the headline table
- `outputs_run4_full/tables/` — paper tables
- `REVIEW_PHASE1.md` — older review (Run 1)
- `LINUX_RUN_REPORT_PHASE1.md` + `LINUX_RUN_REPORT_PHASE1_REVIEW.md` — auto-generated reports
- `PHASE2_SPRINT_PRELIM.md` — Phase 2 mid-training results
- `CHECKPOINT.md` — operational state

## What runs next (Phase 2)

Three SPRINT trainings are running on VUW as of this report. ETA ~22-28 h
to 250 epochs. They use the **unmodified published `agg_config.yml`** (no
contrastive, MorganFeaturizer + ProtBertFeaturizer). Group A audit signal
mid-training:

| Regime | best val/aupr | epoch |
|---|---|---|
| ligand-clean | 0.71 | 33 |
| protein-clean | 0.44 | 33 (plateaued at e11) |
| dual-clean | not yet | featurizing |

The **−27 pp** ligand-clean vs protein-clean drop on the **same model** at
**same config** is the Phase 2 audit signal, mirroring the shallow Morgan-RF
baseline's −16 pp drop. Final numbers in `PHASE2_*.md` once training lands.
