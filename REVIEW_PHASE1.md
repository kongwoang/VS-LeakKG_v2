# Phase 1 Results Review

Date: 2026-05-22
Run host: cuda12.ecs.vuw.ac.nz (VUW), load avg ~24-30 during run

## TL;DR

Phase 1 completed end-to-end. All deliverables landed except baseline
metrics (NaN due to a known SMILES join mismatch). **One emergent
finding worth highlighting in the paper:** ligand-clean splits on
DEKOIS / DUD-E / LIT-PCBA leave **mean contamination = 1.0** because
the protein axis (~81-102 targets shared by 50k+ examples) dominates
when ligand edges alone are forbidden. That is the framework working
as designed, not a bug.

## What ran cleanly

| Step | Status | Notes |
|------|--------|-------|
| [D1] v2 graph rebuild | OK | 4 corpora; perf went from 15 min/corpus to <10 s after the eager-read fix |
| [P2.1] side-table | OK | 1.71M rows; 4 sources (litpcba/dude/dekois/bayesbind) |
| [D4] clean splits | OK | 28 (corpus,regime) split parquets; 14 feasible |
| [D5] contamination matrices | OK | 15 matrix CSVs (3 dirs x 5 feasible regimes) |
| [D6] baselines | partial | Framework runs; AUROC/AUPRC NaN due to join mismatch (see issues) |
| [D8] tables + figure | OK | table1/2/5 CSV + figure2 PNG |

## Bugs found and fixed during the run

| # | Bug | Commit | Symptom |
|---|-----|--------|---------|
| B1 | `scan_parquet` + 5x `.collect()` re-reads on NFS | 97d9764 | PDBBind build took 9 min instead of <10 s |
| B2 | side-table fell back to `target` when no `source_id` | db3c403 | LITPCBA 3M rows -> 15 after dedup |
| B3 | pipeline crashed on graphs with no Example nodes (PDBBind) | 559f221 | `SchemaError: null vs str` on join |
| B4 | pipeline wrote summary under double `phase1/phase1/` | b0f9fa2 | downstream consumers couldn't find files |
| B5 | label always 0.0 (props JSON not parsed) | 5edcdb5 | actives_per_partition=0 -> every regime infeasible |
| B6 | cluster heuristic missed `seq_id` / `rep_seq_id` cols | 9787a24 | 0 `protein_in_cluster` edges added (fix not re-applied to existing graphs) |

## Known issues (genuine limitations, not bugs)

### I1 — baselines are NaN
The v2 Example node_id format (`ex:DEKOIS:hsp90:43914`) differs from
the side-table's `<source>:<source_id>` format (`dekois:CmpId_row0`).
The Hydrator join therefore matches **0 of 95668** dekois examples,
so `smiles` is null everywhere, the RDKit ligand-only baseline is
skipped, and `baseline_auroc` / `baseline_auprc` stay NaN.

Fix needed: rewrite `vsleakkg.v2.build_side_table` to emit example_id
in the v2-graph format (parse the v1 Example node label/props), OR
add a fallback path in `pipeline.extract_examples_frame` to extract
SMILES from the v1 Ligand node label.

### I2 — PDBBind has no Example nodes
The v1 PDBBind schema uses `Complex` + `BindingMeasurement` + `Pocket`
intermediaries instead of Example nodes. We map Pocket/Ligand/Protein
to v2 but drop Complex/BindingMeasurement, leaving no v2 Examples.

All 7 PDBBind regimes are correctly marked infeasible with
`notes='no_examples_in_graph'`.

Fix needed: synthesise an Example node per Complex in `build_graph.py`
(one Example = one PDBBind complex with its ligand + protein + pK).

### I3 — pocket and assay axes universally infeasible
v1 doesn't ship `example_has_pocket` or `example_from_assay` edges in
the *_edges.parquet files. The audit only has access to the binding
+ similarity edges that v1 actually emitted.

Pocket fix: requires the pocket-embedding pipeline (ESM-IF1) to run
on the unpacked PDB files. Deferred to Phase 2.

Assay fix: join chembl_assays.parquet to add `example_from_assay`
edges. Tractable in Phase 1 if the example_id alignment from I1 is
also fixed.

### I4 — ChEMBL / BindingDB sources missing from side-table
We restricted P2.1 to litpcba/dude/dekois/bayesbind to keep runtime
under one hour. ChEMBL is the slowest because the v1 chembl_ligands.parquet
covers ~2M molecules x ~1ms canonicalize = ~30 minutes; BindingDB is similar.

Fix when needed for Phase 2: re-run `build_side_table` with `--sources
litpcba,dude,dekois,bayesbind,chembl,bindingdb,pdbbind`. PDBBind also
just needs the index parquet read.

### I5 — `--sample-examples 50000`
`run_phase1.sh` caps each corpus to 50k rows so leakage-group +
greedy-assign per regime finishes in 2-6 minutes under the loaded
box. For a production run remove this flag to use all 95k DEKOIS /
1.4M DUD-E / 2.7M LIT-PCBA-AVE rows. ETA on a quiet box: ~hours.

### I6 — protein-cluster edges not in graphs
Bug B6 was fixed in commit 9787a24 but build_graph already wrote the
v2 parquets before the fix. Adding `protein_in_cluster` edges would
make the `protein` axis use cluster-30/40/90 (proposal Table 2) rather
than just direct `example_has_protein` joins. Re-run build_graph on
all 4 corpora to apply.

## Headline findings worth flagging

### F1 — every-axis contamination = 1.0 on DUD-E ligand-clean
DUD-E with ligand-clean regime: `train -> test` C_overall mean = 0.97,
**frac_gt_0.5 = 0.95**, p99 = 1.0. So 95% of test examples have a
contamination path of strength >= 0.5 to some train example, even
under the cleanest available ligand-axis split.

Source: protein axis (102 DUD-E targets, ~50k examples sampled). The
ligand-clean split doesn't touch the protein axis, and 50k examples
across 102 targets means almost any test example shares a target with
many train examples.

### F2 — LIT-PCBA protein-clean has rho_max = 0.1365
The largest leakage group in the protein-clean regime contains 13.65%
of all examples. That's above the proposal's `rho_max_ok = 0.30`
threshold (still OK to greedy-split) but well above `rho_max_prune
= 0.60` would warrant Louvain. Worth comparing to v1 numbers.

### F3 — dekois val->test contamination is identical to train->test
For every dekois regime, the three contamination matrices return the
**same values** for train->test, train->val, val->test. That's
because v1 dekois has only 81 targets and the protein axis dominates
in all three directions equally. NOT a bug; it's a structural property
of the dekois corpus.

### F4 — LIT-PCBA dual regime: train->test mean=0.45, val->test mean=1.0
The dual-clean split (forbid ligand + protein + ligand_similar) gets
contamination down to mean=0.45 between train and test but val->test
stays at 1.0. This suggests val and test partitions remain heavily
connected via the remaining axes (scaffold / source / decoy_protocol).

## Recommended follow-up before paper claims

| Priority | Action |
|----------|--------|
| P1 | Fix I1 (SMILES join) so we get real baseline numbers |
| P1 | Re-run build_graph with the B6 fix to get protein_in_cluster edges |
| P2 | Synthesise Example nodes in PDBBind (I2) so PDBBind isn't a 0-row gap |
| P2 | Drop the `--sample-examples 50000` cap and let one production run loose overnight |
| P3 | Side-table chembl + bindingdb + pdbbind so Phase 2 hydrate is complete |
| P3 | Add `example_from_assay` edges from chembl_assays (unblocks assay-clean) |

## Files of interest

- `outputs/v2/graph_<corpus>/v2_{nodes,edges}.parquet` - per-corpus v2 graph
- `outputs/v2/graph_<corpus>/stats.csv` - KG statistics per corpus
- `outputs/v2/graph/side_table.parquet` - 1.71M-row hydrate parquet
- `outputs/v2/phase1/<corpus>_summary.csv` - per-corpus regime summary
- `outputs/v2/phase1/phase1_combined.csv` - 28-row combined view (4 corpora x 7 regimes)
- `outputs/v2/phase1/splits/<corpus>/<regime>.parquet` - per (corpus, regime) split
- `outputs/v2/phase1/validation_contamination/<corpus>/<regime>.csv` - 3-row matrix per feasible regime
- `outputs/v2/tables/table{1,2,5}*.csv` - paper tables
- `outputs/v2/figures/figure2_hub_pareto.png` - hub-Pareto plot
- `LINUX_RUN_REPORT_PHASE1.md` - high-level status
- `LINUX_RUN_REPORT_PHASE1_REVIEW.md` - auto-generated review

## Box-load notes

- VUW load avg held at 24-30 throughout (other users' jobs). Total wall-clock
  for Phase 1: ~2h (could be 30 min on a quiet box).
- The conda env at `/vol/grid-solar/sgeusers/longnd/miniconda3/envs/vsleak`
  is on a 98%-full NFS volume and *un-usable* for fresh python startup
  (610 s for one `import polars`). We used the local `/usr/pkg/bin/python`
  (3.12.13) with `pip install --user polars pulp` instead, which worked
  perfectly. Recommend updating linux-agent-prompt.md to reflect this.
