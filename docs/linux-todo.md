# VS-LeakKG v2 — Linux/GPU work list

Everything in this file requires resources that aren't available on Windows:
big disk for the 40M-edge graph, MMseqs2 for protein clustering, a GPU for
pocket embeddings, or a GPU for ConGLUDe retraining. The Windows-side work
(v2 algorithmic modules + tests) is already done in `src/vsleakkg/v2/`.

Work items are ordered roughly by dependency: do `[D1]` before `[D2]`, etc.

## [D1] Rebuild the graph with v2 schema

Status: pending Linux.

The current graph (`data/processed/mvp2_*.parquet`) was built with v1's
schema (9 edge types, hit-flag scoring). v2 introduces:

- multi-resolution protein-cluster nodes (30/40/90% separately, not merged)
- pocket-similarity edges driven by an embedding cosine
- explicit `example_in_trainset_m` edges per audited model
- IDF and degree-cap mitigation applied at build time

The build pipeline lives in `src/vsleakkg/build_graph.py` (v1). The Linux
agent should:

1. Add a `src/vsleakkg/v2/build_graph.py` that consumes the raw loaders
   (`load_chembl_db`, `load_bindingdb`, `load_pdbbind`, etc.) and emits
   v2-schema parquets (`data/processed/v2_nodes.parquet`,
   `data/processed/v2_edges.parquet`).
2. Apply `HubMitigationConfig` from `v2/schema.py` during build:
   - drop scaffolds with `n_heavy_atoms <= 6` and no substituents
   - shard nodes whose degree exceeds the cap into per-source pieces
   - apply IDF weight to scaffold and assay edges (floor at the default
     weight in Table 2)
3. Re-run MMseqs2 `easy-cluster` at 30 / 40 / 90% identity and emit a
   `ProteinCluster_q` node per resolution.
4. Compute pocket embeddings (ESM-IF1 or chosen encoder) and write the
   pocket-similarity edges (cosine ≥ 0.80).
5. Publish the new zip via `_hf_reupload.py` style flow and bump
   `DATASET_VERSION` to `v2` + the v2 zip filename in
   `scripts/dataset_version.sh` and `scripts/_dataset_version.ps1`.

## [D2] Ingest TrainSet manifests for each audited model

Status: pending Linux + per-model homework.

Each evaluated model needs a manifest of its actual training rows mapped
into the v2 graph's example IDs:

| Model     | Manifest source                                                       |
|-----------|-----------------------------------------------------------------------|
| ConGLUDe  | their `download_data.py` + their train-split CSV                       |
| DrugCLIP  | their training-split JSON / parquet from the released repo             |
| S2Drug    | their training-split CSV                                               |
| LigUnity  | their assay-record training corpus                                     |
| HypSeek   | their training-split CSV                                               |

For each model:

1. Pull the training manifest from the model's release.
2. Build an `id_map` parquet that resolves `(source, identifier)` →
   `example_id` against the v2 graph.
3. Call `v2.trainset.ingest_model_trainset(...)`, merge into the v2 nodes
   and edges via `v2.trainset.merge_into_graph(...)`.
4. Record the `unresolved` count for honesty (proposal section 5.6).

If a model's training corpus isn't auditable, fall back to proxy scoring
against declared provenance sources and clearly flag the limitation.

## [D3] Run Mode B audits per model

Status: pending [D2].

For each evaluated model `m`, compute:

```python
C_m(x_t) = score_overall(edges, reference=D_train^m, queries=D_bench)
```

Persist the result as `outputs/v2/mode_b/<model_id>/contamination.parquet`
plus a one-page summary CSV.

Sanity check: rerun the conglude AUROC drop experiment using the new
`C_conglude` mask. With Mode B applied correctly, the drop should be
substantially larger than the ~0.0001 we measured under the wrong (Mode A
against generic provenance) mask.

## [D4] Generate clean splits for each benchmark

Status: pending [D1].

For each (corpus, regime) pair:

```python
groups = build_leakage_groups(...)
assignment = greedy_assign(groups, examples)
```

Emit `outputs/v2/splits/<corpus>/<regime>.parquet` with columns
`(example_id, partition)` plus a `residual_contamination.csv` summary.
Run all seven regimes: ligand-clean, scaffold-clean, protein-clean,
pocket-clean, assay-clean, dual-clean, strict-clean. Mark infeasible
regimes explicitly (don't silently relax).

Corpora to cover:
- LIT-PCBA AVE
- DUD-E
- DEKOIS-2
- BayesBind V1.5 (use as held-out reference)

## [D5] Validation-contamination matrices

Status: pending [D4].

For each clean regime, compute the three matrices via
`v2.validation_contamination.three_way_contamination(...)`. Output:
`outputs/v2/validation_contamination/<corpus>/<regime>.csv` with the
flattened summary table.

## [D6] Shortcut baselines on v2 splits

Status: pending [D4]. Requires GPU only if the user's `dummy_receptor`
target uses ConGLUDe's protein encoder; otherwise pure CPU.

- ligand-only: `v2.baselines.ligand_only.evaluate_ligand_only(...)`. RDKit
  required for realistic fingerprints. ~CPU minutes per benchmark.
- contamination-NN: `v2.scoring.contamination_nn_label(...)`. CPU minutes.
- dummy-receptor: needs ConGLUDe's encoder loaded on GPU; otherwise CPU
  if using a simpler protein-LM encoder.
- source-only: reuse v1's `source_only_diagnostics.py` against the v2
  splits.

Output one row per (corpus, regime, baseline) into
`outputs/v2/baselines.csv`.

## [D7] ConGLUDe retraining under each clean regime

Status: pending [D4]. Heavy GPU.

Five training runs × three seeds:
- legacy split (reproduces paper number, sanity check)
- ligand-clean
- scaffold-clean
- protein-clean (30% cluster)
- strict-clean

Re-tune learning rate / warmup on the clean val for the main protocol;
also report a "reuse paper hyperparameters" ablation per the proposal
section 5.12.

Output: `outputs/v2/conglude/<regime>/seed<k>/metrics.json`.

## [D8] Final tables and figures

Status: pending [D5] [D6] [D7].

- Table 1: KG statistics + benchmark coverage (template in proposal.tex)
- Table 2: leakage groups & residual contamination per regime
- Table 3: ConGLUDe metrics legacy vs strict-clean
- Table 4: shortcut baselines vs full model per regime
- Table 5: validation-contamination effect (test metric leaky vs clean val)
- Figure 1: performance by contamination decile per model
- Figure 2: leakage-hub Pareto curve
- Figure 3: split-stability under ChEMBL 35 → 36 (appendix)

Code: `src/vsleakkg/v2/final_figures.py` (to be created on Linux side,
mirrors v1's `final_figures.py`).

## Quick wins worth doing first on Linux

If you only have a day of compute:
1. [D1] graph rebuild — unblocks everything else
2. [D2] one model manifest (ConGLUDe) — unblocks the Mode B sanity check
3. [D3] Mode B on ConGLUDe — confirms the methodological fix works
4. [D6] contamination-NN + ligand-only on the existing LIT-PCBA split

That's enough to write the methods paper without the multi-model retraining.
