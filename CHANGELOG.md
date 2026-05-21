# VS-LeakKG v2 — changelog

v2 lives alongside v1. v1 reproducibility is unaffected: every script under
`scripts/` and every module under `src/vsleakkg/` (excluding `src/vsleakkg/v2/`)
behaves identically. v2 modules live under `src/vsleakkg/v2/` and have their
own tests under `tests/v2/`.

This document tracks the methodological changes; for the formal write-up see
`proposal.tex`.

## What changed and why

### 1. Mode A vs Mode B (the headline change)
v1 conflated two questions: "how should we split this corpus?" (Mode A) and
"is this model's reported result contaminated?" (Mode B). The two require
different reference sets:

- **Mode A** scores examples against partitions of the same corpus.
- **Mode B** scores benchmark examples against the *actual training data*
  of a specific model. We model this by adding a `TrainSet_m` node per
  audited model and `example_in_trainset_m` edges.

The conglude null-result we hit (~0 AUROC drop after removing "leaked"
rows) was a direct symptom of this conflation: the leak mask was defined
against a generic provenance DB, not against ConGLUDe's training set.

### 2. Contamination scoring: multiplicative + -log Dijkstra
v1 used a per-axis hit-flag × weight, then max-over-axes. v2 implements the
proposal's path-product formulation:

```
S(π) = ∏_e w_r(e)    w in (0, 1]
C(x_t, A) = max over reference x_i, max over paths π
```

Computed efficiently via the cost transform `c(e) = -log w_r(e)` and
multi-source Dijkstra. Axis decomposition is now done on **axis-specific
subgraphs** (per `schema.AXIS_EDGE_TYPES`), eliminating the v1 ambiguity
about which axis a mixed-type path belongs to.

### 3. Hub-pollution mitigation
v1 had no defence against high-degree hub nodes (trivial scaffolds,
promiscuous targets, large dataset-source supernodes). v2 schema documents
three mitigations via `HubMitigationConfig`:

- trivial-scaffold filter (drop single-ring scaffolds with no substituents)
- degree caps per node type
- IDF-style downweighting on scaffold and assay edges

The cap/IDF logic is parameterised; the build step that applies them is a
Linux task (see v2-linux-todo.md) because it operates on the 40M-edge graph.

### 4. Leakage hubs (new diagnostic)
For each training example we now compute
`H(x_train) = |{ x_test : x_train is the argmax-contamination match }|`.
Top-K hubs identify the small set of training rows that account for most
cross-partition leakage. Often trimming hubs is faster than reconfiguring
the split. Implementation in `v2/hubs.py`.

### 5. Validation-contamination matrices
v1 only reported `C(train→test)`. v2 reports three matrices per regime:

- `C(train→test)`  inflates fit
- `C(train→val)`   inflates checkpoint selection
- `C(val→test)`    inflates final test via val-driven model selection

The validation-leakage experiment compares the test score under (a) the
checkpoint selected on a model's published (potentially leaky) val and
(b) the checkpoint selected on the VS-LeakKG-clean val. The difference is
the headline validation-contamination effect. Implementation in
`v2/validation_contamination.py`.

### 6. Feature leakage vs label leakage
Path-based contamination measures feature similarity. Label leakage (a
genuinely identical `(protein, ligand, label)` row appearing in two
partitions, or the same `(protein, ligand)` with conflicting labels) is
now reported as a separate exact-match check in `v2/label_leakage.py`.

### 7. Giant-component handling
v2 admits the failure mode explicitly:

- if `ρ_max ≤ 0.30`: use connected components directly
- if `0.30 < ρ_max ≤ 0.60`: prune the weakest forbidden relation, recurse
- if `ρ_max > 0.60`: fall back to weighted Louvain community detection

If none of these yields a feasible split, the regime is reported as
`infeasible` rather than silently relaxed. Implementation in
`v2/leakage_groups.py`.

### 8. Group-atomic split assignment
`v2/split.py` implements deterministic greedy assignment with the
proposal's four-term objective (size, label, cover, residual) and minimum
utility constraints (`min_targets_per_partition`, `min_actives_per_partition`).
A PuLP MILP fallback is provided for cases where the greedy can't satisfy
the constraints.

### 9. Shortcut baselines
v2 adds two new baselines and keeps the existing two:

- `v2/baselines/ligand_only.py` — Morgan-RF (sklearn)
- `v2/baselines/dummy_receptor.py` — replaces protein embedding with the
  training-set mean
- contamination-NN — implemented in `v2/scoring.contamination_nn_label`
- source-only — v1's `source_only_diagnostics.py` is reused as-is

### 10. KG versioning is now explicit
`scripts/dataset_version.sh` and `scripts/_dataset_version.ps1` now declare
`DATASET_VERSION` alongside the zip name. v1 stays as the published zip;
the v2 archive is rebuilt on the Linux box (see v2-linux-todo.md).

## Migration notes

- Existing v1 scripts (`run_overnight.py`, `run_mvp_audit.py`, etc.) are
  untouched and continue to read the v1 graph + parquets.
- v2 modules are pure CPU, polars + sklearn + (optional) networkx + pulp.
  No GPU dependencies were added.
- To run the v2 test suite: `pytest tests/v2/ -q`. All 17 tests pass on
  Windows without RDKit installed (ligand-only baseline falls back to a
  hash featuriser).

## What's tested

- `tests/v2/test_scoring.py` — multiplicative + -log Dijkstra equivalence,
  axis decomposition isolation, max-hops bound, contamination-NN argmax
  source recording, unreached-query zero score
- `tests/v2/test_label_leakage.py` — same/conflict label detection,
  three-direction report
- `tests/v2/test_hubs.py` — hub counts, unreached exclusion, top-K
  truncation, concentration metrics
- `tests/v2/test_leakage_groups.py` — simple two-group case,
  isolated-example singletons, giant-component pruning trigger
