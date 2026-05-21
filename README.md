# VS-LeakKG v2

Typed contamination-graph framework for structure-based virtual-screening
benchmark integrity. v2 separates **Mode A** (clean-split construction) from
**Mode B** (model-specific leakage audit), adopts a multiplicative path-product
contamination score solved via `-log` Dijkstra, and adds hub-pollution
mitigation, validation-contamination matrices, leakage-hub diagnostics, and a
giant-component fallback cascade.

For the formal write-up see [`proposal.tex`](proposal.tex).
For the per-change rationale vs v1 see [`CHANGELOG.md`](CHANGELOG.md).
For the Linux/GPU work queue see [`docs/linux-todo.md`](docs/linux-todo.md).

## Relationship to v1

This repo contains **only the v2 algorithmic library and tests**. It does not
ship its own dataset — it reads raw inputs and processed parquets from the v1
repository on disk:

- v1 repo: <https://github.com/kongwoang/VS-LeakKG>
- v1 local checkout (default on Windows): `D:/hoangpc/VS-LeakKG`
- override path via env var: `VSLEAKKG_V1_ROOT=/path/to/VS-LeakKG`

The path resolver lives in `vsleakkg.v2.datapaths`. v2 leaves v1 untouched;
v1 scripts continue to run independently against the v1 graph.

## What's in this repo

| Module | Responsibility |
|--------|----------------|
| `vsleakkg.v2.schema` | Node/edge types, default weights, axis map, hub-mitigation + giant-component + split configs |
| `vsleakkg.v2.scoring` | Multiplicative path-product score; `-log` Dijkstra; per-axis subgraph decomposition; contamination-NN baseline |
| `vsleakkg.v2.label_leakage` | Exact-row overlap (same- vs conflicting-label) across partitions |
| `vsleakkg.v2.hubs` | Leakage-hub diagnostic `H(x_train)` + concentration metrics |
| `vsleakkg.v2.trainset` | Per-model TrainSet manifest ingestion (Mode B node/edge construction) |
| `vsleakkg.v2.validation_contamination` | Three-way matrices `C(train→test)`, `C(train→val)`, `C(val→test)`, val-leakage effect |
| `vsleakkg.v2.leakage_groups` | Connected → prune-weakest → Louvain → infeasible cascade |
| `vsleakkg.v2.split` | Group-atomic split assignment (greedy + PuLP MILP fallback) |
| `vsleakkg.v2.baselines.ligand_only` | Morgan-RF shortcut baseline |
| `vsleakkg.v2.baselines.dummy_receptor` | Replaces protein embedding with the training-set mean |
| `vsleakkg.v2.datapaths` | Resolves the v1 data root (env-var driven) |

## Status

Windows-side: the algorithmic library and 17 unit tests are complete and
pass (~0.4 s). See [`CHANGELOG.md`](CHANGELOG.md).

Linux-side (deferred): graph rebuild with the v2 schema, per-model TrainSet
manifests, Mode B audits, clean splits for 4 corpora × 7 regimes, three-way
validation-contamination matrices, shortcut baselines on v2 splits, ConGLUDe
retraining. See [`docs/linux-todo.md`](docs/linux-todo.md) for the ordered
work queue (`[D1]`–`[D8]`).

## Quick start

```bash
git clone https://github.com/kongwoang/VS-LeakKG_v2.git
cd VS-LeakKG_v2
pip install -e .[graph,milp,dev]   # or just `.[dev]` if you don't need MILP/Louvain
pytest tests/v2 -q
```

You should see `17 passed`.

To run anything that actually consumes the graph, you need the v1 data on disk:

```bash
git clone https://github.com/kongwoang/VS-LeakKG ~/VS-LeakKG
export VSLEAKKG_V1_ROOT=~/VS-LeakKG
# follow the v1 README to fetch the dataset archive from Hugging Face
```

On Windows, the default `VSLEAKKG_V1_ROOT` is `D:/hoangpc/VS-LeakKG`, so no
env var is needed if you check out v1 there.

See [`docs/how-to-run.md`](docs/how-to-run.md) for the full run sequence.

## Tests

```bash
pytest tests/v2 -q
```

The suite covers:
- multiplicative + `-log` Dijkstra equivalence and axis decomposition isolation
- max-hops bound and unreached-query zero score
- contamination-NN argmax-source recording
- label leakage (same-label vs conflicting-label, three-direction report)
- leakage-hub counts and concentration metrics
- leakage-group construction (simple two-group, isolated singletons,
  giant-component pruning trigger)

All tests run pure CPU on Windows and Linux. RDKit is optional; the ligand-only
baseline falls back to a hash featuriser when RDKit is unavailable.

## License

MIT.
