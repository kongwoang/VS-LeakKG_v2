# How to run VS-LeakKG v2

This guide covers the **Windows-runnable** parts (library + tests + smoke
checks). The full Linux pipeline (graph rebuild, Mode B audits, ConGLUDe
retraining) is tracked separately in [`linux-todo.md`](linux-todo.md).

## 1. Install

```bash
git clone https://github.com/kongwoang/VS-LeakKG_v2.git
cd VS-LeakKG_v2
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e .[dev]            # minimal: scoring, hubs, label leakage, splits
pip install -e .[graph,milp,dev] # add networkx/Louvain + PuLP MILP fallback
pip install -e .[chem]           # add RDKit for the realistic ligand-only baseline
```

Python ≥ 3.9 is required.

## 2. Point at the v1 data (only needed when running on real data)

The v2 modules don't need data to run unit tests, but anything that consumes
the actual SBVS corpus expects to find the v1 repo on disk.

```bash
# Linux / macOS
export VSLEAKKG_V1_ROOT=$HOME/VS-LeakKG

# Windows (PowerShell)
$env:VSLEAKKG_V1_ROOT = "D:/hoangpc/VS-LeakKG"
```

Defaults if the env var is unset:
- Windows → `D:/hoangpc/VS-LeakKG`
- Linux/macOS → `~/VS-LeakKG`

You can verify the resolver:

```bash
python -c "from vsleakkg.v2.datapaths import data_root, processed_dir; print(data_root()); print(processed_dir())"
```

## 3. Run the test suite

```bash
pytest tests/v2 -q
```

Expected: `17 passed`. The suite runs in well under a second and needs no
external data.

## 4. Programmatic use of the library

### Contamination score (Mode A)

```python
from vsleakkg.v2.schema import DEFAULT_WEIGHTS
from vsleakkg.v2.scoring import score_overall

edges = [
    # (src_node, dst_node, edge_type, source_id)
    ("ex_train_1", "lig_A", "example_uses_ligand", "chembl"),
    ("ex_test_1",  "lig_A", "example_uses_ligand", "chembl"),
]
scores = score_overall(
    edges=edges,
    reference={"ex_train_1"},
    queries=["ex_test_1"],
    weights=DEFAULT_WEIGHTS,
    max_hops=3,
)
print(scores)   # {"ex_test_1": 0.95}  (exact-ligand weight from Table 2)
```

### Mode B (model-specific audit)

```python
from vsleakkg.v2.trainset import ingest_model_trainset, merge_into_graph

trainset = ingest_model_trainset(
    manifest_path="conglude_train_manifest.csv",
    model_id="conglude",
    id_map={...},   # (source, identifier) -> example_id
)
nodes, edges = merge_into_graph(existing_nodes, existing_edges, trainset)

# Then score benchmark examples against the model's TrainSet_m
from vsleakkg.v2.scoring import score_overall
contam = score_overall(edges, reference={"trainset:conglude"}, queries=bench_ids)
```

### Clean splits

```python
from vsleakkg.v2.leakage_groups import build_leakage_groups
from vsleakkg.v2.split import greedy_assign

groups = build_leakage_groups(
    example_ids=all_examples,
    edges=edges,
    forbidden_relations={"example_uses_ligand", "example_uses_protein_cluster"},
)
assignment = greedy_assign(groups, examples=metadata)
```

### Validation-contamination matrices

```python
from vsleakkg.v2.validation_contamination import three_way_contamination

m = three_way_contamination(
    edges=edges,
    train_ids=train_set,
    val_ids=val_set,
    test_ids=test_set,
)
print(m["train_to_test"].summary())
print(m["train_to_val"].summary())
print(m["val_to_test"].summary())
```

## 5. What the Linux side will add

These entry points don't exist yet; they will be written on the Linux box
because they need the 40 M-edge graph and a GPU. See `docs/linux-todo.md`:

- `vsleakkg.v2.build_graph` — v2-schema graph builder ([D1])
- per-model manifest ingestion scripts ([D2])
- Mode B audit runner ([D3])
- clean-split generators for LIT-PCBA / DUD-E / DEKOIS-2 / BayesBind ([D4])
- shortcut-baseline orchestration on v2 splits ([D6])
- ConGLUDe retraining scaffolding ([D7])

Until they land, the Windows side is exclusively the algorithmic library
and its unit tests.
