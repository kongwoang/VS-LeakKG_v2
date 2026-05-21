# Linux agent prompt — VS-LeakKG v2 end-to-end run

Hand this entire file to the Linux agent. It is self-contained: the agent
does not need any context from the Windows side.

---

## Mission

You are executing **Phase 1** of the VS-LeakKG v2 leakage-audit pipeline
on a Linux GPU box. The Windows team has already built the v2
algorithmic library (scoring, leakage groups, splits, baselines) and
shipped it as a separate repo. Your job in Phase 1 is the **data-only**
work: rebuild the typed contamination graph with the v2 schema,
generate clean splits, compute validation-contamination matrices, run
the data-only shortcut baselines, and produce the tables/figures that
fall out of those.

**Phase 1 explicitly does NOT touch any specific model — no ConGLUDe,
no HypSeek, no DrugCLIP, no S2Drug, no LigUnity.** Anything that
requires a model's published training manifest (Mode B audit) or any
GPU retraining (ConGLUDe retraining) is **deferred to Phase 2** and
must not be attempted in this pass. The "Deferred to Phase 2" section
below is the list. Do not run those steps even if you have time.

The framework you are implementing separates two questions that v1
conflated:

- **Mode A**: how should we split a corpus into train/val/test so the
  three partitions are not connected by shared ligand / scaffold /
  protein / pocket / assay / source / time relations? — **Phase 1**
- **Mode B**: given a *specific* trained model `m`, how contaminated are
  its reported benchmark numbers by overlap between its actual training
  rows and the benchmark? — **Phase 2 (do not run yet)**

The formal write-up lives at `proposal.tex` in the v2 repo. The
algorithmic spec for each module is in its docstring. The full work
queue (Phase 1 + Phase 2) is `docs/linux-todo.md`. Read both before
starting; ignore the `[D2]`, `[D3]`, `[D7]` items for now.

## What is already on the box

- **v1 repo with full data**: the v1 codebase
  (<https://github.com/kongwoang/VS-LeakKG>) is already cloned on this
  machine and the raw dataset archive has already been fetched from
  Hugging Face and unpacked. Ask the user for the path (commonly
  `~/VS-LeakKG`) and export it:

  ```bash
  export VSLEAKKG_V1_ROOT=/path/to/VS-LeakKG     # adjust to actual path
  ```
  
  Verify both the code and the data are there:

  ```bash
  ls "$VSLEAKKG_V1_ROOT/src/vsleakkg/build_graph.py"
  ls "$VSLEAKKG_V1_ROOT/data/raw"            # raw datasets unpacked
  ls "$VSLEAKKG_V1_ROOT/data/processed"      # v1 processed parquets
  ```

  If any of those are missing, stop and ask the user before fetching
  anything yourself — do not re-download the archive.

- **GPU + CUDA toolchain**: assume present. Confirm with `nvidia-smi`
  and the box's `python -c "import torch; print(torch.cuda.is_available())"`.

## What you need to do first

```bash
# 1. Clone the v2 repo (only v2 needs cloning — v1 is already here)
git clone https://github.com/kongwoang/VS-LeakKG_v2.git ~/VS-LeakKG_v2
cd ~/VS-LeakKG_v2

# 2. Set up the v2 environment
use the conda env vsleak and install any thing needed

# 3. Point at the existing v1 checkout
export VSLEAKKG_V1_ROOT=/path/to/VS-LeakKG      # the one already on disk
python -c "from vsleakkg.v2.datapaths import data_root, processed_dir; print(data_root()); print(processed_dir())"

# 4. Verify the test suite passes
pytest tests/v2 -q                               # expect: 17 passed
```

If any step fails, stop and report before continuing.

## Linux compatibility — you ARE allowed to modify files

The v2 repo was authored on Windows. Some files may need small edits to
run cleanly on Linux. You are explicitly permitted to:

- Convert CRLF → LF line endings if a shell script complains. `dos2unix`
  or `sed -i 's/\r$//'` is fine.
- Fix the Windows default in `src/vsleakkg/v2/datapaths.py` if the
  default path doesn't match where v1 actually lives on this box. The
  cleanest fix is to **always** export `VSLEAKKG_V1_ROOT` so no default
  is used; only patch the file if it's confusing.
- Add a Linux equivalent of any Windows-only helper (e.g., port a
  `.ps1` script to `.sh` under `scripts/`). Keep both versions; don't
  delete the `.ps1`.
- Fix any hardcoded `D:/hoangpc/...` paths you find — these should not
  exist in the v2 repo, but if you spot one, replace it with a path
  derived from `vsleakkg.v2.datapaths` or an env var.
- Add `requirements.txt` snapshots if `pip install -e .` fails to
  resolve a dependency cleanly. Document what you pinned and why.

You are **not** permitted to change the algorithmic behaviour of the v2
modules (`schema.py`, `scoring.py`, `hubs.py`, `label_leakage.py`,
`leakage_groups.py`, `split.py`, `trainset.py`,
`validation_contamination.py`, `baselines/*`). If you find an
*algorithmic* bug in any of these, write a failing test first,
document the bug in `docs/issues.md`, and leave the fix for the Windows
team. Linux-compat edits (paths, imports, shebangs) are fine; logic
edits are not.

When you commit your Linux-compat fixes locally, prefix the commit
message with `linux-compat:` so the Windows team can identify them on
review.

## Other constraints

- **Do not push to GitHub.** The credentials on this box do not have
  push permission to either repo. Commit locally if you like (the
  history is your audit trail), but never run `git push`.
- **Treat the v1 repo as read-only.** Use its raw loaders and data,
  but do not commit anything back to it. If you need to write
  per-model artefacts (e.g., the parsed ConGLUDe train manifest), put
  them under `~/VS-LeakKG_v2/outputs/` not under v1.
- **Honest reporting only.** If a regime is infeasible, write
  `infeasible` in the output table — do not silently relax the
  constraints. If a model's training corpus is unauditable, fall back
  to proxy scoring and clearly flag the limitation in the report.

## The Phase 1 pipeline (run in order)

The detailed acceptance criteria for each step are in `docs/linux-todo.md`.
In Phase 1 you run **`[D1]`, `[D4]`, `[D5]`, the data-only subset of `[D6]`,
and a partial `[D8]`** — in that order. Items `[D2]`, `[D3]`, `[D7]`, and
the model-dependent parts of `[D6]` are Phase 2 and listed at the end of
this section under "Deferred to Phase 2".

### [D1] Rebuild graph with v2 schema (~half day)

Write `src/vsleakkg/v2/build_graph.py`. It should import the existing
v1 raw loaders from `$VSLEAKKG_V1_ROOT/src/vsleakkg/` —
`load_chembl_db`, `load_bindingdb`, `load_pdbbind`, `load_dude`,
`load_litpcba`, `load_bayesbind` — and emit v2-schema parquets:

- `outputs/v2/graph/v2_nodes.parquet`
- `outputs/v2/graph/v2_edges.parquet`

The easiest way to import v1 loaders without modifying v1 is to prepend
`$VSLEAKKG_V1_ROOT/src` to `sys.path` at the top of
`build_graph.py`:

```python
import sys
from vsleakkg.v2.datapaths import data_root
sys.path.insert(0, str(data_root() / "src"))
from vsleakkg.load_chembl_db import load_chembl_db   # v1 loader
```

Apply `HubMitigationConfig` from `vsleakkg.v2.schema` during build:

- drop scaffolds with ≤ 6 heavy atoms and no substituents
- shard nodes whose degree exceeds the cap into per-source pieces
- apply IDF weight to scaffold and assay edges (floor at the default
  weight in proposal Table 2)

Re-run MMseqs2 `easy-cluster` at 30 / 40 / 90% identity and emit a
`ProteinCluster_q` node per resolution (keep all three; do not merge).

Compute pocket embeddings (ESM-IF1 or your chosen encoder) and emit
pocket-similarity edges with cosine ≥ 0.80.

Acceptance: `outputs/v2/graph/stats.csv` matching the KG-statistics
template in `proposal.tex` Section 5.1.

### [D2], [D3] — DEFERRED TO PHASE 2

Skip. Do not pull any model's training manifest, do not build
`TrainSet_m` nodes, do not run Mode B audits. These touch specific
models (ConGLUDe, HypSeek, DrugCLIP, S2Drug, LigUnity) and are a
separate phase.

### [D4] Generate clean splits

For each `(corpus, regime)` pair, run `build_leakage_groups(...)`
followed by `greedy_assign(...)`. If greedy violates a constraint,
fall back to the PuLP MILP path.

Corpora: LIT-PCBA AVE, DUD-E, DEKOIS-2, BayesBind V1.5 (held-out
reference).

Regimes: `ligand-clean`, `scaffold-clean`, `protein-clean`,
`pocket-clean`, `assay-clean`, `dual-clean`, `strict-clean` (all seven).

Output: `outputs/v2/splits/<corpus>/<regime>.parquet` with columns
`(example_id, partition)`, plus `residual_contamination.csv`. Regimes
that the cascade flags as infeasible go in the CSV as `infeasible` —
do not silently relax.

### [D5] Validation-contamination matrices

For each `(corpus, regime)`, call
`vsleakkg.v2.validation_contamination.three_way_contamination(...)` and
emit `outputs/v2/validation_contamination/<corpus>/<regime>.csv`.

### [D6] Shortcut baselines — data-only subset

Run **only** the baselines that do not require a specific evaluated
model. On each v2 split:

- `vsleakkg.v2.baselines.ligand_only.evaluate_ligand_only(...)` (RDKit
  required for realistic fingerprints; otherwise the hash fallback is
  fine but flag it). **No model dependency.**
- `vsleakkg.v2.scoring.contamination_nn_label(...)`. **No model
  dependency.**
- v1's `source_only_diagnostics.py` (reuse as-is from
  `$VSLEAKKG_V1_ROOT/src/vsleakkg/`). **No model dependency.**

**Skip in Phase 1:** the dummy-receptor baseline if it requires
ConGLUDe's (or any other audited model's) protein encoder. If a
generic protein-LM encoder (e.g., ESM2 from HuggingFace) is already
loaded on this box, you may run dummy-receptor with that — it does not
count as "touching a model in the audit set". Flag this clearly in the
output row.

Output: one row per `(corpus, regime, baseline)` into
`outputs/v2/baselines.csv`. Add a column `requires_model` (true/false)
and leave Phase 2's model-paired rows absent.

### [D7] ConGLUDe retraining — DEFERRED TO PHASE 2

Skip. No model retraining in this pass.

### [D8] Final tables and figures — partial in Phase 1

Write `src/vsleakkg/v2/final_figures.py` (mirror v1's
`final_figures.py` structure). Produce **only** the tables and figures
that are derivable from data alone:

- **Table 1** KG statistics + benchmark coverage (template in
  `proposal.tex`) — Phase 1
- **Table 2** leakage groups & residual contamination per regime —
  Phase 1
- **Table 5** validation-contamination effect (data side only —
  matrices from `[D5]`; the leaky-vs-clean-val comparison needs Phase 2
  retraining, so emit just the matrix summary for now) — partial
- **Figure 2** leakage-hub Pareto curve (data-side only — uses
  `vsleakkg.v2.hubs` against the v2 graph) — Phase 1

**Skip in Phase 1:** Table 3 (ConGLUDe metrics — needs Phase 2),
Table 4 (shortcut baselines vs full model — full-model number is
Phase 2), Figure 1 (performance by contamination decile per model —
needs Phase 2), Figure 3 (split-stability under ChEMBL 35 → 36 —
defer; can be added in either phase).

Save tables as CSV under `outputs/v2/tables/` and figures as PDF + PNG
under `outputs/v2/figures/`.

## Deferred to Phase 2 (do not run yet)

Listed here for completeness so you don't accidentally start them:

- `[D2]` Per-model TrainSet manifest ingestion (ConGLUDe, HypSeek,
  DrugCLIP, S2Drug, LigUnity)
- `[D3]` Mode B audits per model + the ConGLUDe AUROC-drop sanity check
- `[D6]` model-paired baselines (dummy-receptor with each audited
  model's protein encoder)
- `[D7]` ConGLUDe retraining across regimes × seeds
- `[D8]` Table 3, Table 4, Figure 1 (all need Phase 2 outputs)

When the user is ready for Phase 2 they will send a follow-up prompt.

## Final deliverable (Phase 1)

Write `LINUX_RUN_REPORT.md` at the v2 repo root. It must contain:

1. A **Phase 1 status table**: one row per executed step (`[D1]`,
   `[D4]`, `[D5]`, data-only `[D6]`, partial `[D8]`), columns
   `(status, wall-clock, output path, notes)`. For the deferred steps
   (`[D2]`, `[D3]`, `[D7]`, model-paired `[D6]`, Phase-2 parts of
   `[D8]`) add a separate "Deferred to Phase 2" sub-table that just
   confirms they were intentionally skipped.
2. **Phase 1 headline numbers**:
   - KG statistics summary from `[D1]` (node counts per type, edge
     counts per type, after vs before hub mitigation).
   - For each `(corpus, regime)` in `[D4]`: feasibility status, group
     count, partition sizes, residual contamination at each axis.
   - For each `(corpus, regime)` in `[D5]`: the three-way matrix
     summary `(train→test, train→val, val→test)`.
   - For each `(corpus, regime, baseline)` in `[D6]`: AUROC / AUPRC.
3. Any regimes flagged as infeasible in `[D4]` and which constraint
   forced infeasibility.
4. A list of every Linux-compat edit you made, with file path, one-line
   reason, and the commit hash if you committed locally.
5. Any deviations from the spec (e.g., you used a different protein
   encoder than the proposal mentions because the box doesn't have GPU
   memory for ESM-IF1) — with justification.
6. A "Phase 2 readiness" section listing what's now unblocked for the
   next pass (e.g., "v2 graph is published at `outputs/v2/graph/`; Mode
   B can begin once ConGLUDe's train manifest is parsed").

Do **not** commit `LINUX_RUN_REPORT.md` to the v2 repo. Leave it as an
untracked file and tell the user where it is so they can review and
copy it back.

## Quick-wins ordering (if you only have a day)

If full compute isn't available, run in this order:

1. `[D1]` graph rebuild — unblocks everything else in Phase 1
2. `[D4]` clean splits — at minimum LIT-PCBA across all 7 regimes
3. `[D5]` validation-contamination matrices for the regimes from step 2
4. `[D6]` contamination-NN + ligand-only on the LIT-PCBA splits

That's enough to demonstrate the data-side methodology end-to-end. The
rest of `[D4]`/`[D5]`/`[D6]` (other corpora) can run unattended after.

## Reference list (do not invent URLs)

- v1 repo: <https://github.com/kongwoang/VS-LeakKG>
- v2 repo: <https://github.com/kongwoang/VS-LeakKG_v2>
- ConGLUDe: <https://github.com/ml-jku/conglude>

If any other URL is needed, ask the user for it rather than guessing.

## When you are done

Output a single paragraph to chat with:

- one-sentence status of `[D1]`, `[D4]`, `[D5]`, data-only `[D6]`,
  partial `[D8]`
- confirmation that `[D2]`, `[D3]`, `[D7]` and the model-paired parts
  of `[D6]`/`[D8]` were intentionally skipped per the Phase 1 / Phase 2
  split
- count of Linux-compat edits made
- where `LINUX_RUN_REPORT.md` lives on disk

Then stop. Do not push. Wait for the user to send the Phase 2 prompt
before starting any model-specific work.
