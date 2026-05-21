# Linux agent prompt — VS-LeakKG v2 Phase 2 (Mode B + retraining)

Hand this entire file to the Linux agent. It is self-contained: the
agent does not need any context from the Windows side beyond what is
written here and what lives in the v2 repo's `docs/`.

---

## Mission

You are executing **Phase 2** of the VS-LeakKG v2 leakage-audit
pipeline. Phase 1 (data only — graph rebuild, clean splits,
validation-contamination matrices, data-only baselines) is assumed
**complete** before you start. Phase 2 is the model-specific work:
build the hydrate side-table, train three SBVS models under both
their published splits and our v2 clean splits, run inference on the
released DrugCLIP checkpoint, and emit the final comparison tables.

The three audited models are **SPRINT** (lightweight, smoke-test
target), **LigUnity** (production audit run, most auditable training
data), and **DrugCLIP** (inference-only by default; full retrain is a
stretch goal because each run costs ~350 GPU·h).

The detailed plan for what to do and in what order is in
[`docs/phase2/training-plan.md`](phase2/training-plan.md). The
per-repo investigations with `file:line` references for every patch
are in `docs/phase2/{sprint,ligunity,drugclip}-investigation.md`.
**Read all four before starting.**

## What is already on the box

- **v1 repo with full data** at `$VSLEAKKG_V1_ROOT` (set this env var
  as in Phase 1).
- **Phase 1 outputs** under `~/VS-LeakKG_v2/outputs/v2/` — graph,
  splits, validation-contamination matrices, data-only baselines.
  These were produced by the Phase 1 agent. If they're missing, stop
  and tell the user: Phase 2 cannot proceed without them.
- **The three model repos** cloned shallow under
  `~/_audit_targets/{SPRINT,LigUnity,DrugCLIP}` (cloned by the Windows
  investigation pass). You may re-clone if needed.

## What you need to do first

```bash
cd ~/VS-LeakKG_v2
git pull                                          # latest plan + hydrate module
source .venv/bin/activate                         # or: conda activate vsleak
export VSLEAKKG_V1_ROOT=/path/to/VS-LeakKG        # same as Phase 1
pytest tests/v2 -q                                # expect: 32 passed

ls outputs/v2/graph/v2_edges.parquet              # Phase 1 must be done
ls outputs/v2/splits/lit_pcba/                    # at least one regime should exist
```

If the test count is below 32 or the Phase 1 outputs are missing,
stop. Report and wait.

## Constraints

Same as Phase 1, plus:

- **Do not push to GitHub.** Local commits are fine; prefix any
  Linux-compat commit with `linux-compat:`.
- **Algorithmic behaviour of v2 modules is frozen.** You may add new
  modules (e.g., `vsleakkg.v2.build_side_table`, per-model adapters
  under `vsleakkg.v2.adapters.{sprint,drugclip,ligunity}`) but do not
  change `hydrate.py`, `scoring.py`, `schema.py`, etc.
- **No commercial use** of LigUnity or DrugCLIP outputs (CC BY-NC
  data/weights). Academic audit is fine.
- **Honest reporting.** If a model's training corpus has gaps, report
  the coverage number and proceed with what is recoverable. Don't
  silently drop rows.

## The Phase 2 pipeline (run in order)

### [P2.1] Build the hydrate side-table

Status: blocks everything below. Budget: ~half day.

The side-table maps every v2 graph example_id to the rich-row fields
that model adapters need (SMILES, target sequence, UniProt, labels,
source IDs). The contract is in
`vsleakkg.v2.hydrate.SIDE_TABLE_SCHEMA` and
`SIDE_TABLE_COLUMNS`. Read `src/vsleakkg/v2/hydrate.py` start to
finish before you write a single line.

Add a new module `src/vsleakkg/v2/build_side_table.py` that:

1. Reads the v2 nodes parquet from Phase 1
   (`outputs/v2/graph/v2_nodes.parquet`).
2. For each `Example` node, joins back to v1's raw loaders
   (`load_chembl_db`, `load_bindingdb`, `load_pdbbind`, etc.) to
   recover SMILES, UniProt, labels, source IDs. Use the same
   `sys.path.insert(0, str(data_root() / "src"))` trick the Phase 1
   `build_graph.py` already uses.
3. Canonicalises SMILES via `vsleakkg.v2.hydrate.canonicalize_smiles`.
4. Validates the result via `vsleakkg.v2.hydrate.validate_side_table`.
5. Writes `outputs/v2/graph/side_table.parquet`.

Then add a one-shot **foldseek pass** that fills the
`target_sequence_saprot` column for every UniProt in the side-table
(SPRINT requires this for any SaProt run). Use the existing
`utils/structure_to_saprot.py` in the SPRINT repo as reference; the
output is a per-UniProt parquet that you join in.

**Smoke test before moving on:**

```python
from vsleakkg.v2.hydrate import Hydrator
h = Hydrator.from_parquet("outputs/v2/graph/side_table.parquet")
print(len(h))                                    # should be ~ |Example nodes|
sample = h.hydrate(["chembl:ACT_1", "pdbbind:1abc"])  # adjust IDs to real ones
print(sample.coverage)                           # expect 1.0 for real IDs
```

Acceptance: `Hydrator.from_parquet(...)` returns coverage >= 0.95
against a random sample of 1000 v2 example_ids, and the per-source
counts roughly match the Phase 1 graph statistics.

### [P2.2] SPRINT smoke test (DAVIS, paper split)

Status: blocked by [P2.1]. Budget: ~1 GPU·h + setup.

Goal: prove the whole pipeline (hydrate → CSV → SPRINT `--task
custom` → numbers) end-to-end on the smallest dataset.

1. `pip install -e ~/_audit_targets/SPRINT` (will install
   `lightning==2.4.0`, `torch==2.4.1`, `fair-esm`, etc.).
2. SPRINT-side prep: `data/MERGED/huge_data/download.sh` if you also
   want MERGED for [P2.6]; otherwise skip.
3. Materialise SPRINT-shaped CSVs from the v2 hydrate side-table.
   Write a one-shot script `tools/sprint_csv_from_v2.py` in the v2
   repo that:
   - reads a v2 split parquet for `(corpus="davis", regime="paper")`
   - hydrates each row
   - emits `~/_audit_targets/SPRINT/data/custom/{train,val,test}_foldseek.csv`
     with columns `,SMILES,Target Sequence,Label`
4. Run:
   ```bash
   cd ~/_audit_targets/SPRINT
   ultrafast-train --exp-id smoketest --task custom \
       --config configs/saprot_agg_config.yaml --no-wandb
   ```
5. Compare the resulting `val/aupr` against the SPRINT paper's DAVIS
   number. If it's within noise (~±0.02), the pipeline is sound.

If [P2.2] fails or the number drifts, do NOT continue to [P2.3].
Debug. Most likely cause: SaProt 3Di tokens don't match the upstream
UniProts (cross-check by joining v2 hydrate rows against shipped
DAVIS `*_foldseek.csv` on `uniprot_id`).

### [P2.3] SPRINT production runs

Status: blocked by [P2.2]. Budget: ~100 GPU·h.

For each `(corpus, regime, seed)`:

- corpora: `{DAVIS, BIOSNAP, BindingDB}`
- regimes: `{paper, ligand-clean, scaffold-clean, protein-clean, strict-clean}`
- seeds: 3 (use `--r 0`, `--r 1`, `--r 2` if SPRINT supports it;
  otherwise change `--seed` and re-run)

Save metrics to
`outputs/v2/phase2/sprint/<corpus>/<regime>/seed<k>/metrics.json`.

Acceptance: every (corpus, regime, seed) cell has a `metrics.json`
with at minimum `val_aupr`, `test_aupr`, `test_auroc`.

### [P2.4] LigUnity dependency pinning

Status: blocked by [P2.1]; runs in parallel with [P2.3]. Budget: ~1 day.

LigUnity's README tells you to hand-patch `unicore/options.py:250`
and `unicore_cli/train.py:303` (adding `--validate-begin-epoch`). The
exact `unicore` commit that makes these line numbers valid is **not
recorded anywhere in the repo**.

1. Read `~/_audit_targets/LigUnity/README.md` lines 137-166 to confirm
   the patch contract.
2. Clone `https://github.com/dptech-corp/Uni-Core` and bisect commits
   between 2024-01-01 and the LigUnity repo's commit date (~2025) to
   find a commit where the patch applies cleanly. Test with
   `git apply --check`.
3. Record the resulting commit hash in
   `docs/phase2/dependency-pins.md`:
   ```
   ## unicore
   - commit: <hash>
   - install: pip install git+https://github.com/dptech-corp/Uni-Core@<hash>
   - patch_files: unicore/options.py, unicore_cli/train.py
   ```
4. Also patch `unimol/models/protein_ranking.py:60` to read from
   `facebook/esm2_t12_35M_UR50D` (HuggingFace) instead of the
   hardcoded `/cto_studio/xtalpi_lab/...` path. Commit as
   `linux-compat:`.

### [P2.5] LigUnity retrain

Status: blocked by [P2.4] and [P2.1]. Budget: ~150 GPU·h on 2× A100.

1. Apply the ~80 LOC patch described in
   `docs/phase2/ligunity-investigation.md` "Adaptation plan":
   replace the JSON-driven scrub at `unimol/tasks/train_task.py:540-605`
   and `HGNN/screen_dataset.py:143-193` with a parquet-driven filter
   keyed on `(assay_id, smi)`. Add `--split-parquet` to argparse.
2. Materialise per-partition LigUnity manifests from the v2 hydrate
   side-table: one parquet per `(corpus, regime, partition)`.
3. Run two configurations on the CASF validation set:
   - paper split (use the upstream JSON manifests; this is the
     reproducibility anchor)
   - v2 `strict-clean` split (use the parquet filter)
4. Train the HGNN re-ranker on each checkpoint
   (`HGNN/main.py`, ~2 GPU·h each).
5. Run `bash test.sh ALL pocket_ranking <ckpt> ./result/...` followed
   by `python ensemble_result.py DUDE PCBA DEKOIS`.

Save metrics to
`outputs/v2/phase2/ligunity/<regime>/metrics.json`.

### [P2.6] DrugCLIP inference + (optional) retrain

Status: blocked by [P2.1]. Budget: ~10 GPU·h inference, ~350 GPU·h if
retraining.

**Inference path (default):**

1. Resolve the missing LIT-PCBA eval LMDBs blocker. Two options:
   - find the author's release (check the paper's supplementary, the
     DrugCLIP GDrive folder, the bowen-gao group's other repos)
   - rebuild from raw LIT-PCBA: write
     `py_scripts/build_lit_pcba_lmdbs.py` mirroring
     `py_scripts/write_dude_multi.py` structure.
2. Run `bash test.sh` for DUD-E and LIT-PCBA on the released
   `checkpoint_best.pt`. Save metrics.
3. Build a per-clean-regime DrugCLIP eval LMDB from the v2 split
   parquet. The build is described in
   `docs/phase2/drugclip-investigation.md` "Adaptation plan" step 1-2.
4. Re-run `bash test.sh` against each clean LMDB. Save metrics.

**Retrain path (stretch goal, only if budget permits):**

1. Apply the ~120 LOC patch from
   `docs/phase2/drugclip-investigation.md`: add
   `unimol/data/subset_dataset.py` and 5-7 lines at
   `unimol/tasks/drugclip.py:184`.
2. Write `tools/build_drugclip_v2_split.py` to scan the upstream
   `train.lmdb` and emit `<split>_keys.txt` files.
3. Run `bash drugclip.sh` on **strict-clean only**, single seed.
4. Save metrics.

### [P2.7] Final Phase 2 tables and figures

Status: blocked by [P2.3], [P2.5], [P2.6]. Budget: ~half day.

Write `src/vsleakkg/v2/phase2_figures.py`. Produce:

- **Table 3** Audited-model metrics: paper split vs clean splits.
  Columns: `(model, corpus, regime, seed, auroc, aupr, ef1pct)`.
- **Table 4** Shortcut baselines vs full model per regime (joins
  Phase 1's `outputs/v2/baselines.csv` with `outputs/v2/phase2/.../metrics.json`).
- **Figure 1** Performance by contamination decile per model. Uses
  Phase 1's contamination scores from `outputs/v2/mode_b/` if Mode B
  ran, otherwise uses `outputs/v2/graph/` contamination directly.

Save tables to `outputs/v2/tables/` and figures to
`outputs/v2/figures/`.

## Final deliverable (Phase 2)

Write `LINUX_RUN_REPORT_PHASE2.md` at the v2 repo root (do not
commit). Must contain:

1. **Phase 2 status table**: one row per executed step (`[P2.1]`–
   `[P2.7]`), columns `(status, wall-clock, output path, notes)`.
2. **Headline numbers per model**:
   - **SPRINT**: for each corpus, the AUROC delta `paper → strict-clean`
     (averaged over 3 seeds, with stddev).
   - **LigUnity**: AUROC + EF1% on LIT-PCBA / DUD-E / DEKOIS, paper
     split vs `strict-clean`.
   - **DrugCLIP**: AUROC + EF1% on LIT-PCBA / DUD-E, inference-only,
     under each clean regime.
3. **Side-table coverage** from [P2.1]: what fraction of v2 example_ids
   hydrated successfully; per-source breakdown.
4. **Pinned `unicore` commit** from [P2.4].
5. **DrugCLIP LIT-PCBA LMDB resolution**: where you got it / how you
   rebuilt it.
6. **Linux-compat edits** made during Phase 2 (path, reason, commit
   hash).
7. **Deviations from the plan** with justification.

Do **not** commit `LINUX_RUN_REPORT_PHASE2.md`. Leave it untracked.

## Quick-wins ordering (if you only have ~1 GPU·day)

1. `[P2.1]` side-table — unblocks everything.
2. `[P2.2]` SPRINT DAVIS smoke test — the load-bearing experiment.
3. `[P2.6]` DrugCLIP inference-only on ONE regime (ligand-clean)
   against ONE benchmark (LIT-PCBA).

That's enough to write the methods paper. Skip [P2.3] / [P2.4] /
[P2.5] / [P2.7] if compute is tight.

## When you are done

Output a single paragraph to chat with:

- one-sentence status of each of `[P2.1]`–`[P2.7]`
- the SPRINT DAVIS paper-vs-strict-clean AUROC delta
- whether the LigUnity unicore pin succeeded
- whether DrugCLIP LIT-PCBA LMDBs were resolved
- count of Linux-compat edits made
- where `LINUX_RUN_REPORT_PHASE2.md` lives on disk

Then stop. Do not push.
