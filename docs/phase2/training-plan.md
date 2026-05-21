# VS-LeakKG v2 Phase 2 — cross-cutting training plan

This plan synthesises the three per-repo investigations
(`ligunity-investigation.md`, `drugclip-investigation.md`,
`sprint-investigation.md`) into one ordered roadmap. Each repo has its
own quirks but they share a common shape: take an `(example_id,
partition)` parquet from v2 and (a) train the model under its
**published** split, (b) retrain under our **v2 clean** split, (c)
diff the metrics. The differences are in split-injection cost, GPU
budget, and per-repo blockers.

## TL;DR

| Repo | Split-injection cost | GPU·h per run | Blockers | Recommended scope |
|------|---------------------|---------------|----------|-------------------|
| **SPRINT** | **0 LOC** (drop CSVs into `data/custom/`) | ~1–5 h DAVIS / ~20 h MERGED, 1× A100 | None (just need hydrate side-table) | Both splits × 3 seeds × 3 datasets |
| **LigUnity** | ~80 LOC, 2 files | ~50–100 h, 2× A100 | Unpinned `unicore`; hardcoded ESM2 path | Both splits × 1 seed × 1 dataset |
| **DrugCLIP** | ~120 LOC, 1 new + 1 edit | **~300–400 h, 1× A100** | Missing LIT-PCBA eval LMDBs | Inference-only on v2 splits + 1 retrain stretch |

**Headline recommendation:** lead with SPRINT (smoke test of the whole
pipeline, ~5 GPU·h), then LigUnity (production audit run), then
DrugCLIP **inference-only** unless GPU budget is large. A full
DrugCLIP retrain across regimes × seeds is not realistic on a single
GPU box.

## The one load-bearing dependency — `vsleakkg.v2.hydrate`

All three adaptations fail without it. v2 currently emits
`(example_id, partition)` parquet. Each repo needs richer rows:

- **SPRINT**: `(SMILES, Target Sequence, Label)` with SaProt-style
  3Di-interleaved `Target Sequence` when running with SaProt.
- **LigUnity**: assay dict with `{pockets, ligands[{smi, act}],
  uniprot, assay_id, sequence, version, domain}`.
- **DrugCLIP**: per-record LMDB key + `(smi, pocket_pdb_id)`.

**Deliverable** (one new module, ~150 LOC):

```python
# src/vsleakkg/v2/hydrate.py
def hydrate(example_ids: Iterable[str]) -> polars.DataFrame:
    """Returns columns:
       example_id, smiles, smiles_canonical, inchikey,
       uniprot, target_sequence, target_sequence_saprot, pdb_id,
       chembl_id, bindingdb_id, assay_id, label.
    Sources: the v1 raw loaders + a one-shot foldseek run for SaProt
    sequences keyed by uniprot."""
```

Build this **before** touching any model repo. It is the single bridge
between v2 outputs and every model's data layer.

Open question (from the SPRINT report): does v2 already have this
side-table somewhere we missed? If yes, point the agent at it; if no,
this is the first task of Phase 2.

## Per-repo adaptation summary

### SPRINT — easiest, do first

- **Split mechanism**: `data/<task>/{train,val,test}.csv`, picked by
  `--task` (`datamodules.py:402-404`, `:459-461`). `--task custom` is
  pre-wired to read `data/custom/`.
- **Adaptation**: no source edits. Write SPRINT-shaped CSVs from
  hydrated v2 rows into `~/SPRINT/data/custom/{train,val,test}_foldseek.csv`,
  then run `ultrafast-train --task custom --config configs/saprot_agg_config.yaml`.
- **Risks**: SaProt 3Di tokens (`*_foldseek.csv`) must match the
  upstream UniProt — bring them across by joining v2 rows against
  shipped CSVs on `uniprot_id`. Any OOD UniProt needs a one-off
  foldseek run on AF2 structures.
- **Budget**: ~1 h DAVIS, ~3 h BIOSNAP/BindingDB, ~20 h MERGED. 3
  seeds × 2 splits × 3 datasets ≈ 100 GPU·h.

### LigUnity — production audit run

- **Split mechanism**: NOT a column. Train/test split = which LMDB +
  JSON files the loader opens. Scrub against LIT-PCBA / DUD-E / DEKOIS
  / FEP is encoded as hardcoded JSON lookups under `test_datasets/`.
- **Adaptation**:
  - `unimol/tasks/train_task.py:540-605` — replace the scrub block
    with a parquet-driven filter on `(assay_id, smi)`.
  - `HGNN/screen_dataset.py:143-193` — mirror the same filter for
    HGNN's `load_assayinfo`.
  - Argparse passthrough in `train_task.py:152-288` + `train.sh` (~10
    LOC).
  - **Total ~80 LOC**.
- **Risks**:
  - `unimol/models/protein_ranking.py:60` hardcodes a local ESM2-35M
    path. Patch to `facebook/esm2_t12_35M_UR50D` or local mirror.
  - HGNN re-ranker reuses pre-computed embeddings from the screen
    encoder — must regenerate after each retrain (~2× encoder forward
    cost).
  - README tells you to hand-patch `unicore/options.py:250` and
    `unicore_cli/train.py:303`. **The exact `unicore` commit is not
    pinned anywhere in the repo** — this is the top Phase 2 blocker.
- **Budget**: ~75 GPU·h on 2× A100 per run + <2 h HGNN. 1 seed × 2
  splits × 1 corpus (CASF anchor) ≈ 150 GPU·h.
- **Pre-flight task**: bisect `unicore` commits until the README's
  line-number patch applies cleanly. Record the commit hash in
  `docs/phase2/dependency-pins.md`.

### DrugCLIP — inference-only by default

- **Split mechanism**: same "one LMDB file per split" as LigUnity.
  Loader at `unimol/tasks/drugclip.py:184` reads
  `<split>.lmdb`.
- **Adaptation**:
  - New `unimol/data/subset_dataset.py` (~25 LOC `SubsetWrapper`).
  - Edit `drugclip.py:184-185` (5–7 lines) to honour a sibling
    `<split>_keys.txt`.
  - One-time `tools/build_drugclip_v2_split.py` in the v2 repo
    (~80 LOC) — scans upstream `train.lmdb`, joins against the v2
    parquet, writes keylist files.
  - **Total <120 LOC**.
- **Risks**:
  - LIT-PCBA eval LMDBs are hardcoded at `drugclip.py:623,663,706`
    but **not distributed in the GDrive folder and not described in
    the README**. Either find the author's release or rebuild from
    raw LIT-PCBA before any eval-side comparison can run.
  - Repo self-labels "raw version" (`README.md:10`). Expect dead
    code (`from IPython import embed`, `from xmlrpc.client import
    Boolean`) and unreliable shell scripts.
  - `rdkit==2022.9.5` pin is non-negotiable.
- **Budget**:
  - Inference-only on the released `checkpoint_best.pt` against v2
    splits: ~1 GPU·h per split. Affordable.
  - Full retrain: ~300–400 GPU·h per run. **A regime × seed sweep is
    out of scope.** Maximum feasible: 1 retrain on `strict-clean`
    only, no seeds. That's already ~14 GPU·days.
- **Default plan**: inference-only across all clean regimes. Retrain
  only as a stretch goal.

## Common ID normalization needed

Every model joins back to the v2 graph through some pair of identifiers.
The `hydrate` module must standardise:

- **Ligand**: canonical SMILES via RDKit `MolToSmiles(MolFromSmiles(...))`
  + InChIKey as the primary join key. Both LigUnity and DrugCLIP store
  raw SMILES that need canonicalisation before any cross-join.
- **Target**: UniProt accession is primary; ChEMBL `target_id` and PDB
  ID are secondary. SPRINT also needs `target_sequence_saprot` (3Di
  interleaved).
- **Assay** (LigUnity only): `assay_id` from ChEMBL / BindingDB. Without
  this, LigUnity's filter cannot keep its grouping structure.
- **Label**: float pIC50 (LigUnity) or 0/1 binary (SPRINT). Build both
  columns in `hydrate`; let each adapter pick.

## Phase 2 schedule (one GPU box, ~1 month)

### Week 1 — Foundations

- [ ] Build `vsleakkg.v2.hydrate` side-table.
- [ ] Verify it for one corpus (LIT-PCBA target subset) and one model
      (SPRINT DAVIS).
- [ ] SPRINT smoke test: paper split on DAVIS, single seed. Confirms
      ~1 GPU·h matches paper number within noise.

### Week 2 — SPRINT production runs

- [ ] SPRINT × {DAVIS, BIOSNAP, BindingDB} × {paper, ligand-clean,
      scaffold-clean, protein-clean, strict-clean} × 3 seeds. Budget
      ~100 GPU·h.
- [ ] Pin `unicore` commit for LigUnity (in parallel).

### Week 3 — LigUnity audit

- [ ] LigUnity retrain × {paper, strict-clean} × 1 seed × CASF
      validation set. Budget ~150 GPU·h.
- [ ] HGNN re-ranker retrain on each checkpoint.
- [ ] DrugCLIP inference-only across v2 splits (released ckpt). Budget
      ~10 GPU·h.

### Week 4 — Optional stretches

- [ ] SPRINT on MERGED for LIT-PCBA Table 2 anchor. Budget ~60 GPU·h.
- [ ] DrugCLIP retrain on `strict-clean` only (no seeds). Budget ~350
      GPU·h — only if previous weeks finish under budget.

### Total realistic GPU budget

- Minimum (SPRINT full + LigUnity 1 regime + DrugCLIP inference): ~260
  GPU·h ≈ 11 GPU·days on 1× A100.
- Stretch (above + SPRINT MERGED + 1 DrugCLIP retrain): ~670 GPU·h ≈
  28 GPU·days.

## What to do first when you sit down to Phase 2

1. Read `docs/phase2/ligunity-investigation.md`,
   `docs/phase2/drugclip-investigation.md`,
   `docs/phase2/sprint-investigation.md` in that order.
2. Implement `vsleakkg.v2.hydrate` and write a unit test for it. This
   unblocks every downstream adapter.
3. Stand up SPRINT DAVIS with its **paper** split via the `custom`
   task path (just to verify the hydrate output is consumable end-to-end).
4. Run SPRINT DAVIS on the v2 `ligand-clean` split as the first real
   audit row. Headline number: paper AUROC vs clean-split AUROC.
5. Only after that, start pinning `unicore` for LigUnity.

## Open questions to resolve before launching Phase 2

| # | Question | Owner | Where to look |
|---|----------|-------|---------------|
| 1 | Does v2 already have an `example_id → (smiles, uniprot, …)` side-table? | Windows team | `src/vsleakkg/v2/` |
| 2 | What `unicore` commit makes LigUnity's `--validate-begin-epoch` patch apply cleanly? | Linux agent | `dependency-pins.md` |
| 3 | Where do DrugCLIP's LIT-PCBA `mols.lmdb` / `pockets.lmdb` come from? | Linux agent | GDrive folder + email authors |
| 4 | SaProt 3Di for OOD UniProts — do v2 splits ever introduce them? | Windows team | hydrate spec |
| 5 | Are CC BY-NC clauses on LigUnity / DrugCLIP weights compatible with the audit publication? | Legal | LICENSE files |

## Is Phase 2 hard?

Two real difficulties, neither is GPU:

1. **Training-manifest opacity.** Two of three repos do not publish
   row-level manifests with native ChEMBL / BindingDB primary keys.
   The hydrate side-table is the engineering response. ~1 week of
   work, then it's done forever.
2. **Dependency rot.** LigUnity needs an unpinned `unicore` commit;
   DrugCLIP self-labels "raw version" and ships dead imports. Plan a
   day per repo to wrestle the environment into shape before any
   training runs.

The algorithmic content of Phase 2 — actually swapping splits and
diffing metrics — is straightforward once those two problems are
resolved. **The headline experiment (SPRINT DAVIS paper-vs-clean) can
complete in ~3 GPU·hours once the hydrate layer exists.**
