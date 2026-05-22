# Phase 2 status — 2026-05-22

Phase 1 is fully landed (see `REVIEW_PHASE1.md` + `outputs_run3_pdbbind_pk/`).
This document captures what's set up for Phase 2 and what blocks the
SPRINT smoke test (P2.2).

## What's done

| Item | State | Notes |
|------|-------|-------|
| **vsleak2 venv on workspace** | DONE | `/vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2/`, `python -m venv --system-site-packages --without-pip`. Polars/pulp/vsleakkg.v2 installed via `pip --target`. |
| **SPRINT cloned** | DONE | `/vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/` (https://github.com/abhinadduri/panspecies-dti, commit pinned by --depth=1). Pyproject deps documented. |
| **SPRINT package import** | OK | `ultrafast` package importable. transformers / fair-esm / molfeat / chromadb all OK. |
| **SPRINT data dir** | OK | DAVIS / BindingDB / BIOSNAP / DUDe / MERGED CSVs pre-shipped in the repo. PCBA needs `download_pcba.sh`, MERGED extra needs `download_vsds.sh`. |
| **v2 → SPRINT adapter** | DONE (code) | `tools/sprint_csv_from_v2.py` already written + tested locally; emits `train.csv`/`val.csv`/`test.csv` (+ foldseek variants) from a v2 split + side-table. |

## What blocks the actual smoke test today

### B1 — `import lightning` errors

```
RuntimeError: operator torchvision::nms does not exist
```

Cause: system has `torch 2.10.0+cu126` + `torchvision 0.25.0+cu126`.
`pip install lightning==2.4.0 --target=vsleak2` pulled in `torch 2.12.0`,
which broke the torchvision binary contract. Fix options:

- (A) downgrade `torch` in the venv to `2.10` to match torchvision, then
  pin `lightning` to a version compatible with that torch (probably
  `lightning==2.3.x`).
- (B) install matching torchvision into the venv: `pip install --target
  ... torchvision==X.Y.Z` matching the torch 2.12 you got.
- (C) skip Lightning and run SPRINT's lower-level trainer directly. Not
  trivial — `ultrafast.train.train_cli` is Lightning-based throughout.

(A) is the cleanest; (B) is faster.

### B2 — GPU memory contention

`nvidia-smi` at 14:00 NZST showed (used | free):

```
GPU 0  Quadro RTX 6000  23730 MiB | 293 MiB
GPU 1  Quadro RTX 6000  23536 MiB | 487 MiB
GPU 2  Quadro RTX 6000  20654 MiB | 3369 MiB
```

Other users have ~all the memory. SaProt-650M + a small SPRINT
training batch needs >10 GB, so no card is currently usable. Wait for
a `cohenha+` / `vietoma+` job to release.

### B3 — protein sequences missing from DEKOIS/DUD-E v2 graphs

`Protein` nodes for non-PDBBind corpora carry only the target name
(`pim-2`, `pi3kg`, …), not AA sequence. To emit a SPRINT-shaped CSV
with the `Target Sequence` column populated, we need either:

- pull AA sequence from the v1 raw FASTA / processed pdbbind_proteins
  table (PDBBind works), OR
- a UniProt lookup table to convert DEKOIS/DUD-E target name → UniProt → sequence.

For PDBBind specifically, `pdbbind_proteins.parquet` already has UniProt
+ AA sequence and the v1 `Protein::xxxx` node_id matches the table row.
This is the easiest first SPRINT smoke target.

### B4 — side-table missing target_sequence_saprot

For SPRINT's `saprot_agg_config.yaml` (the headline config), we need
foldseek-derived 3Di-interleaved sequences in `target_sequence_saprot`.
None of our corpora currently has this. Fix: run `foldseek` against AF2
structures for each unique UniProt. ~8h GPU·h work for the corpora.

## Concrete next actions when you resume

1. **Fix B1**: `pip install --target=$ENVDIR/lib/python3.12/site-packages
   --upgrade torchvision` (let it find torch 2.12 compatible). Verify
   `import lightning` works.
2. **Run paper DAVIS smoke**: `cd ~/SPRINT && ultrafast-train --exp-id
   DAVIS --config configs/saprot_agg_config.yaml`. Needs ~10 GB VRAM.
   Watch for SaProt-650M download (~2.5 GB) on first run.
3. **B3 work**: build a `v1_protein_sequence.parquet` (UniProt -> AA seq)
   from the v1 raw FASTA files + chembl_targets.parquet. Patch
   `pipeline.extract_examples_frame` to join target_sequence in.
4. **v2 smoke**: emit SPRINT CSVs from PDBBind protein-clean split via
   `tools/sprint_csv_from_v2.py` (PDBBind has the AA sequences we need).
   `ultrafast-train --task custom --config configs/saprot_agg_config.yaml`.
5. Compare PDBBind paper-split AUROC vs PDBBind v2 protein-clean
   AUROC — first Mode B audit number.

## Update (2026-05-22 15:20 NZST)

Attempted to unblock B1 by installing matching torchvision into the
venv. The script's first step (`import torch` from the venv python)
took >16 minutes wall and never completed — box was effectively
unusable for fresh python startup despite load avg of ~20. Killed the
fix attempt.

This was the same fundamental NFS+box-load problem we saw with the
original conda env on /vol/grid-solar, but now manifesting even on the
fast workspace volume. The venv works fine for already-warm processes
(Phase 1 used it without trouble) but a cold `import torch` against
~5 GB of torch+cuda libs is unworkable when the box is contending.

P2.2 SPRINT DAVIS smoke test is **deferred** until the box is quieter
or until torch is symlinked into local /tmp (a future optimisation).

## What was achievable without a GPU

We did get the codebase + data pipeline + Phase 1 to completion. The
SPRINT clone + dependency install (modulo lightning) is also unblocked.
The `tools/sprint_csv_from_v2.py` adapter is tested locally and ready
to drop in once protein sequences are available.

Phase 2 P2.4-P2.6 (LigUnity, DrugCLIP) have not been touched in this
session because they need many GPU·h of free GPU time and their setup
quirks (unicore commit pin for LigUnity, missing LIT-PCBA LMDBs for
DrugCLIP) consume real engineering time. They remain queued as before.
