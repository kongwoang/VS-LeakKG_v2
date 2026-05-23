# Phase 2 DrugCLIP attempt — env blocker

This document records the DrugCLIP audit attempt on the v2 PDBBind splits and
the specific blocker that prevented training from starting. The data side is
done and reproducible — only the unicore↔torch runtime is wedged.

## What was done

1. **Found PDBBind raw data on VUW.** All 19,037 PDBBind complexes are at
   `/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG/data/raw/PBDBind/extracted/P-L/`
   with per-complex `<pdbid>_protein.pdb`, `<pdbid>_ligand.{mol2,sdf}`, and
   `<pdbid>_pocket.pdb` (pocket already extracted by PDBBind).

2. **Downloaded unimol pretrains.** `mol_pre_no_h_220816.pt` (190 MB) +
   `pocket_pre_220816.pt` (190 MB) from the dptech-corp Uni-Mol v0.1 release at
   `/vol/dl-nguyenb5-solar/users/hoangpc/drugclip_data/pretrains/`.

3. **Wrote `tools/v2_to_drugclip_lmdb.py`** — reads a v2 PDBBind split parquet,
   looks up each example's PDB ID, reads the pre-extracted `pocket.pdb`,
   generates RDKit 3D conformers for the ligand, and writes a pyxis-style LMDB
   in DrugCLIP's expected schema:

   ```python
   {
       "atoms":              [str, ...],   # ligand element symbols
       "coordinates":        [np.ndarray], # list of 3D conformers
       "pocket_atoms":       [str, ...],   # pocket PDB atom names
       "pocket_coordinates": np.ndarray,   # (n_pocket_atoms, 3)
       "smi":                str,
       "pocket":             str,          # pdb id
       "label":              float,        # binarized pK
   }
   ```

   train.lmdb keeps only positives (`label==1`, matching DrugCLIP's
   in-batch-softmax contrastive paradigm). valid.lmdb same. test.lmdb keeps
   both classes for per-pair AUROC scoring at audit time.

4. **Built LMDBs for all 3 v2 splits** on VUW (~10 min each):

   ```
   data/v2_pdbbind_ligand/{train,valid,test}.lmdb   (200M / 43M / 102M)
   data/v2_pdbbind_protein/{train,valid,test}.lmdb  (175M / 55M / 124M)
   data/v2_pdbbind_dual/{train,valid,test}.lmdb     (152M / 67M / 127M)
   ```

   ~95% of examples successfully built; ~5% lost to RDKit conformer-generation
   failures on weird tautomers. Logged in
   `/vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/lmdb_build_*.log`.

5. **Wrote DrugCLIP launcher** (`launch_drugclip.sh`) with the published
   `drugclip.sh` hyperparameters unchanged (lr 1e-3, batch 48, 200 epochs,
   in_batch_softmax loss, drugclip arch), routing through
   `python -m unicore_cli.train`.

## The blocker

Training crashes with a C-level `free(): invalid pointer` immediately after
trainer init, before any forward/backward step:

```
2026-05-23 18:13:20 | INFO | unicore.trainer | loading train data for epoch 1
2026-05-23 18:13:20 | INFO | unicore.tasks.unicore_task | get EpochBatchIterator for epoch 1
2026-05-23 18:13:20 | INFO | unicore.trainer | No existing checkpoint found ...
2026-05-23 18:13:20 | INFO | unicore.trainer | NOTE: your device may support faster training with --fp16
free(): invalid pointer
[python killed]
```

The Python fault handler stack traces to:

```
unicore/optim/adam.py:114 → torch/optim/optimizer.py:405 → torch/_compile.py:47
  → torch._dynamo.aot_compile / convert_frame / __init__
```

i.e., the crash is in `torch._dynamo.aot_compile` during `UnicoreAdam.__init__`
calling `torch.optim.Optimizer.__init__`. The crash reproduces with:
- `--num-workers 0` (rules out DataLoader workers)
- `--no-fp16` (rules out fp16 path)
- `TORCHDYNAMO_DISABLE=1 PYTORCH_DISABLE_TORCH_COMPILE=1` (env vars do not
  short-circuit the aot_compile decorator)
- batch sizes from 48 down to 8 (rules out OOM-class issues)

## Root cause (most likely)

unicore was installed in the shared `vsleak2` venv against the torch present
when SPRINT was set up. Torch 2.12 changed `torch.optim.Optimizer.__init__` to
go through `_compile.inner → aot_compile`, and an ABI mismatch (probably with
numpy 2.x's C ABI) makes `aot_compile` walk into freed memory during dynamo's
internal frame conversion. The "fused_*" CUDA kernels for unicore are missing
(visible at import: `fused_layer_norm is not installed corrected` etc.) which
removes one workaround unicore would normally take.

## Why this isn't quick to fix

Resolving needs one of:
1. **Pin torch ≤ 2.4** (pre-`_compile.inner` decorator). Risk: breaks the
   already-trained SPRINT runtime in the same venv.
2. **Pin numpy < 2.0** to match unicore's C ABI expectations. Risk: same.
3. **Build unicore from source** against current torch/numpy. ~1 hour
   compile + uncertain whether it fixes the dynamo path.
4. **Stand up a fresh venv** specifically for DrugCLIP with pinned versions.
   Cleanest but ~1-2 hours of setup + ProtBert and SPRINT artifacts cannot be
   reused.

Given the audit's 5-7 day budget and that the SPRINT Group A audit (the core
finding) is already shipped, deferring DrugCLIP to a follow-up pass.

## What ships

- `tools/v2_to_drugclip_lmdb.py` — re-runnable LMDB builder, no env clash.
- LMDB artefacts on VUW under `DrugCLIP/data/v2_pdbbind_{ligand,protein,dual}/`
  ready for the next attempt.
- Pretrained unimol weights under `drugclip_data/pretrains/`.
- Launcher template at `/vol/dl-nguyenb5-solar/users/hoangpc/launch_drugclip.sh`.

The next attempt is a venv pin + one `unicore-train` invocation; no data prep
needs to be redone.

## What does NOT ship

- DrugCLIP train/test numbers on v2 splits.
- A second model family in the AUDIT_FINAL.md table.

The audit's headline (protein-axis leakage drops AUROC by 15-17pp on PDBBind,
model-invariant across Morgan-RF and SPRINT) stands on Phase 1 + Phase 2
SPRINT alone. DrugCLIP would have been a third data point — informative but
not load-bearing.
