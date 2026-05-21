# DrugCLIP - Phase 2 adaptation investigation

## Repo snapshot

- Clone: `git clone --depth 1 https://github.com/bowen-gao/DrugCLIP D:/hoangpc/_audit_targets/DrugCLIP`
- Upstream: NeurIPS 2023 paper "DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening" (arXiv:2310.06367).
- README self-describes as: "Currently the code is a raw version, will be updated ASAP" (`README.md:10`).
- Stack: built on `unicore` (DP Tech's fork of fairseq) + Uni-Mol pretrained encoders. No `setup.py` / `requirements.txt` / `environment.yml` is shipped - README only says "same as Uni-Mol" and "**rdkit version should be 2022.9.5**" (`README.md:14-16`).
- Layout:
  - `drugclip.sh`, `test.sh`, `retrieval.sh` - top-level entry scripts.
  - `unimol/tasks/drugclip.py` - the unicore task (data loading + train/eval orchestration).
  - `unimol/models/drugclip.py` - the dual-encoder model with `logit_scale` and 128-d projection heads.
  - `unimol/losses/cross_entropy.py` - InfoNCE-style `in_batch_softmax` loss at `:519`.
  - `unimol/data/` - dataset wrappers (LMDB, affinity, cropping, etc.).
  - `unimol/test.py` - inference entry for DUD-E / LIT-PCBA.
  - `unimol/retrieval.py` - top-k retrieval entry.
  - `py_scripts/lmdb_utils.py`, `py_scripts/write_dude_multi.py` - data-prep helpers.
  - `HomoAug/` - homology-based train-set augmentation pipeline (separate sub-project).
  - `data/dict_mol.txt`, `data/dict_pkt.txt` - atom-type vocab files shipped in-repo.

## Entry points

### Training

README one-liner (`README.md:106`): `bash drugclip.sh`.

The real invocation inside `drugclip.sh:30-45` is (variables expanded for clarity):

```
CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch \
  --nproc_per_node=1 --master_port=10055 $(which unicore-train) \
  data --user-dir ./unimol --train-subset train --valid-subset valid \
  --num-workers 8 --ddp-backend=c10d \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --max-pocket-atoms 256 \
  --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
  --lr-scheduler polynomial_decay --lr 1e-3 --warmup-ratio 0.06 \
  --max-epoch 200 --batch-size 48 --batch-size-valid 128 \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq 1 --seed 1 \
  --tensorboard-logdir tsb_dir --log-interval 100 --log-format simple \
  --validate-interval 1 \
  --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
  --save-dir savedir --tmp-save-dir tmp_save_dir --keep-last-epochs 5 \
  --find-unused-parameters --maximize-best-checkpoint-metric \
  --finetune-pocket-model pocket_pre_220816.pt \
  --finetune-mol-model  mol_pre_no_h_220816.pt
```

Note: `data_path="data"` (`drugclip.sh:3`) - the script expects `data/train.lmdb`, `data/valid.lmdb`, `data/dict_mol.txt`, `data/dict_pkt.txt`. Pretrained Uni-Mol checkpoints are referenced as bare filenames at the cwd.

### Inference / evaluation

README one-liner (`README.md:110`): `bash test.sh`. Retrieval one-liner (`README.md:115`): `bash retrieval.sh`.

`test.sh:7-15` (note the unintended `$data_path` variable - it is unset in the script, so it expands to empty; the literal `"./data"` immediately after is what unicore parses as the positional `data` arg):

```
CUDA_VISIBLE_DEVICES="1" python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
  --results-path ./test --num-workers 8 --ddp-backend=c10d --batch-size 8 \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
  --path checkpoint_best.pt --log-interval 100 --log-format simple \
  --max-pocket-atoms 511 --test-task PCBA      # or DUDE
```

`unimol/test.py:34-65` dispatches on `--test-task`: `DUDE` -> `task.test_dude(model)`, `PCBA` -> `task.test_pcba(model)`. Both walk the per-target subdirs under `./data/DUD-E/raw/all/` or `./data/lit_pcba/` (hardcoded; see "Open questions").

`retrieval.sh:8-18` calls `unimol/retrieval.py` with `--mol-path`, `--pocket-path`, `--emb-dir`.

## Data pipeline

### LMDB schema (per-record fields)

Confirmed by reading `unimol/data/affinity_dataset.py:48-87` (training/valid path) and the README's key table (`README.md:46-58`):

| Key | Type | Used at |
|---|---|---|
| `atoms` | list[str] | `affinity_dataset.py:50` |
| `coordinates` | list of (N,3) arrays (RDKit conformers, up to 10) | `affinity_dataset.py:53,61` |
| `pocket_atoms` | list[str] (numbered names like `CA1`, stripped at `:43-46`) | `affinity_dataset.py:64` |
| `pocket_coordinates` | list of (M,3) arrays | `affinity_dataset.py:67` |
| `smi` | str (SMILES) | `affinity_dataset.py:69` |
| `pocket` | str (PDB ID) | `affinity_dataset.py:70` |
| `mol` | RDKit Mol (per README; not read by trainer) | unused in loader |
| `label` / `affinity` | float, optional - falls back to `1` when absent (`affinity_dataset.py:71-74`) | task line `:197`, `:211` |

For DUD-E / LIT-PCBA `mols.lmdb` records additionally need a `label` (read at `unimol/tasks/drugclip.py:342`, used to score actives vs decoys). `pocket.lmdb` / `pockets.lmdb` records only need `pocket_atoms`, `pocket_coordinates`, `pocket`.

LMDB layout: integer-string keys (`b"0"`, `b"1"`, ...), pickled values, written via `py_scripts/lmdb_utils.py:36-50`. Reader: `unimol/data/lmdb_dataset.py:15-49` - iterates `txn.cursor().iternext(values=False)` to get all keys and uses `f"{idx}".encode("ascii")` for lookup.

### Where the split lives (file:line)

**The split is "one LMDB file per split".** It is NOT a key prefix and NOT a separate keylist - it is just whichever `.lmdb` file is named `<split>.lmdb` under `args.data`.

The decisive line is `unimol/tasks/drugclip.py:184`:

```python
data_path = os.path.join(self.args.data, split + ".lmdb")
```

`split` comes from unicore CLI args `--train-subset train`, `--valid-subset valid` (`drugclip.sh:30`) and `--valid-subset test` (`test.sh:7`). For the paper's split, the downloaded `train_no_test_af.zip` (README `:28-40`) is just `train.lmdb` + `valid.lmdb` + the two `dict_*.txt` files dropped into `data/`. Inside `load_dataset` the only condition is `split.startswith("train")` (`drugclip.py:186`) which toggles the per-epoch conformer random sampling (vs deterministic for valid/test).

Other dataset-shaping calls operate on records, not on the split:
- `RemoveHydrogenPocketDataset` (`drugclip.py:222`)
- `CroppingPocketDataset` (`drugclip.py:229`) - caps pocket atoms at `--max-pocket-atoms` (256 train / 511 test).
- `ResamplingDataset(self.datasets[split])` (`drugclip.py:330`) - per-epoch resample with replacement, sized `1.0 * len(dataset)`, see `unimol/data/resampling_dataset.py:35-65`.

Hardcoded paths to flag:
- `unimol/tasks/drugclip.py:623, 663, 706, 762, 803, 848` - DUD-E / LIT-PCBA paths hardcoded to `./data/DUD-E/raw/all/...` and `./data/lit_pcba/...`. The CLI arg `--results-path` only steers output, not input.
- `py_scripts/write_dude_multi.py:23` defaults `--mol_data_path` to `/data/protein/DUD-E/raw/all` (the author's box).
- `HomoAug/run_HomoAug.py:807` defaults to `/data/protein/AF2DB/AFDB_HC_50.fa`.
- `unimol/tasks/drugclip.py:703` commented-out `/home/gaobowen/DrugClip/...` path.

No hardcoded Google Drive URLs in code - the GDrive folder is documented only in the README (`:20`).

## Adaptation plan

### Inject our v2 clean split

The cleanest hook is the single line `unimol/tasks/drugclip.py:184-185`:

```python
data_path = os.path.join(self.args.data, split + ".lmdb")
dataset = LMDBDataset(data_path)
```

Because v2 emits `(example_id, partition)` parquet, our `example_id` must map to an LMDB record. The LMDB is keyed by integer-strings (`b"0"`, `b"1"`, ...) and the per-record identity available to us is `(smi, pocket)`. Plan:

1. **One-time preprocessing**: scan the upstream `train.lmdb` once, build a parquet `(example_id, lmdb_key, smi, pocket)` (call it `drugclip_index.parquet`). The example_id can be `f"{pocket}|{smi}"` (matches what v2 emits) or just the lmdb_key string for a baseline run. Store under `data/v2_index/`.
2. **Materialise the split**: join v2's `(example_id, partition)` parquet against `drugclip_index.parquet` to produce two text files - `data/v2_clean/train_keys.txt` and `data/v2_clean/valid_keys.txt` - each one LMDB key per line.
3. **Patch the loader** at `unimol/tasks/drugclip.py:184` to wrap `LMDBDataset` with an index-filter when a sibling `<split>_keys.txt` is present. Minimal sketch:

   ```python
   data_path = os.path.join(self.args.data, split + ".lmdb")
   dataset = LMDBDataset(data_path)
   keylist_path = os.path.join(self.args.data, split + "_keys.txt")
   if os.path.exists(keylist_path):
       with open(keylist_path) as fh:
           allowed = {ln.strip() for ln in fh if ln.strip()}
       # LMDBDataset._keys are raw bytes like b"123"
       keep = [i for i, k in enumerate(dataset._keys) if k.decode() in allowed]
       dataset = SubsetWrapper(dataset, keep)   # tiny new class: __getitem__/__len__
   ```

   `SubsetWrapper` is ~10 lines (forwards `__getitem__` through an index list, returns `len(self.indices)`). Keep it in a new file `unimol/data/subset_dataset.py` and re-export from `unimol/data/__init__.py`. All downstream wrappers (`AffinityDataset`, `CroppingPocketDataset`, ...) operate on integer indices and don't care.

4. **Disable / parametrise `ResamplingDataset`** at `drugclip.py:330` for the clean run if we want strict 1 epoch = 1 pass coverage (currently it's `replace=False, size_ratio=1.0` so it already samples len(dataset) without replacement - effectively a shuffle; safe to leave as-is unless we need bitwise determinism).

5. **No changes needed to the test pipeline** for v2 - DUD-E / LIT-PCBA are held-out benchmarks loaded from their own LMDBs, not from the train split.

### Estimated diff

- New: `unimol/data/subset_dataset.py` (~25 lines)
- Edit: `unimol/data/__init__.py` (1 export line)
- Edit: `unimol/tasks/drugclip.py:184-185` (5-7 inserted lines)
- New: `tools/build_drugclip_v2_split.py` outside the repo, in our v2 codebase (~80 lines: scans LMDB, joins parquet, writes keylist).

Total: < 120 lines, no surgery into model / loss / training loop.

### Risks

- The README says ligand `coordinates` has up to 10 RDKit conformers; one is randomly chosen per epoch (`affinity_dataset.py:54-59`). If our v2 example_id is `(smi, pocket)`-keyed and the LMDB has duplicate `(smi, pocket)` rows (HomoAug duplicates a pocket against multiple homologs), our key-list approach is fine but be aware: example_id collisions across LMDB keys are possible. Use the `lmdb_key` itself as canonical example_id to be safe.
- `pocket_atoms` strings carry numeric prefixes (`CA1`, `OE2`, ...) that get stripped at `affinity_dataset.py:42-46`. v2 should not depend on the raw atom-name form.
- `dict_mol.txt` / `dict_pkt.txt` (`drugclip.py:172-173`) come from the upstream zip; reusing them is required for compatibility with the Uni-Mol pretrained encoders.
- Numeric record `label` may be absent in some LMDB records and silently defaults to `1` (`affinity_dataset.py:71-74`). For training this is fine (the loss is in-batch softmax, label unused); for any custom eval we'd need to add it.

## Resource estimate

The repo gives no explicit numbers. Inferring:

- **GPU**: `drugclip.sh:11` sets `n_gpu=1`, `CUDA_VISIBLE_DEVICES="1"`, `--batch-size 48` with fp16. Two Uni-Mol encoders (typical 15-layer transformer) + 128-d projections fit on a 24 GB card at bs=48 fp16; expect ~14-18 GB VRAM.
- **Training time**: `--max-epoch 200`, `--patience 2000` (effectively no early stop). Paper-scale `train_no_test_af` is ~600k records (PDBBind + HomoAug). At bs=48 single-GPU fp16 this is ~12.5k steps/epoch; on an A100 a Uni-Mol-class dual encoder is ~0.4-0.6 s/step => ~1.5-2 h/epoch => **~300-400 GPU-hours / ~13-17 days for a full 200-epoch run**. Note: `--keep-last-epochs 5` and `--validate-interval 1` indicate the authors did run all 200 epochs.
- **LMDB on disk**: not stated. By community reports of `train_no_test_af.zip`, expect ~10-25 GB uncompressed for the training LMDB.
- **Dependencies** (inferred, since README defers to Uni-Mol):
  - PyTorch (Uni-Mol pins 1.11.0 + CUDA 11.3 in its README)
  - `unicore` (DP Tech fork of fairseq) - must be built from source from `https://github.com/dptech-corp/Uni-Core`
  - `rdkit==2022.9.5` (README `:16`, explicit)
  - `lmdb`, `selfies`, `biopandas`, `numpy`, `scikit-learn`, `tqdm`, `IPython` (line 4 imports `from IPython import embed as debug_embedded` - leftover debug)
  - Pretrained checkpoints `mol_pre_no_h_220816.pt`, `pocket_pre_220816.pt` from the same GDrive folder.

## Reproducibility check

- The README does **not** quantitatively promise reproduction. It only says "Currently the code is a raw version, will be updated ASAP" (`:10`) and lists the GDrive folder ("It currently includes the train data, the trained checkpoint and the test data for DUD-E", `:22`).
- **No smoke-test sub-command exists.** Inspecting `test.sh` and `unimol/test.py:61-65` there are only two `--test-task` choices (`DUDE`, `PCBA`), both of which iterate over all targets under `./data/{DUD-E,lit_pcba}/...` via `os.listdir` (`drugclip.py:706, 848`). Recommended workaround: temporarily drop one target subdir into `./data/lit_pcba/` (e.g. `ADRB2/`) to get a ~5-minute end-to-end eval against the released `checkpoint_best.pt`.
- For a training smoke test: set `--max-epoch 1`, point `--train-subset` to a tiny custom `train.lmdb` (~100 records), keep everything else identical. Validates the pipeline end-to-end (incl. the `--finetune-mol-model` + `--finetune-pocket-model` load at `drugclip.py:562-574`) without burning GPU-days.
- Determinism: `--seed 1`, but `ResamplingDataset` re-seeds per epoch with `[42, seed, epoch]` (`resampling_dataset.py:103-109`). Conformer choice in `AffinityDataset` uses `numpy_seed(seed, epoch, index)` (`:55`). Reproducible across runs if all knobs match.

## License

- **Code**: Apache 2.0 (`LICENSE:5-24`). Copyright 2025 AIR Tsinghua, portions adapted from DP Technology 2022.
- **Model weights & outputs**: CC BY-NC 4.0 (`LICENSE:26-44`) - "You may not use the material for commercial purposes". Research / benchmark use is fine.
- **Training data**: README cites PDBBind + HomoAug-augmented set; **no explicit data license file in the repo**. PDBBind itself is "free for academic use, license required for commercial". HomoAug's `LICENSE.txt` lives at `HomoAug/LICENSE.txt` (not inspected here - flag for review).
- **Phase 2 implication**: we can train, eval, and publish numbers under the audit; we should NOT redistribute the LMDB itself nor the trained checkpoint without checking PDBBind terms. Our v2 clean-split parquet is fine to publish - it only contains identifiers, not the molecular data.

## Open questions

1. The README's "raw version" caveat is reflected in code: `unimol/tasks/drugclip.py:703` has a commented-out `/home/gaobowen/...` save-name and `:704` `save_name = ""` - the eval result printer just prints, never saves. We will need to capture stdout if we want metrics in a file.
2. `test.sh:7` references `$data_path` but never sets it - the shell silently expands it to empty. The intended invocation likely is to set `data_path="./data"` like in `drugclip.sh`. Not breaking (the next literal `"./data"` covers it) but a sign the script is unmaintained.
3. `unimol/tasks/drugclip.py:4` imports `from IPython import embed as debug_embedded` and `:9` imports `from xmlrpc.client import Boolean` - clear leftovers from interactive debugging; flagged in case they break in newer Pythons.
4. `unimol/data/__init__.py` re-exports include `NormalizeDockingPoseDataset`, `CrossDistanceDataset`, `TTADockingPoseDataset`, `VAEBindingDataset` etc. that are unused by the DrugCLIP task - dead-code dataset wrappers; ignorable.
5. README mentions a `mol` (RDKit Mol object) field but the loader never reads it (`affinity_dataset.py`). Pickled RDKit Mol objects are version-fragile - if we ever rebuild the LMDB we can safely drop them.
6. The "test data for DUD-E" is on GDrive, but LIT-PCBA test data is NOT mentioned in README - yet `test.py` happily runs `--test-task PCBA`. Where the LIT-PCBA `mols.lmdb` / `pockets.lmdb` come from is unclear. **This is the biggest blocker for full eval reproduction.**
7. Does v2 currently emit example IDs that DrugCLIP can map back to? Need to confirm v2's `example_id` schema (probably `(target_pdb, smiles)`) and add a one-shot `build_drugclip_v2_split.py` joiner before any GPU work starts.
8. The CC BY-NC 4.0 on model outputs may complicate publishing per-example scores in a leakage paper - clarify with legal whether "research benchmark" qualifies as non-commercial.
