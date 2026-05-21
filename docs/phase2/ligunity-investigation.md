# LigUnity — Phase 2 adaptation investigation

## Repo snapshot
- URL: https://github.com/IDEA-XL/LigUnity
- Cloned to: `D:/hoangpc/_audit_targets/LigUnity`
- Last commit: `8ae35ed` "update HGNN code" (2026-03-30)
- Primary language: Python
- Size on disk after shallow clone: 180 MB (most of it is the bundled `test_datasets/` LMDBs for DUD-E / DEKOIS / FEP / OOD; the actual training data lives off-repo on Figshare).
- Paper: Feng et al., *Patterns* 2025 (DOI 10.1016/j.patter.2025.101371). Built on top of Uni-Mol + DrugCLIP, uses the `unicore` training engine (Fairseq-style CLI).

## Entry points

### Training
The repo ships one driver, `train.sh`, with four pre-baked configurations (differing only in `--protein-similarity-thres` / `--rank-weight`). The first block (the canonical screen-pocket model) is `train.sh:27-45`:

```
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT \
    $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
    --num-workers 8 --ddp-backend=c10d \
    --task train_task --loss rank_softmax --arch pocketscreen \
    --max-pocket-atoms 256 \
    --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch \
    --batch-size $batch_size --batch-size-valid $batch_size_valid \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
    --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
    --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-best-checkpoints 8 --keep-last-epochs 10 \
    --find-unused-parameters --maximize-best-checkpoint-metric \
    --finetune-pocket-model $finetune_pocket_model --finetune-mol-model $finetune_mol_model \
    --valid-set CASF --max-lignum 16 --protein-similarity-thres 1.0
```

Hyperparams in `train.sh:1-24`: 2 GPUs, batch 24, 50 epochs, lr 1e-4, fp16, warmup 6 %. Architectures swapped via `--arch pocketscreen` (and `proteinscreen` for the protein-ranking head — `train.sh` only shows the pocket variant, but `test.sh` is parameterized).

Stage 2 — HGNN re-ranker — is run after the screen encoder is trained (README lines 84-89):

```
cd ./HGNN
python main.py --data_root ${path2data} --result_root "../result/pocket_ranking" --test_ckpt ${path2weight_HGNN}
python main.py --data_root ${path2data} --result_root "../result/protein_ranking" --test_ckpt ${path2weight_HGNN}
```

`HGNN/main.py` argparse keys: `--batch_size`, `--embed_dim`, `--lr`, `--epochs` (default 20), `--test_ckpt`, `--data_root`, `--result_root`, `--similarity_thres`.

### Inference / evaluation
`test.sh` (no per-task config — it is dispatched by the first positional arg):

```
CUDA_VISIBLE_DEVICES=0 bash test.sh <ALL|BDB|PDB|FEP> <pocket_ranking|protein_ranking> <weight_path> <results_path>
```

It calls `python ./unimol/test.py "./test_datasets" --user-dir ./unimol --valid-subset test --task test_task --loss rank_softmax --arch $arch --path $weight_path --max-pocket-atoms 511 --test-task $TASK` (`test.sh:10-18`). The dispatch is hardcoded — `--test-task` `ALL` triggers DUD-E + LIT-PCBA + DEKOIS in sequence (see `unimol/tasks/test_task.py` methods `test_dude`/`test_pcba`/`test_dekois`, `test_task.py:842,666,999`). Final-fusion script: `python ensemble_result.py DUDE PCBA DEKOIS` (README:92). For FEP zero/few-shot the bash drivers are `test.sh FEP …` and `test_fewshot.sh FEP <arch> <sup_num> …`.

## Data pipeline

### Files loaded
Tracing `--task train_task` → `unimol/tasks/train_task.py:485 load_dataset()`. For the canonical `split=="train"` branch (`train_task.py:518-524`):

- `${data}/train_lig_all_blend.lmdb`              — ligand 3-D conformers (atoms / coordinates / smi)
- `${data}/train_prot_all_blend.lmdb`             — pocket structures (pocket / pocket_atoms / pocket_coordinates)
- `${data}/train_label_pdbbind_seq.json`          — PDBbind assay labels
- `${data}/train_label_blend_seq_full.json`       — ChEMBLv34 + BindingDBv2024m5 "blend" assay labels (this is the *training manifest* the user mentioned)
- `${data}/uniport40.clstr` / `uniport80.clstr` / `sequence_distance.txt`  — protein-similarity clusters (only loaded if `--protein-similarity-thres` < 1.0; `train_task.py:505-516`)
- `${PROJECT_ROOT}/test_datasets/fep_repeat_ligands_can.json` and `fep_assay_ids.json`   — *de-duplication* lists (training-set leakage scrub vs. FEP test set, applied unconditionally at `train_task.py:540-563`)
- `${PROJECT_ROOT}/test_datasets/dude.json`, `PCBA.json`, `dekois.json`   — uniprot lists removed from training when `--valid-set CASF` (default), `train_task.py:566-580`
- Validation: `${data}/valid_lig.lmdb` + `${data}/valid_prot.lmdb` + `${data}/valid_label_seq.json` (the CASF-2016 holdout — `train_task.py:612-622`).

The "split" therefore is *not* a column on an LMDB. It is a JSON-level filtering operation applied to `pair_label_2` (the blend list). Train/valid/test live in **different files** (different LMDBs + different JSON manifests), and the *only* test→train scrub knob is the JSON file paths in `test_datasets/`.

### Schema
Each entry in `train_label_blend_seq_full.json` is an "assay" dict consumed by `unimol/data/pair_dataset.py PairDataset` (`pair_dataset.py:101-142`). Required keys (read at `pair_dataset.py:125-131`):

- `pockets`        : list[str] of pocket names (must match `pocket_name` in `train_prot_all_blend.lmdb`)
- `ligands`        : list[ {`smi`: SMILES, `act`: float pAct, optionally `rel`: "=" } ]
- `uniprot`        : UniProt accession
- `assay_id`       : ChEMBL / BindingDB assay id (used by the de-dup filter at `train_task.py:560-563`)
- `sequence`       : full UniProt sequence (consumed by the ESM2 encoder in `protein_ranking.py:60`)
- `version`        : ChEMBL version / date (used only by `--valid-set TIME` split, `train_task.py:529`)
- `domain`         : `"pdbbind"` for label-1 entries (re-tagged at `HGNN/screen_dataset.py:151`)

LMDB rows expose: `atoms` (atomic-symbol list), `coordinates` (Nx3 numpy or list-of conformers), `smi`, `pocket`, `pocket_atoms`, `pocket_coordinates` (see `unimol/data/affinity_dataset.py:50-80`).

### Where the split lives (file:line)
- Train / valid filename selection: `unimol/tasks/train_task.py:518-631` (chain of `elif` blocks keyed on `split` and `--valid-set`).
- Test-protein scrub of training set: `unimol/tasks/train_task.py:560-605` (loads `dude.json`, `PCBA.json`, `dekois.json`, `FEP.json` from `test_datasets/` and removes by `uniprot` and `assay_id`).
- FEP-ligand scrub: `unimol/tasks/train_task.py:540-558` (removes assays whose ligands appear in `fep_repeat_ligands_can.json`).
- Per-similarity-threshold protein clustering: `train_task.py:504-516` + `train_task.py:574-579`.
- HGNN replays the same scrub against the same JSONs: `HGNN/screen_dataset.py:143-193` (`load_assayinfo`).
- Few-shot k-shot splits are intra-assay (random or scaffold) inside `train_task.py:317-482`, gated by `--valid-set` ∈ `{TYK2, FEP, TIME, OOD, DEMO}` and `--split-method ∈ {random, scaffold}`.

No deterministic hash of a SMILES/UniProt is ever used; the "split" is the *set of JSON files loaded*.

## Adaptation plan

### Inject our v2 clean split
The lowest-LOC path is to keep the LigUnity LMDBs as-is and replace the JSON manifest filter with a parquet-driven one. The parquet contains `(example_id, partition)` where `example_id` should be defined as `f"{assay_id}__{smi}"` (assay-level row, mirroring how `pair_dataset.py:127-128` flattens ligands per assay) or `(uniprot, smi)` if we want to ignore assay grouping. Two intervention points (only one is needed if we touch the loader, but both keep behaviour consistent):

1. **`unimol/tasks/train_task.py:540-605`** — replace the chain of `repeat_ligands` / `testset_uniprot_lst` / `non_repeat_assayids` filters with a single helper, e.g.:

   ```python
   from vsleakkg.v2.io import load_partition  # external library
   keep_ids = load_partition(self.args.split_parquet, partition=split)  # set[str]
   pair_label_2 = [a for a in pair_label_2
                   if any(f'{a["assay_id"]}__{lig["smi"]}' in keep_ids for lig in a["ligands"])]
   for a in pair_label_2:
       a["ligands"] = [l for l in a["ligands"]
                       if f'{a["assay_id"]}__{l["smi"]}' in keep_ids]
   pair_label_2 = [a for a in pair_label_2 if len(a["ligands"]) >= 3]
   pair_label_1 = [...]  # same against PDBbind manifest if PDBbind rows are in our parquet
   ```

2. **`HGNN/screen_dataset.py:143-193`** — same replacement (this stage independently re-filters via the JSON paths and `data_root/fep_assays.json`).

3. Add `--split-parquet` to the argparse: `unimol/tasks/train_task.py:152-288` (add_args) and a passthrough in `train.sh`. ~10 LOC.

4. Validation: keep `valid_label_seq.json` as the CASF holdout *or* add `partition == "valid"` rows to the parquet and load via the same helper at `train_task.py:612-622`.

### Estimated diff
- `unimol/tasks/train_task.py`: ~40 LOC (10 for argparse + 30 to replace the filter block).
- `HGNN/screen_dataset.py`: ~20 LOC mirror.
- New helper `unimol/utils/v2_split.py`: ~20 LOC (pandas.read_parquet → set of ids).
- `train.sh`: 2-line passthrough.
- **Total ≈ 80 LOC**, single PR-sized.

### Risks
- The schema fix-up assumes the v2 parquet's `example_id` can be deterministically reconstructed from `(assay_id, smi)`. If our generator uses `(uniprot, smi)` instead, the filter still works but conflates assays — flag for caller.
- `pair_dataset.py:65-70` builds a `trainidxmap` that repeats assays proportional to ligand count; after filtering this stays correct as long as we filter *before* `PairDataset(...)` is constructed at `train_task.py:611`. The proposed patch keeps that ordering.
- HGNN stage relies on **pre-computed embeddings** under `result/<arch>/BDB/` and `result/<arch>/PDBBind/` produced by `test.sh ... ALL`. If we change the train split, those embeddings must be regenerated from the new checkpoint; HGNN cannot be retrained in isolation.
- `unimol/models/protein_ranking.py:60-61` hardcodes `/cto_studio/xtalpi_lab/fengbin/DrugCLIP/esm/esm2_t12_35M_UR50D` — must be patched to a local ESM-2 35M path or HF hub id before any protein-ranking run.
- `train_task.py:540-543` unconditionally reads `test_datasets/fep_repeat_ligands_can.json` even when `--valid-set != FEP`. Our patch should preserve this safety scrub or explicitly disable it via a flag.

## Resource estimate
README and `train.sh` say: 2 GPUs (`CUDA_VISIBLE_DEVICES="0,1"`), batch 24, 50 epochs, fp16, `--update-freq 1`. No requirements.txt, no setup.py, no environment.yml is shipped — dependencies must be reverse-engineered from imports:

- `torch` + `torch.distributed.launch` (CUDA implied; fp16 path)
- `unicore` (DeepModeling) — installed via pip; provides `unicore-train` CLI and `LMDBDataset`. README's active-learning section (lines 138-166) patches `unicore/options.py` and `unicore_cli/train.py`, so a *specific* unicore version is required (patch line numbers 250 / 303 suggest unicore commit from 2024).
- `rdkit`, `numpy`, `scipy`, `pandas`, `tqdm`, `scikit-learn`
- `transformers` (for ESM2 35M, `protein_ranking.py:60`)
- Uni-Mol pretrained weights `pretrain/mol_pre_no_h_220816.pt` + `pocket_pre_220816.pt` (`train.sh:11-12`)

Dataset size (Figshare 27966819) — README doesn't quote a number; from the file list (`train_lig_all_blend.lmdb`, `train_prot_all_blend.lmdb`, `uniport40.clstr`, etc.) expect 30-80 GB on disk for the full ChEMBLv34 + BindingDBv2024m5 blend. **Verify before phase 2.**

Wall-clock: paper does not quote training time. Architecture is Uni-Mol-base × 2 (mol + pocket encoder) + ESM2-35M + transformer pair-encoder. Rough estimate: at 50 epochs × ~600k assay rows × batch 24 on 2× A100 80 GB ≈ 48-96 GPU-hours per run. HGNN stage is small (20 epochs, batch 128, single GPU) — probably <2 GPU-h. **The number is not in the README; this is my estimate, not a quote.**

## Reproducibility check
README does *not* claim "one command reproduces Table X". It gives a sequence:
1. Download data from Figshare (`27966819`, `27967422`, `29379161`) and Google Drive.
2. Clone checkpoints from three HF repos (`fengb/LigUnity_VS`, `fengb/LigUnity_pocket_ranking`, `fengb/LigUnity_protein_ranking`).
3. Run `bash test.sh ALL pocket_ranking ...` then `bash test.sh ALL protein_ranking ...` then `cd HGNN && python main.py ...` twice, then `python ensemble_result.py DUDE PCBA DEKOIS`.

There is no Makefile, no `reproduce.sh`, no DVC pipeline. **Gap**: no manifest of expected metric values, no checksum on the Figshare bundle, no pinned version of `unicore`. The README explicitly says "modify the unicore code" for active learning (lines 137-166), which means the public unicore release does not match the one used by the authors.

## License
- `License` file in repo root: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
- README badge claims "Code License: Apache 2.0" + "Data License: CC BY-NC 4.0" (README:7-8), but only one LICENSE file is present and it is CC BY-NC. The Apache-2.0 badge appears to be aspirational / inherited from the Stanford Alpaca template (badge links point at `tatsu-lab/stanford_alpaca`).
- **Net effect for a published audit**: the bundled code+data must be treated as **non-commercial only**. Redistribution of fine-tuned checkpoints derived from LigUnity weights inherits CC BY-NC 4.0. This is compatible with an academic leakage-audit paper, but blocks any commercial product.

## Open questions
1. The Figshare 27966819 bundle is gated behind a click-through; total download size and contents (LMDB-only vs. LMDB + raw SDF/PDB) are not documented in the repo. Need to download once and inventory before estimating disk budget.
2. The "blend" manifest `train_label_blend_seq_full.json` has no published row count. Until we open it we cannot estimate the parquet size needed for our v2 split (likely ~1-3 M (assay, ligand) pairs).
3. `--valid-set CASF` (default) silently removes DUD-E / DEKOIS / LIT-PCBA uniprots from training (`train_task.py:566-580`). For our audit's "paper split" baseline we must decide whether to keep that scrub *on* (matches the paper's reported numbers) or *off* (cleanest A/B against our v2 split). Recommend: keep it on for (a), turn it off for (b) — but this needs caller confirmation.
4. `unimol/models/protein_ranking.py:60` hardcodes a local ESM2-35M path. The HuggingFace hub ID is `facebook/esm2_t12_35M_UR50D` — confirm that swap reproduces author's numbers (i.e. authors didn't fine-tune ESM separately).
5. The patched `unicore` (README:137-166) — what exact commit of `unicore` do we install? `validate-begin-epoch` is added by hand, suggesting the public release post-dates the patch and may have an incompatible `options.py:250`. **This is the most likely blocker** — without a working `unicore-train` we cannot launch any training job.
6. HGNN stage requires per-checkpoint pre-computed embeddings (`result/.../BDB/`, `result/.../PDBBind/`). For Phase 2 we need to budget ~2× the encoder forward-pass cost per retrain.
