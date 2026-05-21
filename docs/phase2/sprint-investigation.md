# SPRINT — Phase 2 adaptation investigation

## Repo snapshot

- Source: `https://github.com/abhinadduri/panspecies-dti` (paper: arXiv 2411.15418, "Scaling Structure Aware Virtual Screening to Billions of Molecules with SPRINT").
- Clone location on this machine: `D:/hoangpc/_audit_targets/SPRINT` (shallow, `--depth 1`).
- Python package name: `ultrafast` (per `pyproject.toml`, version `0.1.0`). Build with `pip install -e .`.
- Key pinned deps (`pyproject.toml` lines 13-29): `lightning==2.4.0`, `torch==2.4.1`, `transformers==4.43.4`, `rdkit==2023.9.5`, `fair-esm==2.0.0`, `molfeat==0.10.1`, `chromadb==0.5.5`, `ml-pyxis` (LMDB wrapper) installed from git. Optional DDP path requires downgrading to `lightning==2.0.8`.
- External binaries / weights: SaProt checkpoint `SaProt_650M_AF2.pt` auto-downloaded from `huggingface.co/westlake-repl/SaProt_650M_AF2` (`featurizers.py:678`); MMseqs2 (`mmseqs`) required only for `--ship-model` similarity-filtered training (`datamodules.py:95-108`); `foldseek` is needed only for re-generating structure-aware sequences from PDB files.
- Repo size after clone: ~80 MB. DAVIS/BindingDB/BIOSNAP CSV splits ship in the repo; MERGED requires running `data/MERGED/huge_data/download.sh` (a `gdown` pull of `merged_data.zip`).

## Entry points

The CLI is defined in `pyproject.toml:32`:
```
[project.scripts]
ultrafast-train = "ultrafast.train:train_cli"
```
`train_cli` lives in `ultrafast/train.py:47-95` and parses argparse args, then dispatches to `train(...)` (`train.py:97`).

### Training

Quoting README.md verbatim (lines 51-60) for DTI prediction on DAVIS:

```
# Reproducing ConPLex
ultrafast-train --exp-id DAVIS --config configs/conplex_config.yaml
# ConPLex-attn
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml --prot_proj agg
# SPRINT-sm
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml
# SPRINT
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml --model-size large
```

Switching dataset is done with `--task {biosnap,bindingdb,davis,biosnap_prot,biosnap_mol,dti_dg,merged,custom}` (`train.py:52-62`). The choice list is hard-coded — `custom` is already wired in.

The headline config `configs/saprot_agg_config.yaml`:
```
task: davis
drug_featurizer: MorganFeaturizer
target_featurizer: SaProtFeaturizer
latent_dimension: 1024
model_size: "small"
prot_proj: "agg"
num_heads_agg: 4
dropout: 0.05
sigmoid_scalar: 5
batch_size: 64
epochs: 250
lr: 1e-5
lr_t0: 10
weight_decay: 0
```
CLI flags override YAML via `config.update(args_overrides)` (`train.py:166-167`).

### Inference / evaluation

Test is run inline at the end of `train(...)` (`train.py:394-400`): `trainer.test(datamodule=datamodule, ckpt_path=best_model_path)`. There is no separate `ultrafast-eval` script. LIT-PCBA evaluation is performed during validation by attaching `PCBAEvaluationCallback` (`train.py:31-37`) when `--eval-pcba` is set. Quoted from README.md lines 65-79:

```
# Setup MMseq2
conda install -c conda-forge -c bioconda mmseqs2
# Running Single Lit-PCBA
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml --task merged --epochs 15 --ship-model --model-size large
--target-protein-id {TARGET} --similarity-threshold {THRESHOLD} --eval-pcba
# SPRINT
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml --epochs 15 --ship-model --model-size large
--target-protein-id "all" --similarity-threshold 0.9
```
(`--task merged` is implied for the SPRINT variant by README convention; the LitPCBA section then sets `--ship-model` so val/test fold into the union and homology-similar proteins to LIT-PCBA targets are MMseqs2-filtered.)

Other CLI entry points (`pyproject.toml:32-36`): `ultrafast-embed`, `ultrafast-store`, `ultrafast-report`, `ultrafast-topk` — these are post-hoc embedding/retrieval utilities, not training.

## Data pipeline

`train_cli -> train(...) -> get_task_dir(config.task)` (`datamodules.py:29-56`) maps the task string to a directory under `data/`. For non-merged tasks the data module is `DTIDataModule` (`datamodules.py:362`). For `--task merged` the trainer swaps to `MergedDataModule` (`train.py:250-253`).

### Files per dataset

`get_task_dir` (`datamodules.py:37-54`) maps:

| `--task` | Directory | What ships in the repo |
| --- | --- | --- |
| `davis` | `data/DAVIS` | `train.csv`, `val.csv`, `test.csv` (+ `*_foldseek.csv` variants used when target featurizer is SaProt). Row counts: 2086 / 3006 / 6011 |
| `biosnap` | `data/BIOSNAP/full_data` | same 6 CSVs; 19238 / 2748 / 5497 rows |
| `biosnap_prot` | `data/BIOSNAP/unseen_protein` | same convention |
| `biosnap_mol` | `data/BIOSNAP/unseen_drug` | same convention |
| `bindingdb` | `data/BindingDB` | 12668 / 6644 / 13289 rows |
| `merged` | `data/MERGED/huge_data` | TSVs `merged_pos_uniq_{train,val,test}_rand.tsv` and `merged_neg_uniq_{train,val,test}_rand.tsv` + `id_to_smiles.npy`, `id_to_sequence.npy`, `id_to_saprot_sequence.npy`, plus pre-built LMDBs `smiles.lmdb`, `SaProt_targets.lmdb`. Fetched by `data/MERGED/huge_data/download.sh` (gdown ID `1JM9b0jRmesZlPrt3OJaG2eKf-XtIx_ea`) |
| `custom` | `data/custom/` | User-supplied `train.csv`, `val.csv`, `test.csv` (or `*_foldseek.csv` if SaProt featurizer) |

When `target_featurizer == SaProtFeaturizer`, `DTIDataModule.__init__` (`datamodules.py:417-420`) silently rewrites paths to the `_foldseek.csv` siblings.

### Schema

For `DTIDataModule` datasets (DAVIS, BindingDB, BIOSNAP*, custom) the columns expected on each CSV are (`datamodules.py:406-408`):
```
SMILES,Target Sequence,Label
```
Plus an unnamed leading integer index column (the loader passes `index_col=0`, `header=0` — `datamodules.py:393-397`). DAVIS/BindingDB CSVs as shipped also include `drug_encoding`, `target_encoding`, `uniprot_id` but only the three named columns are read. For `*_foldseek.csv` variants the `Target Sequence` is a SaProt structure-aware string (residue + 3Di tokens interleaved, e.g. `"M#L#K#F..."` — verified in `data/DAVIS/train_foldseek.csv`).

README lines 193-203 documents the schema for custom training: `SMILES,Target Sequence,Label` where `Label` is `0/1`. CSVs must be renamed `*_foldseek.csv` when using SaProt and contain structure-aware tokens.

For `MergedDataset` (`datamodules.py:1170-1297`) the TSVs use a different schema: `ligand` (SMILES-row key into `id_to_smiles.npy`) and `aa_seq` (UniProt key into `id_to_sequence.npy` / `id_to_saprot_sequence.npy`). Labels are implicit (one TSV per polarity, negatives are subsampled per epoch — `update_epoch_data`, `datamodules.py:1245-1251`).

### Where the split lives (file:line)

The split is materialized as separate files on disk; the loader never re-shuffles within the training run.

- `DTIDataModule` reads the three CSVs by literal filename:
  - `ultrafast/datamodules.py:402-404` sets `self._train_path = Path("train.csv")`, `val.csv`, `test.csv`.
  - `ultrafast/datamodules.py:418-420` swaps to `train_foldseek.csv` / `val_foldseek.csv` / `test_foldseek.csv` when the target featurizer is SaProt.
  - `ultrafast/datamodules.py:459-461` (`setup`) opens them with `pd.read_csv(self._data_dir / self._train_path, ...)`.
- `MergedDataset` reads pre-split TSVs:
  - `ultrafast/datamodules.py:1226-1227`: `pd.read_csv(f'data/MERGED/huge_data/merged_pos_uniq_{split}_rand.tsv', sep='\t')` (and the `_neg_` companion).
- The `--ship-model` path concatenates all three partitions and then drops proteins MMseqs2-similar to LIT-PCBA targets — `datamodules.py:1213-1233`.

There is no in-memory `train_test_split` for the standard DTI datasets; the only random split call is in `EnzPredDataModule` (not used by SPRINT). `MergedDataModule.__init__` accepts `test_size`, `val_size`, `random_state` (`datamodules.py:1309-1311`) but they are ignored — the dataset routes through the precomputed `_{split}_rand.tsv` files.

## Adaptation plan

### Inject our v2 clean split

The cleanest option is the **`custom` task path**: it is the only flag value that is already wired in (`train.py:60`, `datamodules.py:53`) and points at `./data/custom/`, which does not exist in the shipped repo (we create it). Drop-in steps:

1. From the v2 parquet `(example_id, partition)` and the original interaction table for the dataset under audit (e.g. DAVIS), materialize three CSVs with exactly these columns: `,SMILES,Target Sequence,Label` (leading unnamed integer index). For SaProt runs, the `Target Sequence` column must contain SaProt-style strings ("residue#3di_token" interleaved); the easiest way is to copy the relevant rows from the shipped `*_foldseek.csv` keyed by `(SMILES, uniprot_id)` and rename to `train_foldseek.csv`, `val_foldseek.csv`, `test_foldseek.csv`.
2. Place them in `D:/hoangpc/_audit_targets/SPRINT/data/custom/`.
3. Run:
   ```
   ultrafast-train --exp-id custom_v2 --task custom \
       --config configs/saprot_agg_config.yaml --model-size large
   ```

For the merged-dataset path, the equivalent operation is to write `merged_pos_uniq_{train,val,test}_rand.tsv` and `merged_neg_uniq_{train,val,test}_rand.tsv` matching the schema (`ligand`, `aa_seq`) — these are pre-keyed into the LMDBs, so any clean split must reuse the original ligand/protein IDs already in `id_to_smiles.npy` / `id_to_saprot_sequence.npy`. This is heavier; recommend keeping merged for the paper-split anchor only.

### Estimated diff

Zero source-file edits needed for the recommended path. Concretely:
- New dir: `data/custom/` with `train_foldseek.csv`, `val_foldseek.csv`, `test_foldseek.csv` (and plain `*.csv` if running non-SaProt featurizers).
- Optional: a thin Python writer in `VS-LeakKG_v2` that joins the v2 parquet against the source dataset's interaction table and emits SPRINT-shaped CSVs.

If we want to inject a parquet directly (no CSV materialization), we'd subclass `DTIDataModule` and override `setup` (`datamodules.py:458-502`) to read parquet — net ~30 lines, but the CSV route is cheaper and leaves the upstream code untouched.

### Risks

- **Token alignment for SaProt sequences.** The 3Di tokens in `*_foldseek.csv` are precomputed from AF2 structures; they must match the same UniProt as our v2 example. Map by `uniprot_id` from the shipped CSVs rather than regenerating with foldseek. If a v2 example uses a protein not in any shipped CSV, we'd need to run `utils/structure_to_saprot.py`, which requires `foldseek` and a PDB file.
- **Implicit dependency on Morgan canonicalization.** `MorganFeaturizer.smiles_to_morgan` calls `canonicalize(smile)` (`featurizers.py:456`). v2 SMILES should already be canonical, but verify pre-injection.
- **LMDB cache reuse.** `DTIDataModule.prepare_data` writes `Morgan.lmdb` / `SaProt.lmdb` to `data/custom/`. If a re-run uses a different drug/target set, delete the lmdb directories first or the featurizer skips re-encoding (`datamodules.py:430-432`).
- **`index_col=0`.** All CSVs must have an unnamed leading integer index column or `pd.read_csv` will misinterpret `SMILES` as the index.
- **wandb side-effects.** Defaults call `wandb.init`; pass `--no-wandb` (`train.py:84`) for headless audit runs.

## Resource estimate

- **Compute graph**: SaProt-650M (~650M params) runs only inside `SaProtFeaturizer._transform` under `torch.no_grad()` (`featurizers.py:702-704`) and is set to `model.eval()` (`featurizers.py:692`). It is NOT part of the optimizer parameter list — the trainer only sees `DrugTargetCoembeddingLightning` (`model.py:43`), whose trainable parameters are `drug_projector` + `target_projector` (`model.py:70-89`). For `--model-size large` these are `LargeDrugProjector` + `LargeProteinProjector` (`model.py:82-84`); for `small` they are `nn.Linear(2048, 1024)` + `Learned_Aggregation_Layer + nn.Linear`. Approx trainable parameter count: a few million (small) to tens of millions (large) — orders of magnitude smaller than SaProt.
- **GPU memory**: dominated by SaProt forward at featurization time (`batch_size=16` by default in `SaProtFeaturizer.__init__`, `featurizers.py:670`; max sequence len 1024). On A100-40GB this fits with headroom. Training itself is tiny (batch 64, embedding dim 1024).
- **Featurization is one-shot** and cached to LMDB (`datamodules.py:472-476`), so the SaProt cost is paid once per dataset; subsequent epochs are CPU+small-MLP.
- **Epoch count**: config default `epochs: 250` (`configs/saprot_agg_config.yaml:19`). LIT-PCBA recipe uses `--epochs 15` (README:69-73). Optimizer is Adam with `CosineAnnealingWarmRestarts(T_0=10)` (`model.py:165-168`).
- **Training time** (estimated, single A100-40GB):
  - DAVIS, SaProt featurization of unique sequences (small dataset): ~5-15 min.
  - DAVIS, 250 epochs @ batch 64 over ~2k train rows: <1 GPU-hour.
  - BindingDB / BIOSNAP (~13k-19k train rows): 2-4 GPU-hours.
  - MERGED with `--epochs 15 --ship-model` (millions of pos+neg rows, MMseqs2 filter): 8-20 GPU-hours per run is plausible based on dataset scale; the paper claims fast training because the SaProt backbone is frozen.
- **Recommended budget**: ~5 A100-hours per single-dataset retrain (DAVIS/BIOSNAP/BindingDB) end-to-end including featurization. For MERGED (the LIT-PCBA Table 2 anchor): ~20 A100-hours per run. Two runs (paper split + clean split) per dataset.

## Reproducibility check

README explicitly ties the headline result to LIT-PCBA Table 2 (line 108): "Links to download pre-trained models used for Lit-PCBA evaluation in Table 2 are in `checkpoints/README.md`." `checkpoints/README.md` (lines 2-3) states: "Both of the models below were trained on the MERGED datasets with 90% sequence similarity split to the Lit-PCBA sequences (Models in Table 2)." The reproduction command (README:71-73) is:
```
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml --epochs 15 --ship-model --model-size large
--target-protein-id "all" --similarity-threshold 0.9
```
No numeric AUROC/EF1% is quoted in the README itself — the number lives in the paper, Table 2. README does NOT specify replicate count; `replicate: 0` is the default in the YAML (`saprot_agg_config.yaml:25`).

## License

MIT (`LICENSE`, 2024 — Andrew McNutt, Abhinav Adduri, Caleb Ellington, Monica Dayao). Code redistribution is unrestricted under MIT.

Data redistribution is NOT addressed by the LICENSE file. The shipped CSVs include BindingDB, BIOSNAP, DAVIS slices (rederived from the public DTI benchmarks used by ConPLex / MolTrans). MERGED data is hosted on Google Drive (gdown) by the authors. For our audit framework we should cite the original benchmarks and avoid republishing the CSVs verbatim.

## Open questions

1. **Are v2 example IDs sufficient to materialize SPRINT CSVs?** The v2 parquet is `(example_id, partition)`. We need a side-table that maps `example_id -> (SMILES, Target Sequence, Label, uniprot_id)` for the dataset under audit so we can produce SPRINT-shaped CSVs. Where does this side-table live in `VS-LeakKG_v2/src/vsleakkg/v2/`?
2. **SaProt 3Di token availability for new proteins.** If our clean split surfaces UniProts not present in any shipped `*_foldseek.csv`, we need their AF2 structure + foldseek to generate the structure-aware sequence. Does the audit ever introduce out-of-distribution proteins, or do we always remain within the shipped CSV universe?
3. **Which partition does SPRINT's checkpoint-selection rely on?** `ModelCheckpoint` monitors `val/aupr` (`train.py:210, 358-364`). Our clean split needs to maintain a non-empty val partition with both pos and neg labels, or training crashes silently. Confirm v2 split contract.
4. **MERGED swap.** For LIT-PCBA Table 2 reproduction we must regenerate MERGED's pos/neg TSVs under our clean split. Is this in scope for Phase 2 or do we restrict the SPRINT retrain to DAVIS/BIOSNAP/BindingDB?
5. **Replicate count.** Paper numbers presumably aggregate across seeds; current config sets `replicate: 0`. Do we re-run with multiple `--r` values per split?
