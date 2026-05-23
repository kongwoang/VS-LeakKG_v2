# Resume checkpoint — 2026-05-24 01:45 NZST

## Where things stand

| Result | Status | Number |
|---|---|---|
| Morgan-RF random control | done | AUROC 0.8058 |
| Morgan-RF ligand-clean | done | AUROC 0.7070 |
| Morgan-RF protein-clean | done | AUROC 0.5549 |
| Morgan-RF dual-clean | done | AUROC 0.6788 |
| SPRINT random control | done | **AUROC 0.8370** (AUPR 0.88) |
| SPRINT ligand-clean | done | AUROC 0.7619 |
| SPRINT protein-clean | done | AUROC 0.5890 |
| SPRINT dual-clean | done | AUROC 0.7306 |
| DrugCLIP retrain on v2 | **abandoned** | mode collapse — corpus too small |
| DrugCLIP paper ckpt, ligand | done | per-pocket AUROC 0.560 |
| DrugCLIP paper ckpt, protein | done | per-pocket AUROC 0.561 |
| DrugCLIP paper ckpt, dual | done | per-pocket AUROC 0.574 |
| DrugCLIP paper ckpt, random | **building LMDB** | — |

## DrugCLIP story (final)

Two attempts, neither produced a clean Group-A row but both are
informative audit findings:

**Attempt 1 — retrain on v2 train splits**. Mode collapse by epoch 2.
First chain (batch=8, update-freq=1) had gnorm→0; switched to
update-freq=6 (effective batch=48 to match paper) restored gradient flow
but valid_bedroc still plateaued — root cause is **corpus size, not
optimizer**. Paper trains on ~100K+ HomoAug-augmented positives; our v2
train splits ship ≤10K positives. Contrastive in-batch-softmax can't
converge on this scale.

**Attempt 2 — paper checkpoint zero-shot**. Downloaded the published
`checkpoint_best.pt` (1.18 GB; trained on full PDBBind 2020 + HomoAug)
and ran a per-pocket retrieval AUROC on each v2 test split. Result is
essentially flat across regimes (0.56-0.57). Read this as **train-test
contamination dominating** — the paper's training corpus IS PDBBind
2020+HomoAug, our v2 test splits are subsets. The v2 leakage filter does
not separate the published model from its in-domain performance.

Both findings are documented in `AUDIT_FINAL.md` under the
"DrugCLIP — third-model attempt" section. The audit headline (Morgan-RF
+ SPRINT ~25pp drop on protein-clean) is unchanged.

## Pending: random-control LMDB for DrugCLIP paper-ckpt

Random-split DrugCLIP LMDB build is in flight on VUW:
- pid: 1969319 (and 16 mp workers)
- log: `/vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/lmdb_build_random.log`
- ETA: ~15 min total (5 min train + 4 min valid + 4 min test)

When `data/v2_pdbbind_random/test.lmdb` lands, run:
```bash
ssh kongwoang "ssh VUW '
ln -sfn ../dict_mol.txt /vol/dl-nguyenb5-solar/users/hoangpc/DrugCLIP/data/v2_pdbbind_random/dict_mol.txt
ln -sfn ../dict_pkt.txt /vol/dl-nguyenb5-solar/users/hoangpc/DrugCLIP/data/v2_pdbbind_random/dict_pkt.txt
bash /vol/dl-nguyenb5-solar/users/hoangpc/eval_drugclip_paperckpt_retrieval.sh 1
'"
```
(retrieval script currently runs ligand/protein/dual — extend to include
random by editing the for-loop, or run the random one directly).

Then plug random numbers into both DrugCLIP tables in AUDIT_FINAL.md.

## What's running now

| Process | Purpose | nohup pid |
|---|---|---|
| v2_to_drugclip_lmdb random | builds train/valid/test.lmdb for random split | 1969319 |

GPU 2 is idle (no training in flight). GPU 0/1 are someone else's.

## What's already done and committed

| File | Purpose |
|---|---|
| `AUDIT_FINAL.md` | Unified audit deliverable with DrugCLIP findings + Morgan-RF + SPRINT all 4 rows incl. random control |
| `PHASE2_SPRINT_FINAL.md` | SPRINT details + fairness statement |
| `PHASE1_FINAL_REPORT.md` | Phase 1 multi-corpus baseline |
| `tools/build_random_pdbbind_split.py` | random split builder |
| `tools/run_morgan_rf_random.py` | Morgan-RF on random control |
| `tools/v2_to_drugclip_lmdb.py` | DrugCLIP LMDB builder for v2 splits |
| `PHASE2_DRUGCLIP_BLOCKER.md` | Original C-ABI blocker (resolved by pinned drugclip_env) |

## On VUW

Custom artifacts (training infra, not in repo):
- `launch_drugclip.sh` — train launcher (update-freq=6, the principled
  paper-recipe batch but doesn't fix the corpus size)
- `chain_drugclip.sh` — sequential ligand→protein→dual (deprecated, the
  retrain path was abandoned)
- `eval_drugclip.sh` — flat-AUROC eval launcher
- `eval_drugclip_paperckpt.sh` — flat-AUROC across all 3 splits for paper ckpt
- `eval_drugclip_paperckpt_retrieval.sh` — per-pocket retrieval AUROC for paper ckpt
- `DrugCLIP/eval_drugclip_v2.py` — flat AUROC evaluator
- `DrugCLIP/eval_drugclip_v2_retrieval.py` — per-pocket retrieval evaluator
- `SPRINT/run_test_only.py` — SPRINT weights_only=False patch
- Mirror copies live under `D:\hoangpc\VS-LeakKG\.tmp\` on the dev box

Paper-checkpoint cache: `/vol/dl-nguyenb5-solar/users/hoangpc/drugclip_data/paper_ckpt/drugclip_data/checkpoint_best.pt` (1.18 GB, from gdown).

## Envs

- `drugclip_env`: torch 2.4.0+cu121, numpy 1.26.4, unicore from Uni-Core source. Path: `/vol/dl-nguyenb5-solar/users/hoangpc/envs/drugclip_env`.
- `vsleak2`: torch 2.12 + numpy 2.4 (SPRINT + v2 baselines + LMDB build). Path: `/vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2`.
