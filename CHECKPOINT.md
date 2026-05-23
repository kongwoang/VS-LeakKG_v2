# Overnight checkpoint — 2026-05-23 20:40 NZST

Three things are running unattended on VUW. The user can disconnect SSH and
return in ~24h. All processes are nohup'd and survive SSH disconnect.

## Running now

| Process | Purpose | GPU | ETA | nohup pid |
|---|---|---|---|---|
| SPRINT random control train | KG-validation control: same-size-as-protein-clean but random partition | 2 | ~24h | 1693611 |
| DrugCLIP ligand-clean train | Phase 2, second-model audit | 2 | ~6h (50 epochs, batch=8) | 1697836 |
| `chain_drugclip.sh 2 50 8` | Auto-launches DrugCLIP protein + dual sequentially after ligand | 2 | ~18h total | 1704068 |

## Verify everything is alive

```bash
ssh kongwoang "ssh VUW '
ps -p 1693611 -o pid,etime,stat,cmd 2>/dev/null
ps -p 1697836 -o pid,etime,stat,cmd 2>/dev/null
ps -p 1704068 -o pid,etime,stat,cmd 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv'"
```

## Resume tomorrow — verification command

```bash
ssh kongwoang "ssh VUW '
echo === SPRINT random ===
grep -E \"val/aupr.*reached|max_epochs.*reached|Test\" /vol/dl-nguyenb5-solar/users/hoangpc/sprint_runs/v2_pdbbind_random_agg_paper.log | tail -10
echo
for r in ligand protein dual; do
    echo === DrugCLIP \$r ===
    grep -E \"valid_bedroc|best|Test\" /vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/v2_pdbbind_\${r}.log 2>/dev/null | tail -8
    echo
done
echo === Chain log ===
tail -20 /vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/chain.log
'"
```

## When the runs finish — next steps

1. Find each best checkpoint:
   ```bash
   ssh kongwoang "ssh VUW 'ls /vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/v2_pdbbind_*_runs/checkpoint_best.pt
   ls /vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/best_models/v2_pdbbind_random_agg_paper/'"
   ```

2. **SPRINT random test eval** — same `run_test_only.py` wrapper as before
   (it's at `/vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/run_test_only.py`):
   ```bash
   ssh kongwoang "ssh VUW 'cd /vol/dl-nguyenb5-solar/users/hoangpc/SPRINT && \
     CUDA_VISIBLE_DEVICES=2 /vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2/bin/python \
       run_test_only.py \
       --exp-id v2_pdbbind_random_TEST \
       --config configs/agg_config.yml \
       --task v2_pdbbind_random \
       --epochs 0 \
       --checkpoint /vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/best_models/v2_pdbbind_random_agg_paper/v2_pdbbind_random.ckpt \
       --no-wandb'"
   ```

3. **DrugCLIP test eval** — needs a custom evaluator written (no `trainer.test()`
   equivalent in `unicore-train` — use `retrieval.py` or write a forward-pass
   AUROC script). For each regime `<r>`:
   - Load `checkpoint_best.pt`
   - Iterate `test.lmdb`, run pocket + ligand through encoders, dot-product
   - AUROC over `(score, label)` pairs

4. Append the four new rows (SPRINT random + 3 × DrugCLIP) to AUDIT_FINAL.md's
   headline table and commit.

## What's already done and committed (`936ccf0`)

- `AUDIT_FINAL.md` — random control row + leakage interpretation
- `PHASE2_SPRINT_FINAL.md` — explicit fairness statement (what we claim / don't)
- `tools/build_random_pdbbind_split.py` — random split builder
- `tools/run_morgan_rf_random.py` — Morgan-RF on random control (gave 0.8058)
- `tools/v2_to_drugclip_lmdb.py` — earlier; built LMDBs for all 3 DrugCLIP splits
- `PHASE2_DRUGCLIP_BLOCKER.md` — original blocker writeup (now resolved by
  the pinned `drugclip_env` venv, but kept as a record)

## Headline state of the audit (without SPRINT random + DrugCLIP numbers yet)

| Regime | n_test | Morgan-RF | SPRINT | DrugCLIP |
|---|---:|---:|---:|---:|
| random (control) | 5844 | **0.8058** | (training) | — |
| ligand-clean | 4560 | 0.7070 | 0.7619 | (training) |
| protein-clean | 5844 | 0.5549 | 0.5890 | (chain-queued) |
| dual-clean | 5429 | 0.6788 | 0.7306 | (chain-queued) |

Morgan-RF row alone already proves the KG works (random 0.81 vs protein-clean
0.55 = 25pp leakage gap). SPRINT row alone proves it's model-invariant.
DrugCLIP + SPRINT-random rows are confirmations, not load-bearing.

## Notes on env

- `drugclip_env` (pinned: torch 2.4.0+cu121, numpy 1.26.4, unicore from
  github.com/dptech-corp/Uni-Core source) lives at
  `/vol/dl-nguyenb5-solar/users/hoangpc/envs/drugclip_env`. Use this venv for
  any DrugCLIP work. SPRINT keeps using `vsleak2` (torch 2.12 + numpy 2.4)
  unchanged.
- DrugCLIP launch script: `/vol/dl-nguyenb5-solar/users/hoangpc/launch_drugclip.sh`
- DrugCLIP chain script: `/vol/dl-nguyenb5-solar/users/hoangpc/chain_drugclip.sh`
- LMDBs at `/vol/dl-nguyenb5-solar/users/hoangpc/DrugCLIP/data/v2_pdbbind_{ligand,protein,dual}/{train,valid,test}.lmdb`
- unimol pretrains at `/vol/dl-nguyenb5-solar/users/hoangpc/drugclip_data/pretrains/`
