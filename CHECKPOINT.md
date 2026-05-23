# Resume checkpoint — 2026-05-24 01:05 NZST

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
| DrugCLIP ligand-clean | training | best_valid_bedroc 0.154 @ epoch 1 |
| DrugCLIP protein-clean | chain-queued | — |
| DrugCLIP dual-clean | chain-queued | — |

## DrugCLIP retrain — what happened

The first chain (batch=8, update-freq=1) hit **mode collapse** by epoch 2:
loss locked at 3.0 (≈ ln(8), uniform softmax), gnorm → 0.002, gradients
silently dropped, valid_bedroc plateaued at random. Root cause: in-batch
softmax with only 7 negatives per anchor (batch=8) collapses to the trivial
all-embeddings-equal solution.

Fix: `--update-freq 6` => effective batch = 48 (matches paper recipe). At
the new effective batch, gradients stay alive (gnorm 56-75) and training
loss decreases from 2.97 to 2.85 in the first two epochs. valid_bedroc
still peaks at epoch 1 (0.154) because the contrastive in-batch objective
doesn't directly optimize our binary-classification test metric — but the
saved `checkpoint_best.pt` is the right epoch to evaluate.

Broken first-attempt run dirs preserved as `_runs.broken-bs8` /
`.log.broken-bs8` for forensics.

## Running now

| Process | Purpose | GPU | nohup pid |
|---|---|---|---|
| `chain_drugclip.sh 2 50 8` | Auto-launches ligand → protein → dual sequentially (update-freq=6, effective batch 48) | 2 | 1895884 |
| DrugCLIP ligand training | inside the chain | 2 | 1895893 |

ETA: ~140 min/regime × 3 regimes = ~7h total. Expect chain to finish
~08:00 NZST.

## Verify everything is alive

```bash
ssh kongwoang "ssh VUW '
ps -p 1895884 -o pid,etime,stat,cmd 2>/dev/null
ps -p 1895893 -o pid,etime,stat,cmd 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv'"
```

## Resume — verification command

```bash
ssh kongwoang "ssh VUW '
echo === DrugCLIP regime status ===
for r in ligand protein dual; do
    echo --- \$r ---
    grep -E \"valid_bedroc|done training\" /vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/v2_pdbbind_\${r}.log 2>/dev/null | tail -5
    ls -la /vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/v2_pdbbind_\${r}_runs/checkpoint_best.pt 2>/dev/null
done
echo === Chain log ===
tail -10 /vol/dl-nguyenb5-solar/users/hoangpc/drugclip_runs/chain.log'"
```

## When all 3 runs finish — test eval

The DrugCLIP custom test evaluator lives at:
- Script: `DrugCLIP/unimol/eval_drugclip_v2.py` (computes per-row pocket·mol
  dot product → AUROC + AUPR over binary labels in test.lmdb)
- Wrapper: `/vol/dl-nguyenb5-solar/users/hoangpc/eval_drugclip.sh`

For each regime, run on GPU 1 (which is free; GPU 2 hosts the next chain):

```bash
ssh kongwoang "ssh VUW 'cd /vol/dl-nguyenb5-solar/users/hoangpc && \
    for r in ligand protein dual; do
        bash eval_drugclip.sh \$r 1
    done'"
```

Logs land at `drugclip_runs/v2_pdbbind_<regime>_TEST.log`. Each eval is
fast (~1-2 min).

## What's already done and committed

| File | Purpose |
|---|---|
| `AUDIT_FINAL.md` | Unified audit deliverable. Has Morgan-RF + SPRINT all 4 rows including random control |
| `PHASE2_SPRINT_FINAL.md` | SPRINT details + fairness statement |
| `PHASE1_FINAL_REPORT.md` | Phase 1 multi-corpus baseline |
| `tools/build_random_pdbbind_split.py` | random split builder |
| `tools/run_morgan_rf_random.py` | Morgan-RF on random control |
| `tools/v2_to_drugclip_lmdb.py` | DrugCLIP LMDB builder for v2 splits |
| `PHASE2_DRUGCLIP_BLOCKER.md` | Original C-ABI blocker (resolved by pinned drugclip_env) |

## On VUW

Custom artifacts (not in repo, lives on VUW only):
- `/vol/dl-nguyenb5-solar/users/hoangpc/launch_drugclip.sh` — train launcher (update-freq=6 patched in)
- `/vol/dl-nguyenb5-solar/users/hoangpc/chain_drugclip.sh` — sequential ligand→protein→dual
- `/vol/dl-nguyenb5-solar/users/hoangpc/eval_drugclip.sh` — test eval launcher
- `/vol/dl-nguyenb5-solar/users/hoangpc/DrugCLIP/unimol/eval_drugclip_v2.py` — custom evaluator
- `/vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/run_test_only.py` — SPRINT weights_only=False patch

Mirror copies under `D:\hoangpc\VS-LeakKG\.tmp\` for reference (these are
not in the v2 repo deliberately — they belong with the training infra, not
the audit codebase).

## Envs

- `drugclip_env` (DrugCLIP only): torch 2.4.0+cu121, numpy 1.26.4, unicore
  from github.com/dptech-corp/Uni-Core source. Path:
  `/vol/dl-nguyenb5-solar/users/hoangpc/envs/drugclip_env`.
- `vsleak2` (SPRINT and v2 baselines): torch 2.12 + numpy 2.4. Path:
  `/vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2`.

## Tracking

Task IDs of work in flight: #132 (DrugCLIP inference + optional retrain),
#138 (fp16 instability investigation; root cause = small-batch contrastive
collapse), #140 (write custom evaluator — script written, awaiting
checkpoints to run against).
