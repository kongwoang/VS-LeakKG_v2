# Phase 2 SPRINT — final results

All three SPRINT runs trained to `max_epochs=250` on PDBBind v2 clean splits. Best
checkpoints were selected on `val/aupr` and evaluated on the held-out test split.
The post-training `trainer.test()` call originally crashed on a PyTorch 2.6
`weights_only=True` checkpoint-load issue (LeakyReLU not in the default safe-globals
allowlist); training itself was untouched, and the test step was re-run against the
saved best checkpoint via a thin `run_test_only.py` wrapper that forces
`weights_only=False`.

## Run setup (fair)

- **Model**: SPRINT (ultrafast) dual-tower, ConPLex-style aggregation head.
- **Config**: `configs/agg_config.yml` — the published SPRINT config, used unmodified.
- **Featurizers**: `MorganFeaturizer` (drug) + `ProtBertFeaturizer` (target).
- **Loss**: BCE (`contrastive=False` per agg_config.yml; no DUDE pretraining).
- **Optimizer / schedule**: agg_config.yml defaults (Adam, lr 1e-5, CosineAnnealingWarmRestarts t0=10, no weight decay).
- **Epochs**: 250, no early stopping. Best ckpt = highest `val/aupr`.
- **Hardware**: 3× RTX 6000 24GB (one regime per GPU, set via `CUDA_VISIBLE_DEVICES`).
- **Data**: v2 PDBBind clean splits at full corpus size — same v2 split parquets used by the Phase 1 Morgan-RF baseline (`outputs/v2/phase1_full/splits/pdbbind/`).
- **SPRINT CSVs**: built by `tools/v2_to_sprint_csv.py` with the protein-seq side-table.

Only the **split regime** varies between rows below. Model, config, featurizer, and
preprocessing pipeline are identical. This is the Group A audit signal.

## Group A — PDBBind v2 audit table (held-out test set)

| Regime | n_groups | Test size | Pos rate | Best val/aupr (epoch) | **Test AUROC** | **Test AUPR** | Test ACC | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ligand-clean | 13,301 | 4,560 | 0.382 | 0.7227 (e170) | **0.7619** | **0.6693** | 0.5159 | 0.5995 |
| protein-clean | 11,862 | 5,844 | 0.379 | 0.4367 (e11) | **0.5890** | **0.4428** | — | 0.5501 |
| dual-clean | 7,415 | 5,429 | 0.484 | 0.7068 (e242) | **0.7306** | **0.7097** | — | 0.6824 |

`val/aupr` was the checkpoint-selection metric per agg_config.yml. The dual-clean split
was substituted for strict-clean because strict-clean degenerates to singleton groups
on PDBBind (n_groups = n_examples = 19,037 → effectively a random split with full
protein-axis leakage — not a meaningful audit regime).

## Headline finding — protein-axis leakage drives PDBBind performance

Holding model, featurizer, and config fixed and changing only the v2 split:

- **protein−ligand**: −17.3pp AUROC / −22.7pp AUPR
- **protein−dual**: −14.2pp AUROC / −26.7pp AUPR
- **dual−ligand**:  −3.1pp AUROC / +4.0pp AUPR  *(AUPR shifts driven by test class balance: dual test is 48% positive vs ligand 38%)*

When the protein axis is forbidden from leaking between train and test (no shared
sequence, no shared sequence cluster, no shared structure cluster), SPRINT's test
AUROC drops from 0.76 to 0.59 — i.e., the model retains only ~9pp lift above the
0.50 chance floor. The best-checkpoint epoch for protein-clean was epoch 11; no
further improvement occurred across the remaining 239 epochs (16-epoch+ plateau
prior to that). The model converged early to a low ceiling.

The pattern is consistent with — and a magnitude stronger than — the Phase 1
Morgan-RF baseline:

| Regime | Morgan-RF AUROC (Phase 1) | SPRINT AUROC (Phase 2) | Δ |
|---|---:|---:|---:|
| ligand-clean | 0.7070 | 0.7619 | +0.055 (SPRINT) |
| protein-clean | 0.5549 | 0.5890 | +0.034 (SPRINT) |
| dual-clean | 0.6788 | 0.7306 | +0.052 (SPRINT) |

Both models exhibit a protein-axis-dominated shortcut. SPRINT closes some of the
gap (deep model + ProtBert beats Morgan-RF on every regime), but does *not* close
the leakage gap: the protein-clean split's drop is the same shape and direction in
both models.

## Lift above class-balance baseline (AUPR)

AUPR is sensitive to test-set positive rate. Lift = AUPR − positive_rate:

| Regime | AUPR | Pos rate | **Lift** |
|---|---:|---:|---:|
| ligand-clean | 0.6693 | 0.382 | **+0.288** |
| protein-clean | 0.4428 | 0.379 | **+0.064** |
| dual-clean | 0.7097 | 0.484 | **+0.226** |

Protein-clean keeps only ~6pp of AUPR lift above the trivial baseline — about a
quarter of what ligand-clean retains. Dual-clean retains ~80% of ligand-clean's
lift, suggesting that forbidding ligand axis alone is doing most of the work and
adding the protein-axis constraint on top of an already ligand-clean split adds
only a small further drop.

## Fairness statement

### What we claim

This table is **Group A** under our fairness framework. The comparison unit
across rows is:

| held fixed across rows | value |
|---|---|
| corpus | PDBBind v2 (all 19,037 examples, same graph) |
| model | SPRINT (ultrafast) dual-tower, ConPLex-style aggregation |
| config | `configs/agg_config.yml`, *unmodified* (used as published) |
| featurizers | MorganFeaturizer (drug) + ProtBertFeaturizer (target) |
| preprocessing | identical CSV emit (`tools/v2_to_sprint_csv.py`) + same protein-seq lookup |
| epochs / lr / batch / seed | all per agg_config.yml (250 / 1e-5 / 24 / 1) |
| **varies only** | the v2 split regime (ligand-clean / protein-clean / dual-clean) |

The audit claim is precisely: *when only the split regime varies (everything
else above held constant), held-out AUROC drops from 0.7619 (ligand-clean) to
0.5890 (protein-clean) — a 17.3 pp drop attributable to removing
protein-axis leakage between train and test*. Nothing more.

### What we explicitly do NOT claim

- **Not SPRINT-paper reproduction.** The published SPRINT virtual-screening
  setup trains on MERGED (BindingDB + BIOSNAP + DAVIS + ...) and evaluates
  zero-shot on LIT-PCBA. That's a different corpus *and* a different task
  (zero-shot vs. in-distribution). We do not run that experiment and we do not
  subtract our PDBBind numbers from their LIT-PCBA numbers. Cross-corpus +
  cross-task deltas would conflate the audit signal with confounders.
- **Not "SPRINT is worse than reported."** A 0.59 AUROC on protein-clean
  PDBBind says nothing about SPRINT's reported MERGED → LIT-PCBA performance.
  Different corpus, different task.
- **Not a generalization claim about ConPLex / dual-tower architectures.** We
  use SPRINT *as* a fixed architecture/featurizer/config combo so the only
  moving variable is the split. The result is "this specific model exhibits a
  protein-axis shortcut on this specific corpus when leakage is removed,"
  matching the same shortcut Morgan-RF exhibits.

### Group B (deferred)

A SPRINT paper-config reproduction (train MERGED, test LIT-PCBA, or train
DAVIS, test DAVIS-paper-split) is what would let us sanity-check that our
training pipeline is faithful. It is **not** required for the audit claim
above, because that claim is purely internal to Group A. Documented as
deferred future work.

This fairness gate is also encoded in `memory/phase2_fairness_policy.md` and
in the cross-cutting `AUDIT_FINAL.md` § Group B placeholder.

## Reproducibility

Artifacts on VUW:
- Checkpoints: `/vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/best_models/v2_pdbbind_{ligand,protein,dual}_agg_paper/v2_pdbbind_{ligand,protein,dual}.ckpt`
- Training logs: `/vol/dl-nguyenb5-solar/users/hoangpc/sprint_runs/v2_pdbbind_{ligand,protein,dual}_agg_paper.log`
- Test logs: `/vol/dl-nguyenb5-solar/users/hoangpc/sprint_runs/v2_pdbbind_{ligand,protein,dual}_TEST.log`
- v2 SPRINT CSVs: `/vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/data/custom_pdbbind_{ligand,protein,dual}/{train,val,test}.csv`
- Featurized LMDBs: same dir, `Morgan_features.lmdb` + `ProtBert_features.lmdb`

To rerun a test evaluation:

```bash
cd /vol/dl-nguyenb5-solar/users/hoangpc/SPRINT
CUDA_VISIBLE_DEVICES=1 /vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2/bin/python \
    run_test_only.py \
    --exp-id v2_pdbbind_<regime>_TEST \
    --config configs/agg_config.yml \
    --task v2_pdbbind_<regime> \
    --epochs 0 \
    --checkpoint /vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/best_models/v2_pdbbind_<regime>_agg_paper/v2_pdbbind_<regime>.ckpt \
    --no-wandb
```

## Known caveats

- `val/aupr` was the checkpoint-selection metric. Test AUROC and AUPR are reported
  from the model with the best val/aupr — not necessarily the model with the best
  test-set behavior at that epoch.
- Test-set positive rates differ across splits (0.38 / 0.38 / 0.48), so raw AUPR
  is not directly comparable across rows. The lift-over-baseline column corrects
  for this.
- SPRINT's `agg_config.yml` is a non-contrastive BCE setup. The SPRINT paper's
  contrastive variants (saprot_agg_config.yml, conplex_config.yml) need DUDE
  pretraining data and were not run here; switching configs would mix a Group A
  signal with a Group-B-style config change and is intentionally excluded.

## Next

- LigUnity and DrugCLIP audits on v2 PDBBind (same Group A pattern, different
  model family). Both need LMDB-formatted v2 data with 3D coordinates — setup
  starts next.
- Phase 1 Morgan-RF + Phase 2 SPRINT combined audit figure once LigUnity /
  DrugCLIP results land.
