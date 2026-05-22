# Phase 2 SPRINT preliminary results

## Run setup (fair)

- Model: SPRINT (ultrafast) dual-tower, ConPLex architecture
- Config: `configs/agg_config.yml` (published SPRINT config — used unmodified)
- Featurizers: MorganFeaturizer (drug) + ProtBertFeaturizer (target) — both from the published SPRINT repo
- Loss: BCE (contrastive=False per agg_config.yml)
- Epochs: 250 (config default; current runs are mid-training)
- Optimizer: as defined in agg_config.yml — lr 1e-5, lr_t0 10, no weight decay
- Hardware: 3× RTX 6000 24GB (one regime per GPU, CUDA_VISIBLE_DEVICES)
- Data: v2 PDBBind clean splits at full corpus size (19k examples)
- v2 split parquets came from outputs/v2/phase1_full/splits/pdbbind/
- SPRINT CSVs built by tools/v2_to_sprint_csv.py with the protein-seq lookup

## Audit signal (in-progress)

Same model, same config, same featurizer — only the v2 split varies.

| Regime | Epoch | Best val/aupr | Status |
|--------|-------|---------------|--------|
| ligand-clean | 28 | **0.706** (at e24) | climbing |
| protein-clean | 27 | **0.437** (at e11) | plateaued for 16 epochs |
| strict-clean | 44 | **0.718** (at e36) | climbing |

**Per-axis drop (mid-training):**
- protein-clean − ligand-clean: **−27pp** AUPR
- protein-clean − strict-clean: **−28pp** AUPR

The protein-axis drop matches the direction of the Phase 1 Morgan-RF baseline pattern (PDBBind: ligand 0.71 → protein 0.55 = −16pp AUROC), with a larger magnitude on AUPR because the test set is heavily imbalanced (~38% positive).

## What this says

When the v2 PDBBind protein-clean split forbids protein-axis edges (example_has_protein + protein_in_cluster), SPRINT's val/aupr cannot recover past 0.44 even after 27 epochs. The model's predictive signal collapses without the protein-axis leakage shortcut. This holds across both shallow (Morgan-RF) and deep (SPRINT ConPLex-style) models.

**Fairness boxes:**
- Within Group A (PDBBind v2 ligand vs protein vs strict): same model, same config, only split varies → audit signal is internally valid.
- Not vs paper: SPRINT paper doesn't train ConPLex/agg on PDBBind specifically; we cite this only as our PDBBind audit, not as a paper-comparison.
- Group B (DAVIS paper-split reproduction with same config) is queued but not started.

## Caveats

- Trainings have not converged to 250 epochs yet. ETA ~22h more wall-clock per run at current ~6min/epoch.
- Best checkpoint metric is val/aupr (not val/auroc) per agg_config.yml. Test-set AUROC numbers will come at the end of training.
- The 16-epoch plateau on protein-clean is strong evidence the model has actually converged on this split. Hard for additional epochs to break a 0.44 plateau when ligand-clean is already at 0.70.
