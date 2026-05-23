# Resume checkpoint — 2026-05-24 03:35 NZST

## Where things stand

### Group A — PDBBind binary audit (final, shippable)

| Result | Status | Number |
|---|---|---|
| Morgan-RF random control | done | AUROC 0.8058 |
| Morgan-RF ligand-clean | done | AUROC 0.7070 |
| Morgan-RF protein-clean | done | AUROC 0.5549 |
| Morgan-RF dual-clean | done | AUROC 0.6788 |
| SPRINT random control | done | AUROC 0.8370 |
| SPRINT ligand-clean | done | AUROC 0.7619 |
| SPRINT protein-clean | done | AUROC 0.5890 |
| SPRINT dual-clean | done | AUROC 0.7306 |

Both models drop ~25pp from random control to protein-clean. Model-invariant.

### Group C — DUD-E retrieval-native audit (all 102 targets)

Paper-checkpoint zero-shot. Pockets: 65 from PDBBind 2020 + 37 fetched
from RCSB. All 102 targets in scope.

| Regime | n_test | AUROC mean ± std | BEDROC mean ± std |
|---|---:|---:|---:|
| target_random | 22 | 0.450 ± 0.189 | 0.132 ± 0.195 |
| target_clean  | 31 | 0.468 ± 0.174 | 0.103 ± 0.153 |
| active_clean  | 35 | 0.454 ± 0.226 | 0.075 ± 0.129 |
| dual_clean    | 35 | 0.454 ± 0.226 | 0.075 ± 0.129 |

Findings:
- AUROC flat across regimes (0.45-0.47, 2pp range)
- BEDROC ordered random > target_clean > active_clean (expected direction
  for leakage gap) but random→active = 0.057 ± 0.047 (z=1.2, p≈0.23) —
  NOT statistically significant
- Per-target variance still dominates (std 0.19-0.23, range 0.04-0.86)
- active_clean ≡ dual_clean identically (DUD-E scaffold/active sharing
  subsumes Pfam family sharing)
- Honest interpretation: SUBSET-SELECTION effects, not training-time
  leakage gap (frozen paper ckpt doesn't see split filtering)

## Deferred (in priority order)

1. **DrugCLIP retrain with HomoAug-augmentation on a leakage-clean v2
   train split** — the only path to a true *training-time* leakage gap
   audit. Frozen-checkpoint zero-shot (what we did) reveals only
   subset-selection effects.
2. **Group C on LIT-PCBA** — real assay decoys (AVE-defeated benchmark),
   15 targets only, would need a mol2→pocket pipeline (~3-4h). Same
   variance concern though; per-target n is still small.
3. **Group C on DEKOIS 2.0** — 81 targets, stricter property-matched
   decoys. Higher value than LIT-PCBA for variance reduction.
4. **LigUnity retrieval audit** — same Group C protocol, different
   model encoder.

## What's committed

| Commit | Content |
|---|---|
| 3a60d04 | [Group C+] scale DUD-E retrieval audit to all 102 targets |
| d77d843 | [Group C+] fetch_missing_dude_pockets.py — RCSB download + pocket extraction |
| be6afd0 | [Group C] retrieval-native audit tooling for DUD-E (6 scripts under tools/v2_retrieval/) |
| 3133da2 | [Group C] add retrieval-native DUD-E audit section + honest interpretation |
| 4d50ed7 | audit: fill DrugCLIP random control — confirms no leakage gap |
| a234b4d | checkpoint: refresh state after DrugCLIP attempts |
| 86a740a | audit: document DrugCLIP attempts — retrain + paper-ckpt blockers |
| 55b6fa2 | audit: SPRINT random control = 0.8370; DrugCLIP retrain in flight |

## Artifacts (on VUW)

- `outputs/v2_retrieval/graph_dude/v2_target_node.parquet` — 102 targets
- `outputs/v2_retrieval/graph_dude/v2_active_of_target.parquet` — 22,805 edges
- `outputs/v2_retrieval/graph_dude/v2_target_in_family.parquet` — 82 families
- `outputs/v2_retrieval/splits/dude/{target_random,target_clean,active_clean,dual_clean}.parquet`
- `outputs/v2_retrieval/diagnostics/dude_contamination.csv`
- `outputs/v2_retrieval/results/dude/<regime>_per_target.csv`
- `DrugCLIP/data/dude_retrieval/<target>/{pocket,mols}.lmdb` × 65 targets, 80142 mol rows
- `DrugCLIP/eval_dude_retrieval.py` — per-target retrieval evaluator
- `eval_dude_retrieval.py` driver wrapper at `/vol/.../run_dude_eval_all_regimes.sh`

## Audit is shippable on commits up to `3133da2`

The Group A binary audit headline (model-invariant 25pp leakage drop on
PDBBind protein-clean) is the primary deliverable. Group C is a
proof-of-protocol that adds a second audit track for retrieval models;
its findings are documented honestly (protocol validated, but per-target
variance + frozen-checkpoint limits prevent a true training-time gap
claim).
