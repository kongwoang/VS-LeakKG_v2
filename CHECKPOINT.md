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

### Group C — DUD-E retrieval-native audit (proof-of-protocol complete)

Paper-checkpoint zero-shot, 65/102 DUD-E targets (PDBBind-overlapping):

| Regime | n_test | AUROC mean ± std | BEDROC mean ± std |
|---|---:|---:|---:|
| target_random | 18 | 0.458 ± 0.216 | 0.112 ± 0.214 |
| target_clean  | 18 | 0.468 ± 0.247 | 0.156 ± 0.224 |
| active_clean  | 23 | 0.382 ± 0.204 | 0.091 ± 0.182 |
| dual_clean    | 23 | 0.382 ± 0.204 | 0.091 ± 0.182 |

Findings:
- Protocol works (smoke test aa2ar BEDROC=0.95)
- Per-target variance dominates (std≈0.2, range 0.12-0.86 AUROC)
- active_clean ≡ dual_clean (DUD-E scaffold/active sharing subsumes Pfam)
- 8pp regime gap is within SE — not statistically significant at n=18-23
- Honest interpretation: SUBSET-SELECTION effects, not training-time
  leakage gap (frozen paper ckpt doesn't see split filtering)

## Deferred (in priority order)

1. **Extend Group C to all 102 DUD-E targets** — fetch the 37 missing
   PDBs from RCSB + extract pockets. Quadruples per-regime n, would let
   us detect ~4pp gaps with significance.
2. **Group C on LIT-PCBA** — real assay decoys, AVE-defeated benchmark.
   Most rigorous retrieval corpus; isolates synthetic-decoy bias.
3. **Group C on DEKOIS 2.0** — stricter property-matched decoys.
4. **DrugCLIP retrain with HomoAug-augmentation** — would yield a true
   training-time leakage-gap audit on the v2 splits.
5. **LigUnity retrieval audit** — same Group C protocol.

## What's committed

| Commit | Content |
|---|---|
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
