# VS-LeakKG v2 Phase 1 review

## C1-C2 — per-corpus graph outputs

| corpus | nodes | edges | stats | n_nodes | n_edges | fail |
|--------|-------|-------|-------|---------|---------|------|
| pdbbind | ✓ (1899 kB) | ✓ (549 kB) | ✓ (1 kB) | 55201 | 19166 |  |
| dekois | ✓ (7540 kB) | ✓ (7698 kB) | ✓ (1 kB) | 245634 | 468236 |  |
| dude | ✓ (89336 kB) | ✓ (108017 kB) | ✓ (1 kB) | 3047091 | 6882433 |  |
| litpcba_ave | ✓ (49162 kB) | ✓ (114127 kB) | ✓ (1 kB) | 3138734 | 8325167 |  |

## C3 — side-table

- exists: True
- rows: 1710073
- sources: {"bayesbind": 21037, "dekois": 88152, "dude": 1196111, "litpcba": 404773}
- **FAIL**: missing sources: ['chembl', 'bindingdb', 'pdbbind']

## C4-C7 — pipeline per-corpus

| corpus | summary | regime_count | feasible | fail |
|--------|---------|--------------|----------|------|
| pdbbind | ✓ (0 kB) | 7 | 0 |  |
| dekois | ✓ (0 kB) | 7 | 5 |  |
| dude | ✓ (0 kB) | 7 | 5 |  |
| litpcba | ✓ (0 kB) | 7 | 5 | AUROC out of [0,1]: [('pocket', 'nan'), ('assay', 'nan'), ('strict', 'nan')]; missing VC for feasible regimes: ['strict'] |

## C8 — final tables + figure

- table1: ✓ (1 kB)
- table2: ✓ (1 kB)
- table5: ✓ (3 kB)
- figure2: ✓ (40 kB)

## Summary

- ✗ **2 critical failure(s)**:
  - side-table: missing sources: ['chembl', 'bindingdb', 'pdbbind']
  - pipeline[litpcba]: AUROC out of [0,1]: [('pocket', 'nan'), ('assay', 'nan'), ('strict', 'nan')]; missing VC for feasible regimes: ['strict']