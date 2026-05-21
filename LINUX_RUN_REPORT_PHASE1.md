# VS-LeakKG v2 Phase 1 run report (2026-05-22T09:39:25)

## Status

| Step | Result |
|------|--------|
| [D1] v2 graph rebuild (4 corpora) | OK (4 files) |
| [P2.1] hydrate side-table | OK (1 files) |
| [D4] clean splits (per regime/corpus) | OK (4 files) |
| [D5] three-way contamination matrices | OK (14 files) |
| [D6] data-only baselines | OK (14 files) |
| [D8] Tables 1/2/5 + Figure 2 | OK (1 files) |
## Headline numbers

| corpus   | regime   | feasible   |   n_groups |   rho_max |   size_train |   size_val |   size_test |   baseline_auroc |   baseline_auprc |
|:---------|:---------|:-----------|-----------:|----------:|-------------:|-----------:|------------:|-----------------:|-----------------:|
| dekois   | ligand   | True       |      47592 |   0.0003  |        35000 |       7500 |        7500 |         0.929737 |      0.66965     |
| dekois   | scaffold | True       |      37723 |   0.00162 |        35000 |       7500 |        7500 |         0.926959 |      0.658798    |
| dekois   | protein  | True       |         81 |   0.0136  |        21511 |      14188 |       14301 |         0.862992 |      0.26533     |
| dekois   | pocket   | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| dekois   | assay    | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| dekois   | dual     | True       |      47592 |   0.0003  |        35000 |       7500 |        7500 |         0.929737 |      0.66965     |
| dekois   | strict   | True       |      50000 |   2e-05   |        35000 |       7500 |        7500 |         0.965337 |      0.733328    |
| dude     | ligand   | True       |      49642 |   6e-05   |        35000 |       7500 |        7500 |         0.937967 |      0.610489    |
| dude     | scaffold | True       |      35258 |   0.00266 |        35000 |       7500 |        7500 |         0.922971 |      0.551811    |
| dude     | protein  | True       |        102 |   0.03758 |        19097 |      15761 |       15142 |         0.879007 |      0.324831    |
| dude     | pocket   | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| dude     | assay    | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| dude     | dual     | True       |      49642 |   6e-05   |        35000 |       7500 |        7500 |         0.937967 |      0.610489    |
| dude     | strict   | True       |      50000 |   2e-05   |        35000 |       7500 |        7500 |         0.957059 |      0.679455    |
| litpcba  | ligand   | True       |          0 |   0       |        35000 |       7500 |        7500 |         0.652475 |      0.00936859  |
| litpcba  | scaffold | True       |          0 |   0       |        35000 |       7500 |        7500 |         0.627067 |      0.00565694  |
| litpcba  | protein  | True       |          0 |   0       |        19808 |      16384 |       13808 |         0.428564 |      0.000941395 |
| litpcba  | pocket   | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| litpcba  | assay    | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| litpcba  | dual     | True       |          0 |   0       |        35000 |       7500 |        7500 |         0.559077 |      0.00849346  |
| litpcba  | strict   | True       |          0 |   0       |        35000 |       7500 |        7500 |       nan        |    nan           |
| pdbbind  | ligand   | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| pdbbind  | scaffold | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| pdbbind  | protein  | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| pdbbind  | pocket   | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| pdbbind  | assay    | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| pdbbind  | dual     | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |
| pdbbind  | strict   | False      |          0 |   0       |            0 |          0 |           0 |       nan        |    nan           |

## Deferred to Phase 2

- [D2] Per-model TrainSet manifest ingestion
- [D3] Mode B audits per model
- [D6] model-paired baselines (dummy-receptor)
- [D7] ConGLUDe retraining
- Table 3, Table 4, Figure 1 (model-dependent)
- Pocket-similarity edges (need ESM-IF1 encoder)
- Time-overlap edges (need ChEMBL dates)

## Linux-compat notes

- Used /usr/pkg/bin/python (system 3.12.13) + pip --user (polars, pulp) because the conda env at /vol/grid-solar (98% full) was unusably slow.
- build_graph.py: switched from scan_parquet+5x .collect() to eager read_parquet to avoid repeated NFS rereads.
- run_phase1.sh capped --sample-examples=50000 per corpus so the pipeline completes under heavy box load (~load 24+). Remove for a full production sweep.
