# Session checkpoint — 2026-05-23 (Phase 2 SPRINT training overnight)

## Live overnight (Phase 2 SPRINT)

Three SPRINT trainings nohup'd on VUW, will run ~22-28h to completion of
the configured 250 epochs from `agg_config.yml` (unmodified, published).

| GPU | Regime | exp_id | PID | Status as of 06:14 NZST |
|-----|--------|--------|-----|--------------------------|
| 0 | ligand | v2_pdbbind_ligand_agg_paper | 970059 | epoch 33, best val/aupr 0.71 |
| 1 | protein | v2_pdbbind_protein_agg_paper | 984088 | epoch 33, best val/aupr 0.437 (plateaued since epoch 11) |
| 2 | dual | v2_pdbbind_dual_agg_paper | 1422504 | featurizing 79/742 ProtBert batches |

After each `trainer.fit()` completes, Lightning auto-runs `trainer.test()` so
the final test-set AUROC/AUPRC will be in the log.

To verify after disconnect:
```
ssh kongwoang "ssh VUW 'for r in ligand protein dual; do echo === \$r ===; grep \"val/aupr\\|Test\" /vol/dl-nguyenb5-solar/users/hoangpc/sprint_runs/v2_pdbbind_\${r}_agg_paper.log | tail -5; done'"
```

Audit signal already clear at mid-training: protein-clean val/aupr 0.44 vs
ligand-clean 0.71 (-27pp). Strict-clean killed because it degenerated to
a random split (singleton groups on PDBBind). Replaced with dual-clean
(real constraint).

---

# Session checkpoint — 2026-05-22 (UPDATED Run 2)

## Resume order on next session

1. Reconnect SSH bridge (Windows → kongwoang Tailscale → VUW).
2. Check whether the **litpcba pipeline** at PID 681123 on VUW finished. If yes, `rerun_pipeline.log` will have a `[done rerun_pipeline]` line.
3. **Critical**: rebuild PDBBind graph with `corpus=pdbbind` ONLY (commit `d4a5e1b` adds BindingMeasurement pK extraction). Then rerun pdbbind pipeline. PDBBind will go from 0/7 feasible to ~5-6/7 feasible once labels are real.
4. Regenerate `phase1_combined.csv` + tables/figures.
5. Then C (drop --sample-examples cap) — many hours, unattended.

## Run 2 progress (the new run with A+B applied)

| Step | State | Notes |
|------|-------|-------|
| `rerun_d1.sh` (A+B = rebuild all 4 graphs) | **DONE** | seq_id fix added 35586 protein_in_cluster edges per corpus; PDBBind got 19037 synthesized Examples but labels still = 0.0 (commit d4a5e1b fixes this) |
| `rerun_pipeline.sh` pdbbind | DONE (broken) | 0/7 feasible because pdbbind labels were all 0.0 (build with old code); baseline crashed with IndexError, now guarded in commit `d4a5e1b` |
| `rerun_pipeline.sh` dekois | **DONE** | 7 regimes, 5 feasible, baselines AUROC 0.86–0.97 same as Run 1 |
| `rerun_pipeline.sh` dude | **DONE** | 7 regimes, 5 feasible, baselines same as Run 1 |
| `rerun_pipeline.sh` litpcba | **RUNNING** at PID 681123 (10:53 start) | ~25 min more expected. Strict regime may OOM-page again |

## Was running at this checkpoint

- `bash /tmp/rerun_pipeline.sh` PID 664902 — orchestrator
- `python -m vsleakkg.v2.pipeline ... litpcba` PID 681123 — running through litpcba regimes
- No finalize.sh active (the rerun_pipeline.sh script handles its own finalize step at the end)

## What to do on resume

```bash
# 1. Check it landed
ssh kongwoang "ssh VUW 'tail -50 /vol/dl-nguyenb5-solar/users/hoangpc/rerun_pipeline.log'"

# 2. Quick check pdbbind got a non-zero feasible count after the rerun
ssh kongwoang "ssh VUW 'cat /vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v2/outputs/v2/phase1/pdbbind_summary.csv'"

# 3. If pdbbind feasible=0 (highly likely - it ran before the pK fix), rebuild + rerun JUST pdbbind:
ssh kongwoang "ssh VUW 'cd /vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v2 && /usr/pkg/bin/python -m vsleakkg.v2.build_graph --output-dir outputs/v2/graph_pdbbind --corpus pdbbind && rm -f outputs/v2/phase1/pdbbind_summary.csv outputs/v2/phase1/splits/pdbbind/*.parquet outputs/v2/phase1/validation_contamination/pdbbind/*.csv outputs/v2/phase1/baselines/pdbbind/*.csv && /usr/pkg/bin/python -m vsleakkg.v2.pipeline --graph-dir outputs/v2/graph_pdbbind --side-table outputs/v2/graph/side_table.parquet --output-dir outputs/v2/phase1 --corpus-tag pdbbind --sample-examples 50000'"

# 4. Regenerate phase1_combined + tables + figure
ssh kongwoang "ssh VUW 'cd /vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v2 && /usr/pkg/bin/python -m vsleakkg.v2.final_figures --repo-root .'"

# 5. Pull updated outputs to Windows for git archive
```

## Local Run 1 archive already committed

- `outputs_run1_50k/phase1/{pdbbind,dekois,dude,litpcba}_summary.csv` + `phase1_combined.csv`
- `outputs_run1_50k/tables/table{1,2,5}*.csv`
- `outputs_run1_50k/figures/figure2_hub_pareto.png`
- `LINUX_RUN_REPORT_PHASE1.md` (Run 1)
- `LINUX_RUN_REPORT_PHASE1_REVIEW.md` (Run 1)
- `REVIEW_PHASE1.md` (manual Run 1 review)

## Commits this session

```
d4a5e1b fix(pdbbind): extract pK from BM props + guard single-class baseline
c687437 data: archive Run 1 outputs (sample-examples=50k cap)
e4e88ae feat(build_graph): synthesize PDBBind Example nodes from Complex (step B)
4a31ae9 fix(pipeline): pull SMILES from v1 Ligand node label
f61720c docs: Phase 1 results review
5edcdb5 fix(pipeline): extract label from v1 Example node props
b0f9fa2 fix(pipeline): write summary directly under output_dir
559f221 fix(pipeline): tolerate corpora with no Example nodes (PDBBind)
db3c403 fix(side-table): never fall back source_id to target column
9787a24 fix(build_graph): recognise seq_id / rep_seq_id in cluster parquets
3f5529a feat(review): tools/phase1_review.py
97d9764 perf(build_graph): eager read_parquet
ace8be9 feat(sprint): tools/sprint_csv_from_v2.py
b6716b8 feat(figures): Phase 1 final tables + Figure 2
9bcd9d7 feat(pipeline): end-to-end Phase 1 driver
4cf7cc3 feat(graph): v1->v2 schema mapper + side-table builder
```

---

# Original checkpoint (Run 1)



## SSH chain (will need to re-establish)

- Windows → kongwoang Tailscale IP `100.73.27.51` → ssh VUW (ProxyJump vic_gateway, user `longnd`, host `cuda12.ecs.vuw.ac.nz`)
- Mux config on **kongwoang** at `~/.ssh/config` (ControlMaster auto for VUW+vic_gateway).
- After reconnect, re-warm the VUW master: `ssh kongwoang "ssh -fN VUW"`.
- Working dir on VUW: `/vol/dl-nguyenb5-solar/users/hoangpc/`
- Repo on VUW: `/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v2/`
- Conda env at `/vol/grid-solar/.../envs/vsleak` is unusable. Use `/usr/pkg/bin/python` (3.12.13) with `pip install --user polars pulp` already done.

## What ran successfully

| Step | Output | Path on VUW |
|------|--------|-------------|
| Env bootstrap | system python + polars 1.40.1 + pulp 3.3.1 | `~/.local/lib/python3.12/site-packages` |
| v2 clone | git repo | `~/VS-LeakKG_v2/` (where ~ = `/vol/dl-nguyenb5-solar/users/hoangpc`) |
| **[D1]** build_graph × 4 corpora | 4 graph parquet pairs + stats.csv | `outputs/v2/graph_{pdbbind,dekois,dude,litpcba_ave}/` |
| **[P2.1]** side-table (litpcba/dude/dekois/bayesbind) | 1.71M-row parquet, 96MB | `outputs/v2/graph/side_table.parquet` |
| **[D4/D5/D6]** pipeline pdbbind | summary CSV (0/7 regimes — no Example nodes) | `outputs/v2/phase1/pdbbind_summary.csv` |
| **[D4/D5/D6]** pipeline dekois | 5/7 feasible, baselines AUROC 0.86–0.97 | `outputs/v2/phase1/dekois_summary.csv` |
| **[D4/D5/D6]** pipeline dude | 5/7 feasible, baselines AUROC 0.88–0.96 | `outputs/v2/phase1/dude_summary.csv` |
| **[D4/D5/D6]** pipeline litpcba | 4-5/7 feasible (last regime finishing as of checkpoint) | `outputs/v2/phase1/litpcba_summary.csv` (writing) |

## What was running at checkpoint time

- `run_phase1.sh` PID 634153 — orchestrator, alive
- `vsleakkg.v2.pipeline litpcba` PID 646794 — running the **strict regime baseline** (RandomForest fit on 35k × 2048 features). Strict split parquet already written at `outputs/v2/phase1/splits/litpcba/strict.parquet`. RSS 4.3 GB, in D-state on NFS. Just slow because of box load.
- `phase1_finalize.sh` PID 634174 — polling every 60 s, waiting for the pipeline to exit. When it does, it'll:
  1. Run `vsleakkg.v2.final_figures` to render `tables/table{1,2,5}*.csv` + `figures/figure2_hub_pareto.png`.
  2. Run `tools/phase1_review.py` and write `LINUX_RUN_REPORT_PHASE1_REVIEW.md`.
  3. Generate `LINUX_RUN_REPORT_PHASE1.md` with combined headline numbers.

## What still needs to run (was queued but not started)

Per the user's "do all of ABCD" plan:

| Letter | Action | Status |
|--------|--------|--------|
| **A** | Re-run `build_graph` for all 4 corpora with the **seq_id / rep_seq_id** fix from commit `9787a24` — this adds protein_in_cluster edges (~75k for PDBBind) so the protein axis uses the v1 multi-resolution MMseqs2 clusters. ETA ~30 s. | pending |
| **B** | Modify `build_graph.py` to synthesize Example nodes from PDBBind Complex (one Example per (Complex, Ligand, Protein, BindingMeasurement) tuple). Re-run for pdbbind. Then pdbbind pipeline can produce non-zero regimes. | code change pending |
| **C** | Drop `--sample-examples 50000` cap in `run_phase1.sh` and rerun full pipeline. ETA: dekois ~10 min, dude ~hours, litpcba ~hours. | pending C, waits for A+B+current run |
| **D** | Pocket-similarity edges via ESM-IF1 (needs GPU + AF2 structures or PDB pockets). ETA: half-day eng + 6-12 GPU·h. | pending, may not complete this session |

## Local commits on Windows (all pushed to GitHub)

Last 10 commits in `main`:

```
4a31ae9 fix(pipeline): pull SMILES from v1 Ligand node label (fixes baselines)
f61720c docs: Phase 1 results review
5edcdb5 fix(pipeline): extract label from v1 Example node props (JSON)
b0f9fa2 fix(pipeline): write summary directly under output_dir
559f221 fix(pipeline): tolerate corpora with no Example nodes (PDBBind)
db3c403 fix(side-table): never fall back source_id to target column
9787a24 fix(build_graph): recognise seq_id / rep_seq_id in cluster parquets
3f5529a feat(review): tools/phase1_review.py — post-run output sanity checker
97d9764 perf(build_graph): eager read_parquet + drop the 5x .collect() rereads
b6716b8 feat(figures): Phase 1 final tables + Figure 2 hub-Pareto renderer
```

GitHub: `https://github.com/kongwoang/VS-LeakKG_v2`
Local: `D:/hoangpc/VS-LeakKG_v2`
Tests on Windows: **44/44 passing**.

## Box state at checkpoint

- VUW load avg 24, was up to 81 mid-run, down to 24 now
- 3× Quadro RTX 6000 24 GB (mostly used by other users)
- `/vol/dl-nguyenb5-solar` (working dir + outputs) at 11% used, fast NFS
- `/vol/grid-solar` (conda env, slow NFS) at 98% — avoided
- `/tmp` (local tmpfs) at 47 GB / 19 GB free

## How to resume

After re-establishing SSH and ensuring run_phase1.sh + finalize have produced `LINUX_RUN_REPORT_PHASE1.md`:

1. **Verify Phase 1 outputs** — re-read `outputs/v2/phase1/phase1_combined.csv` and `LINUX_RUN_REPORT_PHASE1.md`. Confirm litpcba has 7 regime rows with baseline numbers populated.

2. **Begin step A**: relaunch `python -m vsleakkg.v2.build_graph` with the seq_id-aware code already on disk. Confirm `protein_in_cluster_<30,50,90>` edge counts > 0 in each stats.csv.

3. **Begin step B**: edit `src/vsleakkg/v2/build_graph.py` to synthesize Example nodes from PDBBind Complex. Add tests. Rerun pdbbind build_graph.

4. **Begin step C**: edit `run_phase1.sh` to drop `--sample-examples 50000`. Launch and monitor.

5. **Begin step D** (separate session — needs GPU): clone facebookresearch/esm or equivalent. Embed pocket residues for every PDB pocket in v1 data. Write pocket_similar edges. Rerun build_graph for pdbbind. Rerun pipeline.
