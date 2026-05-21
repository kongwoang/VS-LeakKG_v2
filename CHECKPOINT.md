# Session checkpoint — 2026-05-22

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
