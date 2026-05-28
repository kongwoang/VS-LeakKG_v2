"""Run a Stage 1/2/3 corpus end-to-end.

Order:
    1. Splitter runs (subset of the protocol's full split list — the runner
       prints which were skipped and why).
    2. compute_split_quality.py per split.
    3. compute_model_metrics.py per split.

DataSAIL splitter is currently a stub; runner records it as "skipped:
datasail_integration_pending" and continues. AVE may take long; runs last
in the splitter loop.
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import time
from pathlib import Path


import os
MMSEQS_PATH_DIR = "/vol/dl-nguyenb5-solar/users/hoangpc/bin"
DATASAIL_PYPATH = "/vol/dl-nguyenb5-solar/users/hoangpc/envs/datasail_pkgs"

def sh(cmd: str, env: dict | None = None) -> tuple[int, str]:
    print(f"\n$ {cmd}", flush=True)
    real_env = os.environ.copy()
    if env: real_env.update(env)
    real_env["PATH"] = f"{MMSEQS_PATH_DIR}:{real_env.get('PATH', '')}"
    real_env["PYTHONPATH"] = f"{DATASAIL_PYPATH}:{real_env.get('PYTHONPATH', '')}"
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=real_env)
    print(p.stdout[-3000:] if p.stdout else "", flush=True)
    if p.returncode != 0:
        print("[STDERR]", p.stderr[-2000:], flush=True)
    return p.returncode, (p.stdout + p.stderr)


# Each tuple: (label, splitter module, extra-args, mode, eligible-corpora-or-None)
MODE_A_SPLITS = [
    ("random_modeA",          "splitter_random",  "",                                "A", None),
    ("scaffold_modeA",        "splitter_scaffold", "",                               "A", None),
    ("drugood_scaffold_modeA","splitter_drugood", "--axis scaffold",                 "A", None),
    ("drugood_size_modeA",    "splitter_drugood", "--axis size",                     "A", None),
    ("kg_ligand_modeA",       "splitter_kg",      "--axis ligand --kg-splits {KGSPLITS}", "A", None),
    ("kg_scaffold_modeA",     "splitter_kg",      "--axis scaffold --kg-splits {KGSPLITS}", "A", None),
    ("ave_ligand_modeA",      "splitter_ave",
     "--stats-out {OUT}/litpcba/data/ave_stats.csv --subset-dir {SUBSET}", "A", None),
]

MODE_B_SPLITS = [
    ("random_modeB",            "splitter_random",   "",                                 "B", None),
    ("scaffold_modeB",          "splitter_scaffold", "",                                 "B", None),
    ("protein_modeB",           "splitter_protein",  "--protein-meta {PROT_META}",       "B", None),
    ("drugood_scaffold_modeB",  "splitter_drugood",  "--axis scaffold",                  "B", None),
    ("drugood_size_modeB",      "splitter_drugood",  "--axis size",                      "B", None),
    ("drugood_protein_modeB",   "splitter_drugood",  "--axis protein",                   "B", None),
    ("drugood_family_modeB",    "splitter_drugood",  "--axis protein_family",            "B", None),
    ("kg_protein_modeB",        "splitter_kg",       "--axis protein --kg-splits {KGSPLITS}", "B", None),
    ("kg_dual_modeB",           "splitter_kg",       "--axis dual --kg-splits {KGSPLITS}",    "B", None),
    ("datasail_s1_ligand_modeB","splitter_datasail", "--datasail-mode s1_ligand --protein-meta {PROT_META}", "B", None),
    ("datasail_s1_protein_modeB","splitter_datasail","--datasail-mode s1_protein --protein-meta {PROT_META}", "B", None),
    ("datasail_s2_modeB",       "splitter_datasail", "--datasail-mode s2 --protein-meta {PROT_META}",        "B", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["litpcba", "dekois", "dude"])
    ap.add_argument("--out-root", default=Path("outputs/splits_review"), type=Path)
    ap.add_argument("--python", default="/vol/dl-nguyenb5-solar/users/hoangpc/envs/drugclip_env/bin/python")
    ap.add_argument("--kg-splits-root",
                    default=Path("outputs/v2/phase1/splits"), type=Path)
    ap.add_argument("--skip-models", action="store_true", help="skip the Morgan-RF/1-NN stage")
    ap.add_argument("--skip-ave",    action="store_true", help="skip AVE (long)")
    args = ap.parse_args()

    corpus = args.corpus
    out  = args.out_root / corpus
    manifest = out / "corpus_manifest.parquet"
    subset_dir = out / "manifests"
    splits_dir = out / "splits"
    data_dir   = out / "data"
    splits_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)
    kg_splits = args.kg_splits_root / corpus
    prot_meta = out / "protein_meta.parquet"

    summary: list[dict] = []

    # Run AVE first (it may write subset_<target>.parquet that other splitters consume).
    splits_to_run = [s for s in MODE_A_SPLITS + MODE_B_SPLITS]
    splits_to_run.sort(key=lambda x: 0 if "ave" in x[0] else 1)

    for label, mod, extra, mode, _eligible in splits_to_run:
        if args.skip_ave and "ave" in label:
            summary.append({"label": label, "status": "skipped_by_flag"}); continue
        out_split = splits_dir / f"{label}.parquet"
        cmd = (f"{args.python} -m tools.splits_review.{mod} "
               f"--manifest {manifest} --subset-dir {subset_dir} "
               f"--mode {mode} --out {out_split} "
               f"{extra.format(OUT=args.out_root, SUBSET=subset_dir, KGSPLITS=kg_splits, PROT_META=prot_meta)}")
        t0 = time.time()
        rc, output = sh(cmd)
        rt = time.time() - t0
        status = "ok" if rc == 0 else f"fail_rc{rc}"
        if "DATASAIL_SOLVER_LIMITED" in output:
            status = "solver_limited"
        elif "mmseqs not on PATH" in output or "MMseqs is not installed" in output:
            status = "tool_unavailable_mmseqs"
        summary.append({"label": label, "status": status, "runtime_s": rt})

    # Quality scoring per split.
    qA = data_dir / "table_split_quality_modeA.csv"
    qB = data_dir / "table_split_quality_modeB.csv"
    for p in (qA, qB):
        if p.exists(): p.unlink()

    for label, _, _, mode, _ in splits_to_run:
        out_split = splits_dir / f"{label}.parquet"
        if not out_split.exists():
            continue
        out_csv = qA if mode == "A" else qB
        cmd = (f"{args.python} -m tools.splits_review.compute_split_quality "
               f"--manifest {manifest} --split {out_split} --corpus {corpus} "
               f"--mode {mode} --splitter {label} --out-csv {out_csv}")
        sh(cmd)

    # Model metrics per split.
    if not args.skip_models:
        mA = data_dir / "table_split_modelmetrics_modeA.csv"
        mB = data_dir / "table_split_modelmetrics_modeB.csv"
        for p in (mA, mB):
            if p.exists(): p.unlink()

        for label, _, _, mode, _ in splits_to_run:
            out_split = splits_dir / f"{label}.parquet"
            if not out_split.exists():
                continue
            out_csv = mA if mode == "A" else mB
            cmd = (f"{args.python} -m tools.splits_review.compute_model_metrics "
                   f"--manifest {manifest} --split {out_split} --corpus {corpus} "
                   f"--mode {mode} --splitter {label} --out-csv {out_csv}")
            sh(cmd)

    # Print final summary.
    print("\n=== STAGE SUMMARY ===")
    for r in summary:
        rt = f"{r.get('runtime_s', 0):.1f}s" if "runtime_s" in r else ""
        print(f"  {r['label']:<35} {r['status']:<25} {rt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
