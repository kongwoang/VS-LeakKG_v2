"""Subset LigUnity training label JSONs per Group D regime.

For each regime (paper-clean, target-clean, active-clean, scaffold-clean,
dual-clean), write filtered copies of the two train label files:
  <regime>/train_label_blend_seq_full.json
  <regime>/train_label_pdbbind_seq.json

The two source files are concatenated index-wise to match what
build_ligunity_train_kg.py iterated. We reproduce the same filter logic
here (not relying on the surviving_assays_<regime>.json since pdbbind
entries lack assay_id), so the per-regime label JSONs are emitted
directly with full data row content preserved.

After this, training a regime requires only:
  data_path = <out-dir>/<regime>/
  (symlink the shared .lmdb, .clstr, *.json files into <regime>/)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def canon_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return Chem.MolToSmiles(m)
    except Exception:
        return ""


def scaffold_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc else ""
    except Exception:
        return ""


def parse_clstr(path: Path) -> dict[str, str]:
    cluster_of: dict[str, str] = {}
    current_cluster = None
    pat = re.compile(r">(?:sp|tr)\|([A-Za-z0-9_]+)\|")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = "clstr_" + line.split()[1]
            else:
                m = pat.search(line)
                if m and current_cluster is not None:
                    cluster_of[m.group(1)] = current_cluster
    return cluster_of


def precompute_assay_smiles(a: dict) -> tuple[set[str], set[str]]:
    smi_set: set[str] = set()
    scaf_set: set[str] = set()
    for lig in a.get("ligands", []):
        smi = canon_smi(lig.get("smi", ""))
        if not smi:
            continue
        smi_set.add(smi)
        sc = scaffold_smi(smi)
        if sc:
            scaf_set.add(sc)
    return smi_set, scaf_set


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ligunity-data", required=True, type=Path)
    p.add_argument("--ligunity-test-dir", required=True, type=Path)
    p.add_argument("--v2-retrieval-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path,
                   help="Per-regime label dirs go here: <out-root>/<regime>/")
    args = p.parse_args()

    print("loading sources...")
    blend = json.load((args.ligunity_data / "train_label_blend_seq_full.json").open())
    pdb   = json.load((args.ligunity_data / "train_label_pdbbind_seq.json").open())
    print(f"  blend assays:   {len(blend):,}")
    print(f"  pdbbind assays: {len(pdb):,}")

    cluster_of = parse_clstr(args.ligunity_data / "uniport40.clstr")

    import polars as pl
    test_uniprots: set[str] = set()
    for fname in ("dude.json", "dekois.json", "PCBA.json"):
        rows = json.load((args.ligunity_test_dir / fname).open())
        for r in rows:
            if r and r[0]:
                test_uniprots.add(r[0])
    test_clusters = {cluster_of[u] for u in test_uniprots if u in cluster_of}

    test_actives_smi: set[str] = set()
    test_actives_scaf: set[str] = set()
    for corpus_dir in ("graph_dude", "graph_dekois", "graph_litpcba"):
        p_act = args.v2_retrieval_root / corpus_dir / "v2_active_of_target.parquet"
        if not p_act.exists():
            continue
        df = pl.read_parquet(p_act)
        for s in df["smiles_canonical"].to_list():
            cs = canon_smi(s)
            if cs:
                test_actives_smi.add(cs)
        if "scaffold_smiles" in df.columns:
            for s in df["scaffold_smiles"].drop_nulls().to_list():
                cs = canon_smi(s)
                if cs:
                    test_actives_scaf.add(cs)

    print(f"  test uniprots: {len(test_uniprots)}; test clusters: {len(test_clusters)}")
    print(f"  test active smi: {len(test_actives_smi):,}; scaffolds: {len(test_actives_scaf):,}")

    # Precompute per-assay smiles/scaffolds for both sources (slow part).
    print("canonicalising blend ligands...")
    blend_aux = [precompute_assay_smiles(a) for a in blend]
    print("canonicalising pdbbind ligands...")
    pdb_aux = [precompute_assay_smiles(a) for a in pdb]

    def keep_paper(a):
        return a.get("uniprot", "") not in test_uniprots

    def keep_target(a):
        if not keep_paper(a):
            return False
        u = a.get("uniprot", "")
        return not (u in cluster_of and cluster_of[u] in test_clusters)

    def keep_active(a, aux):
        if not keep_paper(a):
            return False
        return not (aux[0] & test_actives_smi)

    def keep_scaffold(a, aux):
        if not keep_paper(a):
            return False
        return not (aux[1] & test_actives_scaf)

    def keep_dual(a, aux):
        return keep_target(a) and keep_active(a, aux)

    regimes = {
        "paper_clean":   ("paper-clean",   lambda a, aux: keep_paper(a)),
        "target_clean":  ("target-clean",  lambda a, aux: keep_target(a)),
        "active_clean":  ("active-clean",  lambda a, aux: keep_active(a, aux)),
        "scaffold_clean":("scaffold-clean",lambda a, aux: keep_scaffold(a, aux)),
        "dual_clean":    ("dual-clean",    lambda a, aux: keep_dual(a, aux)),
    }

    for tag, (label, fn) in regimes.items():
        out_dir = args.out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        kept_blend = [blend[i] for i in range(len(blend)) if fn(blend[i], blend_aux[i])]
        kept_pdb   = [pdb[i]   for i in range(len(pdb))   if fn(pdb[i],   pdb_aux[i])]
        n_pairs = sum(len(a.get("ligands", [])) for a in kept_blend + kept_pdb)
        (out_dir / "train_label_blend_seq_full.json").write_text(json.dumps(kept_blend))
        (out_dir / "train_label_pdbbind_seq.json").write_text(json.dumps(kept_pdb))
        print(f"  [{label:14s}] wrote {out_dir}  "
              f"blend={len(kept_blend):>6,d}  pdbbind={len(kept_pdb):>6,d}  pairs={n_pairs:>8,d}")


if __name__ == "__main__":
    main()
