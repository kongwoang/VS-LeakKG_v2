"""Group D — Build LigUnity train-corpus KG and report per-regime filter sizes.

For each of 5 regimes (paper-clean, target-clean, active-clean,
scaffold-clean, dual-clean), report:
  - n_train_assays surviving
  - n_train_pairs (assay × ligand) surviving
  - n_train_unique_uniprots
  - n_train_unique_ligands

before any retraining. The five regimes are filters on top of LigUnity's
existing training corpus (ChEMBL + BindingDB + PDBBind blend, ~43k
assays).

Filter axes:
  paper-clean (random)   — LigUnity's own default: exact-uniprot exclusion
                           of DUD-E + DEKOIS + LIT-PCBA test target UniProts.
  target-clean           — paper-clean PLUS remove train assays whose
                           uniprot is in the same uniport40 cluster (≥40%
                           seq-id) as any test target's uniprot.
  active-clean           — paper-clean PLUS remove train assays sharing
                           any canonical SMILES with test target actives.
  scaffold-clean         — paper-clean PLUS remove train assays sharing
                           any Bemis-Murcko scaffold (among actives) with
                           test target actives.
  dual-clean             — target-clean ∩ active-clean.

Outputs (additive, under outputs/v2_retrieval/ligunity_train_kg/):
  filter_summary.csv        — per-regime counts
  surviving_assays_<r>.json — list of assay_ids that survive each regime
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# ---------- uniport40 cluster parser ----------
def parse_clstr(path: Path) -> dict[str, str]:
    """CD-HIT .clstr → {uniprot: cluster_id}.

    CD-HIT line format: `<idx>\\t<len>aa, >sp|UNIPROT|NAME...|... at X%`
    or `>tr|UNIPROT|NAME...`. We extract the UniProt accession from the
    pipe-delimited sp/tr header.
    """
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


# ---------- RDKit canonicalisation + scaffold ----------
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ligunity-data", required=True, type=Path,
                   help="LigUnity/data/ root (must contain train_label_*.json + uniport40.clstr)")
    p.add_argument("--ligunity-test-dir", required=True, type=Path,
                   help="LigUnity/test_datasets/ root (contains dude.json / dekois.json / PCBA.json)")
    p.add_argument("--v2-retrieval-root", required=True, type=Path,
                   help="outputs/v2_retrieval/ root (for graph_<corpus>/v2_active_of_target.parquet)")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load train assays ----
    print("loading train labels...")
    blend = json.load((args.ligunity_data / "train_label_blend_seq_full.json").open())
    pdb   = json.load((args.ligunity_data / "train_label_pdbbind_seq.json").open())
    train_assays = blend + pdb
    print(f"  blend assays:   {len(blend):,}")
    print(f"  pdbbind assays: {len(pdb):,}")
    print(f"  total:          {len(train_assays):,}")

    # build per-assay summary
    print("canonicalising train ligands + scaffolds...")
    train_rows = []
    for a in train_assays:
        uni = a.get("uniprot", "")
        ligs = a.get("ligands", [])
        smi_set = set()
        scaf_set = set()
        for lig in ligs:
            smi = canon_smi(lig.get("smi", ""))
            if not smi:
                continue
            smi_set.add(smi)
            sc = scaffold_smi(smi)
            if sc:
                scaf_set.add(sc)
        train_rows.append({
            "assay_id":  a.get("assay_id", str(len(train_rows))),
            "uniprot":   uni,
            "n_ligands": len(smi_set),
            "smiles":    list(smi_set),
            "scaffolds": list(scaf_set),
        })

    # ---- load test target uniprots from the three retrieval corpora ----
    print("loading test target uniprots...")
    test_corpora = {}
    for fname, label in [
        ("dude.json", "DUD-E"),
        ("dekois.json", "DEKOIS"),
        ("PCBA.json", "LIT-PCBA"),
    ]:
        path = args.ligunity_test_dir / fname
        if not path.exists():
            print(f"  [skip] {fname}")
            continue
        rows = json.load(path.open())
        unis = [r[0] for r in rows if r and r[0]]
        test_corpora[label] = unis
        print(f"  {label}: {len(unis)} uniprots ({len(set(unis))} unique)")
    test_uniprots = set()
    for u in test_corpora.values():
        test_uniprots.update(u)
    print(f"  total unique test uniprots: {len(test_uniprots)}")

    # ---- load uniport40 cluster file ----
    print("loading uniport40 clusters...")
    cluster_of = parse_clstr(args.ligunity_data / "uniport40.clstr")
    print(f"  {len(cluster_of):,} uniprot→cluster entries")
    test_clusters = {cluster_of[u] for u in test_uniprots if u in cluster_of}
    n_mapped = sum(1 for u in test_uniprots if u in cluster_of)
    print(f"  {n_mapped}/{len(test_uniprots)} test uniprots mapped to uniport40 clusters")
    print(f"  {len(test_clusters)} unique test clusters")

    # ---- load test actives from v2_retrieval graph parquets ----
    print("loading test actives + scaffolds...")
    test_actives_smi  = set()
    test_actives_scaf = set()
    for corpus_dir in ("graph_dude", "graph_dekois", "graph_litpcba"):
        p_act = args.v2_retrieval_root / corpus_dir / "v2_active_of_target.parquet"
        if not p_act.exists():
            print(f"  [skip] {p_act}")
            continue
        df = pl.read_parquet(p_act)
        smis = [canon_smi(s) for s in df["smiles_canonical"].to_list()]
        smis = [s for s in smis if s]
        test_actives_smi.update(smis)
        if "scaffold_smiles" in df.columns:
            sc = [canon_smi(s) for s in df["scaffold_smiles"].drop_nulls().to_list()]
            sc = [s for s in sc if s]
            test_actives_scaf.update(sc)
        print(f"  {corpus_dir}: +{len(smis)} smiles, +{len(sc) if 'scaffold_smiles' in df.columns else 0} scaffolds")
    print(f"  total test active SMILES (canon, unique): {len(test_actives_smi):,}")
    print(f"  total test active scaffolds (canon, unique): {len(test_actives_scaf):,}")

    # ---- evaluate each filter regime ----
    print()
    print("=" * 70)
    print("Per-regime filter survival")
    print("=" * 70)

    def evaluate(name: str, keep_fn) -> dict:
        kept = [r for r in train_rows if keep_fn(r)]
        n_assays = len(kept)
        n_pairs = sum(r["n_ligands"] for r in kept)
        n_uni = len({r["uniprot"] for r in kept if r["uniprot"]})
        n_lig_unique = len({s for r in kept for s in r["smiles"]})
        print(f"  {name:18s} assays={n_assays:>6,d}  pairs={n_pairs:>9,d}  "
              f"uniq_uniprots={n_uni:>5,d}  uniq_ligands={n_lig_unique:>7,d}")
        return {
            "regime": name,
            "n_train_assays": n_assays,
            "n_train_pairs": n_pairs,
            "n_unique_uniprots": n_uni,
            "n_unique_ligands": n_lig_unique,
        }

    # paper-clean = remove exact-uniprot matches of test set
    def keep_paper(r):
        return r["uniprot"] not in test_uniprots

    # target-clean = paper-clean + remove uniport40-cluster siblings
    def keep_target(r):
        if not keep_paper(r):
            return False
        u = r["uniprot"]
        if u in cluster_of and cluster_of[u] in test_clusters:
            return False
        return True

    # active-clean = paper-clean + remove ligand-SMILES overlap
    def keep_active(r):
        if not keep_paper(r):
            return False
        for s in r["smiles"]:
            if s in test_actives_smi:
                return False
        return True

    # scaffold-clean = paper-clean + remove scaffold overlap
    def keep_scaffold(r):
        if not keep_paper(r):
            return False
        for sc in r["scaffolds"]:
            if sc in test_actives_scaf:
                return False
        return True

    # dual-clean = target-clean ∧ active-clean
    def keep_dual(r):
        return keep_target(r) and keep_active(r)

    # also report "no filter" as a sanity row
    rows = []
    rows.append(evaluate("(no filter)",   lambda r: True))
    rows.append(evaluate("paper-clean",   keep_paper))
    rows.append(evaluate("target-clean",  keep_target))
    rows.append(evaluate("active-clean",  keep_active))
    rows.append(evaluate("scaffold-clean", keep_scaffold))
    rows.append(evaluate("dual-clean",    keep_dual))

    df = pl.DataFrame(rows)
    df.write_csv(args.out_dir / "filter_summary.csv")
    print()
    print(f"wrote {args.out_dir/'filter_summary.csv'}")

    # Save per-regime surviving assay-ids
    for regime_name, fn in [
        ("paper_clean",   keep_paper),
        ("target_clean",  keep_target),
        ("active_clean",  keep_active),
        ("scaffold_clean", keep_scaffold),
        ("dual_clean",    keep_dual),
    ]:
        kept_ids = [r["assay_id"] for r in train_rows if fn(r)]
        (args.out_dir / f"surviving_assays_{regime_name}.json").write_text(json.dumps(kept_ids))
    print(f"wrote per-regime surviving_assays_*.json under {args.out_dir}")


if __name__ == "__main__":
    main()
