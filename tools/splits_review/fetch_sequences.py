"""Build per-corpus FASTA + uniprot mapping for protein/family axes.

Outputs:
    outputs/splits_review/<corpus>/protein_meta.parquet
        target_id, uniprot, sequence, family

Sources (per corpus):
    dude    : copy /vol/.../SPRINT/data/DUDe/dude_targets.fasta (102 seqs)
    dekois  : concat /vol/.../LigUnity/test_datasets/DEKOIS_2.0x/<t>/<t>_prot/<t>_pocket_5.0.fasta
    litpcba : fetch from UniProt using the hard-coded mapping below (15 targets)

Family lookup uses a small embedded table (UniProt KEYWORDS family classes).
If UniProt fetch fails for a LIT-PCBA target, that target's `sequence` is
left null and the protein splitter will flag it as target_split_control
rather than silently fall back.
"""
from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path
import polars as pl


# Hard-coded LIT-PCBA target -> UniProt mapping (from the paper, Tran-Nguyen 2020).
LITPCBA_UNIPROT = {
    "ADRB2":     "P07550",
    "ALDH1":     "P00352",
    "ESR1_ago":  "P03372",
    "ESR1_ant":  "P03372",
    "FEN1":      "P39748",
    "GBA":       "P04062",
    "IDH1":      "O75874",
    "KAT2A":     "Q92830",
    "MAPK1":     "P28482",
    "MTORC1":    "P42345",   # MTOR
    "OPRK1":     "P41145",
    "PKM2":      "P14618",
    "PPARG":     "P37231",
    "TP53":      "P04637",
    "VDR":       "P11473",
}

# Coarse family labels; primarily used by the protein_family axis. We use
# broad UniProt class names so the axis has 3-5 buckets per corpus rather
# than 1 family per target (which would collapse with `protein`).
LITPCBA_FAMILY = {
    "ADRB2":    "GPCR",
    "OPRK1":    "GPCR",
    "ALDH1":    "Oxidoreductase",
    "IDH1":     "Oxidoreductase",
    "FEN1":     "Nuclease",
    "GBA":      "Hydrolase",
    "KAT2A":    "Transferase",
    "MAPK1":    "Kinase",
    "MTORC1":   "Kinase",
    "PKM2":     "Transferase",
    "TP53":     "DNA_binding_TF",
    "ESR1_ago": "NuclearReceptor",
    "ESR1_ant": "NuclearReceptor",
    "PPARG":    "NuclearReceptor",
    "VDR":      "NuclearReceptor",
}


def fetch_uniprot_sequence(uid: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            txt = r.read().decode()
        body = "".join(line for line in txt.splitlines() if not line.startswith(">"))
        return body or None
    except Exception as e:
        print(f"  WARN: UniProt fetch failed for {uid}: {e}")
        return None


def build_litpcba(out: Path) -> None:
    rows = []
    for tid, uid in LITPCBA_UNIPROT.items():
        seq = fetch_uniprot_sequence(uid)
        rows.append({
            "target_id": tid, "uniprot": uid,
            "sequence": seq, "family": LITPCBA_FAMILY.get(tid, "UNK"),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)
    n_ok = sum(1 for r in rows if r["sequence"])
    print(f"litpcba protein_meta: {n_ok}/{len(rows)} sequences fetched -> {out}")


def build_dude(fasta_path: Path, out: Path) -> None:
    # Parse the SPRINT DUDe fasta. Headers look like ">cxcr4 <unknown description>".
    target_id, seq = None, []
    rows = []
    with fasta_path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if target_id is not None:
                    rows.append({"target_id": target_id.upper(), "uniprot": None,
                                 "sequence": "".join(seq), "family": "UNK"})
                target_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if target_id is not None:
            rows.append({"target_id": target_id.upper(), "uniprot": None,
                         "sequence": "".join(seq), "family": "UNK"})
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)
    print(f"dude protein_meta: {len(rows)} sequences -> {out}")


def build_dekois(dekois_root: Path, out: Path) -> None:
    rows = []
    for tdir in sorted(dekois_root.glob("*")):
        if not tdir.is_dir():
            continue
        tname = tdir.name
        fa = tdir / f"{tname}_prot" / f"{tname}_pocket_5.0.fasta"
        if not fa.exists():
            print(f"  WARN: dekois fasta missing for {tname}")
            continue
        seq = []
        with fa.open() as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    continue
                seq.append(line)
        rows.append({"target_id": tname.upper(), "uniprot": None,
                     "sequence": "".join(seq), "family": "UNK"})
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)
    print(f"dekois protein_meta: {len(rows)} sequences -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=["litpcba", "dude", "dekois"])
    ap.add_argument("--dude-fasta",  type=Path,
                    default=Path("/vol/dl-nguyenb5-solar/users/hoangpc/SPRINT/data/DUDe/dude_targets.fasta"))
    ap.add_argument("--dekois-root", type=Path,
                    default=Path("/vol/dl-nguyenb5-solar/users/hoangpc/LigUnity/test_datasets/DEKOIS_2.0x"))
    ap.add_argument("--out",         required=True, type=Path)
    args = ap.parse_args()

    if args.corpus == "litpcba":
        build_litpcba(args.out)
    elif args.corpus == "dude":
        build_dude(args.dude_fasta, args.out)
    elif args.corpus == "dekois":
        build_dekois(args.dekois_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
