#!/usr/bin/env python3
"""DrugCLIP retrieval-native evaluator for the Group C DUD-E audit.

Reads per-target pocket.lmdb (1 entry) + mols.lmdb (actives + decoys with
labels). For each target:
  - encode pocket once (using model.pocket_model directly)
  - iterate mols through task.load_dataset → encode + score
  - cosine = (pocket_emb · mol_emb)
  - compute per-target BEDROC (α=80.5), ROC-AUC, EF1%, EF5%

Reports mean ± std across target set, and a per-target table.

Layout expected:
  <data-root>/dict_mol.txt   (symlink to DrugCLIP/data/dict_mol.txt)
  <data-root>/dict_pkt.txt   (symlink to DrugCLIP/data/dict_pkt.txt)
  <data-root>/<target>/pocket.lmdb
  <data-root>/<target>/mols.lmdb
  <data-root>/<target>/dict_mol.txt   (per-target symlinks, created here if missing)
  <data-root>/<target>/dict_pkt.txt

Run:
  CUDA_VISIBLE_DEVICES=2 python eval_dude_retrieval.py \
    <data-root> --user-dir ./unimol \
    --task drugclip --loss in_batch_softmax --arch drugclip \
    --max-pocket-atoms 256 --batch-size 64 --num-workers 4 \
    --fp16 --seed 1 \
    --path <ckpt> --out-csv <per-target csv>
"""
import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
import torch
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
from sklearn.metrics import roc_auc_score

import unicore
from unicore import checkpoint_utils, distributed_utils, options, tasks, utils

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("eval_dude_retrieval")


def ensure_dict_symlinks(target_dir: Path, source_dicts_dir: Path) -> None:
    """Ensure target_dir/dict_{mol,pkt}.txt exist (symlink from source if not)."""
    for name in ("dict_mol.txt", "dict_pkt.txt"):
        tgt = target_dir / name
        if tgt.exists() or tgt.is_symlink():
            continue
        src = source_dicts_dir / name
        if not src.exists():
            raise SystemExit(f"missing source dict {src}")
        tgt.symlink_to(os.path.relpath(src, target_dir))


def encode_mols_for_target(model, task, mols_lmdb_path: Path, args, use_cuda: bool):
    """Iterate mols.lmdb via task.load_mols_dataset (the mol-only path used by
    DrugCLIP's own DUD-E/LIT-PCBA evaluators). Returns (mol_embs, labels)."""
    import torch.utils.data as tud
    ds = task.load_mols_dataset(str(mols_lmdb_path), "atoms", "coordinates")
    loader = tud.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=ds.collater, shuffle=False,
    )

    mol_embs, labels = [], []
    with torch.no_grad():
        for sample in loader:
            if use_cuda:
                sample = utils.move_to_cuda(sample)
            net = sample["net_input"]
            if args.fp16 and "mol_src_distance" in net:
                net["mol_src_distance"] = net["mol_src_distance"].half()
            mt = net["mol_src_tokens"]
            mp = mt.eq(model.mol_model.padding_idx)
            mx = model.mol_model.embed_tokens(mt)
            n = net["mol_src_distance"].size(-1)
            mg = model.mol_model.gbf_proj(
                model.mol_model.gbf(net["mol_src_distance"], net["mol_src_edge_type"])
            )
            ma = mg.permute(0, 3, 1, 2).contiguous().view(-1, n, n)
            mo = model.mol_model.encoder(mx, padding_mask=mp, attn_mask=ma)[0]
            emb = model.mol_project(mo[:, 0, :])
            emb = emb / emb.norm(dim=1, keepdim=True)
            mol_embs.append(emb.detach().cpu().float())
            # 'target' here is the float label (per load_mols_dataset definition)
            lab = sample["target"]
            if torch.is_tensor(lab):
                labels.append(lab.detach().cpu().float())
            else:
                labels.append(torch.tensor(lab, dtype=torch.float32))

    return torch.cat(mol_embs, dim=0), torch.cat(labels, dim=0).long()


def encode_pocket(model, pocket_lmdb_path: Path, task, args, use_cuda: bool) -> torch.Tensor:
    """Encode the single pocket entry directly via the pocket encoder."""
    env = lmdb.open(str(pocket_lmdb_path), subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        row = pickle.loads(txn.get(b"0"))
    env.close()
    atoms = row["pocket_atoms"]
    coords = np.asarray(row["pocket_coordinates"], dtype=np.float32)

    max_atoms = args.max_pocket_atoms
    if len(atoms) > max_atoms:
        center = coords.mean(axis=0)
        dist = np.linalg.norm(coords - center, axis=1)
        keep = np.argsort(dist)[:max_atoms]
        keep.sort()
        atoms = [atoms[i] for i in keep]
        coords = coords[keep]

    pkt_dict = task.pocket_dictionary
    bos, eos = pkt_dict.bos(), pkt_dict.eos()
    tokens = [bos] + [pkt_dict.index(a) for a in atoms] + [eos]
    n_token = len(tokens)
    coords_full = np.zeros((n_token, 3), dtype=np.float32)
    coords_full[1:-1] = coords
    diff = coords_full[None, :, :] - coords_full[:, None, :]
    distmat = np.linalg.norm(diff, axis=-1).astype(np.float32)
    tok_arr = np.array(tokens, dtype=np.int64)
    et = (tok_arr[:, None] * len(pkt_dict)) + tok_arr[None, :]

    t_tok = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    t_dist = torch.tensor(distmat).unsqueeze(0)
    t_et = torch.tensor(et, dtype=torch.long).unsqueeze(0)
    if use_cuda:
        t_tok = t_tok.cuda(); t_dist = t_dist.cuda(); t_et = t_et.cuda()
    if args.fp16:
        t_dist = t_dist.half()

    with torch.no_grad():
        pp = t_tok.eq(model.pocket_model.padding_idx)
        px = model.pocket_model.embed_tokens(t_tok)
        gbf = model.pocket_model.gbf_proj(model.pocket_model.gbf(t_dist, t_et))
        attn = gbf.permute(0, 3, 1, 2).contiguous().view(-1, n_token, n_token)
        out = model.pocket_model.encoder(px, padding_mask=pp, attn_mask=attn)[0]
        rep = out[:, 0, :]
        emb = model.pocket_project(rep)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.detach().cpu().float().squeeze(0)


def per_target_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return dict(auroc=float("nan"), bedroc=float("nan"),
                    ef1=float("nan"), ef5=float("nan"))
    auroc = roc_auc_score(labels, scores)
    order = np.argsort(-scores)
    sorted_labels = labels[order].reshape(-1, 1).astype(np.int32)
    bedroc = CalcBEDROC(sorted_labels, 0, 80.5)
    enrichment = CalcEnrichment(sorted_labels, 0, [0.01, 0.05])
    return dict(auroc=float(auroc), bedroc=float(bedroc),
                ef1=float(enrichment[0]), ef5=float(enrichment[1]))


def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    data_root = Path(args.data)
    # Make sure data_root has dict_*.txt; else look in parent
    source_dicts = data_root if (data_root / "dict_mol.txt").exists() else data_root.parent
    if not (source_dicts / "dict_mol.txt").exists():
        raise SystemExit(f"dict_mol.txt not found in {data_root} or {data_root.parent}")
    ensure_dict_symlinks(data_root, source_dicts)

    logger.info(f"loading checkpoint: {args.path}")
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()
    model.eval()

    target_names = (args.targets.split(",") if args.targets
                    else sorted([d.name for d in data_root.iterdir()
                                 if d.is_dir() and (d / "pocket.lmdb").exists()]))
    logger.info(f"evaluating {len(target_names)} targets")

    rows = []
    for tname in target_names:
        tdir = data_root / tname
        pocket_lmdb = tdir / "pocket.lmdb"
        mols_lmdb = tdir / "mols.lmdb"
        if not pocket_lmdb.exists() or not mols_lmdb.exists():
            logger.warning(f"  skip {tname}: missing lmdb")
            continue
        ensure_dict_symlinks(tdir, source_dicts)

        try:
            pkt_emb = encode_pocket(model, pocket_lmdb, task, args, use_cuda)
            mol_emb, labels = encode_mols_for_target(model, task, mols_lmdb, args, use_cuda)
        except Exception as e:
            logger.warning(f"  skip {tname}: {type(e).__name__}: {e}")
            continue

        scores = (mol_emb @ pkt_emb.unsqueeze(-1)).squeeze(-1).numpy()
        labels_np = labels.numpy()
        m = per_target_metrics(scores, labels_np)
        rows.append({
            "target_name": tname,
            "n_actives": int(labels_np.sum()),
            "n_decoys": int((1 - labels_np).sum()),
            **m,
        })
        logger.info(f"  {tname:8s}  n_act={int(labels_np.sum()):4d}  n_dec={int((1-labels_np).sum()):4d}  "
                    f"AUROC={m['auroc']:.3f}  BEDROC={m['bedroc']:.3f}  EF1%={m['ef1']:.2f}  EF5%={m['ef5']:.2f}")

    df = pl.DataFrame(rows)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(args.out_csv)
        logger.info(f"wrote per-target table → {args.out_csv}")

    if df.shape[0] > 0:
        print()
        print(f"=== Aggregated across {df.shape[0]} targets ===")
        for col in ("auroc", "bedroc", "ef1", "ef5"):
            vals = df[col].drop_nulls().to_numpy()
            if len(vals) == 0:
                continue
            print(f"  {col:7s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"median={np.median(vals):.4f}  min={np.min(vals):.4f}  max={np.max(vals):.4f}")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--targets", default=None, type=str,
                        help="comma-separated target names; omit to use all dirs")
    parser.add_argument("--out-csv", default=None, type=str)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
