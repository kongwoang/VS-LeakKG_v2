#!/bin/bash
# Prepare a per-regime data dir for LigUnity training.
# Sets up data_regimes/<regime>/ with symlinks to shared lmdbs + dicts +
# pretrains, plus regime-specific train_label_*.json copies.
#
# Usage: ./prepare_ligunity_regime.sh <regime>
#   regime ∈ {paper_clean, target_clean, active_clean, scaffold_clean, dual_clean}

set -euo pipefail
REGIME="$1"
ROOT=/vol/dl-nguyenb5-solar/users/hoangpc/LigUnity
SRC=$ROOT/data            # shared LMDB + dicts + clstr
KGSRC=$ROOT/data_kg/$REGIME  # regime-specific filtered labels
DEST=$ROOT/data_regimes/$REGIME

mkdir -p "$DEST"

# Shared files (symlink — never copy)
for f in train_prot_all_blend.lmdb train_lig_all_blend.lmdb \
         valid_lig.lmdb valid_prot.lmdb valid_label_seq.json \
         uniport40.clstr uniport80.clstr \
         pocket_name2idx_train_blend.json mol_smi2idx_train_blend.json \
         align_res; do
    if [ -e "$SRC/$f" ] && [ ! -e "$DEST/$f" ]; then
        ln -sfn "$SRC/$f" "$DEST/$f"
    fi
done

# Per-regime labels — must be copied (regime-specific subset)
if [ "$REGIME" = "paper_clean_baseline" ] || [ "$REGIME" = "vanilla" ]; then
    # Plain LigUnity baseline = use full unfiltered labels + their built-in
    # --protein-similarity-thres 1.0
    ln -sfn "$SRC/train_label_blend_seq_full.json" "$DEST/train_label_blend_seq_full.json"
    ln -sfn "$SRC/train_label_pdbbind_seq.json"   "$DEST/train_label_pdbbind_seq.json"
else
    for f in train_label_blend_seq_full.json train_label_pdbbind_seq.json; do
        if [ -f "$KGSRC/$f" ]; then
            cp -f "$KGSRC/$f" "$DEST/$f"
        else
            echo "ERROR: missing $KGSRC/$f" >&2
            exit 1
        fi
    done
fi

# Pretrained encoders (used by --finetune-mol-model / --finetune-pocket-model)
mkdir -p "$DEST/pretrain"
for f in mol_pre_no_h_220816.pt pocket_pre_220816.pt; do
    src=/vol/dl-nguyenb5-solar/users/hoangpc/drugclip_data/pretrains/$f
    if [ -f "$src" ] && [ ! -e "$DEST/pretrain/$f" ]; then
        ln -sfn "$src" "$DEST/pretrain/$f"
    fi
done

echo "Prepared $DEST"
ls -la "$DEST" | tail -20
