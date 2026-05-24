#!/bin/bash
# Chain runner for Group D: 5 LigUnity trainings + eval, sequential.
# Each regime: prepare data dir → train → wait → no-op (eval is done
# afterwards via a separate script since LigUnity eval needs paper-native
# test data fetched separately).
#
# Usage: nohup ./chain_ligunity_groupd.sh <gpu> [start_regime] >chain.log 2>&1 &
#   gpu: single GPU index for all trainings (sequential)
#   start_regime: optional, skip earlier regimes (e.g. "active_clean")

set -euo pipefail

GPU="${1:-0}"
START="${2:-paper_clean}"
ROOT=/vol/dl-nguyenb5-solar/users/hoangpc

REGIMES=(paper_clean target_clean active_clean scaffold_clean dual_clean)

# Honor START — skip until we find it
SKIP=true
for R in "${REGIMES[@]}"; do
    if [ "$R" = "$START" ]; then SKIP=false; fi
    if $SKIP; then continue; fi

    echo "$(date) ===== Regime $R ====="

    # Ensure data dir is prepared (idempotent)
    bash $ROOT/prepare_ligunity_regime.sh "$R" || { echo "[$R] prep failed"; continue; }

    SAVE=$ROOT/LigUnity/runs/$R
    LOG=$ROOT/LigUnity/runs/${R}.log
    mkdir -p "$SAVE"

    # Skip if best ckpt already exists (resume-friendly)
    if ls "$SAVE"/checkpoint_best*.pt >/dev/null 2>&1; then
        echo "$(date) [$R] checkpoint_best exists, skipping training"
        continue
    fi

    echo "$(date) [$R] launching training on GPU $GPU"
    bash $ROOT/launch_ligunity_train.sh "$R" "$GPU"
    sleep 20

    # Wait for training to exit
    while pgrep -fa "unicore-train.*data_regimes/$R" >/dev/null 2>&1; do
        sleep 120
    done
    echo "$(date) [$R] training exited"

    if ls "$SAVE"/checkpoint_best*.pt >/dev/null 2>&1; then
        echo "$(date) [$R] best ckpt produced"
    else
        echo "$(date) [$R] WARNING no checkpoint_best produced — check $LOG"
    fi
done

echo "$(date) ===== chain complete ====="
