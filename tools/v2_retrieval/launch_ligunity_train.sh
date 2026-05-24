#!/bin/bash
# Launch LigUnity training for one Group D regime.
#
# Paper-native settings (from LigUnity/train.sh):
#   - arch pocket_ranking, loss rank_softmax, task train_task
#   - batch_size 24, batch_size_valid 32, lr 1e-4, warmup 0.06
#   - max_epoch 50 (paper default; we keep)
#   - max_pocket_atoms 256, max_lignum 16
#   - --protein-similarity-thres 1.0 because our pre-filter handled it
#
# Usage: ./launch_ligunity_train.sh <regime> <gpu_indices>
#   regime ∈ {paper_clean, target_clean, active_clean, scaffold_clean, dual_clean}
#   gpu_indices e.g. "0,1" or "1"

set -euo pipefail
REGIME="$1"
GPUS="$2"
NGPU=$(echo "$GPUS" | awk -F, '{print NF}')
ROOT=/vol/dl-nguyenb5-solar/users/hoangpc
ENV=$ROOT/envs/drugclip_env
PY=$ENV/bin/python

DATA=$ROOT/LigUnity/data_regimes/$REGIME
SAVE=$ROOT/LigUnity/runs/$REGIME
LOG=$ROOT/LigUnity/runs/${REGIME}.log
mkdir -p "$SAVE" "$ROOT/LigUnity/runs"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$GPUS
# LigUnity's PairDataset.__len__ reads WORLD_SIZE directly from env.
# Set it explicitly so single-GPU and multi-GPU both work.
export WORLD_SIZE=$NGPU

cd $ROOT/LigUnity

if [ "$NGPU" -gt 1 ]; then
    LAUNCHER="$PY -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=$((10000 + RANDOM % 10000))"
else
    LAUNCHER="$PY"
fi

UNICORE_TRAIN=$ENV/bin/unicore-train

$LAUNCHER $UNICORE_TRAIN $DATA --user-dir ./unimol \
    --train-subset train --valid-subset valid \
    --num-workers 4 --ddp-backend=c10d \
    --task train_task --loss rank_softmax --arch pocket_ranking \
    --max-pocket-atoms 256 \
    --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr 1e-4 --warmup-ratio 0.06 \
    --max-epoch 50 \
    --batch-size 24 --batch-size-valid 32 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --update-freq 1 --seed 1 \
    --log-interval 100 --log-format simple \
    --validate-interval 1 \
    --best-checkpoint-metric valid_bedroc --patience 2000 \
    --all-gather-list-size 2048000 \
    --save-dir $SAVE --tmp-save-dir $SAVE/tmp \
    --keep-best-checkpoints 4 --keep-last-epochs 4 \
    --find-unused-parameters \
    --maximize-best-checkpoint-metric \
    --finetune-pocket-model $DATA/pretrain/pocket_pre_220816.pt \
    --finetune-mol-model $DATA/pretrain/mol_pre_no_h_220816.pt \
    --valid-set CASF \
    --max-lignum 16 \
    --protein-similarity-thres 1.0 \
    > "$LOG" 2>&1 &

echo "Launched ${REGIME} (pid=$!) on GPUs=${GPUS} → log=${LOG}"
