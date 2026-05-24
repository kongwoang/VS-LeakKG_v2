#!/bin/bash
# 1-epoch smoke test for LigUnity training (validates infra before full run).
set -euo pipefail
REGIME="${1:-paper_clean}"
GPUS="${2:-0,2}"
NGPU=$(echo "$GPUS" | awk -F, '{print NF}')
ROOT=/vol/dl-nguyenb5-solar/users/hoangpc
ENV=$ROOT/envs/drugclip_env
PY=$ENV/bin/python
DATA=$ROOT/LigUnity/data_regimes/$REGIME
SAVE=$ROOT/LigUnity/runs/${REGIME}_smoke
LOG=$ROOT/LigUnity/runs/${REGIME}_smoke.log
mkdir -p "$SAVE" "$ROOT/LigUnity/runs"
rm -f "$SAVE"/*.pt 2>/dev/null || true

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$GPUS
cd $ROOT/LigUnity

if [ "$NGPU" -gt 1 ]; then
    LAUNCHER="$PY -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=$((10000 + RANDOM % 10000))"
else
    LAUNCHER="$PY"
fi

$LAUNCHER $ENV/bin/unicore-train $DATA --user-dir ./unimol \
    --train-subset train --valid-subset valid \
    --num-workers 4 --ddp-backend=c10d \
    --task train_task --loss rank_softmax --arch pocket_ranking \
    --max-pocket-atoms 256 \
    --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr 1e-4 --warmup-ratio 0.06 \
    --max-epoch 1 \
    --batch-size 24 --batch-size-valid 32 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --update-freq 1 --seed 1 \
    --log-interval 50 --log-format simple \
    --validate-interval 1 \
    --best-checkpoint-metric valid_bedroc --patience 2000 \
    --all-gather-list-size 2048000 \
    --save-dir $SAVE --tmp-save-dir $SAVE/tmp \
    --keep-best-checkpoints 1 --keep-last-epochs 1 \
    --find-unused-parameters \
    --maximize-best-checkpoint-metric \
    --finetune-pocket-model $DATA/pretrain/pocket_pre_220816.pt \
    --finetune-mol-model $DATA/pretrain/mol_pre_no_h_220816.pt \
    --valid-set CASF \
    --max-lignum 16 \
    --protein-similarity-thres 1.0 \
    >"$LOG" 2>&1
echo "Training exited $?  Log: $LOG"
tail -20 "$LOG"
