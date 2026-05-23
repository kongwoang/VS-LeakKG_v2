#!/bin/bash
# Run paper-ckpt retrieval eval on each Group C DUD-E split regime.
# Reports per-target metrics + per-regime aggregate.
#
# Usage: ./run_dude_eval_all_regimes.sh <gpu_index>

set -euo pipefail

GPU="${1:-2}"
ROOT=/vol/dl-nguyenb5-solar/users/hoangpc
ENV=$ROOT/envs/drugclip_env
PY=$ENV/bin/python
CKPT=$ROOT/drugclip_data/paper_ckpt/drugclip_data/checkpoint_best.pt
DATA=$ROOT/DrugCLIP/data/dude_retrieval
SPLITS=$ROOT/VS-LeakKG_v2/outputs/v2_retrieval/splits/dude
OUT=$ROOT/VS-LeakKG_v2/outputs/v2_retrieval/results/dude

mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=1
cd $ROOT/DrugCLIP

for regime in target_random target_clean active_clean dual_clean; do
    echo
    echo "================================================"
    echo "  DUD-E retrieval-native eval — $regime (paper ckpt)"
    echo "================================================"
    # Pull test target names from the split parquet
    TARGETS=$($PY -c "
import polars as pl
df = pl.read_parquet('$SPLITS/${regime}.parquet')
test = df.filter(pl.col('partition')=='test')['target_id'].to_list()
names = [t.split(':')[-1] for t in test]
print(','.join(names))
")
    echo "Test targets ($regime): $TARGETS"
    echo

    $PY eval_dude_retrieval.py $DATA \
        --user-dir ./unimol \
        --task drugclip --loss in_batch_softmax --arch drugclip \
        --max-pocket-atoms 256 --batch-size 64 --num-workers 4 \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --seed 1 \
        --path $CKPT \
        --log-format simple --log-interval 100 \
        --targets "$TARGETS" \
        --out-csv $OUT/${regime}_per_target.csv \
        2>&1 | tee $OUT/${regime}_run.log | tail -80
done

echo
echo "All regimes done. Results under $OUT/"
ls -la $OUT/
