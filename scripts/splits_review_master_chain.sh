#!/usr/bin/env bash
# Master chain for the KG vs split-frameworks benchmark.
# Runs LIT-PCBA → DEKOIS → DUD-E end-to-end, with all gating logic.
# Designed to be launched under nohup and to survive 7+ hours.

set -uo pipefail

ROOT=/vol/dl-nguyenb5-solar/users/hoangpc
REPO=$ROOT/VS-LeakKG_v2
PY=$ROOT/envs/drugclip_env/bin/python
export PATH=$ROOT/bin:$PATH
export PYTHONPATH=$ROOT/envs/datasail_pkgs:${PYTHONPATH:-}

cd "$REPO"

LOG=$REPO/outputs/splits_review/_chain.log
STATUS=$REPO/outputs/splits_review/_chain_status.txt
mkdir -p "$(dirname "$LOG")"
exec 1>>"$LOG" 2>&1

ts() { date +"%Y-%m-%d %H:%M:%S"; }
echo "[$(ts)] === Chain start ==="
echo "PID=$$"
echo "PATH=$PATH"
echo "PYTHONPATH=$PYTHONPATH"

set_status() {
    echo "[$(ts)] $1" | tee -a "$STATUS"
}

run_stage() {
    local corpus=$1; shift
    set_status "stage $corpus: start"
    nice -n 19 $PY -u -m tools.splits_review.run_stage --corpus "$corpus" "$@"
    set_status "stage $corpus: done rc=$?"
}

# -------- LIT-PCBA --------
set_status "LIT-PCBA: prepare manifest + sequences"
$PY -m tools.splits_review.fetch_sequences --corpus litpcba \
    --out outputs/splits_review/litpcba/protein_meta.parquet
$PY -m tools.splits_review.make_corpus_manifest --v2-root outputs/v2 --corpus litpcba \
    --protein-meta outputs/splits_review/litpcba/protein_meta.parquet \
    --out outputs/splits_review/litpcba/corpus_manifest.parquet

run_stage litpcba

# -------- DEKOIS --------
set_status "DEKOIS: prepare manifest + sequences"
$PY -m tools.splits_review.fetch_sequences --corpus dekois \
    --out outputs/splits_review/dekois/protein_meta.parquet
$PY -m tools.splits_review.make_corpus_manifest --v2-root outputs/v2 --corpus dekois \
    --protein-meta outputs/splits_review/dekois/protein_meta.parquet \
    --out outputs/splits_review/dekois/corpus_manifest.parquet

run_stage dekois

# -------- DUD-E --------
set_status "DUD-E: prepare manifest + sequences"
$PY -m tools.splits_review.fetch_sequences --corpus dude \
    --out outputs/splits_review/dude/protein_meta.parquet
$PY -m tools.splits_review.make_corpus_manifest --v2-root outputs/v2 --corpus dude \
    --protein-meta outputs/splits_review/dude/protein_meta.parquet \
    --out outputs/splits_review/dude/corpus_manifest.parquet

# DUD-E has 22.9k actives + 1.4M decoys — Morgan-RF is the slowest step.
# Run with the same flags as the others; the splitter falls back / marks
# solver-limited where it must.
run_stage dude

# -------- Stats + report --------
set_status "stats + report"
for corpus in litpcba dekois dude; do
    qA=outputs/splits_review/$corpus/data/table_split_quality_modeA.csv
    qB=outputs/splits_review/$corpus/data/table_split_quality_modeB.csv
    mA=outputs/splits_review/$corpus/data/table_split_modelmetrics_modeA.csv
    mB=outputs/splits_review/$corpus/data/table_split_modelmetrics_modeB.csv
    sout=outputs/splits_review/$corpus/data/table_stat_tests.csv
    if [ -f "$qA" ] || [ -f "$qB" ]; then
        $PY -m tools.splits_review.compute_stats \
            --quality "$qA" --metrics "$mA" --corpus "$corpus" --mode A --out-csv "$sout" || true
    fi
done
$PY -m tools.splits_review.build_report

set_status "=== CHAIN COMPLETE ==="
echo "[$(ts)] === Chain done ==="
