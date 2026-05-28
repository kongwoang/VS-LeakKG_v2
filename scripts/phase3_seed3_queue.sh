#!/usr/bin/env bash
# Phase 3 seed=3 OPPORTUNISTIC queue.
# Polls every 60s for a free GPU (util<10% AND mem_used<1 GiB).
# Claims it immediately and runs the next pending (regime, seed=3) job.
# Runs alongside the existing seed=2 queue (which owns GPU 2 by convention).
#
# Outputs:
#   - LigUnity/runs/phase3_seed3_queue.log : structured ndjson events
#   - LigUnity/runs/phase3_seed3_queue.out : driver stdout/stderr
#   - LigUnity/runs/<regime>_seed3/        : ckpt dir
#   - LigUnity/runs/<regime>_seed3.log     : train stdout/stderr
#   - LigUnity/runs/<regime>_seed3_eval.log: eval stdout/stderr

set -uo pipefail

ROOT=/vol/dl-nguyenb5-solar/users/hoangpc
SCRIPTS=$ROOT/VS-LeakKG_v2/scripts
LAUNCH=$SCRIPTS/launch_ligunity_seed.sh
EVAL=$SCRIPTS/eval_ligunity_regime.sh
RUNS=$ROOT/LigUnity/runs
QLOG=$RUNS/phase3_seed3_queue.log

SEED=3
POLL_SEC=60
UTIL_MAX=10        # %
MEM_MAX_MIB=1024   # MiB used on the GPU by anyone
LOAD_MAX=28        # 1-min loadavg gate before launching training

mkdir -p "$RUNS"

log_event() {
    local kind=$1; local regime=$2; local seed=$3; local status=$4
    shift 4
    local extra="$*"
    local ts=$(date +"%Y-%m-%d %H:%M:%S")
    if [ -n "$extra" ]; then
        printf '{"ts":"%s","kind":"%s","regime":"%s","seed":%s,"status":"%s",%s}\n' \
            "$ts" "$kind" "$regime" "$seed" "$status" "$extra" >> "$QLOG"
    else
        printf '{"ts":"%s","kind":"%s","regime":"%s","seed":%s,"status":"%s"}\n' \
            "$ts" "$kind" "$regime" "$seed" "$status" >> "$QLOG"
    fi
    # Echo to stderr so $(wait_for_free_gpu) never captures log lines into the GPU index.
    echo "[$ts] $kind $regime seed=$seed $status $extra" >&2
}

wait_for_pid() {
    local pid=$1
    while kill -0 "$pid" 2>/dev/null; do sleep 120; done
}

# Print the index of the first GPU that meets util<UTIL_MAX and mem_used<MEM_MAX_MIB.
# Empty string if none free.
find_free_gpu() {
    local line idx util mem
    while IFS=',' read -r idx util mem; do
        idx=${idx// /}
        util=${util// /}; util=${util%\%}; util=${util// /}
        mem=${mem// /}; mem=${mem%MiB}; mem=${mem// /}
        if [ -z "$idx" ]; then continue; fi
        if [ "${util:-100}" -lt "$UTIL_MAX" ] && [ "${mem:-99999}" -lt "$MEM_MAX_MIB" ]; then
            echo "$idx"
            return 0
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
    return 0
}

wait_for_free_gpu() {
    local regime=$1 seed=$2
    local first=1
    while true; do
        local gpu
        gpu=$(find_free_gpu)
        if [ -n "$gpu" ]; then
            # Quick double-check 5 s later to avoid a transient blip
            sleep 5
            local gpu2
            gpu2=$(find_free_gpu)
            if [ "$gpu" = "$gpu2" ]; then
                local la=$(awk '{print $1}' /proc/loadavg)
                local la_int=${la%.*}
                if [ "${la_int:-0}" -lt "$LOAD_MAX" ]; then
                    echo "$gpu"
                    return 0
                else
                    log_event poll "$regime" "$seed" load_high "\"gpu\":$gpu,\"loadavg\":\"$la\""
                fi
            fi
        fi
        if [ $first -eq 1 ]; then
            log_event poll "$regime" "$seed" waiting_for_gpu "\"util_max\":$UTIL_MAX,\"mem_max_mib\":$MEM_MAX_MIB,\"poll_sec\":$POLL_SEC"
            first=0
        fi
        sleep "$POLL_SEC"
    done
}

train_regime_seed() {
    local regime=$1 seed=$2
    local save_dir=$RUNS/${regime}_seed${seed}
    local log=$RUNS/${regime}_seed${seed}.log

    if [ -f "$save_dir/checkpoint_best.pt" ]; then
        log_event train "$regime" "$seed" skip "\"reason\":\"checkpoint_best.pt already exists\""
        return 0
    fi

    local gpu
    gpu=$(wait_for_free_gpu "$regime" "$seed")
    log_event train "$regime" "$seed" gpu_claimed "\"gpu\":$gpu"

    log_event train "$regime" "$seed" launching "\"save_dir\":\"$save_dir\",\"log\":\"$log\",\"gpu\":$gpu"
    local start_ts=$(date +%s)
    local pid
    pid=$(bash "$LAUNCH" "$regime" "$seed" "$gpu" 2>>"$RUNS/phase3_seed3_queue.out")
    if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
        log_event train "$regime" "$seed" FAILED "\"reason\":\"launch returned no live pid\",\"pid_str\":\"${pid:-}\",\"gpu\":$gpu"
        return 1
    fi

    log_event train "$regime" "$seed" running "\"pid\":$pid,\"gpu\":$gpu,\"start_ts\":$start_ts"
    wait_for_pid "$pid"
    local end_ts=$(date +%s)
    local dur=$(( end_ts - start_ts ))

    if [ -f "$save_dir/checkpoint_best.pt" ]; then
        log_event train "$regime" "$seed" complete \
            "\"pid\":$pid,\"gpu\":$gpu,\"end_ts\":$end_ts,\"duration_s\":$dur,\"ckpt\":\"$save_dir/checkpoint_best.pt\""
        echo "$gpu"   # return GPU on stdout for eval reuse
        return 0
    else
        log_event train "$regime" "$seed" FAILED \
            "\"reason\":\"no checkpoint_best.pt produced\",\"pid\":$pid,\"gpu\":$gpu,\"duration_s\":$dur"
        return 1
    fi
}

eval_regime_seed() {
    local regime=$1 seed=$2 gpu=$3
    local dir=${regime}_seed${seed}
    local ckpt=$RUNS/${dir}/checkpoint_best.pt
    local results=$RUNS/${dir}/eval

    if [ ! -f "$ckpt" ]; then
        log_event eval "$regime" "$seed" SKIPPED "\"reason\":\"no ckpt at $ckpt\""
        return 1
    fi
    if [ -d "$results/DEKOIS" ] && [ "$(ls -A $results/DEKOIS 2>/dev/null | wc -l)" -gt 70 ]; then
        log_event eval "$regime" "$seed" skip "\"reason\":\"eval/DEKOIS already populated\",\"results\":\"$results\""
        return 0
    fi

    log_event eval "$regime" "$seed" launching "\"ckpt\":\"$ckpt\",\"gpu\":$gpu"
    local start_ts=$(date +%s)
    bash "$EVAL" "$dir" "$gpu" >> "$RUNS/${dir}_eval.log" 2>&1
    local rc=$?
    local end_ts=$(date +%s)
    local dur=$(( end_ts - start_ts ))

    if [ $rc -eq 0 ]; then
        log_event eval "$regime" "$seed" complete "\"exit\":$rc,\"gpu\":$gpu,\"duration_s\":$dur,\"results\":\"$results\""
    else
        log_event eval "$regime" "$seed" FAILED "\"exit\":$rc,\"gpu\":$gpu,\"duration_s\":$dur"
    fi
    return $rc
}

# ============================================================================
# Main queue
# ============================================================================

log_event queue . . start "\"seed\":$SEED,\"poll_sec\":$POLL_SEC,\"util_max\":$UTIL_MAX,\"mem_max_mib\":$MEM_MAX_MIB"

QUEUE=(
    "dual_clean"
    "scaffold_clean"
    "random_clean_dual"
    "random_clean_scaffold"
    "paper_clean"
)
# Allow override (e.g., to re-run only the regimes that failed): export QUEUE_OVERRIDE="a b c"
if [ -n "${QUEUE_OVERRIDE:-}" ]; then
    # shellcheck disable=SC2206
    QUEUE=( $QUEUE_OVERRIDE )
fi
for regime in "${QUEUE[@]}"; do
    used_gpu=$(train_regime_seed "$regime" "$SEED" || true)
    # If train failed (no gpu echoed) or skipped (no gpu echoed but ckpt exists),
    # fall back to eval-time GPU search.
    if [ -z "${used_gpu:-}" ]; then
        if [ -f "$RUNS/${regime}_seed${SEED}/checkpoint_best.pt" ]; then
            used_gpu=$(wait_for_free_gpu "$regime" "$SEED")
        else
            continue
        fi
    fi
    eval_regime_seed "$regime" "$SEED" "$used_gpu" || true
done

log_event queue . . complete
echo "=== Phase 3 seed=3 queue done at $(date) ==="
