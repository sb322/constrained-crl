#!/usr/bin/env bash
# run_depth_sweep.sh — depth ablation sweep for Constrained CRL
#
# Launches one run per depth level (L ∈ {2, 4, 6, 8}) × N_SEEDS seeds.
# All runs share the same wandb group so they appear together in the
# Pareto-frontier plots (reward vs constraint_violation).
#
# Usage:
#   bash run_depth_sweep.sh                    # runs on GPU 0
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_depth_sweep.sh
#   N_SEEDS=3 bash run_depth_sweep.sh          # 3 seeds per depth
#
# Requirements:
#   - Python environment with all C-CRL dependencies installed
#   - WANDB_API_KEY exported in environment (or --track False to disable)

set -euo pipefail

# ── Sweep configuration ───────────────────────────────────────────────────────
DEPTHS=(2 4 6 8)
N_SEEDS=${N_SEEDS:-2}         # seeds: 0 … N_SEEDS-1
WANDB_GROUP=${WANDB_GROUP:-"depth_ablation_v1"}
ENV_ID=${ENV_ID:-"humanoid_big_maze"}
EVAL_ENV_ID=${EVAL_ENV_ID:-"humanoid_big_maze_eval"}
TOTAL_STEPS=${TOTAL_STEPS:-100_000_000}
NUM_EPOCHS=${NUM_EPOCHS:-100}
COST_BUDGET=${COST_BUDGET:-0.1}
TRACK=${TRACK:-true}
PARALLEL=${PARALLEL:-false}   # set PARALLEL=true to background all jobs

echo "==========================================================="
echo "  C-CRL Depth Ablation Sweep"
echo "  depths:  ${DEPTHS[*]}"
echo "  seeds:   0 .. $((N_SEEDS - 1))"
echo "  group:   $WANDB_GROUP"
echo "  env:     $ENV_ID"
echo "  budget:  $COST_BUDGET"
echo "  parallel: $PARALLEL"
echo "==========================================================="

PIDS=()

for DEPTH in "${DEPTHS[@]}"; do
    for SEED in $(seq 0 $((N_SEEDS - 1))); do
        RUN_TAG="d${DEPTH}_s${SEED}"
        LOG_FILE="logs/${WANDB_GROUP}_${RUN_TAG}.log"
        mkdir -p logs

        CMD="python train.py \
            --env_id            $ENV_ID \
            --eval_env_id       $EVAL_ENV_ID \
            --depth             $DEPTH \
            --seed              $SEED \
            --total_env_steps   $TOTAL_STEPS \
            --num_epochs        $NUM_EPOCHS \
            --use_constraints   True \
            --cost_budget_d     $COST_BUDGET \
            --track             $TRACK \
            --wandb_group       $WANDB_GROUP"

        echo ""
        echo "--- Launching: depth=$DEPTH seed=$SEED → $LOG_FILE"
        echo "    $CMD"

        if [ "$PARALLEL" = "true" ]; then
            # Background: log to file, collect PID for wait
            eval "$CMD" > "$LOG_FILE" 2>&1 &
            PIDS+=($!)
            echo "    PID $!"
        else
            # Sequential: stream output and also save log
            eval "$CMD" 2>&1 | tee "$LOG_FILE"
        fi
    done
done

# If running in parallel, wait for all jobs and report exit codes
if [ "$PARALLEL" = "true" ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo "All jobs launched.  Waiting for ${#PIDS[@]} processes ..."
    FAILED=0
    for PID in "${PIDS[@]}"; do
        if wait "$PID"; then
            echo "  PID $PID finished OK"
        else
            echo "  PID $PID FAILED (exit $?)"
            FAILED=$((FAILED + 1))
        fi
    done
    echo ""
    if [ $FAILED -eq 0 ]; then
        echo "Sweep complete — all runs succeeded."
    else
        echo "Sweep complete — $FAILED run(s) failed.  Check logs/ for details."
        exit 1
    fi
else
    echo ""
    echo "Sweep complete."
fi
