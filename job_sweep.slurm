#!/bin/bash -l
#SBATCH --job-name=ccrl_sweep
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=high_ag2682
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11

# ── NJIT Wulver HPC — Hyperparameter sweep ────────────────────
# Sweeps over cost_budget_d × lambda_lr × seed
# Usage:  sbatch job_sweep.slurm
# Prereq: uv sync

module purge
mkdir -p logs checkpoints

# ── Sweep grid ─────────────────────────────────────────────────
# SR-CPO sweep: budget × PID_Ki × seed
BUDGETS=(0.05 0.1 0.2)
PID_KIS=(0.003 0.001)
SEEDS=(0 42)

# Decode SLURM_ARRAY_TASK_ID into grid indices
N_BUDGETS=${#BUDGETS[@]}
N_KIS=${#PID_KIS[@]}
N_SEEDS=${#SEEDS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
REMAINING=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
KI_IDX=$(( REMAINING % N_KIS ))
BUDGET_IDX=$(( REMAINING / N_KIS ))

BUDGET=${BUDGETS[$BUDGET_IDX]}
KI=${PID_KIS[$KI_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: budget=$BUDGET, pid_ki=$KI, seed=$SEED"

uv run train.py \
    --env_id "ant_big_maze" \
    --eval_env_id "ant_big_maze_eval" \
    --num_epochs 100 \
    --total_env_steps 100000000 \
    --critic_depth 8 \
    --actor_depth 8 \
    --actor_skip_connections 4 \
    --critic_skip_connections 4 \
    --batch_size 512 \
    --vis_length 1000 \
    --save_buffer 0 \
    --use_constraints True \
    --cost_epsilon 0.1 \
    --cost_tau 0.05 \
    --cost_budget_d "$BUDGET" \
    --pid_kp 0.1 \
    --pid_ki "$KI" \
    --pid_kd 0.001 \
    --lambda_max 100.0 \
    --cost_critic_lr 3e-4 \
    --cost_critic_width 256 \
    --cost_critic_depth 4 \
    --cost_critic_tau 0.005 \
    --dual_estimator_goals 64 \
    --seed "$SEED" \
    --track True \
    --wandb_project "constrained-crl" \
    --wandb_group "sweep_budget_pid"
