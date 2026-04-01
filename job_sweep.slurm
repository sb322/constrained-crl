#!/bin/bash
#SBATCH --job-name=ccrl_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
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
BUDGETS=(0.05 0.1 0.2)
LAMBDA_LRS=(1e-3 5e-4)
SEEDS=(0 42)

# Decode SLURM_ARRAY_TASK_ID into grid indices
N_BUDGETS=${#BUDGETS[@]}
N_LRS=${#LAMBDA_LRS[@]}
N_SEEDS=${#SEEDS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
REMAINING=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
LR_IDX=$(( REMAINING % N_LRS ))
BUDGET_IDX=$(( REMAINING / N_LRS ))

BUDGET=${BUDGETS[$BUDGET_IDX]}
LR=${LAMBDA_LRS[$LR_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: budget=$BUDGET, lambda_lr=$LR, seed=$SEED"

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
    --alpha_cost 0.5 \
    --sigma_wall 1.0 \
    --cost_budget_d "$BUDGET" \
    --lambda_lr "$LR" \
    --lambda_max 100.0 \
    --cost_critic_lr 3e-4 \
    --cost_critic_width 256 \
    --cost_critic_depth 4 \
    --cost_critic_tau 0.005 \
    --seed "$SEED" \
    --track True \
    --wandb_project "constrained-crl" \
    --wandb_group "sweep_budget_lr"
