#!/bin/bash
#SBATCH --job-name=ccrl_ant_big
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── NJIT Wulver HPC — Constrained CRL ─────────────────────────
# Usage:  sbatch job_constrained.slurm
# Prereq: uv sync  (run once after cloning)

module purge

# Create log directory
mkdir -p logs checkpoints

# ── Ant Big Maze — Constrained ─────────────────────────────────
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
    --contact_threshold 0.1 \
    --cost_budget_d 0.1 \
    --lambda_init 0.0 \
    --lambda_lr 1e-3 \
    --lambda_max 100.0 \
    --cost_critic_lr 3e-4 \
    --cost_critic_width 256 \
    --cost_critic_depth 4 \
    --cost_critic_tau 0.005 \
    --cost_discount 0.99 \
    --seed 0 \
    --track True \
    --wandb_project "constrained-crl"
