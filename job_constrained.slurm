#!/bin/bash -l
#SBATCH --job-name=ccrl_ant_big
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_40g:1
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
    --cost_epsilon 0.1 \
    --cost_tau 0.05 \
    --cost_budget_d 0.1 \
    --pid_kp 0.1 \
    --pid_ki 0.003 \
    --pid_kd 0.001 \
    --lambda_max 100.0 \
    --cost_critic_lr 3e-4 \
    --cost_critic_width 256 \
    --cost_critic_depth 4 \
    --cost_critic_tau 0.005 \
    --cost_discount 0.99 \
    --dual_estimator_goals 64 \
    --seed 0 \
    --track True \
    --wandb_project "constrained-crl"
