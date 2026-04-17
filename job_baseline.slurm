#!/bin/bash -l
#SBATCH --job-name=crl_ant_big_baseline
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

# ── NJIT Wulver HPC — Baseline CRL (no constraints) ───────────
# Usage:  sbatch job_baseline.slurm
# Prereq: uv sync  (run once after cloning)

module purge

mkdir -p logs checkpoints

# ── Ant Big Maze — Unconstrained baseline ──────────────────────
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
    --use_constraints False \
    --seed 0 \
    --track True \
    --wandb_project "constrained-crl"
