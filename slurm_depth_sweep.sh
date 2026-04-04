#!/bin/bash
#SBATCH --job-name=ccrl_depth
#SBATCH --array=0-7                  # 8 jobs: 4 depths x 2 seeds
#SBATCH --gres=gpu:1                 # 1 GPU per job
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/ccrl_depth_%A_%a.out
#SBATCH --error=logs/ccrl_depth_%A_%a.err

# ── Map array index → (depth, seed) ──────────────────────────────────────────
# Array layout:  index = depth_idx * N_SEEDS + seed
#   0 → depth=2 seed=0 |  1 → depth=2 seed=1
#   2 → depth=4 seed=0 |  3 → depth=4 seed=1
#   4 → depth=6 seed=0 |  5 → depth=6 seed=1
#   6 → depth=8 seed=0 |  7 → depth=8 seed=1

DEPTHS=(2 4 6 8)
N_SEEDS=2

DEPTH_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
DEPTH=${DEPTHS[$DEPTH_IDX]}

echo "======================================================"
echo "  SLURM Array Job: $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "  depth=$DEPTH  seed=$SEED"
echo "  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# ── Environment setup ─────────────────────────────────────────────────────────
# Adjust the conda env name / module loads to match your Wulver setup
module purge
module load cuda/12.1              # adjust to your CUDA version on Wulver

# Activate your conda/venv environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ccrl                # change to your env name

# ── Project directory ─────────────────────────────────────────────────────────
PROJECT_DIR=~/constrained-crl/constrained_crl   # adjust if different
cd $PROJECT_DIR

# ── Log directory ─────────────────────────────────────────────────────────────
mkdir -p logs

# ── Run ───────────────────────────────────────────────────────────────────────
python train.py \
    --env_id            humanoid_big_maze \
    --eval_env_id       humanoid_big_maze_eval \
    --depth             $DEPTH \
    --seed              $SEED \
    --total_env_steps   100000000 \
    --num_epochs        100 \
    --use_constraints   True \
    --cost_budget_d     0.1 \
    --track             True \
    --wandb_group       depth_ablation_v1

echo "Job finished: depth=$DEPTH seed=$SEED"
