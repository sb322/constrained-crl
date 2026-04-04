#!/bin/bash -l
#SBATCH --job-name=ccrl_depth_sweep
#SBATCH --output=ccrl_sweep.%j.out
#SBATCH --error=ccrl_sweep.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=high_ag2682
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=4G
#SBATCH --time=192:00:00

# Single job — all 8 (depth × seed) runs sequentially on one A100.

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

WORKDIR="/mmfs1/home/sb3222/projects/constrained-crl"
VENV="$WORKDIR/.venv"
PYTHON="$VENV/bin/python"
CUDA_BIN="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0/bin"

cd "$WORKDIR/constrained_crl" || exit 1

export PATH="$VENV/bin:$CUDA_BIN:$PATH"

PY_SITE=$("$PYTHON" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export LD_LIBRARY_PATH="$PY_SITE/nvidia/cudnn/lib:$PY_SITE/nvidia/cusolver/lib:$PY_SITE/nvidia/cublas/lib:$PY_SITE/nvidia/cuda_runtime/lib:$PY_SITE/nvidia/cuda_nvrtc/lib:$PY_SITE/nvidia/cufft/lib:$PY_SITE/nvidia/cuda_cupti/lib"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "===== PATH CHECK ====="
echo "$PATH"
echo "===== PYTHON CHECK ====="
which python
"$PYTHON" --version
echo "===== JAX CHECK ====="
"$PYTHON" - <<'PY'
import jax
print("devices =", jax.devices())
print("default_backend =", jax.default_backend())
PY

# ── Sequential depth sweep ────────────────────────────────────────────────────
DEPTHS=(2 4 6 8)
SEEDS=(0 1)

echo "===== START DEPTH SWEEP ====="
for DEPTH in "${DEPTHS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- depth=$DEPTH  seed=$SEED  $(date) ---"

        "$PYTHON" train.py \
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

        echo "--- DONE: depth=$DEPTH  seed=$SEED  $(date) ---"
    done
done

echo "===== ALL RUNS COMPLETE ====="
