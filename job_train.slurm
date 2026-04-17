#!/bin/bash -l
#SBATCH --job-name=ccrl_train
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

# ══════════════════════════════════════════════════════════════════════════════
#  NJIT Wulver HPC — SR-CPO full training (single seed, Ant Big Maze)
#  Submit:  sbatch job_train.slurm
# ══════════════════════════════════════════════════════════════════════════════

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

WORKDIR="/mmfs1/home/sb3222/projects/constrained-crl"
VENV="$WORKDIR/.venv"
PYTHON="$VENV/bin/python"

cd "$WORKDIR/constrained_crl" || exit 1
mkdir -p logs checkpoints

export PATH="$VENV/bin:$PATH"

PY_SITE=$("$PYTHON" - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)

PIP_NVIDIA_LIBS=""
for subpkg in cudnn cusolver cublas cuda_runtime cuda_nvrtc cufft cuda_cupti cusparse; do
    d="$PY_SITE/nvidia/$subpkg/lib"
    [ -d "$d" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$d"
done
SYS_CUDA_LIB="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0/lib64"
export LD_LIBRARY_PATH="${PIP_NVIDIA_LIBS}:${SYS_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
nvidia-smi | head -20

"$PYTHON" - <<'PYCHECK'
import jax
print("jax.devices()     =", jax.devices())
print("default_backend() =", jax.default_backend())
assert jax.default_backend() == "gpu", "JAX did not detect the GPU"
PYCHECK
if [ $? -ne 0 ]; then echo "JAX GPU check failed."; exit 1; fi

# ── Full SR-CPO training run ──────────────────────────────────────────────
"$PYTHON" train.py \
    --env_id                "ant_big_maze" \
    --eval_env_id           "ant_big_maze_eval" \
    --num_epochs            100 \
    --total_env_steps       100000000 \
    --critic_depth          8 \
    --actor_depth           8 \
    --actor_skip_connections 4 \
    --critic_skip_connections 4 \
    --batch_size            512 \
    --vis_length            1000 \
    --save_buffer           0 \
    --use_constraints       True \
    --cost_epsilon          0.1 \
    --cost_tau              0.05 \
    --cost_budget_d         0.1 \
    --pid_kp                0.1 \
    --pid_ki                0.003 \
    --pid_kd                0.001 \
    --lambda_max            100.0 \
    --cost_critic_lr        3e-4 \
    --cost_critic_width     256 \
    --cost_critic_depth     4 \
    --cost_critic_tau       0.005 \
    --cost_discount         0.99 \
    --dual_estimator_goals  64 \
    --seed                  0 \
    --track                 True \
    --wandb_project         "constrained-crl" \
    --wandb_group           "main_run"

echo "===== TRAIN RUN COMPLETE ====="
