#!/bin/bash -l
#SBATCH --job-name=ccrl_sweep
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11

# ══════════════════════════════════════════════════════════════════════════════
#  NJIT Wulver HPC — SR-CPO hyperparameter sweep
#  Grid:  cost_budget_d × pid_ki × seed   (3 × 2 × 2 = 12 tasks)
#  Submit:  sbatch job_sweep_new.slurm
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

# ── Sweep grid ─────────────────────────────────────────────────────────────
BUDGETS=(0.05 0.1 0.2)
PID_KIS=(0.003 0.001)
SEEDS=(0 42)

N_BUDGETS=${#BUDGETS[@]}
N_KIS=${#PID_KIS[@]}
N_SEEDS=${#SEEDS[@]}

SEED_IDX=$((  SLURM_ARRAY_TASK_ID               % N_SEEDS   ))
REM=$((       SLURM_ARRAY_TASK_ID               / N_SEEDS   ))
KI_IDX=$((    REM                               % N_KIS     ))
BUDGET_IDX=$((REM                               / N_KIS     ))

BUDGET=${BUDGETS[$BUDGET_IDX]}
KI=${PID_KIS[$KI_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "===== ARRAY TASK $SLURM_ARRAY_TASK_ID ====="
echo "budget=$BUDGET  pid_ki=$KI  seed=$SEED"
nvidia-smi | head -5

"$PYTHON" - <<'PYCHECK'
import jax
assert jax.default_backend() == "gpu", "JAX did not detect the GPU"
print("jax.devices() =", jax.devices())
PYCHECK
if [ $? -ne 0 ]; then echo "JAX GPU check failed."; exit 1; fi

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
    --cost_budget_d         "$BUDGET" \
    --pid_kp                0.1 \
    --pid_ki                "$KI" \
    --pid_kd                0.001 \
    --lambda_max            100.0 \
    --cost_critic_lr        3e-4 \
    --cost_critic_width     256 \
    --cost_critic_depth     4 \
    --cost_critic_tau       0.005 \
    --cost_discount         0.99 \
    --dual_estimator_goals  64 \
    --seed                  "$SEED" \
    --track                 True \
    --wandb_project         "constrained-crl" \
    --wandb_group           "sweep_budget_pid"

echo "===== ARRAY TASK $SLURM_ARRAY_TASK_ID DONE ====="
