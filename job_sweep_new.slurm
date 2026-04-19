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

cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "cd to SLURM_SUBMIT_DIR failed"; exit 1; }
mkdir -p logs checkpoints

if   [ -x "./.venv/bin/python" ];        then VENV="$(pwd)/.venv"
elif [ -x "../.venv/bin/python" ];       then VENV="$(cd .. && pwd)/.venv"
else echo "FATAL: no .venv/bin/python found in $(pwd) or $(dirname $(pwd))"; exit 1
fi
PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
echo "Using PYTHON=$PYTHON  (cwd=$(pwd))"

PY_SITE=$("$PYTHON" - <<'PY'
import site
for p in site.getsitepackages():
    print(p)
PY
)

PIP_NVIDIA_LIBS=""
for SITE in $PY_SITE; do
    # nvjitlink MUST precede cusparse so the loader finds the venv's newer
    # libnvJitLink.so.12 (with 12.9 symbols) before any system CUDA 12.8 copy.
    for subpkg in nvjitlink cudnn cusolver cublas cuda_runtime cuda_nvrtc cufft cuda_cupti cusparse nccl; do
        d="$SITE/nvidia/$subpkg/lib"
        [ -d "$d" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$d"
    done
done

if ! echo "$PIP_NVIDIA_LIBS" | grep -q cusolver; then
    echo "[warn] canonical nvidia/*/lib layout not found — falling back to find()"
    FOUND=$(find "$VENV" -name 'libcusolver.so*' -not -path '*/torch/*' 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        EXTRA=$(dirname "$FOUND")
        PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$EXTRA"
        NVIDIA_ROOT=$(echo "$EXTRA" | sed 's|/nvidia/.*||')/nvidia
        if [ -d "$NVIDIA_ROOT" ]; then
            for sub in "$NVIDIA_ROOT"/*/lib; do
                [ -d "$sub" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS}:$sub"
            done
        fi
    fi
fi

SYS_CUDA_LIB="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0/lib64"
export LD_LIBRARY_PATH="${PIP_NVIDIA_LIBS}:${SYS_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# ── stdout/stderr flushing (critical for visible epoch progress) ──
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

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

echo ""
echo "===== PIP NVIDIA PACKAGES ====="
"$PYTHON" -m pip list 2>/dev/null | grep -i -E '^(nvidia|jax|jaxlib)' || echo "  (none found)"

echo ""
echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

echo ""
echo "===== cuSOLVER LOAD CHECK ====="
"$PYTHON" - <<'PYCHECK_LIB'
import ctypes, os, sys
paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
loaded = False
for p in paths:
    for name in ('libcusolver.so.11', 'libcusolver.so'):
        lib = os.path.join(p, name)
        if os.path.exists(lib):
            print(f"  found: {lib}")
            try:
                ctypes.CDLL(lib)
                print(f"  loaded OK: {lib}")
                loaded = True
                break
            except Exception as e:
                print(f"  load FAILED: {e}")
    if loaded:
        break
if not loaded:
    print("  FATAL: libcusolver.so could not be loaded from any LD_LIBRARY_PATH entry.")
    sys.exit(1)
PYCHECK_LIB
if [ $? -ne 0 ]; then echo "cuSOLVER preflight failed."; exit 1; fi

"$PYTHON" - <<'PYCHECK'
import jax
assert jax.default_backend() == "gpu", "JAX did not detect the GPU"
print("jax.devices() =", jax.devices())
PYCHECK
if [ $? -ne 0 ]; then echo "JAX GPU check failed."; exit 1; fi

# Same throughput/visibility fixes as job_train.slurm: num_envs=512,
# num_epochs=1000, num_minibatches=8, -u / PYTHONUNBUFFERED.
"$PYTHON" -u train.py \
    --env_id                "ant_big_maze" \
    --eval_env_id           "ant_big_maze_eval" \
    --num_epochs            1000 \
    --total_env_steps       100000000 \
    --num_envs              512 \
    --num_minibatches       8 \
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
