#!/bin/bash -l
#SBATCH --job-name=ccrl_depth
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-7

# ══════════════════════════════════════════════════════════════════════════════
#  NJIT Wulver HPC — SR-CPO DEPTH ABLATION
#
#  Research question:
#    How does network depth affect CMDP feasibility and constraint-violation
#    behavior of SR-CPO on Ant Big Maze?  Specifically, does deeper actor/
#    critic accelerate convergence of the Lagrangian (λ) and of the
#    cost-critic estimate (Ĵ_c) toward the budget d, or does it just
#    increase compile/step cost without improving constraint satisfaction?
#
#  Design:
#    4 depths × 2 seeds = 8 array tasks, each a ~1–2 h run after its first
#    jit-trace compile. Env, horizon, cost structure, PID gains, budget,
#    env-step budget are held fixed at the baseline config — only actor_depth,
#    critic_depth, actor_skip_connections, critic_skip_connections vary.
#
#  Grid:
#    DEPTHS  = (2 4 6 8)               → actor_depth = critic_depth
#    SEEDS   = (0 42)
#    skip    = depth / 2               (so depth=2→skip=1, ..., depth=8→skip=4)
#
#    task_id  →  (depth, seed)
#       0     →  (2,  0)    4     →  (6,  0)
#       1     →  (2, 42)    5     →  (6, 42)
#       2     →  (4,  0)    6     →  (8,  0)
#       3     →  (4, 42)    7     →  (8, 42)
#
#  Wall budget per task:  5 h   (ceiling; most depths finish in well under 2 h)
#  Env-step budget:       20M   (200 epochs × ~100k env-steps/epoch)
#  wandb_group:           "depth_ablation"
#
#  Submit:  sbatch job_depth_sweep.slurm
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

# ── JAX compilation cache ─────────────────────────────────────────────────
export JAX_COMPILATION_CACHE_DIR="$(pwd)/.jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ── Depth × seed grid ─────────────────────────────────────────────────────
DEPTHS=(2 4 6 8)
SEEDS=(0 42)

N_DEPTHS=${#DEPTHS[@]}
N_SEEDS=${#SEEDS[@]}

SEED_IDX=$((  SLURM_ARRAY_TASK_ID               % N_SEEDS  ))
DEPTH_IDX=$(( SLURM_ARRAY_TASK_ID               / N_SEEDS  ))

DEPTH=${DEPTHS[$DEPTH_IDX]}
SEED=${SEEDS[$SEED_IDX]}
SKIP=$(( DEPTH / 2 ))

echo "===== ARRAY TASK $SLURM_ARRAY_TASK_ID ====="
echo "depth=$DEPTH  skip=$SKIP  seed=$SEED"
nvidia-smi | head -5

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

# ── Depth-ablation run (20M env-steps / 200 epochs at baseline config) ────
"$PYTHON" -u train.py \
    --env_id                "ant_big_maze" \
    --eval_env_id           "ant_big_maze_eval" \
    --num_epochs            200 \
    --total_env_steps       20000000 \
    --num_envs              512 \
    --num_minibatches       8 \
    --critic_depth          "$DEPTH" \
    --actor_depth           "$DEPTH" \
    --actor_skip_connections "$SKIP" \
    --critic_skip_connections "$SKIP" \
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
    --seed                  "$SEED" \
    --track                 True \
    --wandb_project         "constrained-crl" \
    --wandb_group           "depth_ablation"

echo "===== ARRAY TASK $SLURM_ARRAY_TASK_ID DONE ====="
