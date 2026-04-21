#!/bin/bash -l
#SBATCH --job-name=ccrl_debug
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ══════════════════════════════════════════════════════════════════════════════
#  NJIT Wulver HPC — SR-CPO debug / smoke test
# ══════════════════════════════════════════════════════════════════════════════
#  Short run (≤30 min) to verify:
#    1. JAX detects the A100 (40g MIG slice)
#    2. Brax envs register and reset
#    3. Buffer prefill, SGD step, and evaluator all execute end-to-end
#    4. wandb logging is reachable
#  Use this BEFORE launching the real 24h job.
#  Submit:  sbatch job_debug.slurm
# ══════════════════════════════════════════════════════════════════════════════

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

# Run from the directory you submitted sbatch from — no hard-coded path.
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "cd to SLURM_SUBMIT_DIR failed"; exit 1; }
mkdir -p logs checkpoints

# Locate the venv: .venv in this dir, else one level up.
if   [ -x "./.venv/bin/python" ];        then VENV="$(pwd)/.venv"
elif [ -x "../.venv/bin/python" ];       then VENV="$(cd .. && pwd)/.venv"
else echo "FATAL: no .venv/bin/python found in $(pwd) or $(dirname $(pwd))"; exit 1
fi
PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
echo "Using PYTHON=$PYTHON  (cwd=$(pwd))"

# ── LD_LIBRARY_PATH: pip-installed nvidia libs first, then system CUDA ──
PY_SITE=$("$PYTHON" - <<'PY'
import site
# Print ALL site-package dirs (getsitepackages may return a dir that isn't the one
# with nvidia libs in some venvs).
for p in site.getsitepackages():
    print(p)
PY
)

PIP_NVIDIA_LIBS=""
for SITE in $PY_SITE; do
    for subpkg in cudnn cusolver cublas cuda_runtime cuda_nvrtc cufft cuda_cupti cusparse nccl; do
        d="$SITE/nvidia/$subpkg/lib"
        [ -d "$d" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$d"
    done
done

# Fallback: if canonical layout missed it, locate libcusolver.so anywhere under the venv.
if ! echo "$PIP_NVIDIA_LIBS" | grep -q cusolver; then
    echo "[warn] canonical nvidia/*/lib layout not found — falling back to find()"
    FOUND=$(find "$VENV" -name 'libcusolver.so*' -not -path '*/torch/*' 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        EXTRA=$(dirname "$FOUND")
        PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$EXTRA"
        # Also add sibling nvidia libs if they're all in the same tree.
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# ── Preflight ────────────────────────────────────────────────────────────
echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
nvidia-smi | head -20

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
if [ $? -ne 0 ]; then echo "cuSOLVER preflight failed — JAX will not see the GPU."; exit 1; fi

echo ""
echo "===== JAX GPU CHECK ====="
"$PYTHON" - <<'PYCHECK'
import jax
print("jax.devices()     =", jax.devices())
print("default_backend() =", jax.default_backend())
assert jax.default_backend() == "gpu", "JAX did not detect the GPU"
PYCHECK
if [ $? -ne 0 ]; then echo "JAX GPU check failed."; exit 1; fi

# ── Tiny SR-CPO run  (≈ 2 epochs, ~5 min on A100 40g) ─────────────────────
"$PYTHON" train.py \
    --env_id            "ant_u_maze" \
    --eval_env_id       "ant_u_maze_eval" \
    --num_epochs        2 \
    --total_env_steps   500000 \
    --num_envs          64 \
    --eval_envs         16 \
    --batch_size        128 \
    --unroll_length     32 \
    --num_minibatches   4 \
    --num_update_epochs 1 \
    --min_replay_size   512 \
    --max_replay_size   2000 \
    --episode_length    200 \
    --vis_length        200 \
    --actor_depth       4 \
    --critic_depth      4 \
    --cost_critic_depth 4 \
    --save_buffer       0 \
    --use_constraints   True \
    --cost_epsilon      0.1 \
    --cost_tau          0.05 \
    --cost_budget_d     0.1 \
    --pid_kp            0.1 \
    --pid_ki            0.003 \
    --pid_kd            0.001 \
    --lambda_max        100.0 \
    --cost_critic_lr    3e-4 \
    --cost_critic_tau   0.005 \
    --cost_discount     0.99 \
    --dual_estimator_goals 16 \
    --seed              0 \
    --track             False \
    --wandb_project     "constrained-crl"

echo "===== DEBUG RUN COMPLETE ====="
