#!/bin/bash -l
#SBATCH --job-name=ccrl_probe
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
#  NJIT Wulver HPC — FAST PROBE for SR-CPO (Constrained CRL)
#
#  Purpose:
#    Before launching long runs, confirm that the pipeline actually completes
#    an epoch at the FULL env + network config (ant_big_maze, depth 8) within
#    a few minutes.  If epoch 1 does not return in < 10 minutes here, there's
#    a real bug in training_epoch to hunt — otherwise the full run will work.
#
#  Config differences vs job_train.slurm (everything else identical):
#    --total_env_steps 500000   (was 100000000) → 1/200 of the total work
#    --num_epochs      5        (was 1000)      → 5 epoch prints expected
#    --num_minibatches 4        (was 32)        → smaller SGD inner-loop
#    --num_update_epochs 1      (was 4)         → "
#    --eval_every 5             (eval only at the end)
#    --wandb_group "probe"      (keeps probes separate in wandb)
#
#  Expected wall time:  ≤ 10 minutes end-to-end.
#
#  Submit:  sbatch job_probe.slurm
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

echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
nvidia-smi | head -20

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

# ── Fast probe — full env + networks, tiny workload ──────────────────────
"$PYTHON" -u train.py \
    --env_id                "ant_big_maze" \
    --eval_env_id           "ant_big_maze_eval" \
    --num_epochs            5 \
    --total_env_steps       500000 \
    --num_envs              512 \
    --num_minibatches       4 \
    --num_update_epochs     1 \
    --eval_every            5 \
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
    --wandb_group           "probe"

echo "===== PROBE RUN COMPLETE ====="
