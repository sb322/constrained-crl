#!/bin/bash -l
#SBATCH --job-name=ccrl_depth_sweep
#SBATCH --output=ccrl_sweep.%j.out
#SBATCH --error=ccrl_sweep.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=high_ag2682
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=192:00:00
#SBATCH --constraint=a100

# Single job — all (depth × seed) runs sequentially on one A100.
# Requests a full A100 (not MIG slice) via --constraint and generic gpu:1.

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

WORKDIR="/mmfs1/home/sb3222/projects/constrained-crl"
VENV="$WORKDIR/.venv"
PYTHON="$VENV/bin/python"

cd "$WORKDIR/constrained_crl" || exit 1

export PATH="$VENV/bin:$PATH"

# ── LD_LIBRARY_PATH: pip nvidia packages FIRST, system CUDA as fallback ──
PY_SITE=$("$PYTHON" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

# Pip-installed nvidia libs (these contain cuDNN, cuBLAS, cuSOLVER, etc.)
PIP_NVIDIA_LIBS=""
for subpkg in cudnn cusolver cublas cuda_runtime cuda_nvrtc cufft cuda_cupti cusparse; do
    d="$PY_SITE/nvidia/$subpkg/lib"
    [ -d "$d" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$d"
done

# System CUDA libs as fallback (in case pip packages are incomplete)
SYS_CUDA_LIB="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0/lib64"

export LD_LIBRARY_PATH="${PIP_NVIDIA_LIBS}:${SYS_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# JAX memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# ── Diagnostic checks ────────────────────────────────────────────────────────
echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 | head -20

echo ""
echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

echo ""
echo "===== cuSOLVER CHECK ====="
# Verify cuSOLVER is actually loadable
"$PYTHON" -c "
import ctypes, os
for p in os.environ.get('LD_LIBRARY_PATH','').split(':'):
    lib = os.path.join(p, 'libcusolver.so')
    if os.path.exists(lib):
        print(f'  found: {lib}')
        try:
            ctypes.CDLL(lib)
            print(f'  loaded OK')
            break
        except Exception as e:
            print(f'  load FAILED: {e}')
else:
    print('  WARNING: libcusolver.so not found in any LD_LIBRARY_PATH entry')
"

echo ""
echo "===== JAX GPU CHECK ====="
"$PYTHON" - <<'PYCHECK'
import jax
devices = jax.devices()
backend = jax.default_backend()
print(f"devices = {devices}")
print(f"default_backend = {backend}")
if backend != "gpu":
    print("FATAL: JAX did not detect GPU. Aborting.")
    import sys; sys.exit(1)
print("GPU OK — proceeding with sweep.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "JAX GPU check failed — aborting job."
    exit 1
fi

# ── Sequential depth sweep ────────────────────────────────────────────────────
DEPTHS=(2 4 6 8)
SEEDS=(0 1)

echo ""
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
