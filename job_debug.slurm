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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# ── Preflight ────────────────────────────────────────────────────────────
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
