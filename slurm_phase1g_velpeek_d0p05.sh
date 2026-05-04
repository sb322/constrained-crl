#!/bin/bash -l
#SBATCH --job-name=ccrl_1g_velpeek_d05
#SBATCH --output=phase1g_velpeek_d0p05.%j.out
#SBATCH --error=phase1g_velpeek_d0p05.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE-1g BUDGET SWEEP — d=0.05  (peek, 15 epochs, velocity_quadratic)
# ─────────────────────────────────────────────────────────────────────────────
#  One of four peeks queued together to map the binding-regime budget curve
#  under velocity-coupled cost. Sweep: d ∈ {0.05, 0.02, 0.01, 0.005}.
#  Each peek is 15 epochs / 1.5M env steps / ~3.5h wallclock.
#
#  Decision rule across the 4 peeks (applied in the morning):
#    Find the LARGEST d at which λ̃ rises above 0 by epoch 10 AND
#    actor_loss stays bounded. That d is the "natural binding budget"
#    for this env + cost combination. Submit the 50-epoch full run at
#    that budget for the headline H2 result.
# ═════════════════════════════════════════════════════════════════════════════

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

WORKDIR="/mmfs1/home/sb3222/projects/constrained-crl"
VENV="$WORKDIR/.venv"
PYTHON="$VENV/bin/python"
cd "$WORKDIR" || exit 1
export PATH="$VENV/bin:$PATH"

PY_SITE=$("$PYTHON" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
PIP_NVIDIA_LIBS=""
for subpkg in cudnn cusolver cublas cuda_runtime cuda_nvrtc cufft cuda_cupti cusparse nvjitlink nccl; do
    d="$PY_SITE/nvidia/$subpkg/lib"
    [ -d "$d" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$d"
done
SYS_CUDA_LIB="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0/lib64"
export LD_LIBRARY_PATH="${PIP_NVIDIA_LIBS}:${SYS_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="/mmfs1/home/sb3222/.cache/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
nvidia-smi 2>&1 | head -10

echo ""
echo "===== STATIC DIFF + PROBE VERIFICATION ====="
"$PYTHON" - <<'PYCHECK'
import pathlib
src        = pathlib.Path("train.py").read_text()
src_env    = pathlib.Path("envs/ant_maze.py").read_text()
src_buffer = pathlib.Path("buffer.py").read_text()
assert "alpha_max: float = 1.0" in src
assert src.count("sa_norm_safe = jnp.sqrt(") == 2
assert "jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-8" not in src
assert ('cost_type not in ("quadratic", "sigmoid", "velocity_quadratic")' in src_env)
assert 'cost = v_xy_sq * cost' in src_env
assert "hard = v_xy_sq * hard" not in src_env
for m in ['"v_xy_norm"', '"vel_cost_mult"']:
    assert m in src_env
for m in ['extras["v_xy_norm"] = jnp.squeeze', 'extras["vel_cost_mult"] = jnp.squeeze']:
    assert m in src_buffer
for m in ['"mean_v_xy_norm":      jnp.mean(v_xy_norm)',
          '"mean_vel_cost_mult":  jnp.mean(vel_cost_mult)']:
    assert m in src
assert "f\"         nan[obs_c=" in src
assert "f\"         grad[c=" in src
assert "f\"         actor[α=" in src
assert "|v_xy|={log_dict['mean_v_xy_norm']" in src
print("Phase-1g budget-sweep gate verified.")
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== JAX GPU CHECK ====="
"$PYTHON" - <<'PYCHECK'
import jax
print(f"devices = {jax.devices()}, backend = {jax.default_backend()}")
import sys
if jax.default_backend() != "gpu":
    sys.exit(1)
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== LAUNCH d=0.05 peek  $(date) ====="
"$PYTHON" train.py \
    --env_id ant_big_maze --eval_env_id ant_big_maze_eval \
    --episode_length 1000 \
    --total_env_steps 1500000 --num_epochs 15 \
    --num_envs 256 --eval_envs 32 --eval_every 5 \
    --unroll_length 62 --num_minibatches 32 --num_update_epochs 4 --batch_size 256 \
    --seed 0 \
    --obs_dim 29 --goal_start_idx 0 --goal_end_idx 3 \
    --actor_depth 4 --critic_depth 4 --cost_critic_depth 4 \
    --actor_network_width 256 --critic_network_width 256 \
    --actor_lr 3e-4 --critic_lr 3e-4 --alpha_lr 3e-4 \
    --use_constraints True \
    --cost_type velocity_quadratic \
    --cost_epsilon 2.0 --cost_epsilon_hard 0.1 --cost_tau 0.05 \
    --cost_budget_d 0.05 \
    --pid_kp 0.1 --pid_ki 0.003 --pid_kd 0.001 --lambda_max 100.0 \
    --wandb_project constrained-crl \
    --wandb_group phase1g_velpeek_d0p05 \
    --track True

EXIT_CODE=$?
echo "===== DONE d=0.05 peek  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
