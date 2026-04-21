#!/bin/bash -l
#SBATCH --job-name=ccrl_phase1b_fast
#SBATCH --output=phase1b_fast.%j.out
#SBATCH --error=phase1b_fast.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ═════════════════════════════════════════════════════════════════════════════
#  Phase-1b FAST SMOKE — de-risk the 18 h overnight run.
#
#  300 k env steps, 3 epochs, ~30–45 min on A100.  The ONLY purpose is to
#  catch the v1a failure modes before committing overnight:
#
#    (a) mean_step_cost ≡ 0             → compact-support didn't land
#    (b) λ̃ ≡ 0 AND Ĵ_c > d             → PID integrator broken
#    (c) actor_loss diverging to −10⁵   → cost penalty not applied to policy
#    (d) NaN anywhere                   → gradient path issue
#
#  If any of (a)–(d) appear in the first 3 epochs of this smoke, do NOT
#  submit slurm_phase1_smoke.sh — fix first.  If all four are clean,
#  the overnight run is safe to submit.
#
#  Identical hyperparameters to the overnight job EXCEPT:
#    --total_env_steps  5 M → 300 k       (≈40× shorter)
#    --num_epochs        50 → 3
#    --eval_every         5 → 1           (want all 3 eval points)
#    --wandb_group phase1b_smoke → phase1b_fastsmoke
#    SBATCH --time      18 h → 1 h
# ═════════════════════════════════════════════════════════════════════════════

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

WORKDIR="/mmfs1/home/sb3222/projects/constrained-crl"
VENV="$WORKDIR/.venv"
PYTHON="$VENV/bin/python"

cd "$WORKDIR/constrained_crl" || exit 1

export PATH="$VENV/bin:$PATH"

# Same LD_LIBRARY_PATH setup as overnight script
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
echo "PYTHON=$("$PYTHON" --version 2>&1)"
nvidia-smi 2>&1 | head -5

# ── Tier 1 preflight: run probe_minimal.py to verify cost plumbing ────────
echo ""
echo "===== TIER 1 — ENV PROBE ====="
"$PYTHON" probe_minimal.py 2>&1 | tail -25
PROBE_EXIT=$?
if [ $PROBE_EXIT -ne 0 ]; then
    echo "probe_minimal.py failed — aborting before training launch."
    exit $PROBE_EXIT
fi

# ── Tier 2 fast smoke training ────────────────────────────────────────────
echo ""
echo "===== TIER 2 — FAST SMOKE TRAINING  $(date) ====="

"$PYTHON" train.py \
    --env_id            ant_big_maze \
    --eval_env_id       ant_big_maze_eval \
    --episode_length    1000 \
    --total_env_steps   300000 \
    --num_epochs        3 \
    --num_envs          256 \
    --eval_envs         32 \
    --eval_every        1 \
    --unroll_length     62 \
    --num_minibatches   32 \
    --num_update_epochs 4 \
    --batch_size        256 \
    --seed              0 \
    --obs_dim           29 \
    --goal_start_idx    0 \
    --goal_end_idx      3 \
    --actor_depth       4 \
    --critic_depth      4 \
    --cost_critic_depth 4 \
    --actor_network_width  256 \
    --critic_network_width 256 \
    --actor_lr          3e-4 \
    --critic_lr         3e-4 \
    --alpha_lr          3e-4 \
    --use_constraints   True \
    --cost_type         quadratic \
    --cost_epsilon      2.0 \
    --cost_epsilon_hard 0.1 \
    --cost_tau          0.05 \
    --cost_budget_d     0.15 \
    --pid_kp            0.1 \
    --pid_ki            0.003 \
    --pid_kd            0.001 \
    --lambda_max        100.0 \
    --wandb_project     constrained-crl \
    --wandb_group       phase1b_fastsmoke \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== FAST SMOKE DONE  $(date)  exit=$EXIT_CODE ====="
echo ""
echo "───── GATE FOR OVERNIGHT SUBMISSION ────────────────────────────────"
echo "Check wandb group phase1b_fastsmoke for these 4 signals over 3 epochs:"
echo "  1. train/mean_step_cost  >  0.05                 (cost field alive)"
echo "  2. NOT (lambda_tilde ≡ 0  AND  jhat_c > 0.15)   (PID can respond)"
echo "  3. |train/actor_loss|     <  1e4                 (cost penalty bites)"
echo "  4. all losses finite                             (no NaN)"
echo ""
echo "If all four pass:  sbatch slurm_phase1_smoke.sh"
echo "If any fail:       inspect the .out file BEFORE resubmitting."
echo "────────────────────────────────────────────────────────────────────"
exit $EXIT_CODE
