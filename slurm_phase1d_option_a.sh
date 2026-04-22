#!/bin/bash -l
#SBATCH --job-name=ccrl_phase1d_optA
#SBATCH --output=phase1d_option_a.%j.out
#SBATCH --error=phase1d_option_a.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE-1d — OPTION-A STABILITY TEST  (feasible CMDP, λ̃ expected ≈ 0)
# ─────────────────────────────────────────────────────────────────────────────
#  Purpose: test whether the scaling-crl canonical stabilizer stack is
#  sufficient to keep `train/actor_loss` bounded when the CMDP is strictly
#  feasible and the Lagrangian decays to zero. This isolates representation-
#  level stability from constraint enforcement.
#
#  Option-A modifications (landed in train.py, not overridden on the CLI):
#    • critic energy:  −√(‖φ−ψ‖² + 1e-12)  →  ⟨φ, ψ⟩  (inner product)
#    • normalize_observations: False → True  (restores JaxGCRL upstream default)
#    • nu_f: log(N) ≈ 5.5 → 1.0  (no actor-loss shrinkage of CRL signal)
#
#  Unchanged from Phase-1b (loose budget d=0.15 — FEASIBLE regime, NOT d=0.05):
#    env_id=ant_big_maze, num_envs=256, unroll_length=62, depth=4,
#    batch_size=256, PID=(0.1, 0.003, 0.001), 5M steps, 50 epochs,
#    cost_type=quadratic, ε_train=2.0, ε_hard=0.1, seed=0.
#
#  Why loose budget here (not tight d=0.05):
#    Phase-1b on `rvsu555e` established that d=0.15 produces J_c_hat ≈ 0.019,
#    so the CMDP is strictly feasible and λ̃ decays to 0 within a handful of
#    epochs. That regime is the WORST CASE for Option-A: the Lagrangian is
#    silent, so any residual instability in the CRL actor must be absorbed by
#    the architectural stabilizers alone. If actor_loss stays bounded here,
#    the dot-product + obs-norm + ν_f=1 combination is sufficient for
#    constrained-CRL stability at λ̃ ≈ 0. If it still blows up to O(10⁵), the
#    Lagrangian is structurally required as a representation regularizer in
#    our architecture — which is itself a publishable finding (see
#    crl_formulation_verification.md §7).
#
#  Metrics to inspect on wandb (group phase1d_option_a):
#
#    PRIMARY (stability):
#      train/actor_loss            EXPECTED |·| < 1e3 throughout (was 4.86e5)
#      train/critic_loss           EXPECTED bounded, finite, no NaN
#      train/log_alpha             EXPECTED bounded, plateau near log(α*)
#      train/lambda_tilde          EXPECTED ≈ 0 after warmup (~epoch 3–5)
#
#    SECONDARY (constraint accounting — sanity only, not the test):
#      train/jhat_c                EXPECTED < 0.15 (feasible) throughout
#      train/mean_step_cost        same band as Phase-1b: [0.10, 0.35] early
#      train/hard_violation_rate   EXPECTED ≤ 0.05
#
#    TERTIARY (task progress — informational):
#      eval/episode_reward         any upward trend is a win at 5M steps
#      eval/mean_d_wall            EXPECTED in [0, 5]
#      eval/success_rate           not expected to be strong at 5M steps
#
#  Pass criterion for Option-A stability test:
#    (a) |train/actor_loss|_max over all epochs < 1e3
#    (b) no NaN in any loss
#    (c) train/lambda_tilde ≤ 1.0 for ≥ 80 % of epochs (confirms feasible regime
#        was actually reached, so the test is well-posed)
#
#  Failure signature — interpret:
#    |actor_loss| → O(10⁵) again  →  CRL actor is unstable under λ̃=0 even
#      with dot + obs-norm + ν_f=1. Escalate to Option-B (reframe paper around
#      the emergent-Lagrangian-regularizer observation).
#    actor_loss bounded but critic diverges → dot energy hit a different
#      pathology; consider `cosine` energy (hard-normalize φ, ψ).
#    lambda_tilde > 1 persistently → Option-A changed the shape of J_c_hat
#      unexpectedly (obs-norm changes state-distribution seen by cost critic);
#      verify cost-critic calibration via C4 audit.
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

# ── LD_LIBRARY_PATH: pip nvidia packages FIRST, system CUDA as fallback ──
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 | head -20

echo ""
echo "===== OPTION-A DIFF VERIFICATION ====="
"$PYTHON" - <<'PYCHECK'
# Verify the three Option-A knobs are in the shipped code, not just in git.
# If any of these asserts fail, the wrong train.py is on PYTHONPATH.
import ast, pathlib
src = pathlib.Path("train.py").read_text()
assert "normalize_observations: bool = True" in src, \
    "normalize_observations default is not True — Option-A not applied"
assert "nu_f: float = 1.0" in src, \
    "nu_f default is not 1.0 — Option-A not applied"
assert 'jnp.einsum("ik,jk->ij", sa_repr, g_repr)' in src, \
    "critic_loss_fn still uses non-dot energy — Option-A not applied"
assert "f_sa_g  = jnp.sum(sa_repr * g_repr, axis=-1)" in src, \
    "actor_loss_fn still uses non-dot energy — Option-A not applied"
assert "-jnp.sqrt(_d2 + 1e-12)" not in src, \
    "old L2 critic logits still present — Option-A not fully applied"
print("Option-A diff verified in train.py.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Option-A diff verification failed — aborting job."
    exit 1
fi

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
print("GPU OK — proceeding with Option-A smoke run.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "JAX GPU check failed — aborting job."
    exit 1
fi

# ── Phase-1d preflight: same env check as Phase-1b/1c ─────────────────────────
echo ""
echo "===== PHASE-1D ENV PREFLIGHT ====="
"$PYTHON" - <<'PYCHECK'
import envs  # registers ant/humanoid mazes
from brax import envs as brax_envs
import jax

env = brax_envs.get_environment(
    "ant_big_maze",
    exclude_current_positions_from_observation=False,
)
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
env = VmapWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))

keys = jax.random.split(jax.random.PRNGKey(0), 2)
state = env.reset(keys)
print(f"obs.shape        = {state.obs.shape}")
print(f"action_size      = {env.action_size}")
print(f"torso xy @ reset = {state.pipeline_state.x.pos[:, 0, :2]}")
print(f"obs[:, 0:3]      = {state.obs[:, 0:3]}   # should be [x, y, z]")
PYCHECK

echo ""
echo "===== PHASE-1D OPTION-A LAUNCH  $(date) ====="

# Note: we do NOT pass --normalize_observations or --nu_f on the CLI. Option-A
# defaults live in train.py and must be used as shipped. Overriding them on
# the CLI would silently defeat the purpose of this run.

"$PYTHON" train.py \
    --env_id            ant_big_maze \
    --eval_env_id       ant_big_maze_eval \
    --episode_length    1000 \
    --total_env_steps   5000000 \
    --num_epochs        50 \
    --num_envs          256 \
    --eval_envs         32 \
    --eval_every        5 \
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
    --wandb_group       phase1d_option_a \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1D DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
