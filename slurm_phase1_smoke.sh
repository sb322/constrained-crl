#!/bin/bash -l
#SBATCH --job-name=ccrl_phase1b_smoke
#SBATCH --output=phase1b_smoke.%j.out
#SBATCH --error=phase1b_smoke.%j.err
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
#  Phase-1b smoke training run — SR-CPO on ant_big_maze, seed 0, 5 M env
#  steps, 50 epochs, **compact-support quadratic cost**.
#
#  Motivation: Phase-1a v2-overnight (job 962826, COMPLETED 11:00:06) showed
#  vacuous safety — hard_viol ≡ 0, mean_step_cost ≡ 0, λ̃ ≡ 0 for the entire
#  run, and actor_loss diverged to −485,495 with no constraint pressure.
#  Root cause: the sigmoid cost σ((ε − d_wall)/τ) with ε=0.1, τ=0.05 has
#  *infinite support* but numerically collapses to ~10⁻¹⁵ for the d_wall ≈ 1.75
#  the ant actually occupies in the corridor interior.  Cost field is flat,
#  cost critic has nothing to fit, PID stays at 0, constraint is not active.
#
#  Fix: switch to the compact-support quadratic shaping cost
#      c_train(s) = (1 − d_wall(s)/ε_train)²   for d_wall ≤ ε_train,  else 0
#  with ε_train = 0.3 (shaping band) and ε_hard = 0.1 (violation threshold).
#  Hard-indicator and CMDP budget use ε_hard independently of ε_train so
#  shaping is decoupled from accounting (cf. CBF safety literature).  Bounded
#  cost ∈ [0, 1] keeps Q_c ∈ [0, 100] under γ_c=0.99 so the PID dual gradient
#  stays well-scaled.  Budget raised d=0.1 → 0.15 to give the dual some
#  feasible slack during the learning phase.
#
#  Pass criteria (see phase1_criteria.md — v1b update):
#    • C1  epoch  0–5:   mean_step_cost > 0.1 near spawn, std across steps > 0.05
#    • C2  epoch 40–50:  mean cost strictly descending vs epoch 0–5 (≥10 % rel)
#    • C3  epoch <30:    λ̃ transiently > 0 when Ĵ_c > budget (PID responsive)
#    • C4  throughout:   |train/mean_step_cost − eval/mean_quad_cost| ≤ 3×  AND
#                        eval/mean_hard_cost ≤ train/mean_step_cost
#    • C5  throughout:   no NaN in any loss AND actor_loss stays finite-bounded
#                        (no −5 × 10⁵ explosion)
#
#  Failure signatures to watch for:
#    • mean_step_cost ≡ 0 throughout           → compact-support fix didn't land
#                                                 (check env_kwargs pipeline)
#    • hard_viol ≡ 0 AND mean_d_wall ≈ 2       → shape OK, agent never approaches
#                                                 wall → relax ε_train further
#                                                 or lower cost_budget_d
#    • λ saturates lambda_max immediately      → lower pid_kp (0.1 → 0.03)
#    • NaN in actor_loss or cost_critic_loss   → gradient path issue
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
#   (identical to slurm_depth_sweep.sh — required for JAX to find cuSOLVER)
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

# JAX memory settings — unchanged from sweep
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Stream Python stdout/stderr live (no block-buffering under SLURM redirect).
# Without this, epoch-print lines sit in a 64 KB buffer and only flush at exit,
# leaving the .out file looking "stuck" mid-training.
export PYTHONUNBUFFERED=1

# NEW: persist JIT compile cache across runs.  First run populates it (~60 s
# MJX ant compile); every subsequent run loads from cache in <5 s.
export JAX_COMPILATION_CACHE_DIR="/mmfs1/home/sb3222/.cache/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ── Diagnostic checks (same as sweep) ────────────────────────────────────────
echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 | head -20

echo ""
echo "===== cuSOLVER CHECK ====="
"$PYTHON" -c "
import ctypes, os
for p in os.environ.get('LD_LIBRARY_PATH','').split(':'):
    lib = os.path.join(p, 'libcusolver.so')
    if os.path.exists(lib):
        print(f'  found: {lib}')
        try:
            ctypes.CDLL(lib); print(f'  loaded OK'); break
        except Exception as e:
            print(f'  load FAILED: {e}')
else:
    print('  WARNING: libcusolver.so not found')
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
print("GPU OK — proceeding with smoke run.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "JAX GPU check failed — aborting job."
    exit 1
fi

# ── Phase-1 preflight: print env obs/action shapes ───────────────────────────
#   Catches obs_dim / goal_slice mismatches BEFORE 2 h of training.
echo ""
echo "===== PHASE-1 ENV PREFLIGHT ====="
"$PYTHON" - <<'PYCHECK'
import envs  # registers ant/humanoid mazes
from brax import envs as brax_envs
import jax

# Construct exactly as train.py does for ant_big_maze under Phase-1.
env = brax_envs.get_environment(
    "ant_big_maze",
    exclude_current_positions_from_observation=False,  # Phase-1 CRL-goal fix
)
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
env = VmapWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))

keys = jax.random.split(jax.random.PRNGKey(0), 2)
state = env.reset(keys)
print(f"obs.shape        = {state.obs.shape}")
print(f"action_size      = {env.action_size}")
print(f"info keys        = {list(state.info.keys())}")
print(f"metrics keys     = {list(state.metrics.keys())}")
print(f"torso xy @ reset = {state.pipeline_state.x.pos[:, 0, :2]}")
print(f"obs[:, 0:3]      = {state.obs[:, 0:3]}   # should be [x, y, z]")
if "cost" in state.info:
    print(f"info['cost']     = {state.info['cost']}")
    print(f"info['d_wall']   = {state.info['d_wall']}")
PYCHECK

# ═════════════════════════════════════════════════════════════════════════════
#  Phase-1b training launch  — compact-support quadratic cost
#
#  Prior-run post-mortem (job 962826, COMPLETED Tue Apr 21 2026):
#    • 11 h wall, all 50 epochs, exit 0, MaxRSS 3.8 GB.
#    • Phase-0 / Phase-1 plumbing still works (obs[:,0:3]=[x,y,z], preflight
#      clean, info/metrics populated), BUT safety signal is vacuous —
#      train/mean_step_cost ≡ 0, eval/mean_hard_cost ≡ 0, λ̃ ≡ 0, actor_loss
#      diverges to −485,495.  Cause: sigmoid cost σ((0.1 − d_wall)/0.05)
#      ≈ 10⁻¹⁵ for d_wall ≈ 1.75 → flat cost field → no gradient for the
#      cost critic → PID never bites → CRL InfoNCE loss drifts unbounded.
#
#  v1b changes (rationale):
#    • --cost_type           quadratic  (new)  — compact-support (1−d/ε)²
#    • --cost_epsilon        0.1 → 2.0          — shaping bandwidth ε_train,
#        calibrated to ant_big_maze corridor half-width.  With scale=4.0
#        and half_wall=2.0, interior d_wall ∈ [0, 2.0]; a dense-grid sweep
#        (scripts/outputs/scale_check.py) shows ε_train=2.0 gives nonzero
#        cost at 87 % of open-cell positions (E[c]≈0.26 under uniform policy).
#        The originally-discussed ε_train=0.3 would cover only 9 % — same
#        vacuous-safety failure mode as the sigmoid, just shifted scale.
#    • --cost_epsilon_hard   0.1        (new)  — violation threshold ε_hard
#        kept tight so CMDP accounting is stringent (5 % of interior has
#        d_wall<0.1, consistent with "near-collision" semantics).
#    • --cost_budget_d       0.1 → 0.15         — small feasible slack on Ĵ_c
#        over the hard indicator.  May need downstream retuning once we see
#        the actual Ĵ_c magnitude at start of training.
#    • --cost_tau retained in CLI but ignored under cost_type=quadratic
#
#  Unchanged — reasons already documented:
#    env_id=ant_big_maze, num_envs=256, unroll_length=62, depth=4,
#    obs_dim=29 (auto-corrects to 28), goal_start:end=0:3,
#    total_env_steps=5M, num_epochs=50, eval_every=5, walltime=18h,
#    PID=(0.1, 0.003, 0.001), seed=0.
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "===== PHASE-1B TRAINING LAUNCH  $(date) ====="

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
    --wandb_group       phase1b_smoke \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1B DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
