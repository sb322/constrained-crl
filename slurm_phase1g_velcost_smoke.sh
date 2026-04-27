#!/bin/bash -l
#SBATCH --job-name=ccrl_1g_velcost_smoke
#SBATCH --output=phase1g_velcost_smoke.%j.out
#SBATCH --error=phase1g_velcost_smoke.%j.err
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
#  PHASE-1g — VELOCITY-QUADRATIC COST SMOKE (correctness check, 3 epochs)
# ─────────────────────────────────────────────────────────────────────────────
#  Context.
#  Phase-1f at d=0.15 (job 1005888, 50 epochs) passed all stability criteria
#  but `λ̃` stayed at 0 throughout — `Ĵ_c` plateaued at 0.019, far below the
#  budget. The unconstrained reward-maximizer on `ant_big_maze` with
#  cost_type=quadratic naturally avoids walls (open corridors, goals in
#  open cells), so reward and cost are decoupled in state space and the
#  CMDP reduces to the unconstrained MDP at any reasonable d.
#
#  Phase-1g introduces a NEW CMDP-respecting cost type, "velocity_quadratic":
#
#       c(s, a) = ‖v_xy‖² · (1 − d_wall(s)/ε_train)² · 1{d_wall(s) < ε_train}
#
#  Cost fires only when the agent is BOTH near a wall AND moving fast.
#  Reward (goal-reaching) pulls toward speed; cost pulls toward slowing
#  down whenever near walls. Reward and cost overlap in state space, so
#  the unconstrained reward-maximizer should now have J_c > 0 by
#  construction. Hard violation stays purely geometric: 1{d_wall < ε_hard}.
#
#  This run is a 3-epoch CORRECTNESS SMOKE, not the binding-regime test.
#  Its job is to verify:
#     (1) The new cost code path runs without NaN.
#     (2) `mean_step_cost` is non-trivially positive (proof of life:
#         velocity-coupled cost actually fires on the trained policy).
#     (3) `mean_v_xy_norm` and `mean_vel_cost_mult` are populated and
#         finite (the env diagnostics are wired through transitions →
#         flatten_crl_fn → cost_critic_loss_fn → log_dict).
#     (4) `Ĵ_c` is non-trivially larger than under cost_type=quadratic
#         at the same epoch count (proof of coupling: velocity multiplier
#         pushes J_c above the geometric-only cost's natural rate).
#     (5) All Phase-1f stability invariants still hold:
#         - all NaN flags = 0
#         - α capped at 1.0, α_clip = 1.00 every epoch
#         - row-L2 contract on φ, ψ unbroken
#         - actor_loss bounded
#
#  After this smoke passes, Cyam decides the binding-regime budget for
#  the 50-epoch run. The smoke's mean_step_cost / Ĵ_c values calibrate
#  what `d` would actually bind under velocity-coupled cost.
#
#  No other knobs change vs. Phase-1f production:
#     τ = 0.1 (default)
#     row-L2 with autograd-safe sqrt-with-eps-inside
#     α_max = 1.0
#     PID = (0.1, 0.003, 0.001), λ_max = 100
#     ε_train = 2.0, ε_hard = 0.1, depth = 4, width = 256
#     d = 0.15 (Phase-1f budget — kept unchanged for apples-to-apples
#               comparison of J_c trajectories under the two cost types).
#     The point of THIS run is to measure the new cost's J_c, NOT to
#     bind the constraint. Budget tightening is a SEPARATE experiment.
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
echo "===== PHASE-1g STATIC DIFF + PROBE VERIFICATION ====="
"$PYTHON" - <<'PYCHECK'
# Verify (a) all surviving Phase-1f invariants, AND (b) the new
# velocity_quadratic cost type is wired through env → buffer → train.py.
# If any assert fires, the wrong commits are on disk; abort before burning
# the cluster slot.
import pathlib
src        = pathlib.Path("train.py").read_text()
src_env    = pathlib.Path("envs/ant_maze.py").read_text()
src_buffer = pathlib.Path("buffer.py").read_text()

# --- 1. Surviving Option-A + Phase-1f knobs (unchanged) ----------------------
assert "normalize_observations: bool = True" in src
assert "nu_f: float = 1.0" in src
assert "tau: float = 0.1" in src
assert "alpha_max: float = 1.0" in src, \
    "α-cap regressed; Phase-1g requires it for stability"

# --- 2. Autograd-safe row-L2 (unchanged from Phase-1f) -----------------------
assert src.count("sa_norm_safe = jnp.sqrt(") == 2
assert src.count("g_norm_safe  = jnp.sqrt(") == 2
assert "jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-8" not in src, \
    "stale unsafe row-L2 form re-entered the code"

# --- 3. Phase-1g velocity-quadratic cost (NEW for this run) ------------------
# 3a. Env: validator accepts the new cost type
assert ('cost_type not in ("quadratic", "sigmoid", "velocity_quadratic")'
        in src_env), \
    "ant_maze.py validator does not accept cost_type='velocity_quadratic'"

# 3b. Env: _compute_safety_cost has a velocity_quadratic branch that
#     multiplies the geometric cost by ‖v_xy‖²
assert 'self._cost_type in ("quadratic", "velocity_quadratic")' in src_env, \
    "ant_maze.py dispatch branch for velocity_quadratic missing"
assert 'cost = v_xy_sq * cost' in src_env, \
    "velocity-quadratic cost = ‖v_xy‖² · geom_cost is not applied in ant_maze.py"

# 3c. Env: hard_violation must stay GEOMETRIC (1{d_wall < ε_hard}). The
#     velocity-coupling does not touch `hard`; verify by checking that
#     the velocity-quadratic branch only multiplies `cost`, not `hard`.
#     Negative pattern: there must be NO `hard = v_xy_sq * hard` anywhere.
assert "hard = v_xy_sq * hard" not in src_env, \
    "hard_violation must NOT be velocity-coupled — keep it geometric"

# 3d. Env: new diagnostics emitted in info/metrics
for marker in ['"v_xy_norm"', '"vel_cost_mult"',
               'v_xy_norm=v_xy_norm', 'vel_cost_mult=vel_cost_mult']:
    assert marker in src_env, f"env-side diagnostic missing — {marker}"

# 3e. train.py: collect_step + prefill_one + dummy_transition all
#     populate the new keys in transition.extras
for marker in ['"v_xy_norm":       _vnorm,',
               '"vel_cost_mult":   _vmult,',
               '"v_xy_norm":       jnp.zeros(()),',
               '"vel_cost_mult":   jnp.zeros(()),']:
    assert marker in src, f"train.py extras wiring missing — {marker}"

# 3f. buffer.py: flatten_crl_fn passes the new keys through
for marker in ['extras["v_xy_norm"] = jnp.squeeze',
               'extras["vel_cost_mult"] = jnp.squeeze']:
    assert marker in src_buffer, f"flatten_crl_fn pass-through missing — {marker}"

# 3g. train.py: cost_critic_loss_fn aggregates and returns the new keys
for marker in ['"mean_v_xy_norm":      jnp.mean(v_xy_norm)',
               '"mean_vel_cost_mult":  jnp.mean(vel_cost_mult)',
               '"mean_v_xy_norm":      cc_m["mean_v_xy_norm"]',
               '"mean_vel_cost_mult":  cc_m["mean_vel_cost_mult"]']:
    assert marker in src, f"cost_critic_loss_fn / metrics wiring missing — {marker}"

# 3h. train.py: log_dict + per-epoch print line
assert 'float(jnp.mean(epoch_metrics["mean_v_xy_norm"]))' in src, \
    "log_dict aggregation missing for mean_v_xy_norm"
assert 'float(jnp.mean(epoch_metrics["mean_vel_cost_mult"]))' in src, \
    "log_dict aggregation missing for mean_vel_cost_mult"
assert "|v_xy|={log_dict['mean_v_xy_norm']" in src, \
    "per-epoch print line missing |v_xy| token"
assert "vel_mult={log_dict['mean_vel_cost_mult']" in src, \
    "per-epoch print line missing vel_mult token"

# --- 4. Probes alive (forward NaN, grad/param, prefill, epoch-1 forensics) ---
assert "f\"         nan[obs_c=" in src, \
    "per-epoch forward NaN one-liner missing — probes were stripped"
assert "f\"         grad[c=" in src, \
    "per-epoch grad/param one-liner missing — probes were stripped"
assert "[prefill probe] buffer NaN anywhere:" in src, \
    "post-prefill buffer-NaN probe missing — probes were stripped"
assert "[epoch1 forensics]" in src, \
    "epoch-1 time-series dump missing — probes were stripped"
assert "f\"         actor[α=" in src, \
    "per-epoch actor-component one-liner missing — probes were stripped"

print("Phase-1g velocity-quadratic cost wiring + Phase-1f invariants verified.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Phase-1g static diff failed — aborting job."
    exit 1
fi

echo ""
echo "===== JAX GPU CHECK ====="
"$PYTHON" - <<'PYCHECK'
import jax
print(f"devices = {jax.devices()}")
print(f"default_backend = {jax.default_backend()}")
if jax.default_backend() != "gpu":
    print("FATAL: JAX did not detect GPU. Aborting.")
    import sys; sys.exit(1)
PYCHECK
[ $? -ne 0 ] && exit 1

# ── Env preflight: sanity on the new cost type's reset behavior ──────────────
echo ""
echo "===== PHASE-1g ENV PREFLIGHT ====="
"$PYTHON" - <<'PYCHECK'
import envs  # registers ant/humanoid mazes
from brax import envs as brax_envs
import jax, jax.numpy as jnp

# Build an env with the velocity_quadratic cost type explicitly. At reset the
# agent is stationary, so v_xy = 0 → cost should be 0 by construction.
env = brax_envs.get_environment(
    "ant_big_maze",
    exclude_current_positions_from_observation=False,
    cost_type="velocity_quadratic",
)
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
env = VmapWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))

keys  = jax.random.split(jax.random.PRNGKey(0), 4)
state = env.reset(keys)
print(f"obs.shape         = {state.obs.shape}")
print(f"action_size       = {env.action_size}")
print(f"info[cost]        = {state.info['cost']}     # should be 0 at reset (v=0)")
print(f"info[d_wall]      = {state.info['d_wall']}")
print(f"info[v_xy_norm]   = {state.info['v_xy_norm']}     # should be 0 at reset")
print(f"info[vel_cost_mult] = {state.info['vel_cost_mult']}  # should be 0 at reset")
assert float(jnp.max(state.info["cost"])) == 0.0, \
    "velocity_quadratic cost must be 0 at reset (v=0); env wiring broken"
assert float(jnp.max(state.info["v_xy_norm"])) <= 1e-5, \
    "v_xy_norm must be ~0 at reset; env wiring broken"

# Step once with a random action — velocity should now be nonzero.
key = jax.random.PRNGKey(42)
action = jax.random.uniform(key, (4, env.action_size), minval=-1.0, maxval=1.0)
state = env.step(state, action)
print(f"\nAfter one random step:")
print(f"info[cost]        = {state.info['cost']}")
print(f"info[d_wall]      = {state.info['d_wall']}")
print(f"info[v_xy_norm]   = {state.info['v_xy_norm']}")
print(f"info[vel_cost_mult] = {state.info['vel_cost_mult']}")
assert float(jnp.max(jnp.abs(state.info["v_xy_norm"]))) > 0.0, \
    "v_xy_norm must be nonzero after a real env step; velocity wiring broken"
print("\nPhase-1g env preflight OK — velocity-quadratic cost wiring is live.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Phase-1g env preflight failed — aborting job."
    exit 1
fi

echo ""
echo "===== PHASE-1g VELOCITY-QUADRATIC SMOKE LAUNCH  $(date) ====="

# 3 epochs / 300K env steps. Production knobs except cost_type and duration.
"$PYTHON" train.py \
    --env_id            ant_big_maze \
    --eval_env_id       ant_big_maze_eval \
    --episode_length    1000 \
    --total_env_steps   300000 \
    --num_epochs        3 \
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
    --cost_type         velocity_quadratic \
    --cost_epsilon      2.0 \
    --cost_epsilon_hard 0.1 \
    --cost_tau          0.05 \
    --cost_budget_d     0.15 \
    --pid_kp            0.1 \
    --pid_ki            0.003 \
    --pid_kd            0.001 \
    --lambda_max        100.0 \
    --wandb_project     constrained-crl \
    --wandb_group       phase1g_velcost_smoke \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1g VELOCITY-QUADRATIC SMOKE DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
