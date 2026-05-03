#!/bin/bash -l
#SBATCH --job-name=ccrl_1g_velcost_full
#SBATCH --output=phase1g_velcost_full.%j.out
#SBATCH --error=phase1g_velcost_full.%j.err
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
#  PHASE-1g — VELOCITY-QUADRATIC COST × BINDING BUDGET (H2 TEST)
#  50 epochs / 5M env steps / d = 0.05 / cost_type = velocity_quadratic
# ─────────────────────────────────────────────────────────────────────────────
#  Context.
#  Phase-1f at d=0.15 (job 1005888): all stability criteria passed, λ̃ = 0
#    throughout (Ĵ_c = 0.019 < d).
#  Phase-1e at d=0.05 (job 1010573): same — Ĵ_c = 0.016 < 0.05.
#  Phase-1g smoke (job 1016438, 3 epochs, velocity_quadratic): coupling
#    engaged. mean_step_cost grew 0.022 → 0.091, Ĵ_c grew 4–5× faster than
#    plain quadratic, all stability invariants held.
#
#  THIS RUN — full 50-epoch test in the binding regime.
#
#  H2: under reward-cost coupling, a binding constraint (λ̃ > 0) interacts
#      cleanly with the row-L2 + α-cap stack. Agent learns to slow near walls.
#
#  Pass criteria (all four):
#    (a) |actor_loss|_max < 1e3
#    (b) no NaN
#    (c) median λ̃ over last 20 epochs > 0.1   (constraint binds)
#    (d) final Ĵ_c ≤ d + 0.02 = 0.07          (constraint met)
#
#  Sole CLI differences from Phase-1f production:
#    --cost_type quadratic → velocity_quadratic
#    --cost_budget_d 0.15 → 0.05
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
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 | head -20

echo ""
echo "===== PHASE-1g STATIC DIFF + PROBE VERIFICATION ====="
"$PYTHON" - <<'PYCHECK'
import pathlib
src        = pathlib.Path("train.py").read_text()
src_env    = pathlib.Path("envs/ant_maze.py").read_text()
src_buffer = pathlib.Path("buffer.py").read_text()

assert "normalize_observations: bool = True" in src
assert "nu_f: float = 1.0" in src
assert "tau: float = 0.1" in src
assert "alpha_max: float = 1.0" in src
assert src.count("sa_norm_safe = jnp.sqrt(") == 2
assert src.count("g_norm_safe  = jnp.sqrt(") == 2
assert "jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-8" not in src

assert ('cost_type not in ("quadratic", "sigmoid", "velocity_quadratic")'
        in src_env)
assert 'self._cost_type in ("quadratic", "velocity_quadratic")' in src_env
assert 'cost = v_xy_sq * cost' in src_env
assert "hard = v_xy_sq * hard" not in src_env
for marker in ['"v_xy_norm"', '"vel_cost_mult"',
               'v_xy_norm=v_xy_norm', 'vel_cost_mult=vel_cost_mult']:
    assert marker in src_env, f"env diagnostic missing — {marker}"
for marker in ['"v_xy_norm":       _vnorm,',
               '"vel_cost_mult":   _vmult,',
               '"v_xy_norm":       jnp.zeros(()),',
               '"vel_cost_mult":   jnp.zeros(()),']:
    assert marker in src, f"train.py extras wiring missing — {marker}"
for marker in ['extras["v_xy_norm"] = jnp.squeeze',
               'extras["vel_cost_mult"] = jnp.squeeze']:
    assert marker in src_buffer, f"buffer wiring missing — {marker}"
for marker in ['"mean_v_xy_norm":      jnp.mean(v_xy_norm)',
               '"mean_vel_cost_mult":  jnp.mean(vel_cost_mult)',
               '"mean_v_xy_norm":      cc_m["mean_v_xy_norm"]',
               '"mean_vel_cost_mult":  cc_m["mean_vel_cost_mult"]']:
    assert marker in src, f"metrics wiring missing — {marker}"
assert 'float(jnp.mean(epoch_metrics["mean_v_xy_norm"]))' in src
assert 'float(jnp.mean(epoch_metrics["mean_vel_cost_mult"]))' in src
assert "|v_xy|={log_dict['mean_v_xy_norm']" in src
assert "vel_mult={log_dict['mean_vel_cost_mult']" in src

assert "f\"         nan[obs_c=" in src
assert "f\"         grad[c=" in src
assert "[prefill probe] buffer NaN anywhere:" in src
assert "[epoch1 forensics]" in src
assert "f\"         actor[α=" in src

print("Phase-1g full-run gate verified.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Phase-1g full-run static diff failed — aborting job."
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

echo ""
echo "===== PHASE-1g ENV PREFLIGHT ====="
"$PYTHON" - <<'PYCHECK'
import envs
from brax import envs as brax_envs
import jax, jax.numpy as jnp

env = brax_envs.get_environment(
    "ant_big_maze",
    exclude_current_positions_from_observation=False,
    cost_type="velocity_quadratic",
)
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
env = VmapWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))

keys  = jax.random.split(jax.random.PRNGKey(0), 4)
state = env.reset(keys)
print(f"At reset: cost = {state.info['cost']}, v_xy_norm = {state.info['v_xy_norm']}")
assert float(jnp.max(state.info["cost"])) == 0.0
assert float(jnp.max(state.info["v_xy_norm"])) <= 1e-5

key = jax.random.PRNGKey(42)
action = jax.random.uniform(key, (4, env.action_size), minval=-1.0, maxval=1.0)
state = env.step(state, action)
print(f"After 1 step: cost = {state.info['cost']}, v_xy_norm = {state.info['v_xy_norm']}")
assert float(jnp.max(jnp.abs(state.info["v_xy_norm"]))) > 0.0
print("Phase-1g env preflight OK.")
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== PHASE-1g VELOCITY-QUADRATIC FULL-RUN LAUNCH  $(date) ====="

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
    --cost_type         velocity_quadratic \
    --cost_epsilon      2.0 \
    --cost_epsilon_hard 0.1 \
    --cost_tau          0.05 \
    --cost_budget_d     0.05 \
    --pid_kp            0.1 \
    --pid_ki            0.003 \
    --pid_kd            0.001 \
    --lambda_max        100.0 \
    --wandb_project     constrained-crl \
    --wandb_group       phase1g_velcost_d0p05 \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1g VELOCITY-QUADRATIC FULL-RUN DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
