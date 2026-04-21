"""
probe_minimal.py — one-step diagnostic.  Prints at every stage so we can
pinpoint which step is hanging.  No JIT on outer loop, no vmap wrapper
unless explicitly requested, no data collection.  Run this FIRST.

Usage:
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python probe_minimal.py
"""
import os
import sys
import time

def _t(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

_t("Python started")
_t(f"  JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")
_t(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")

_t("importing jax...")
import jax
import jax.numpy as jnp
_t(f"  jax.devices() = {jax.devices()}")

_t("importing brax...")
from brax import envs as brax_envs
_t("  brax ok")

_t("importing constrained_crl envs (registers ant/humanoid mazes)...")
import envs  # noqa: F401
_t("  envs ok")

_t("constructing ant_big_maze (enable_cost=True, cost_type=quadratic)...")
# v1b: compact-support quadratic with ε_train=2.0 (calibrated to ant_big_maze
# corridor half-width) and ε_hard=0.1 (tight violation threshold).
env = brax_envs.get_environment(
    "ant_big_maze",
    enable_cost=True,
    cost_type="quadratic",
    cost_epsilon=2.0,
    cost_epsilon_hard=0.1,
    cost_tau=0.05,
    exclude_current_positions_from_observation=False,
)
_t(f"  env constructed: {type(env).__name__}  action_size={env.action_size}")

_t("wrapping with EpisodeWrapper + VmapWrapper (num_envs=2)...")
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
env = VmapWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))
_t("  wrappers ok")

_t("splitting reset keys for num_envs=2...")
rng = jax.random.PRNGKey(0)
reset_keys = jax.random.split(rng, 2)
_t("  keys ok")

_t("calling env.reset(...)  — this triggers the first JIT compile...")
t0 = time.time()
env_state = env.reset(reset_keys)
_t(f"  env_state = {type(env_state).__name__}")
_t(f"  env.reset traced (still async on device)")

_t("blocking on env.reset result (np.asarray of pipeline_state.x.pos)...")
import numpy as np
xy = np.asarray(env_state.pipeline_state.x.pos[:, 0, :2])
dt = time.time() - t0
_t(f"  reset() compile+exec took {dt:.1f}s")
_t(f"  torso xy at reset: {xy.tolist()}")

_t("reading info keys...")
info_keys = list(env_state.info.keys()) if hasattr(env_state.info, "keys") else "???"
_t(f"  info keys: {info_keys}")
if "cost" in env_state.info:
    _t(f"  cost at reset: {np.asarray(env_state.info['cost']).tolist()}")
    _t(f"  d_wall at reset: {np.asarray(env_state.info['d_wall']).tolist()}")
    _t(f"  hard_violation at reset: {np.asarray(env_state.info['hard_violation']).tolist()}")
else:
    _t("  !!!  info does NOT contain 'cost' — ant_maze.reset() patch didn't run")

_t("attempting ONE step...")
action = jnp.zeros((2, env.action_size))
t0 = time.time()
env_state2 = env.step(env_state, action)
xy2 = np.asarray(env_state2.pipeline_state.x.pos[:, 0, :2])
dt = time.time() - t0
_t(f"  step() compile+exec took {dt:.1f}s")
_t(f"  torso xy after step: {xy2.tolist()}")

if "cost" in env_state2.info:
    _t(f"  cost after step: {np.asarray(env_state2.info['cost']).tolist()}")
    _t(f"  d_wall after step: {np.asarray(env_state2.info['d_wall']).tolist()}")

# ── v1b expected values at spawn ─────────────────────────────────────────
# At ant_big_maze R cell (1,1) with scale=4.0: agent at (4.0, 4.0), nearest
# wall surface at d=2.0 exactly.  Under quadratic with ε_train=2.0, that
# gives cost = (1 - 2.0/2.0)² = 0.0 exactly.  This is EXPECTED — the
# constraint should be inactive at spawn.  The meaningful check is that
# cost > 0 when the agent is *close* to a wall, so let's also probe a
# near-wall position manually.
_t("probing cost at a near-wall position (synthetic)...")
from cost_utils import compact_quadratic_cost, get_wall_centers
wc, hw = get_wall_centers("ant_big_maze")
near_wall = jnp.array([[4.0, 2.1]], dtype=jnp.float32)  # d_wall=0.1
c_near, d_near, h_near = compact_quadratic_cost(
    near_wall, wc, hw,
    cost_epsilon_train=2.0, cost_epsilon_hard=0.1,
)
_t(f"  at (4.0, 2.1): d_wall={float(d_near[0]):.4f}  "
   f"cost={float(c_near[0]):.4f}  hard={int(h_near[0])}")
_t(f"  EXPECTED:     d_wall=0.1000  cost=0.9025  hard=1")
_t(f"  If these match, the compact-support cost field is live.")

_t("DONE — all stages completed successfully.")
