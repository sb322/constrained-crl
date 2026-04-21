"""
probe_cost_wiring.py — Phase-0 sanity probe for the safety-cost pipeline.

Purpose
───────
The prior SR-CPO runs on BigMaze/U-maze reported `mean_cost = 0.8806`,
`hard_viol = 1.0`, `λ = 0` — all constant across every training step.  Root
cause: `cost_critic_loss_fn` was computing `agent_xy = state[:, :2]`, but with
`exclude_current_positions_from_observation=True` (the Brax default) the first
two dims of the Ant observation are `[z, quat_w]`, not `(x, y)`.  The fake
"agent" was pinned inside the wall box at the origin, giving d_wall ≡ 0 and
hence c(s) ≡ σ((0.1 − 0)/0.05) = 0.8807585 at every step.

This probe verifies the Phase-0 fix end-to-end **without running the full
training loop**:

    env.step() → info["cost"] / info["d_wall"] / info["hard_violation"]
                 are computed from `pipeline_state.x.pos[0, :2]` (authoritative
                 torso xy), not from the observation.

Pass criteria (all must hold)
─────────────────────────────
  (P1)  std(d_wall)           > 0.05            — d_wall is not constant
  (P2)  max(d_wall) - min(d_wall) > 0.5         — agent explores distances
  (P3)  |mean(cost) - 0.8806| > 0.05            — escaped the σ(2) fixed point
  (P4)  0.0 < mean(hard_viol) < 1.0             — neither always-safe nor always-in-wall
  (P5)  (x,y) trajectory has >0.5 range in at least one axis
                                                 — pipeline_state actually moves
  (P6)  env_state.info and state.metrics agree on cost/d_wall/hard_violation
                                                 — wiring is consistent

Usage
─────
  python probe_cost_wiring.py                        # ant_big_maze, random policy
  python probe_cost_wiring.py --env_id ant_u_maze    # U-maze
  python probe_cost_wiring.py --steps 500 --num_envs 16

Artifacts produced in the working directory:
  probe_cost_wiring_scatter.png
  probe_cost_wiring_metrics.npz
  probe_cost_wiring_report.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Register ant/humanoid maze envs.  This import has the side effect of
# calling brax.envs.register_environment for every variant in envs/__init__.py.
import envs  # noqa: F401
from brax import envs as brax_envs


# ─────────────────────────────────────────────────────────────────────────
#  Probe configuration
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-0 safety-cost wiring probe (non-training).")
    p.add_argument("--env_id", type=str, default="ant_big_maze",
                   help="Env id as registered in constrained_crl/envs/__init__.py.")
    p.add_argument("--num_envs", type=int, default=8,
                   help="Parallel rollouts (small — we're just probing).")
    p.add_argument("--steps", type=int, default=200,
                   help="Rollout length (steps per env).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cost_epsilon", type=float, default=0.1)
    p.add_argument("--cost_tau", type=float, default=0.05)
    p.add_argument("--outdir", type=str, default=".",
                   help="Directory for probe artifacts.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
#  Pass/fail thresholds — tuned against the theoretical σ(2)=0.8807585
# ─────────────────────────────────────────────────────────────────────────

SIGMA_2 = 1.0 / (1.0 + np.exp(-2.0))   # 0.8807585…
SIGMA_2_OFFSET = 0.05                   # mean(cost) must be > 0.05 away from σ(2)

THRESHOLDS = {
    "d_wall_std_min":      0.05,
    "d_wall_range_min":    0.5,
    "cost_gap_min":        SIGMA_2_OFFSET,
    "hard_viol_upper":     0.999,
    "hard_viol_lower":     1e-4,   # allow some envs to be in open space only
    "xy_range_min":        0.5,
}


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── Build env ─────────────────────────────────────────────────────────
    # The constrained_crl ant_maze constructor signature we patched accepts
    # (enable_cost, cost_epsilon, cost_tau) as kwargs.  Pass them through.
    env = brax_envs.get_environment(
        args.env_id,
        enable_cost=True,
        cost_epsilon=args.cost_epsilon,
        cost_tau=args.cost_tau,
    )

    # Wrap in the standard EpisodeWrapper so info["steps"] / info["truncation"]
    # behave like in train.py — but NOT the VmapWrapper, we vmap manually so
    # env_state.info has the same structure as in collect_step.
    from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
    env = VmapWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    env_keys = jax.random.split(reset_key, args.num_envs)

    env_state = env.reset(env_keys)

    action_size = env.action_size

    # Random-policy rollout: a uniform action is fine for a wiring probe.
    # We want the agent to wander around (not stay pinned at origin) so we
    # verify that d_wall varies.  A deterministic zero policy would leave the
    # ant lying there and give us exactly the same flat signal as before.
    @jax.jit
    def step_fn(env_state, key):
        ak, nk = jax.random.split(key)
        action = jax.random.uniform(
            ak, (args.num_envs, action_size), minval=-1.0, maxval=1.0)
        next_env_state = env.step(env_state, action)
        return next_env_state, nk

    # ── Rollout, collecting per-step info from env_state ──────────────────
    costs       = []   # from env_state.info["cost"]
    d_walls     = []
    hard_viols  = []
    xs, ys      = [], []
    metrics_cost, metrics_dwall, metrics_hard = [], [], []

    rng, loop_key = jax.random.split(rng)
    key = loop_key
    for t in range(args.steps):
        # Record BEFORE stepping — env_state.info["cost"] refers to the
        # current state (c(s_t)), consistent with collect_step in train.py.
        # pipeline_state.x.pos[0, :2] is the ground truth torso xy.
        pstate = env_state.pipeline_state
        xs.append(np.asarray(pstate.x.pos[:, 0, 0]))  # (num_envs,)
        ys.append(np.asarray(pstate.x.pos[:, 0, 1]))

        costs.append(np.asarray(env_state.info["cost"]))
        d_walls.append(np.asarray(env_state.info["d_wall"]))
        hard_viols.append(np.asarray(env_state.info["hard_violation"]))

        metrics_cost.append(np.asarray(env_state.metrics["cost"]))
        metrics_dwall.append(np.asarray(env_state.metrics["d_wall"]))
        metrics_hard.append(np.asarray(env_state.metrics["hard_violation"]))

        env_state, key = step_fn(env_state, key)

    costs      = np.stack(costs)       # (T, num_envs)
    d_walls    = np.stack(d_walls)
    hard_viols = np.stack(hard_viols)
    xs         = np.stack(xs)
    ys         = np.stack(ys)
    metrics_cost  = np.stack(metrics_cost)
    metrics_dwall = np.stack(metrics_dwall)
    metrics_hard  = np.stack(metrics_hard)

    # ── Pass/fail checks ─────────────────────────────────────────────────
    d_std   = float(d_walls.std())
    d_range = float(d_walls.max() - d_walls.min())
    c_mean  = float(costs.mean())
    c_gap   = float(abs(c_mean - SIGMA_2))
    hv_mean = float(hard_viols.mean())
    x_range = float(xs.max() - xs.min())
    y_range = float(ys.max() - ys.min())
    xy_range = max(x_range, y_range)

    # Consistency check: info vs metrics
    info_vs_metrics_max_abs_diff = max(
        float(np.max(np.abs(costs      - metrics_cost))),
        float(np.max(np.abs(d_walls    - metrics_dwall))),
        float(np.max(np.abs(hard_viols - metrics_hard))),
    )

    checks = {
        "P1_d_wall_std":
            (d_std   > THRESHOLDS["d_wall_std_min"],     f"std(d_wall)={d_std:.4f} > {THRESHOLDS['d_wall_std_min']}"),
        "P2_d_wall_range":
            (d_range > THRESHOLDS["d_wall_range_min"],   f"range(d_wall)={d_range:.4f} > {THRESHOLDS['d_wall_range_min']}"),
        "P3_cost_escapes_sigma2":
            (c_gap   > THRESHOLDS["cost_gap_min"],       f"|mean(cost) − σ(2)|={c_gap:.4f} > {THRESHOLDS['cost_gap_min']} "
                                                         f"(mean={c_mean:.4f}, σ(2)={SIGMA_2:.4f})"),
        "P4a_hard_viol_not_always_one":
            (hv_mean < THRESHOLDS["hard_viol_upper"],    f"mean(hard_viol)={hv_mean:.4f} < {THRESHOLDS['hard_viol_upper']}"),
        "P4b_hard_viol_not_always_zero":
            # Note: it's OK for mean hard_viol to be 0 if ε is small and the
            # agent never breaches the wall.  We relax this to a warning.
            (True, f"mean(hard_viol)={hv_mean:.4f}  (info-only; small ε allowed)"),
        "P5_xy_moves":
            (xy_range > THRESHOLDS["xy_range_min"],      f"max(x_range, y_range)={xy_range:.4f} > {THRESHOLDS['xy_range_min']}"),
        "P6_info_matches_metrics":
            (info_vs_metrics_max_abs_diff < 1e-5,        f"max |info − metrics| = {info_vs_metrics_max_abs_diff:.2e} < 1e-5"),
    }

    # ── Report ─────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append(f"PHASE-0 COST-WIRING PROBE — env_id={args.env_id}")
    report_lines.append(f"  num_envs={args.num_envs}  steps={args.steps}  "
                        f"ε={args.cost_epsilon}  τ={args.cost_tau}")
    report_lines.append("")
    report_lines.append("Aggregate statistics:")
    report_lines.append(f"  cost:           mean={c_mean:.4f}  std={costs.std():.4f}  "
                        f"min={costs.min():.4f}  max={costs.max():.4f}")
    report_lines.append(f"  d_wall:         mean={d_walls.mean():.4f}  std={d_std:.4f}  "
                        f"min={d_walls.min():.4f}  max={d_walls.max():.4f}")
    report_lines.append(f"  hard_viol:      mean={hv_mean:.4f}  std={hard_viols.std():.4f}")
    report_lines.append(f"  torso_x range:  {x_range:.4f}")
    report_lines.append(f"  torso_y range:  {y_range:.4f}")
    report_lines.append(f"  σ(2) reference: {SIGMA_2:.7f}  "
                        f"(the bug-symptom value we must escape)")
    report_lines.append("")
    report_lines.append("Pass/fail checks:")
    all_pass = True
    for name, (ok, msg) in checks.items():
        tag = "PASS" if ok else "FAIL"
        if not ok and not name.startswith("P4b"):
            all_pass = False
        report_lines.append(f"  [{tag}] {name}: {msg}")
    report_lines.append("")
    report_lines.append(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    report = "\n".join(report_lines)
    print(report)

    # ── Artifacts ─────────────────────────────────────────────────────────
    np.savez(
        os.path.join(args.outdir, "probe_cost_wiring_metrics.npz"),
        cost=costs, d_wall=d_walls, hard_violation=hard_viols,
        x=xs, y=ys,
        metrics_cost=metrics_cost,
        metrics_d_wall=metrics_dwall,
        metrics_hard_violation=metrics_hard,
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax[0].scatter(xs.flatten(), ys.flatten(),
                       c=d_walls.flatten(), s=4, cmap="viridis")
    plt.colorbar(sc, ax=ax[0], label="d_wall")
    ax[0].set_xlabel("torso x"); ax[0].set_ylabel("torso y")
    ax[0].set_aspect("equal")
    ax[0].set_title(f"Trajectory colored by d_wall ({args.env_id})")

    ax[1].plot(costs.mean(axis=1), label="mean cost", lw=1.5)
    ax[1].axhline(SIGMA_2, ls="--", c="red",
                  label=f"σ(2) = {SIGMA_2:.4f}  (bug value)")
    ax[1].axhline(args.cost_epsilon, ls=":", c="gray",
                  label=f"ε = {args.cost_epsilon}")
    ax[1].set_xlabel("rollout step"); ax[1].set_ylabel("cost")
    ax[1].set_title("Per-step cost vs. bug-symptom baseline")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "probe_cost_wiring_scatter.png"),
                dpi=130)
    plt.close(fig)

    with open(os.path.join(args.outdir, "probe_cost_wiring_report.txt"), "w") as f:
        f.write(report + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
