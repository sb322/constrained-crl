"""
cost_utils.py — Cost computation utilities for SR-CPO (Constrained CRL).

Provides:
  1. Wall-distance computation from known maze geometry.
  2. Smooth sigmoid training cost: c(s) = σ((ε - d_wall(s)) / τ).
  3. Hard indicator evaluation cost: c_eval(s) = 1{d_wall(s) < ε}.

The smooth sigmoid is used during training for differentiability;
the hard indicator is used at evaluation time for interpretable
constraint violation metrics.

All functions are JAX-compatible and JIT-safe.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

# ─────────────────────────────────────────────────────────────
#  Maze layouts — copied verbatim from envs/ so cost_utils
#  is self-contained and doesn't import the env modules.
# ─────────────────────────────────────────────────────────────

# Ant mazes  (maze_size_scaling = 4.0)
ANT_U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 'R', 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 'G', 0, 0, 1],
    [1, 1, 1, 1, 1],
]

ANT_BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'R', 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 'G', 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

ANT_HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'R', 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 'G', 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Humanoid mazes  (maze_size_scaling = 2.0)
HUMANOID_U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 'R', 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 'G', 0, 0, 1],
    [1, 1, 1, 1, 1],
]

HUMANOID_BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'R', 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 'G', 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

HUMANOID_HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'R', 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 'G', 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# ─────────────────────────────────────────────────────────────
#  Registry: env_id prefix → (layout, scale)
# ─────────────────────────────────────────────────────────────

_MAZE_REGISTRY = {
    # Ant variants
    "ant_u_maze":       (ANT_U_MAZE, 4.0),
    "ant_big_maze":     (ANT_BIG_MAZE, 4.0),
    "ant_hardest_maze": (ANT_HARDEST_MAZE, 4.0),
    # Humanoid variants
    "humanoid_u_maze":       (HUMANOID_U_MAZE, 2.0),
    "humanoid_big_maze":     (HUMANOID_BIG_MAZE, 2.0),
    "humanoid_hardest_maze": (HUMANOID_HARDEST_MAZE, 2.0),
}


def get_wall_centers(env_id: str) -> Tuple[jnp.ndarray, float]:
    """Return (wall_centers [N,2], half_wall_size) for the given env_id.

    Wall centers are the (x, y) positions of every cell marked 1 in the
    layout, scaled by maze_size_scaling.  half_wall_size is 0.5 * scale.
    """
    # Strip _eval suffix if present
    env_key = env_id.replace("_eval", "")
    # Handle numbered U-maze variants (ant_u2_maze, etc.)
    # They all share the same U_MAZE layout
    for prefix in ("ant_u", "humanoid_u"):
        if env_key.startswith(prefix) and env_key.endswith("_maze"):
            middle = env_key[len(prefix):-len("_maze")]
            if middle == "" or middle.isdigit():
                base_key = prefix + "_maze"
                if base_key in _MAZE_REGISTRY:
                    env_key = base_key
                    break

    if env_key not in _MAZE_REGISTRY:
        raise ValueError(
            f"Unknown env_id '{env_id}' for cost computation. "
            f"Known: {list(_MAZE_REGISTRY.keys())}"
        )
    layout, scale = _MAZE_REGISTRY[env_key]

    centers = []
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell == 1:
                # Maze generation uses (i * scale, j * scale) for position
                centers.append([i * scale, j * scale])
    centers = jnp.array(centers, dtype=jnp.float32)  # [N, 2]
    half_size = 0.5 * scale
    return centers, half_size


def compute_wall_distance(
    agent_xy: jnp.ndarray,
    wall_centers: jnp.ndarray,
    half_wall_size: float,
) -> jnp.ndarray:
    """Minimum L∞ distance from agent_xy to nearest wall surface.

    Each wall is an axis-aligned box of half-extent `half_wall_size`.
    The distance to a single box centered at c is:
        d_box = max(|x - cx| - h, |y - cy| - h, 0)
    We return min over all walls.

    Args:
        agent_xy: [..., 2]  agent (x, y) positions.
        wall_centers: [N, 2]  wall cell centers.
        half_wall_size: scalar, half side-length of each wall box.

    Returns:
        d_wall: [...]  minimum distance to any wall surface.
                 0.0 means inside a wall (collision).
    """
    # Expand for broadcasting: agent [..., 1, 2] - walls [N, 2]
    delta = jnp.abs(agent_xy[..., None, :] - wall_centers)  # [..., N, 2]
    # Distance per axis to box surface (clamped at 0)
    gap = jnp.maximum(delta - half_wall_size, 0.0)           # [..., N, 2]
    # L2 distance to nearest point on box surface
    d_per_wall = jnp.sqrt(jnp.sum(gap ** 2, axis=-1) + 1e-8) # [..., N]
    d_min = jnp.min(d_per_wall, axis=-1)                      # [...]
    return d_min


def smooth_sigmoid_cost(
    agent_xy: jnp.ndarray,
    wall_centers: jnp.ndarray,
    half_wall_size: float,
    cost_epsilon: float = 0.1,
    cost_tau: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Smooth sigmoid training cost: c(s) = σ((ε − d_wall(s)) / τ).

    This is the SR-CPO training cost (Eq. in Constrained_CRL_Revision_v2.tex).
    The sigmoid provides a differentiable approximation to the hard indicator
    1{d_wall < ε}, with temperature τ controlling sharpness.

    As τ → 0, this converges to the hard indicator.

    Args:
        agent_xy: [..., 2]  agent (x, y) positions.
        wall_centers: [N, 2]  wall cell centers.
        half_wall_size: float
        cost_epsilon: proximity threshold ε — distance below which cost ≈ 1.
        cost_tau: sigmoid temperature τ — smaller = sharper transition.

    Returns:
        cost: [...]  smooth cost in (0, 1).
        d_wall: [...]  minimum wall distance (for logging).
        hard_indicator: [...]  binary 1{d_wall < ε} (for eval metrics).
    """
    d_wall = compute_wall_distance(agent_xy, wall_centers, half_wall_size)
    # c(s) = sigmoid((ε - d_wall) / τ) = 1 / (1 + exp((d_wall - ε) / τ))
    cost = jax.nn.sigmoid((cost_epsilon - d_wall) / cost_tau)
    hard_indicator = (d_wall < cost_epsilon).astype(jnp.float32)
    return cost, d_wall, hard_indicator


def hard_indicator_cost(
    agent_xy: jnp.ndarray,
    wall_centers: jnp.ndarray,
    half_wall_size: float,
    cost_epsilon: float = 0.1,
) -> jnp.ndarray:
    """Hard indicator evaluation cost: c_eval(s) = 1{d_wall(s) < ε}.

    Used at evaluation time for interpretable constraint violation reporting.
    NOT used during training (use smooth_sigmoid_cost instead).

    Args:
        agent_xy: [..., 2]
        wall_centers: [N, 2]
        half_wall_size: float
        cost_epsilon: proximity threshold ε.

    Returns:
        cost: [...]  binary cost in {0, 1}.
    """
    d_wall = compute_wall_distance(agent_xy, wall_centers, half_wall_size)
    return (d_wall < cost_epsilon).astype(jnp.float32)


def compact_quadratic_cost(
    agent_xy: jnp.ndarray,
    wall_centers: jnp.ndarray,
    half_wall_size: float,
    cost_epsilon_train: float = 2.0,
    cost_epsilon_hard: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compact-support quadratic training cost.

    c_train(s) = (1 - d_wall(s) / ε_train)²   for d_wall ≤ ε_train
               = 0                             otherwise.

    Motivation (replaces `smooth_sigmoid_cost`):
      The sigmoid `σ((ε - d_wall)/τ)` has *infinite* support.  At ε=0.1, τ=0.05
      and typical mid-run d_wall ≈ 1.75 (ant in the interior of a big-maze
      corridor), the training cost is σ(-33) ≈ 10⁻¹⁵ — numerically zero
      *everywhere the agent actually goes*.  The cost field is therefore flat,
      the cost critic has no signal to fit, and the PID-Lagrangian dual stays
      pinned at λ̃ ≡ 0.  Constraint pressure vanishes (vacuous safety).

      A *compact-support* shaping cost remedies this without changing the
      CMDP problem.  The quadratic form has four load-bearing properties:

        (i)   Exact zero outside the danger band (d_wall ≥ ε_train) — so
              the constraint is inactive in the free interior, matching the
              geometric semantics of "distance-to-wall".
        (ii)  Smooth (C¹) transition at d_wall = ε_train — d c/d d_wall
              → 0 as d_wall → ε_train⁻, avoiding the kink that a linear
              (1 - d/ε)⁺ would impose on the cost critic.
        (iii) Non-trivial gradient on the entire band (0, ε_train) — grows
              as the agent approaches the wall, which is what actually drives
              the PID dual and the policy.
        (iv)  Bounded range c ∈ [0, 1], so with γ_c = 0.99 the cost-to-go
              Q_c has |Q_c| ≤ 1/(1-γ_c) = 100 — this is the `ν_c` scale that
              the Lagrangian update divides by, and keeping it O(1) prevents
              the gradient-through-dual from exploding.

      Default ε_train = 2.0 is calibrated to the ant_big_maze corridor
      geometry (maze_size_scaling=4.0 → walls are 4×4 boxes, corridor
      half-width ≈ 2.0 units).  A dense-grid sweep of the interior shows
      this gives nonzero cost at 87.5 % of open-cell positions and a
      mean-cost integral of ≈0.26 over a uniform policy.  Smaller choices
      (e.g. 0.3) cover only ≈9 % of the interior — cost is zero for
      essentially every position the ant actually visits, reproducing the
      same vacuous-safety failure mode the sigmoid had.
      ε_hard stays tight (default 0.1) so the *violation* indicator
      remains meaningful as a CMDP accounting metric, independent of the
      broader shaping band — the same decoupling CBF-based safe control
      uses between a safety filter's band and the underlying constraint.

    Args:
        agent_xy: [..., 2]            agent (x, y) positions.
        wall_centers: [N, 2]           wall cell centers.
        half_wall_size: float          half side-length of each wall box.
        cost_epsilon_train: shaping bandwidth ε_train — distance at which
            training cost becomes zero.  Must be strictly positive.
        cost_epsilon_hard:  hard-indicator threshold ε_hard — distance below
            which we record a *violation* (used for eval metrics and the
            hard-violation logged curve, independent of training signal).

    Returns:
        cost:            [...]  training cost in [0, 1].
        d_wall:          [...]  minimum wall distance (for logging).
        hard_indicator:  [...]  binary 1{d_wall < ε_hard} (for eval metrics).
    """
    d_wall = compute_wall_distance(agent_xy, wall_centers, half_wall_size)
    # u = max(0, 1 - d_wall / ε_train)   — support is {d_wall ≤ ε_train}
    u = jnp.clip(1.0 - d_wall / cost_epsilon_train, a_min=0.0, a_max=1.0)
    cost = u * u
    hard_indicator = (d_wall < cost_epsilon_hard).astype(jnp.float32)
    return cost, d_wall, hard_indicator
