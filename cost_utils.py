"""
cost_utils.py — Cost computation utilities for Constrained CRL.

Provides:
  1. Wall-distance computation from known maze geometry.
  2. Binary collision detection from MuJoCo contact forces.
  3. Hybrid step cost: c(s,a,s') = α·1{contact} + (1-α)·exp(-d̄/σ).

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


def hybrid_cost(
    agent_xy: jnp.ndarray,
    wall_centers: jnp.ndarray,
    half_wall_size: float,
    alpha_cost: float = 0.5,
    sigma_wall: float = 1.0,
    contact_threshold: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Hybrid step cost: c = α·1{collision} + (1-α)·exp(-d/σ).

    Args:
        agent_xy: [..., 2]
        wall_centers: [N, 2]
        half_wall_size: float
        alpha_cost: weight on binary collision term.
        sigma_wall: length-scale for proximity soft-cost.
        contact_threshold: distance below which we declare collision.

    Returns:
        cost: [...]  hybrid cost in [0, 1].
        d_wall: [...]  minimum wall distance (for logging).
        collision: [...]  binary collision indicator.
    """
    d_wall = compute_wall_distance(agent_xy, wall_centers, half_wall_size)
    collision = (d_wall < contact_threshold).astype(jnp.float32)
    proximity = jnp.exp(-d_wall / sigma_wall)
    cost = alpha_cost * collision + (1.0 - alpha_cost) * proximity
    return cost, d_wall, collision
