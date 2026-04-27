import os
from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import xml.etree.ElementTree as ET

# Phase-0 patch (2026-04-20): env now emits safety cost in state.info so the
# training loop no longer has to reach into the observation's first two dims
# (which were [z, quat_w] — NOT torso xy — under the default
# exclude_current_positions_from_observation=True). The authoritative torso
# position comes from pipeline_state.x.pos[0, :2].
from cost_utils import smooth_sigmoid_cost, compact_quadratic_cost


RESET = R = 'r'
GOAL = G = 'g'

U_MAZE = [[1, 1, 1, 1, 1],
           [1, R, G, G, 1],
           [1, 1, 1, G, 1],
           [1, G, G, G, 1],
           [1, 1, 1, 1, 1]]

U_MAZE_EVAL = [[1, 1, 1, 1, 1],
                [1, R, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, G, G, G, 1],
                [1, 1, 1, 1, 1]]

U_MAZE_SINGLE_EVAL = [[1, 1, 1, 1, 1],
                       [1, R, 0, 0, 1],
                       [1, 1, 1, 0, 1],
                       [1, G, 0, 0, 1],
                       [1, 1, 1, 1, 1]]

U_MAZE_EVAL_1f2f3f4f5f = [[1, 1, 1, 1, 1],
                            [1, R, G, G, 1],
                            [1, 1, 1, G, 1],
                            [1, 0, G, G, 1],
                            [1, 1, 1, 1, 1]]

U_MAZE_EVAL_1f2f3f4f = [[1, 1, 1, 1, 1],
                          [1, R, G, G, 1],
                          [1, 1, 1, G, 1],
                          [1, 0, 0, G, 1],
                          [1, 1, 1, 1, 1]]

U_MAZE_EVAL_1f2f3f = [[1, 1, 1, 1, 1],
                        [1, R, G, G, 1],
                        [1, 1, 1, G, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]]

U_MAZE_EVAL_5f6f = [[1, 1, 1, 1, 1],
                      [1, R, 0, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, G, G, 0, 1],
                      [1, 1, 1, 1, 1]]

U2_MAZE = [[1, 1, 1, 1, 1, 1],
            [1, R, G, G, G, 1],
            [1, 1, 1, 1, G, 1],
            [1, G, G, G, G, 1],
            [1, 1, 1, 1, 1, 1]]

U2_MAZE_EVAL = [[1, 1, 1, 1, 1, 1],
                 [1, R, 0, 0, 0, 1],
                 [1, 1, 1, 1, 0, 1],
                 [1, G, G, G, G, 1],
                 [1, 1, 1, 1, 1, 1]]

U3_MAZE = [[1, 1, 1, 1, 1, 1, 1],
            [1, R, G, G, G, G, 1],
            [1, 1, 1, 1, 1, G, 1],
            [1, G, G, G, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1]]

U3_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1],
                 [1, R, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 1],
                 [1, G, G, G, G, G, 1],
                 [1, 1, 1, 1, 1, 1, 1]]

U3_MAZE_SINGLE_EVAL = [[1, 1, 1, 1, 1, 1, 1],
                         [1, R, 0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1, 0, 1],
                         [1, G, 0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1]]

U4_MAZE = [[1, 1, 1, 1, 1],
            [1, G, G, G, 1],
            [1, R, 1, G, 1],
            [1, 1, 1, G, 1],
            [1, G, 1, G, 1],
            [1, G, G, G, 1],
            [1, 1, 1, 1, 1]]

U4_MAZE_EVAL = [[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, R, 1, 0, 1],
                  [1, 1, 1, 0, 1],
                  [1, G, 1, 0, 1],
                  [1, G, G, G, 1],
                  [1, 1, 1, 1, 1]]

U5_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, G, G, G, G, G, G, 1],
            [1, R, 1, 1, 1, 1, G, 1],
            [1, 1, 1, 1, 1, 1, G, 1],
            [1, G, 1, 1, 1, 1, G, 1],
            [1, G, G, G, G, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

U5_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 1],
                  [1, R, 1, 1, 1, 1, 0, 1],
                  [1, 1, 1, 1, 1, 1, 0, 1],
                  [1, G, 1, 1, 1, 1, G, 1],
                  [1, G, G, G, G, G, G, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1]]

U5_MAZE_SINGLE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 0, 1],
                         [1, R, 1, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1],
                         [1, G, 1, 1, 1, 1, 0, 1],
                         [1, 0, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1]]

U6_MAZE = [[1, 1, 1, 1, 1, 1, 1],
            [1, G, G, G, G, G, 1],
            [1, R, 1, 1, 1, G, 1],
            [1, 1, 1, 1, 1, G, 1],
            [1, G, 1, 1, 1, G, 1],
            [1, G, G, G, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1]]

U6_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 1],
                  [1, R, 1, 1, 1, 0, 1],
                  [1, 1, 1, 1, 1, 0, 1],
                  [1, G, 1, 1, 1, G, 1],
                  [1, G, G, G, G, G, 1],
                  [1, 1, 1, 1, 1, 1, 1]]

U7_MAZE = [[1, 1, 1, 1, 1, 1],
            [1, G, G, G, G, 1],
            [1, R, 1, 1, G, 1],
            [1, 1, 1, 1, G, 1],
            [1, G, 1, 1, G, 1],
            [1, G, G, G, G, 1],
            [1, 1, 1, 1, 1, 1]]

U7_MAZE_EVAL = [[1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 1],
                  [1, R, 1, 1, 0, 1],
                  [1, 1, 1, 1, 0, 1],
                  [1, G, 1, 1, G, 1],
                  [1, G, G, G, G, 1],
                  [1, 1, 1, 1, 1, 1]]

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, R, G, 1, 1, G, G, 1],
             [1, G, G, 1, G, G, G, 1],
             [1, 1, G, G, G, 1, 1, 1],
             [1, G, G, 1, G, G, G, 1],
             [1, G, 1, G, G, 1, G, 1],
             [1, G, G, G, 1, G, G, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, R, 0, 1, 1, G, G, 1],
                   [1, 0, 0, 1, 0, 0, G, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1],
                   [1, 0, 0, 1, 0, 0, 0, 1],
                   [1, 0, 1, G, 0, 1, G, 1],
                   [1, 0, G, G, 1, G, G, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, R, G, G, G, 1, G, G, G, G, G, 1],
                 [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                 [1, G, G, G, G, G, G, 1, G, G, G, 1],
                 [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                 [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                 [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                 [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_HEIGHT = 0.5


def find_robot(structure, size_scaling):
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                return i * size_scaling, j * size_scaling


def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])
    return jp.array(goals)


def make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "u_maze_eval":
        maze_layout = U_MAZE_EVAL
    elif maze_layout_name == "u_maze_single_eval":
        maze_layout = U_MAZE_SINGLE_EVAL
    elif maze_layout_name == "u_maze_eval_1f2f3f4f5f":
        maze_layout = U_MAZE_EVAL_1f2f3f4f5f
    elif maze_layout_name == "u_maze_eval_1f2f3f4f":
        maze_layout = U_MAZE_EVAL_1f2f3f4f
    elif maze_layout_name == "u_maze_eval_1f2f3f":
        maze_layout = U_MAZE_EVAL_1f2f3f
    elif maze_layout_name == "u_maze_eval_5f6f":
        maze_layout = U_MAZE_EVAL_5f6f
    elif maze_layout_name == "u2_maze":
        maze_layout = U2_MAZE
    elif maze_layout_name == "u2_maze_eval":
        maze_layout = U2_MAZE_EVAL
    elif maze_layout_name == "u3_maze":
        maze_layout = U3_MAZE
    elif maze_layout_name == "u3_maze_eval":
        maze_layout = U3_MAZE_EVAL
    elif maze_layout_name == "u3_maze_single_eval":
        maze_layout = U3_MAZE_SINGLE_EVAL
    elif maze_layout_name == "u4_maze":
        maze_layout = U4_MAZE
    elif maze_layout_name == "u4_maze_eval":
        maze_layout = U4_MAZE_EVAL
    elif maze_layout_name == "u5_maze":
        maze_layout = U5_MAZE
    elif maze_layout_name == "u5_maze_eval":
        maze_layout = U5_MAZE_EVAL
    elif maze_layout_name == "u6_maze":
        maze_layout = U6_MAZE
    elif maze_layout_name == "u6_maze_eval":
        maze_layout = U6_MAZE_EVAL
    elif maze_layout_name == "u7_maze":
        maze_layout = U7_MAZE
    elif maze_layout_name == "u7_maze_eval":
        maze_layout = U7_MAZE_EVAL
    elif maze_layout_name == "u5_maze_single_eval":
        maze_layout = U5_MAZE_SINGLE_EVAL
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "big_maze_eval":
        maze_layout = BIG_MAZE_EVAL
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'assets', "ant_maze.xml")
    robot_x, robot_y = find_robot(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    # Collect wall centers while we build the XML so the safety-cost function
    # sees EXACTLY the same geometry as the simulator (no drift between
    # cost_utils._MAZE_REGISTRY and the XML).
    wall_centers_list = []
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling,
                                      j * maze_size_scaling,
                                      MAZE_HEIGHT / 2 * maze_size_scaling),
                    size="%f %f %f" % (0.5 * maze_size_scaling,
                                       0.5 * maze_size_scaling,
                                       MAZE_HEIGHT / 2 * maze_size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )
                wall_centers_list.append([i * maze_size_scaling,
                                          j * maze_size_scaling])

    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data")
    torso.set("data", f"{robot_x} {robot_y} " + data)

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    wall_centers = jp.asarray(wall_centers_list, dtype=jp.float32)  # [N, 2]
    half_wall_size = 0.5 * maze_size_scaling
    return xml_string, possible_goals, wall_centers, half_wall_size


class AntMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        # Safety-cost hyperparameters. enable_cost=True makes step() populate
        # state.info with cost / d_wall / hard_violation scalars computed from
        # the ant's torso position. Set enable_cost=False for the unconstrained
        # baseline (keys still present but always zero, so the collector
        # codepath is uniform).
        #
        # cost_type selects the training-cost shape:
        #   - "quadratic" (default): c_train = (1 - d_wall/ε_train)²  on the
        #       compact support d_wall ∈ [0, ε_train]; 0 elsewhere.  Hard
        #       violation uses a *separate* threshold ε_hard (usually
        #       ε_hard ≤ ε_train) so shaping is decoupled from accounting.
        #       This is the default after the v2-overnight run 962826 showed
        #       the sigmoid produces a flat cost field over the corridor
        #       interior (σ(-33) ≈ 10⁻¹⁵ for d_wall=1.75, ε=0.1, τ=0.05),
        #       pinning λ̃ ≡ 0 and making safety vacuous.
        #   - "velocity_quadratic": c_train = ‖v_xy‖² · (1 - d_wall/ε_train)²
        #       on the same compact support. Couples cost with motion: cost
        #       fires only when the ant is BOTH near a wall AND moving fast.
        #       Reward (goal-reaching) pulls toward speed; cost pulls toward
        #       slowing down whenever near walls. Reward and cost overlap in
        #       state space, so the unconstrained reward-maximizer has
        #       J_c > 0 and the CMDP can have a binding regime even with
        #       moderate budgets. v_xy is the torso (x, y) velocity computed
        #       in step() as (pos_t+1 − pos_t) / dt; at reset() the agent is
        #       stationary so v_xy = 0 and cost = 0 (correct boundary).
        #       Hard violation stays purely geometric: 1{d_wall < ε_hard}.
        #   - "sigmoid":    legacy c_train = σ((ε - d_wall)/τ).  Kept for
        #       ablation/back-compat; ignores cost_epsilon_hard.
        enable_cost: bool = True,
        cost_type: str = "quadratic",
        cost_epsilon: float = 2.0,
        cost_epsilon_hard: float = 0.1,
        cost_tau: float = 0.05,
        **kwargs,
    ):
        xml_string, possible_goals, wall_centers, half_wall_size = make_maze(
            maze_layout_name, maze_size_scaling
        )
        sys = mjcf.loads(xml_string)
        self.possible_goals = possible_goals

        # Safety-cost config (stored as plain Python attrs so JIT picks them up
        # as constants; wall_centers is a jax array bound at trace time).
        self._enable_cost = enable_cost
        if cost_type not in ("quadratic", "sigmoid", "velocity_quadratic"):
            raise ValueError(
                f"Unknown cost_type '{cost_type}'. "
                "Expected one of: 'quadratic', 'sigmoid', 'velocity_quadratic'."
            )
        self._cost_type = cost_type
        # For quadratic: cost_epsilon is the *shaping* bandwidth ε_train
        # (default 0.3).  For sigmoid legacy path: cost_epsilon is the hard
        # threshold used inside the sigmoid argument (default 0.1 there).
        self._cost_epsilon = cost_epsilon
        self._cost_epsilon_hard = cost_epsilon_hard
        self._cost_tau = cost_tau
        self._wall_centers = wall_centers          # [N, 2] float32
        self._half_wall_size = half_wall_size      # scalar

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.005)
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        _, target = self._random_target(rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        # Safety-cost scalars at initial state.  Even when enable_cost=False we
        # keep the keys in info so the collector pipeline has a uniform schema.
        # At reset() the agent is stationary, so velocity_xy = 0 and the
        # "velocity_quadratic" cost is automatically 0. v_xy_norm and
        # vel_cost_mult are emitted for schema uniformity.
        agent_xy = pipeline_state.x.pos[0, :2]
        cost0, d_wall0, hard0, v_xy_norm0, vel_cost_mult0 = self._compute_safety_cost(
            agent_xy
        )

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
            # Safety-cost metrics (rollout-level aggregation is done by the
            # training loop; these are per-step scalars).
            "cost": cost0,
            "d_wall": d_wall0,
            "hard_violation": hard0,
            # Phase-1g velocity-quadratic diagnostics.
            "v_xy_norm": v_xy_norm0,
            "vel_cost_mult": vel_cost_mult0,
        }
        info = {
            "seed": 0,
            "cost": cost0,
            "d_wall": d_wall0,
            "hard_violation": hard0,
            "v_xy_norm": v_xy_norm0,
            "vel_cost_mult": vel_cost_mult0,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def _compute_safety_cost(
        self,
        agent_xy: jax.Array,
        velocity_xy: jax.Array = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute (cost, d_wall, hard_violation, v_xy_norm, vel_cost_mult) at
        a single agent (x, y).

        Dispatches on self._cost_type:
          - "quadratic":           cost = (1 − d_wall/ε_train)²  on d_wall ≤ ε_train,
                                   hard = 1{d_wall < ε_hard}   (separate threshold).
          - "velocity_quadratic":  cost = ‖v_xy‖² · (1 − d_wall/ε_train)²  on the
                                   same compact support. Couples cost with motion:
                                   reward-maximizing trajectories that pass close
                                   to walls now incur cost (because they're moving),
                                   so J_c(π_unconstrained) > 0 and the CMDP can
                                   have a binding regime. Hard violation is still
                                   purely geometric — 1{d_wall < ε_hard} — so the
                                   accounting threshold is decoupled from the
                                   shaped training cost.
          - "sigmoid"  :           cost = σ((ε − d_wall)/τ),    hard = 1{d_wall < ε}.

        When self._enable_cost is False, returns zeros for cost and hard_violation
        but still returns correct d_wall, v_xy_norm, and vel_cost_mult (useful for
        diagnostics even in the unconstrained baseline).

        Args:
            agent_xy:    shape [2] — ant torso (x, y) in maze frame.
            velocity_xy: shape [2] — ant torso (x, y) velocity. If None, treated
                         as zeros (relevant only at reset(), where the ant is
                         stationary and the velocity-coupled cost is 0 by
                         construction; this matches the geometric "quadratic"
                         path's behavior at reset).

        Returns:
            cost:           scalar training cost (disabled → 0).
            d_wall:         scalar, L2 distance from torso to nearest wall surface.
            hard_violation: scalar ∈ {0, 1} (disabled → 0).
            v_xy_norm:      scalar, ‖v_xy‖₂ — diagnostic, tells us how fast the
                            torso is moving in the maze plane.
            vel_cost_mult:  scalar, ‖v_xy‖² — the velocity multiplier applied
                            on top of the geometric quadratic shell. 0 for the
                            "quadratic" / "sigmoid" cost types (so the
                            diagnostic is uniform across cost types).
        """
        if velocity_xy is None:
            velocity_xy = jp.zeros_like(agent_xy)
        v_xy_sq   = jp.sum(velocity_xy * velocity_xy)
        v_xy_norm = jp.sqrt(v_xy_sq + 1e-12)

        if self._cost_type in ("quadratic", "velocity_quadratic"):
            cost, d_wall, hard = compact_quadratic_cost(
                agent_xy, self._wall_centers, self._half_wall_size,
                cost_epsilon_train=self._cost_epsilon,
                cost_epsilon_hard=self._cost_epsilon_hard,
            )
            if self._cost_type == "velocity_quadratic":
                # Multiply the geometric quadratic shell by ‖v_xy‖² to couple
                # cost with motion. Hard violation stays purely geometric — it
                # is the accounting indicator, not the shaped training signal.
                cost = v_xy_sq * cost
                vel_cost_mult = v_xy_sq
            else:
                vel_cost_mult = jp.zeros_like(v_xy_sq)
        else:  # "sigmoid" — legacy path, kept for ablation
            cost, d_wall, hard = smooth_sigmoid_cost(
                agent_xy, self._wall_centers, self._half_wall_size,
                cost_epsilon=self._cost_epsilon,
                cost_tau=self._cost_tau,
            )
            vel_cost_mult = jp.zeros_like(v_xy_sq)

        if not self._enable_cost:
            cost = jp.zeros_like(cost)
            hard = jp.zeros_like(hard)
        return cost, d_wall, hard, v_xy_norm, vel_cost_mult

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}

        # ── Phase-0 safety-cost wiring ───────────────────────────────────────
        # Compute (cost, d_wall, hard_violation) from the authoritative torso
        # position (pipeline_state.x.pos[0, :2]), NOT from the observation.
        # Rationale: exclude_current_positions_from_observation=True strips
        # (x, y) from obs, so obs[:, :2] is [z, quat_w] — wrong two dims.
        # This caused a flat cost signal (σ(2)=0.8806) across all v1/v2 runs.
        #
        # NOTE on ordering: velocity is computed BEFORE the safety cost so the
        # "velocity_quadratic" cost type can read v_xy. For the "quadratic" /
        # "sigmoid" paths the velocity is unused inside the cost dispatch but
        # still surfaced via v_xy_norm / vel_cost_mult diagnostics.
        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt

        agent_xy    = pipeline_state.x.pos[0, :2]
        velocity_xy = velocity[:2]
        cost, d_wall, hard_violation, v_xy_norm, vel_cost_mult = (
            self._compute_safety_cost(agent_xy, velocity_xy)
        )
        info["cost"]            = cost
        info["d_wall"]          = d_wall
        info["hard_violation"]  = hard_violation
        info["v_xy_norm"]       = v_xy_norm
        info["vel_cost_mult"]   = vel_cost_mult

        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2., dtype=float)

        reward = -dist + healthy_reward - ctrl_cost - contact_cost

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
            # Safety-cost per-step scalars (flat, not averaged — the training
            # loop is responsible for mean/hist aggregation over batches).
            cost=cost,
            d_wall=d_wall,
            hard_violation=hard_violation,
            # Phase-1g velocity-quadratic diagnostics (always populated for
            # schema uniformity; vel_cost_mult is 0 for non-velocity cost types).
            v_xy_norm=v_xy_norm,
            vel_cost_mult=vel_cost_mult,
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jp.concatenate([qpos] + [qvel] + [target_pos])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a random target location."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return rng, jp.array(self.possible_goals[idx])[0]
