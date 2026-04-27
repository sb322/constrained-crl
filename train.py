"""
train.py — SR-CPO: Surrogate-Reward Constrained Policy Optimization

Drop-in replacement for scaling-crl/train.py that adds:
  • CostCritic  Q_c(s,a,g)  trained with plain cost-to-go (no entropy)
  • Smooth sigmoid cost  c(s) = σ((ε − d_wall) / τ)
  • Preconditioned actor loss  L = E[α·log π − f/ν_f + λ̃·Q_c/ν_c]
  • PID-Lagrangian dual update with critic-based estimator
  • Calibration logging (Kendall τ) for theory-practice gap monitoring

All original CRL logic (InfoNCE critic, residual networks, SAC
entropy, geometric goal relabeling) is preserved exactly.
"""

import functools
import os
import pickle
import time
from dataclasses import dataclass
from typing import NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from brax import envs
from flax.training.train_state import TrainState

# NOTE: flatten_crl_fn is a *static method* on the class, NOT a module export.
from buffer import TrajectoryUniformSamplingQueue
from evaluator import CrlEvaluator
from cost_utils import get_wall_centers, smooth_sigmoid_cost, hard_indicator_cost


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class Args:
    # ── Original CRL args ──────────────────────────────────────
    env_id: str = "humanoid_big_maze"
    eval_env_id: str = "humanoid_big_maze_eval"
    episode_length: int = 1000
    total_env_steps: int = 100_000_000
    num_epochs: int = 100
    num_envs: int = 256
    eval_envs: int = 64
    eval_every: int = 1
    action_repeat: int = 1
    unroll_length: int = 62
    num_minibatches: int = 32
    num_update_epochs: int = 4
    seed: int = 0
    batch_size: int = 256
    normalize_observations: bool = True  # Option-A: restore JaxGCRL/scaling-crl default
    max_replay_size: int = 10_000
    min_replay_size: int = 8_192
    deterministic_eval: bool = True
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Phase-1g alpha cap. Job 994166 forensics decomposed actor_loss = α·log_p −
    # f_term and showed every component except α is stationary across 15 epochs
    # at production τ=0.1: α grows ~26%/epoch (1.12 → 27.9), pushing actor_loss
    # from +1.1 to −148. Extrapolated to 50 epochs: α≈97k, actor_loss≈−530k.
    # Mechanism: target_entropy = −action_size = −8 is unreachable for the
    # tanh-Gaussian under contrastive shaping (achievable entropy ≈ +5.5 nats
    # vs. target +8), so the auto-α update has no equilibrium and α grows
    # unboundedly. Default cap 1.0 caps actor_loss at ≈ |1·log_p − f_term| ≈ 1.
    # Auto-α still adapts in [0, α_max]; the cap only forecloses the
    # unbounded-divergence mode.
    alpha_max: float = 1.0
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    actor_skip_connections: int = 0
    critic_skip_connections: int = 0
    obs_dim: int = 265          # HumanoidMaze: state_dim=268, last 3 = target xyz
    goal_start_idx: int = 0    # root x,y,z live at obs[0:3] (pipeline_state.q[:3])
    goal_end_idx: int = 3
    vis_length: int = 200
    save_buffer: int = 0
    track: bool = True
    wandb_project: str = "constrained-crl"
    wandb_entity: str = ""
    wandb_group: str = ""

    # ── SR-CPO Constrained additions ─────────────────────────────
    use_constraints: bool = True
    # Training-cost shape — see envs/ant_maze.py::__init__ docstring.
    #   "quadratic":  c_train(s) = (1 − d_wall/ε_train)²  on d_wall ≤ ε_train, else 0
    #                 c_hard (s) = 1{d_wall < ε_hard}                (independent threshold)
    #   "sigmoid"  :  c_train(s) = σ((ε − d_wall)/τ)                 (legacy; ignores ε_hard)
    # "quadratic" is the default after run 962826 showed the sigmoid
    # collapses to ~10⁻¹⁵ outside the tight d_wall<ε band and produces
    # vacuous safety.  Hard-indicator C4 calibration uses ε_hard.
    cost_type: str = "quadratic"
    # Under quadratic: ε_train — shaping bandwidth (compact support).  Default
    # 2.0 matches the ant_big_maze corridor half-width (maze_size_scaling=4.0,
    # half_wall_size=2.0 → interior d_wall ∈ [0, 2.0] for typical trajectories).
    # A dense-grid sweep of the interior shows ε_train=2.0 gives nonzero cost
    # at ~88 % of interior positions; ε_train=0.3 covers only ~9 %, leaving
    # the cost field as vacuous as the original sigmoid.
    # Under sigmoid (legacy): ε — threshold inside σ((ε − d_wall)/τ).
    cost_epsilon: float = 2.0
    # Hard-indicator threshold ε_hard — only used under cost_type="quadratic".
    # Kept strictly smaller than cost_epsilon so the shaping band properly
    # surrounds the violation band (ε_hard < ε_train decouples accounting
    # from shaping, mirroring the CBF safety-filter design).
    cost_epsilon_hard: float = 0.1
    cost_tau: float = 0.05           # sigmoid temperature τ (legacy path only)
    # CMDP budget  — interpreted against the hard-indicator under quadratic.
    cost_budget_d: float = 0.15
    lambda_max: float = 100.0
    # PID-Lagrangian dual update (replaces simple gradient ascent)
    pid_kp: float = 0.1             # proportional gain
    pid_ki: float = 0.003           # integral gain
    pid_kd: float = 0.001           # derivative gain
    # Gradient preconditioning scales
    # ν_f = log(N) where N = batch_size (InfoNCE normalization constant)
    # ν_c = 1/(1-γ) (cost-to-go scale)
    # These are computed from other args at init; set to 0 for auto.
    nu_f: float = 1.0               # Option-A: no preconditioning shrinkage, match scaling-crl actor scale. 0 = auto → log(batch_size) (legacy)
    nu_c: float = 0.0               # 0 = auto → 1/(1-gamma)
    # Phase-1f: temperature for cosine InfoNCE logits ⟨φ̂, ψ̂⟩ / τ.
    # For this run we pin τ = 0.1 (per experiment spec: clean stability test
    # in isolation of representation scaling). Paper-standard value is
    # τ = 1/√D = 0.125 (D=64 = final Dense in SA_encoder/G_encoder); τ=0.1
    # is slightly sharper than that, well within the Wang 2025 / JaxGCRL
    # operating range. Keep overridable via --tau for a τ-tuning branch.
    # One knob per run: do NOT bundle τ changes with other interventions.
    tau: float = 0.1
    # Cost critic architecture
    cost_critic_lr: float = 3e-4
    cost_critic_width: int = 256
    cost_critic_depth: int = 4
    cost_critic_skip_connections: int = 0
    cost_critic_tau: float = 0.005   # Polyak averaging coefficient
    cost_discount: float = 0.99      # γ_c for Bellman backup
    # Initial-state estimator: number of goals to sample for Ĵ_c
    dual_estimator_goals: int = 64
    # Depth ablation: depth > 0 overrides actor_depth, critic_depth,
    # and cost_critic_depth simultaneously (for the sweep script).
    # depth = 0 means use the three individual depth args above.
    depth: int = 0


# ═══════════════════════════════════════════════════════════════
#  Network building blocks  (verbatim from upstream)
# ═══════════════════════════════════════════════════════════════

lecun_unfirom = nn.initializers.lecun_normal()  # sic — preserves upstream typo
bias_init = nn.initializers.zeros


def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x); x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x); x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x); x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x); x = activation(x)
    return x + identity


class SA_encoder(nn.Module):
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, s, a):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                for _ in range(4):
                    x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
                    x = normalize(x); x = activation(x)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, g):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = g
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                for _ in range(4):
                    x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
                    x = normalize(x); x = activation(x)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int = 8
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, x):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                for _ in range(4):
                    x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
                    x = normalize(x); x = activation(x)
        means = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_stds = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_stds = jnp.clip(log_stds, -5, 2)
        return means, log_stds


# ── NEW: Cost Critic ──────────────────────────────────────────

class CostCritic(nn.Module):
    """Q_c(s, a, g) → scalar.  Same residual architecture as SA_encoder."""
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, s, a, g):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = jnp.concatenate([s, a, g], axis=-1)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                for _ in range(4):
                    x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
                    x = normalize(x); x = activation(x)
        x = nn.Dense(1, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x.squeeze(-1)


# ═══════════════════════════════════════════════════════════════
#  Transition and TrainingState
# ═══════════════════════════════════════════════════════════════

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    # SR-CPO constraint additions
    cost_critic_state: TrainState
    cost_critic_target_params: dict
    # PID-Lagrangian state
    lambda_tilde: jnp.ndarray     # λ̃ = current dual variable (after PID)
    pid_e_prev: jnp.ndarray       # e_{k-1} for derivative term
    pid_sum_e: jnp.ndarray        # Σ e_j for integral term


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main(args: Args):

    # ── Unified depth override ─────────────────────────────────
    # When --depth D is passed (D > 0), override all three network depths.
    # This is the primary knob for the depth ablation sweep.
    if args.depth > 0:
        args.actor_depth = args.depth
        args.critic_depth = args.depth
        args.cost_critic_depth = args.depth

    # ── Wandb ──────────────────────────────────────────────────
    _eff_depth = args.depth if args.depth > 0 else args.actor_depth
    run_name = f"{args.env_id}_ccrl_d{_eff_depth}_s{args.seed}_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            config=vars(args),
            name=run_name,
        )

    # ── RNG ────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    rng, env_key, eval_key, buf_key = jax.random.split(rng, 4)

    # ── Environment ────────────────────────────────────────────
    # Use brax's training wrapper so state.info["state_extras"] is populated.
    # This gives us seed (episode ID) and truncation flags that flatten_crl_fn needs.
    #
    # PHASE-1 FIX: Ant envs default to `exclude_current_positions_from_observation=True`,
    # which strips torso (x, y) from the observation.  With that, the CRL
    # goal-relabel slice `future_state[:, 0:3]` becomes `[z, quat_x, quat_y]`
    # instead of `[x, y, z]`, so the contrastive critic learns on a pose-
    # ambiguous goal and the task reward stops descending.  Humanoid mazes
    # don't have this problem (humanoid_maze._get_obs does NOT strip xy), so
    # we only override the kwarg for ant_* envs.  The safety-cost computation
    # uses pipeline_state.x.pos[0, :2] internally (Phase-0 fix) and is
    # unaffected either way.
    env_kwargs = {}
    if args.env_id.startswith("ant_"):
        env_kwargs["exclude_current_positions_from_observation"] = False
    # Safety-cost kwargs — only the ant_maze env currently honors these; other
    # envs ignore unknown kwargs via **kwargs passthrough.  We still gate on
    # ant_* to keep the humanoid path exactly as before.
    if args.use_constraints and args.env_id.startswith("ant_"):
        env_kwargs["cost_type"]          = args.cost_type
        env_kwargs["cost_epsilon"]       = args.cost_epsilon
        env_kwargs["cost_epsilon_hard"]  = args.cost_epsilon_hard
        env_kwargs["cost_tau"]           = args.cost_tau
    raw_env = envs.get_environment(args.env_id, **env_kwargs)
    env = envs.training.wrap(
        raw_env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
    )
    eval_env_id = args.eval_env_id if args.eval_env_id else args.env_id
    eval_env_kwargs = {}
    if eval_env_id.startswith("ant_"):
        eval_env_kwargs["exclude_current_positions_from_observation"] = False
    if args.use_constraints and eval_env_id.startswith("ant_"):
        eval_env_kwargs["cost_type"]          = args.cost_type
        eval_env_kwargs["cost_epsilon"]       = args.cost_epsilon
        eval_env_kwargs["cost_epsilon_hard"]  = args.cost_epsilon_hard
        eval_env_kwargs["cost_tau"]           = args.cost_tau
    # Wrap eval env with brax's training wrappers so the evaluator can pass a
    # batched PRNG key of shape (num_eval_envs,) to reset(). Without VmapWrapper
    # the underlying ant_maze.reset sees shape (N,) and jax.random.split raises
    # "split accepts a single key, but was given a key array of shape (N,)".
    eval_env = envs.training.wrap(
        envs.get_environment(eval_env_id, **eval_env_kwargs),
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
    )

    action_size = raw_env.action_size
    goal_dim = args.goal_end_idx - args.goal_start_idx
    obs_size = args.obs_dim + goal_dim

    # Auto-detect actual observation size from the environment.
    # args.obs_dim may differ from the brax version installed on this machine.
    # We probe with a tiny reset to get the ground-truth shape, then correct
    # args.obs_dim and obs_size so networks, buffer, and transitions all agree.
    _probe_keys = jax.random.split(jax.random.PRNGKey(0), 2)
    _probe_obs_size = int(env.reset(_probe_keys).obs.shape[-1])
    if _probe_obs_size != obs_size:
        print(f"[obs_dim auto-correct] args.obs_dim={args.obs_dim} → "
              f"{_probe_obs_size - goal_dim}  (env obs={_probe_obs_size}, goal={goal_dim})")
        args.obs_dim = _probe_obs_size - goal_dim
        obs_size = _probe_obs_size

    # ── Wall geometry (for cost computation) ───────────────────
    if args.use_constraints:
        wall_centers, half_wall_size = get_wall_centers(args.env_id)
    else:
        wall_centers = jnp.zeros((1, 2), dtype=jnp.float32)
        half_wall_size = 1.0

    # ── Networks ───────────────────────────────────────────────
    rng, actor_key, sa_key, g_key, cost_key = jax.random.split(rng, 5)

    sa_encoder = SA_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
    )
    g_encoder = G_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
    )
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder_params  = g_encoder.init(g_key,  np.ones([1, goal_dim]))

    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    actor = Actor(
        action_size=action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr),
    )

    log_alpha_init = jnp.zeros((), dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params=log_alpha_init,
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    cost_critic = CostCritic(
        network_width=args.cost_critic_width,
        network_depth=args.cost_critic_depth,
        skip_connections=args.cost_critic_skip_connections,
    )
    cost_critic_params = cost_critic.init(
        cost_key,
        np.ones([1, args.obs_dim]),
        np.ones([1, action_size]),
        np.ones([1, goal_dim]),
    )
    cost_critic_state = TrainState.create(
        apply_fn=cost_critic.apply,
        params=cost_critic_params,
        tx=optax.adam(learning_rate=args.cost_critic_lr),
    )
    cost_critic_target_params = jax.tree_util.tree_map(jnp.copy, cost_critic_params)

    # ── Auto-compute preconditioning scales ──────────────────
    if args.nu_f == 0.0:
        args.nu_f = float(jnp.log(args.batch_size))  # log(N), InfoNCE scale
    if args.nu_c == 0.0:
        args.nu_c = 1.0 / (1.0 - args.gamma)         # 1/(1-γ), cost-to-go scale
    print(f"  Preconditioning: ν_f={args.nu_f:.4f}, ν_c={args.nu_c:.4f}")

    # ── PID-Lagrangian initial state ──────────────────────────
    lambda_tilde = jnp.zeros((), dtype=jnp.float32)
    pid_e_prev = jnp.zeros((), dtype=jnp.float32)
    pid_sum_e = jnp.zeros((), dtype=jnp.float32)

    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        cost_critic_state=cost_critic_state,
        cost_critic_target_params=cost_critic_target_params,
        lambda_tilde=lambda_tilde,
        pid_e_prev=pid_e_prev,
        pid_sum_e=pid_sum_e,
    )

    # ── Replay buffer ──────────────────────────────────────────
    # Dummy transition MUST match the extras structure we'll store at collection
    # time (after flatten_crl_fn is applied the structure changes, but the buffer
    # stores the raw collected format).
    dummy_transition = Transition(
        observation=jnp.zeros(obs_size),
        action=jnp.zeros(action_size),
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        extras={
            "state_extras": {
                "seed": jnp.zeros(()),
                "truncation": jnp.zeros(()),
            },
            "state":      jnp.zeros(args.obs_dim),   # s_t
            "next_state": jnp.zeros(args.obs_dim),   # s_{t+1} — required by
                                                      # cost_critic_loss_fn.
                                                      # Must match the shape that
                                                      # collect_step / prefill_one
                                                      # actually insert.
            # Phase-0 safety-cost scalars, emitted by the env's step() from the
            # authoritative torso position (pipeline_state.x.pos[0, :2]).
            # These replace the old state[:, :2] slicing path in
            # cost_critic_loss_fn, which was reading [z, quat_w] under the
            # default exclude_current_positions_from_observation=True.
            "cost":            jnp.zeros(()),
            "d_wall":          jnp.zeros(()),
            "hard_violation":  jnp.zeros(()),
            # Phase-1g velocity-quadratic diagnostics — must be in the dummy
            # so the buffer's pytree structure matches what collect_step /
            # prefill_one actually insert. 0 is the correct placeholder
            # (matches the "cost_type=quadratic" default's vel_cost_mult).
            "v_xy_norm":       jnp.zeros(()),
            "vel_cost_mult":   jnp.zeros(()),
        },
    )

    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=args.max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=args.batch_size * args.num_minibatches * args.num_update_epochs,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
    )
    buffer_state = replay_buffer.init(buf_key)

    # ── Evaluator ──────────────────────────────────────────────
    # actor_step must match: (training_state, env, env_state, extra_fields) -> (env_state, transition)
    def actor_step(training_state, env, env_state, extra_fields=()):
        obs = env_state.obs
        state = obs[:, :args.obs_dim]
        goal  = obs[:, args.obs_dim:]
        observation = jnp.concatenate([state, goal], axis=1)
        means, _ = actor.apply(training_state.actor_state.params, observation)
        actions = nn.tanh(means)                  # deterministic for eval
        next_env_state = env.step(env_state, actions)
        transition = Transition(
            observation=obs,
            action=actions,
            reward=next_env_state.reward,
            discount=1.0 - next_env_state.done,
            extras={},
        )
        return next_env_state, transition

    # Eval-time hard-indicator cost — uses ε_hard under the quadratic scheme so
    # "violation" is measured against the tight threshold (not the shaping band).
    # Under the legacy sigmoid path ε_hard is irrelevant; we fall back to ε.
    _eps_hard = (
        args.cost_epsilon_hard if args.cost_type == "quadratic" else args.cost_epsilon
    )
    if args.use_constraints:
        def eval_cost_fn(xy: jnp.ndarray) -> jnp.ndarray:
            return hard_indicator_cost(
                xy, wall_centers, half_wall_size,
                cost_epsilon=_eps_hard,
            )
    else:
        eval_cost_fn = None

    # Calibration-tracking smooth cost at eval time.  Must mirror the *training*
    # cost shape so C4 (|train_cost − eval_smooth_cost|) is meaningful.
    if args.use_constraints:
        if args.cost_type == "quadratic":
            from cost_utils import compact_quadratic_cost
            _eps_train = args.cost_epsilon
            _eps_hard_local = args.cost_epsilon_hard
            def eval_smooth_cost_fn(xy: jnp.ndarray) -> jnp.ndarray:
                cost, _, _ = compact_quadratic_cost(
                    xy, wall_centers, half_wall_size,
                    cost_epsilon_train=_eps_train,
                    cost_epsilon_hard=_eps_hard_local,
                )
                return cost
        else:  # legacy sigmoid
            def eval_smooth_cost_fn(xy: jnp.ndarray) -> jnp.ndarray:
                cost, _, _ = smooth_sigmoid_cost(
                    xy, wall_centers, half_wall_size,
                    cost_epsilon=args.cost_epsilon,
                    cost_tau=args.cost_tau,
                )
                return cost
    else:
        eval_smooth_cost_fn = None

    evaluator = CrlEvaluator(
        actor_step=actor_step,
        eval_env=eval_env,
        num_eval_envs=args.eval_envs,
        episode_length=args.episode_length,
        key=eval_key,
        cost_fn=eval_cost_fn,
        cost_budget_d=args.cost_budget_d,
        obs_dim=args.obs_dim,
        smooth_cost_fn=eval_smooth_cost_fn,
    )

    # ──────────────────────────────────────────────────────────
    #  Loss functions
    # ──────────────────────────────────────────────────────────

    def critic_loss_fn(critic_params, transitions, key):
        """InfoNCE contrastive loss — unchanged from upstream."""
        sa_enc_p = critic_params["sa_encoder"]
        g_enc_p  = critic_params["g_encoder"]
        obs          = transitions.observation
        action       = transitions.action
        future_state = transitions.extras["future_state"]
        goal         = future_state[:, args.goal_start_idx:args.goal_end_idx]

        sa_repr = sa_encoder.apply(sa_enc_p, obs[:, :args.obs_dim], action)
        g_repr  = g_encoder.apply(g_enc_p, goal)

        # Phase-1f: cosine-InfoNCE energy. Row-L2 normalize φ, ψ so
        #   ⟨φ̂, ψ̂⟩  =  cos(φ, ψ)  ∈  [−1, 1],
        # making the critic score genuinely norm-bounded. The Phase-1d Option-A
        # comment here used to claim that LayerNorm *inside* the residual blocks
        # anchored ‖φ‖, ‖ψ‖; it does not (LayerNorm normalizes hidden activations
        # per-sample, not the final encoder output — trailing Dense layers can
        # rescale arbitrarily). Without explicit row-L2 normalization, ⟨φ, ψ⟩ is
        # unbounded above, and the actor escapes by growing ‖φ‖‖ψ‖, which is
        # what destroyed Phase-1b (|a_loss| → −4.86e5) AND Phase-1d Option-A
        # (|a_loss| → −4.67e5) at λ̃ = 0. Dividing by args.tau controls the
        # softmax temperature for the InfoNCE log-likelihood.

        # Diagnostic: mean ‖φ‖, ‖ψ‖ BEFORE normalization — the Phase-1d escape
        # route (growing representations to drive ⟨φ,ψ⟩ → ∞) will show here
        # even while the post-normalization cosine logits stay bounded. If
        # these grow monotonically while critic_loss stays flat, the row-L2
        # is masking a runaway encoder — the signal we care about per the
        # stability-isolation experiment spec.
        sa_repr_norm = jnp.mean(jnp.linalg.norm(sa_repr, axis=-1))
        g_repr_norm  = jnp.mean(jnp.linalg.norm(g_repr,  axis=-1))

        # Phase-1f NaN-forensics probes (pure observability, no causal change).
        # After Phase-1f smokes at both τ=0.1 and τ=1.0 produced identical
        # step-1 NaN signatures, we need to localize NaN origin: buffer obs,
        # encoder forward, zero-norm at the row-L2 divisor, or post-division.
        sa_has_nan_c  = jnp.any(jnp.isnan(sa_repr)).astype(jnp.float32)
        g_has_nan_c   = jnp.any(jnp.isnan(g_repr)).astype(jnp.float32)
        obs_has_nan_c = jnp.any(jnp.isnan(obs)).astype(jnp.float32)
        sa_norm_min_c = jnp.min(jnp.linalg.norm(sa_repr, axis=-1))
        g_norm_min_c  = jnp.min(jnp.linalg.norm(g_repr,  axis=-1))

        # Phase-1f NaN-fix.  Original form was
        #     sa_repr / (jnp.linalg.norm(sa_repr, axis=-1) + 1e-8)
        # which is forward-finite (divisor ≥ 1e-8) but autograd-NaN whenever
        # sum(sa²) underflows to 0.  jnp.linalg.norm internally computes
        # sqrt(sum(x²)); JAX's gradient of sqrt(s) is 1/(2·sqrt(s)), so at
        # s = 0 the backward returns +inf, and inf · 0 from the chain rule
        # produces NaN even though the forward divisor is 1e-8.  Putting
        # the epsilon INSIDE the sqrt makes the radicand strictly positive,
        # so both forward and backward are finite.  Job 988003 epoch-1
        # forensics: critic forward finite, c_grad_nan first at flat=0 —
        # the textbook signature of this autograd singularity.
        sa_norm_safe = jnp.sqrt(
            jnp.sum(sa_repr * sa_repr, axis=-1, keepdims=True) + 1e-12
        )
        g_norm_safe  = jnp.sqrt(
            jnp.sum(g_repr  * g_repr,  axis=-1, keepdims=True) + 1e-12
        )
        sa_repr = sa_repr / sa_norm_safe
        g_repr  = g_repr  / g_norm_safe
        logits = jnp.einsum("ik,jk->ij", sa_repr, g_repr) / args.tau
        logits_has_nan_c = jnp.any(jnp.isnan(logits)).astype(jnp.float32)
        logsumexp_val = jax.nn.logsumexp(logits, axis=1)
        loss = -jnp.mean(jnp.diag(logits) - logsumexp_val)
        loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp_val ** 2)

        I = jnp.eye(logits.shape[0])
        correct  = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I)       / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        return loss, (logits_pos, logits_neg, jnp.mean(correct),
                      jnp.mean(logsumexp_val), sa_repr_norm, g_repr_norm,
                      sa_has_nan_c, g_has_nan_c, obs_has_nan_c,
                      sa_norm_min_c, g_norm_min_c, logits_has_nan_c)

    def actor_loss_fn(actor_params, critic_params, log_alpha,
                      cost_critic_params, lambda_tilde, transitions, key):
        """Preconditioned SR-CPO actor loss:

            L = E[α·log π  −  f(s,a,g)/ν_f  +  λ̃·Q_c(s,a,g)/ν_c]

        where ν_f = log(N) and ν_c = 1/(1−γ) are fixed preconditioning
        scales that put the contrastive surrogate and cost critic on
        comparable gradient magnitude.  λ̃ is the PID dual variable.
        """
        obs          = transitions.observation
        future_state = transitions.extras["future_state"]
        state        = obs[:, :args.obs_dim]
        goal         = future_state[:, args.goal_start_idx:args.goal_end_idx]
        observation  = jnp.concatenate([state, goal], axis=1)

        # Reparameterized action sample (SAC-style)
        means, log_stds = actor.apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)

        # Phase-1f: cosine-InfoNCE energy, matching critic_loss_fn. Row-L2
        # normalize φ, ψ so f(s, a, g) = ⟨φ̂, ψ̂⟩ / τ ∈ [−1/τ, 1/τ]. This is the
        # actual norm bound on the actor's reward-critic term, replacing the
        # broken Phase-1d claim that LayerNorm alone bounded ‖φ‖, ‖ψ‖.
        sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, action)
        g_repr  = g_encoder.apply(critic_params["g_encoder"],  goal)

        # Diagnostic (pre-normalization): mean ‖φ‖, ‖ψ‖ seen by the actor.
        # Should stay finite. If it grows while actor_loss also grows, the
        # actor is exploiting encoder rescaling as its minimization route —
        # exactly the failure mode the row-L2 is meant to foreclose.
        sa_repr_norm_actor = jnp.mean(jnp.linalg.norm(sa_repr, axis=-1))
        g_repr_norm_actor  = jnp.mean(jnp.linalg.norm(g_repr,  axis=-1))

        # Phase-1f NaN-forensics probes (actor path). Mirrors the critic-side
        # probes so we can tell whether the actor's action already produced
        # NaN before reaching the encoder, or whether the encoder itself
        # produced NaN on a finite input.
        action_has_nan_a = jnp.any(jnp.isnan(action)).astype(jnp.float32)
        action_max_a     = jnp.max(jnp.abs(action))
        sa_has_nan_a     = jnp.any(jnp.isnan(sa_repr)).astype(jnp.float32)
        g_has_nan_a      = jnp.any(jnp.isnan(g_repr)).astype(jnp.float32)
        sa_norm_min_a    = jnp.min(jnp.linalg.norm(sa_repr, axis=-1))
        g_norm_min_a     = jnp.min(jnp.linalg.norm(g_repr,  axis=-1))

        # Phase-1f NaN-fix (actor side, identical reasoning to the critic
        # side above). Move epsilon inside sqrt so the radicand is always
        # strictly positive; otherwise grad of sqrt(0) = +inf and chain
        # rule produces NaN gradients despite finite forward.
        sa_norm_safe = jnp.sqrt(
            jnp.sum(sa_repr * sa_repr, axis=-1, keepdims=True) + 1e-12
        )
        g_norm_safe  = jnp.sqrt(
            jnp.sum(g_repr  * g_repr,  axis=-1, keepdims=True) + 1e-12
        )
        sa_repr = sa_repr / sa_norm_safe
        g_repr  = g_repr  / g_norm_safe
        f_sa_g  = jnp.sum(sa_repr * g_repr, axis=-1) / args.tau
        f_has_nan_a = jnp.any(jnp.isnan(f_sa_g)).astype(jnp.float32)

        alpha = jnp.exp(log_alpha)

        # Phase-1g actor-loss component forensics. After job 990871, the
        # 50-epoch Phase-1f run with row-L2 + τ=0.1 produced a_loss = −484K
        # despite all NaN flags clean and f_sa_g ∈ [−10, 10] by Cauchy-Schwarz
        # on the unit sphere. Since
        #     a_loss = α·mean(log_prob) − mean(f_sa_g)/ν_f
        # and the second term is bounded by 10/ν_f, the −484K must originate
        # in α·mean(log_prob). These probes split that into its causal pieces:
        #
        #   alpha_metric    : α = exp(log_α). If α ≫ 1 and grows monotonically,
        #                     the SAC entropy temperature is unbounded above —
        #                     symptom of entropy-target chasing.
        #   log_prob_mean   : mean(log_prob) = mean(Gaussian logpdf − tanh-sat).
        #                     If hugely positive, distribution is concentrated
        #                     (low entropy, σ small or saturating actions).
        #                     If hugely negative, distribution is spread out
        #                     and α·log_prob can blow up negative.
        #   alpha_logprob   : α·mean(log_prob). The dominant term in a_loss.
        #                     Direct read on whether this is the divergent term.
        #   gaussian_logp_mean: mean(Gaussian logpdf only, before tanh corr).
        #                     Splits log_prob into its two contributions.
        #   sat_correction_mean: mean(−log((1−a²)+1e-6)) summed over action dims.
        #                     With |a|=1, each dim contributes +log(1e6) ≈ +13.8;
        #                     summed over 8 ant action dims → +110. If this
        #                     term dominates log_prob, tanh-saturation is the
        #                     direct driver of α·log_prob magnitude.
        #   log_std_mean    : mean(log σ). Lower-clipped at −5 (σ≈0.0067), upper
        #                     at +2 (σ≈7.4). If pinned to the lower clip, policy
        #                     has collapsed to a near-deterministic peak.
        #   f_term_mean     : mean(f_sa_g)/ν_f. Should stay bounded by 10 under
        #                     the row-L2 contract.  Probe so we can confirm.
        sat_correction_per_dim = jnp.log((1 - jnp.square(action)) + 1e-6)
        # gaussian_logp_per_dim is the un-corrected Gaussian density log,
        # before subtracting the tanh-Jacobian. summed over action dims.
        gaussian_logp_full = jax.scipy.stats.norm.logpdf(
            x_ts, loc=means, scale=stds
        ).sum(-1)
        sat_correction_full = sat_correction_per_dim.sum(-1)

        alpha_metric        = alpha
        log_prob_mean       = jnp.mean(log_prob)
        alpha_logprob_mean  = alpha * log_prob_mean
        gaussian_logp_mean  = jnp.mean(gaussian_logp_full)
        sat_correction_mean = jnp.mean(sat_correction_full)
        log_std_mean        = jnp.mean(log_stds)
        f_term_mean         = jnp.mean(f_sa_g) / args.nu_f

        # Preconditioned: divide f by ν_f to normalize InfoNCE scale
        actor_loss = jnp.mean(alpha * log_prob - f_sa_g / args.nu_f)

        if args.use_constraints:
            qc_pi = cost_critic.apply(cost_critic_params, state, action, goal)
            # Preconditioned: divide Q_c by ν_c to normalize cost-to-go scale
            actor_loss = actor_loss + jnp.mean(lambda_tilde * qc_pi / args.nu_c)
        else:
            qc_pi = jnp.zeros_like(f_sa_g)

        return actor_loss, (log_prob, jnp.mean(qc_pi),
                            sa_repr_norm_actor, g_repr_norm_actor,
                            sa_has_nan_a, g_has_nan_a,
                            sa_norm_min_a, g_norm_min_a,
                            action_has_nan_a, action_max_a, f_has_nan_a,
                            alpha_metric, log_prob_mean, alpha_logprob_mean,
                            gaussian_logp_mean, sat_correction_mean,
                            log_std_mean, f_term_mean)

    def alpha_loss_fn(log_alpha, log_prob):
        return -jnp.mean(log_alpha * (log_prob + action_size))

    def cost_critic_loss_fn(cost_critic_params, cost_critic_target_params,
                            actor_params, transitions, key):
        """TD(0) Bellman loss for Q_c with plain cost-to-go (NO entropy).

        Target: y = c(s) + γ_c · Q_c^target(s', a', g)
        where a' ~ π(·|s',g) and V_c(s',g) = E_{a'~π}[Q_c(s',a',g)]
        (plain expectation, no -α·log π term — this is cost-to-go, not
        soft value).  See SR-CPO theory §5.3.

        Uses smooth sigmoid cost c(s) = σ((ε - d_wall)/τ).
        """
        obs          = transitions.observation
        action       = transitions.action
        future_state = transitions.extras["future_state"]
        state        = obs[:, :args.obs_dim]
        # s_{t+1} for the Bellman target.  transitions.extras["state"] is s_t,
        # NOT s_{t+1} — using it collapsed Q_c into a stationary self-reference
        # and violated the contractive backup.  The collector / flatten_crl_fn
        # now provide explicit "next_state" = s_{t+1}.
        next_state   = transitions.extras["next_state"]
        goal         = future_state[:, args.goal_start_idx:args.goal_end_idx]

        # Phase-0 fix: read the per-step cost directly from transition.extras.
        # Previously we recomputed c(s) from state[:, :2], which — under the
        # default exclude_current_positions_from_observation=True — was the
        # slice [z, quat_w] (not torso xy).  d_wall therefore collapsed to 0
        # inside the wall box at the origin, and c(s) ≡ σ((ε-0)/τ) = 0.8806.
        # The env now computes c, d_wall, 1{d_wall<ε} from the authoritative
        # pipeline_state.x.pos[0, :2] and emits them into state.info; the
        # collector lifts them into transition.extras; flatten_crl_fn
        # preserves them.  Here we just read them out.
        cost           = transitions.extras["cost"]
        d_wall         = transitions.extras["d_wall"]
        hard_indicator = transitions.extras["hard_violation"]
        # Phase-1g velocity-quadratic diagnostics. NOT used in the Bellman
        # target — they are env-emitted side-channels, just for logging.
        v_xy_norm      = transitions.extras["v_xy_norm"]
        vel_cost_mult  = transitions.extras["vel_cost_mult"]

        # Next-state action from current policy (plain sample, no entropy)
        next_obs = jnp.concatenate([next_state, goal], axis=1)
        means_next, log_stds_next = actor.apply(actor_params, next_obs)
        key, subkey = jax.random.split(key)
        x_next = means_next + jnp.exp(log_stds_next) * jax.random.normal(
            subkey, shape=means_next.shape)
        action_next = nn.tanh(x_next)

        # Plain cost-to-go target:
        #
        #     y_t = c(s_t) + γ_c · (1 − d_t) · Q_c^target(s_{t+1}, a', g)
        #
        # where d_t = 1{episode terminated at step t}.  The (1 − d_t) mask is
        # ESSENTIAL: on terminal transitions, s_{t+1} is the post-reset state,
        # and bootstrapping through it injects noise into the Q_c target
        # (violates the Bellman contraction).  `transitions.discount` is
        # stored as (1 − done) by the collector, so we can use it directly.
        #
        # NOTE: No −α·log π term here. Cost critic is plain cost-to-go, not
        # a soft value (see SR-CPO theory §5.3).
        done_mask = transitions.discount                    # 1 − d_t
        qc_target = cost_critic.apply(cost_critic_target_params, next_state, action_next, goal)
        td_target = jax.lax.stop_gradient(
            cost + args.cost_discount * done_mask * qc_target
        )

        qc_online = cost_critic.apply(cost_critic_params, state, action, goal)
        loss = jnp.mean((qc_online - td_target) ** 2)

        return loss, {
            "cost_critic_loss": loss,
            "mean_step_cost":   jnp.mean(cost),
            "mean_d_wall":      jnp.mean(d_wall),
            "hard_violation_rate": jnp.mean(hard_indicator),
            "mean_qc":          jnp.mean(qc_online),
            # Phase-1g velocity-quadratic diagnostics. mean_v_xy_norm tells
            # us the typical torso speed in the maze plane; mean_vel_cost_mult
            # tells us the typical velocity multiplier on the geometric cost
            # shell (= 0 for non-velocity cost types). Together with
            # mean_step_cost they let us decompose what the velocity-coupled
            # cost is actually doing on the trained policy.
            "mean_v_xy_norm":      jnp.mean(v_xy_norm),
            "mean_vel_cost_mult":  jnp.mean(vel_cost_mult),
        }

    # ──────────────────────────────────────────────────────────
    #  SGD step — NO @jax.jit; it lives inside jax.lax.scan
    # ──────────────────────────────────────────────────────────

    def _critic_based_dual_estimator(cost_critic_params, actor_params,
                                     initial_obs, goals, key):
        """Ĵ_c = (1-γ) · (1/M) · Σ_m V_c(s_0, g_m).

        Estimates the expected discounted cost under the current policy
        from initial states, using the cost critic.  This replaces the
        biased batch-average estimator from the old code.

        Args:
            cost_critic_params: current cost critic parameters.
            actor_params: current actor parameters.
            initial_obs: [B, obs_dim] initial observations (s_0 ~ ρ_0).
            goals: [M, goal_dim] sampled goals.
            key: PRNG key for action sampling.

        Returns:
            j_c_hat: scalar estimate of J_c(π_θ).
        """
        # For each (s_0, g) pair, sample a ~ π(·|s_0, g) and get Q_c
        M = goals.shape[0]
        B = initial_obs.shape[0]
        # Use first B states, tile goals: [B, M, ...]
        s0 = initial_obs[:, None, :].repeat(M, axis=1)  # [B, M, obs_dim]
        g  = goals[None, :, :].repeat(B, axis=0)         # [B, M, goal_dim]
        s0_flat = s0.reshape(-1, initial_obs.shape[-1])   # [B*M, obs_dim]
        g_flat  = g.reshape(-1, goals.shape[-1])          # [B*M, goal_dim]

        obs_flat = jnp.concatenate([s0_flat, g_flat], axis=-1)
        means, log_stds = actor.apply(actor_params, obs_flat)
        a_flat = nn.tanh(means + jnp.exp(log_stds) * jax.random.normal(
            key, shape=means.shape))

        qc_vals = cost_critic.apply(cost_critic_params, s0_flat, a_flat, g_flat)
        # V_c(s_0, g) ≈ Q_c(s_0, a, g) for a ~ π
        v_c = qc_vals.reshape(B, M).mean(axis=0)  # [M] — average over s_0
        j_c_hat = (1.0 - args.gamma) * jnp.mean(v_c)
        return j_c_hat

    # Phase-1f gradient/param NaN-forensics helpers (used inside sgd_step).
    # _grads_have_nan returns 1.0 if any leaf of the gradient pytree contains
    #   NaN — fingerprints "loss → grad chain produced NaN".
    # _grads_global_norm = sqrt(sum(‖leaf‖²)) — finite-but-huge means runaway.
    # _params_have_nan returns 1.0 if any post-update param leaf is NaN —
    #   fingerprints "optimizer step poisoned the params".
    def _grads_have_nan(grads):
        leaves = jax.tree_util.tree_leaves(grads)
        flags = jnp.stack([jnp.any(jnp.isnan(x)) for x in leaves])
        return jnp.any(flags).astype(jnp.float32)

    def _grads_global_norm(grads):
        leaves = jax.tree_util.tree_leaves(grads)
        sq = sum(jnp.sum(x * x) for x in leaves)
        return jnp.sqrt(sq)

    def _params_have_nan(params):
        leaves = jax.tree_util.tree_leaves(params)
        flags = jnp.stack([jnp.any(jnp.isnan(x)) for x in leaves])
        return jnp.any(flags).astype(jnp.float32)

    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key, cost_key, dual_key = jax.random.split(key, 5)

        # ── InfoNCE Critic update ─────────────────────────────
        # aux = (logits_pos, logits_neg, accuracy, logsumexp,
        #        mean ‖φ‖ pre-normalize, mean ‖ψ‖ pre-normalize,
        #        + Phase-1f NaN-forensics probes:
        #        sa_nan_c, g_nan_c, obs_nan_c, sa_norm_min_c, g_norm_min_c, logits_nan_c)
        (c_loss, (lp, ln, acc, lse, sa_rn_crit, g_rn_crit,
                  sa_nan_c, g_nan_c, obs_nan_c,
                  sa_nmin_c, g_nmin_c, logits_nan_c)), c_grads = \
            jax.value_and_grad(critic_loss_fn, has_aux=True)(
            training_state.critic_state.params, transitions, critic_key)
        # Gradient diagnostics (Phase-1f forensics):
        #   c_grad_nan = 1 if any leaf is NaN  →  loss → grad chain broke
        #   c_grad_norm reports magnitude; finite-but-huge means runaway
        c_grad_nan  = _grads_have_nan(c_grads)
        c_grad_norm = _grads_global_norm(c_grads)
        critic_state = training_state.critic_state.apply_gradients(grads=c_grads)
        c_params_nan = _params_have_nan(critic_state.params)

        # ── Actor update (preconditioned Lagrangian) ──────────
        # aux = (log_prob, mean_qc_pi, mean ‖φ‖ pre-normalize, mean ‖ψ‖ pre-normalize,
        #        + Phase-1f NaN-forensics probes:
        #        sa_nan_a, g_nan_a, sa_norm_min_a, g_norm_min_a,
        #        action_nan_a, action_max_a, f_nan_a)
        (a_loss, (log_prob, mean_qc_pi, sa_rn_act, g_rn_act,
                  sa_nan_a, g_nan_a, sa_nmin_a, g_nmin_a,
                  action_nan_a, action_max_a, f_nan_a,
                  alpha_metric_a, log_prob_mean_a, alpha_logprob_mean_a,
                  gaussian_logp_mean_a, sat_correction_mean_a,
                  log_std_mean_a, f_term_mean_a)), a_grads = \
            jax.value_and_grad(actor_loss_fn, has_aux=True)(
            training_state.actor_state.params,
            critic_state.params,
            training_state.alpha_state.params,
            training_state.cost_critic_state.params,
            training_state.lambda_tilde,
            transitions,
            actor_key,
        )
        # Actor-side gradient diagnostics
        a_grad_nan  = _grads_have_nan(a_grads)
        a_grad_norm = _grads_global_norm(a_grads)
        actor_state = training_state.actor_state.apply_gradients(grads=a_grads)
        a_params_nan = _params_have_nan(actor_state.params)

        # ── SAC entropy alpha ─────────────────────────────────
        al_loss, al_grads = jax.value_and_grad(alpha_loss_fn)(
            training_state.alpha_state.params, log_prob)
        alpha_state = training_state.alpha_state.apply_gradients(grads=al_grads)
        # Phase-1g α-cap. Job 994166 forensics confirmed actor_loss diverges
        # purely via α (every other actor-loss component is stationary). Cap
        # log_alpha at log(args.alpha_max) so α ∈ [0, alpha_max]. The auto-α
        # update remains active inside that range; we only foreclose the
        # unbounded-divergence mode where target_entropy is unreachable.
        # alpha_clip_active = 1 if the cap binds this step; useful diagnostic
        # for whether auto-α is naturally below the cap or pinned to it.
        log_alpha_cap = jnp.log(jnp.maximum(args.alpha_max, 1e-12))
        clipped_log_alpha = jnp.minimum(alpha_state.params, log_alpha_cap)
        alpha_clip_active = (alpha_state.params > log_alpha_cap).astype(jnp.float32)
        alpha_state = alpha_state.replace(params=clipped_log_alpha)

        # ── Cost critic + PID dual update ─────────────────────
        if args.use_constraints:
            # Cost critic TD update
            (cc_loss, cc_m), cc_grads = jax.value_and_grad(
                cost_critic_loss_fn, has_aux=True)(
                training_state.cost_critic_state.params,
                training_state.cost_critic_target_params,
                actor_state.params,
                transitions,
                cost_key,
            )
            cc_grad_nan  = _grads_have_nan(cc_grads)
            cc_grad_norm = _grads_global_norm(cc_grads)
            cost_critic_state_new = training_state.cost_critic_state.apply_gradients(
                grads=cc_grads)
            cc_params_nan = _params_have_nan(cost_critic_state_new.params)

            # Polyak target update
            tau = args.cost_critic_tau
            target_params_new = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1.0 - tau) * tp,
                cost_critic_state_new.params,
                training_state.cost_critic_target_params,
            )

            # ── Critic-based dual estimator ───────────────────
            # Use initial observations from the batch as s_0 ~ ρ_0
            # and goals from the batch for the Monte Carlo estimate.
            initial_obs = transitions.observation[:, :args.obs_dim]
            future_state = transitions.extras["future_state"]
            goals = future_state[:args.dual_estimator_goals,
                                 args.goal_start_idx:args.goal_end_idx]
            j_c_hat = _critic_based_dual_estimator(
                cost_critic_state_new.params, actor_state.params,
                initial_obs[:args.dual_estimator_goals],
                goals, dual_key,
            )

            # ── PID-Lagrangian update (Stooke et al., 2020) ──────
            #
            #     e_k  = Ĵ_c(π_θ) − d                (constraint error)
            #     S_k  = clamp( S_{k−1} + e_k, ±S_max )   (anti-windup)
            #     λ̃_k  = clip( K_p·e_k + K_i·S_k + K_d·(e_k − e_{k−1}),
            #                   0, λ_max )
            #
            # Anti-windup rationale.  Without the clamp on S_k, once λ̃
            # saturates at λ_max the integral keeps accumulating during a
            # persistent violation, so when the violation subsides λ̃
            # lags badly before descending.  We bound S_k so that the
            # integral term alone cannot exceed λ_max:  |K_i·S_max| ≤ λ_max
            # ⇒  S_max = λ_max / max(K_i, 1e−8).
            e_k = j_c_hat - args.cost_budget_d
            s_max = args.lambda_max / jnp.maximum(args.pid_ki, 1e-8)
            pid_sum_e_new = jnp.clip(
                training_state.pid_sum_e + e_k, -s_max, s_max
            )
            lambda_tilde_new = jnp.clip(
                args.pid_kp * e_k
                + args.pid_ki * pid_sum_e_new
                + args.pid_kd * (e_k - training_state.pid_e_prev),
                0.0, args.lambda_max,
            )

            j_c_hat_metric = j_c_hat
        else:
            cost_critic_state_new = training_state.cost_critic_state
            target_params_new = training_state.cost_critic_target_params
            lambda_tilde_new = training_state.lambda_tilde
            e_k = jnp.zeros(())
            pid_sum_e_new = training_state.pid_sum_e
            cc_loss = jnp.zeros(())
            cc_m = {k: jnp.zeros(()) for k in
                    ["cost_critic_loss", "mean_step_cost", "mean_d_wall",
                     "hard_violation_rate", "mean_qc",
                     "mean_v_xy_norm", "mean_vel_cost_mult"]}
            j_c_hat_metric = jnp.zeros(())
            cc_grad_nan   = jnp.zeros(())
            cc_grad_norm  = jnp.zeros(())
            cc_params_nan = jnp.zeros(())

        new_ts = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
            cost_critic_state=cost_critic_state_new,
            cost_critic_target_params=target_params_new,
            lambda_tilde=lambda_tilde_new,
            pid_e_prev=e_k,
            pid_sum_e=pid_sum_e_new,
        )

        metrics = {
            "critic_loss":      c_loss,
            "actor_loss":       a_loss,
            "alpha_loss":       al_loss,
            "alpha":            jnp.exp(alpha_state.params),
            "logits_pos":       lp,
            "logits_neg":       ln,
            "accuracy":         acc,
            "logsumexp":        lse,
            # Phase-1f representation-scaling diagnostics (pre-normalize norms).
            # *_critic come from the critic pass (goals from future buffer);
            # *_actor come from the actor pass (goals from current policy).
            # Both should stay bounded if row-L2 is genuinely isolating the
            # stability question from encoder-rescale escapes.
            "sa_repr_norm_critic": sa_rn_crit,
            "g_repr_norm_critic":  g_rn_crit,
            "sa_repr_norm_actor":  sa_rn_act,
            "g_repr_norm_actor":   g_rn_act,
            # Phase-1f NaN-forensics — scalar boolean flags (0.0 / 1.0) and mins.
            # Any flag > 0 at epoch 1 names the NaN origin site:
            #   obs_nan_c = 1  → buffer contamination (env emitted NaN into buffer)
            #   sa_nan_c / g_nan_c = 1 (with obs_nan_c = 0) → encoder forward NaN
            #   *_norm_min = 0  → row-L2 divides by +1e-8 on zero vector (not NaN but suspect)
            #   logits_nan_c = 1 (with sa_nan_c = g_nan_c = 0) → cosine/τ pathology
            #   action_nan_a = 1 → actor emits NaN action; upstream cause = actor enc NaN
            #   action_max_a very large → tanh saturation precursor (should stay ≤ 1.0)
            #   f_nan_a = 1  → actor reward term is NaN even with finite φ̂, ψ̂
            "nan_obs_critic":      obs_nan_c,
            "nan_sa_critic":       sa_nan_c,
            "nan_g_critic":        g_nan_c,
            "nan_logits_critic":   logits_nan_c,
            "sa_norm_min_critic":  sa_nmin_c,
            "g_norm_min_critic":   g_nmin_c,
            "nan_sa_actor":        sa_nan_a,
            "nan_g_actor":         g_nan_a,
            "nan_action_actor":    action_nan_a,
            "nan_f_actor":         f_nan_a,
            "sa_norm_min_actor":   sa_nmin_a,
            "g_norm_min_actor":    g_nmin_a,
            "action_max_actor":    action_max_a,
            # Phase-1f gradient/param NaN-forensics. These localize *which* of
            # the three gradient steps within sgd_step first produces NaN, and
            # whether the optimizer apply produced NaN params (vs. NaN-grad
            # → finite params via Adam's epsilon). Reading order:
            #   c_grad_nan  = 1 → critic_loss_fn → grad chain produced NaN
            #     (with all forward probes finite, this points at the autograd
            #      path through row-L2 / cosine / xent, not the forward eval)
            #   c_grad_norm finite-but-huge → runaway, not NaN. Combine with
            #     c_params_nan to disambiguate "Adam clamped" vs "params survived"
            #   c_params_nan = 1 → critic_state.apply_gradients produced NaN
            #     params (Adam moments saturated or NaN propagated through
            #     update). Critical: NaN params here will poison every
            #     subsequent forward pass — actor reads critic.params via
            #     stop_grad in actor_loss_fn? No — actor reads its own params.
            #     But cost_critic uses actor_state.params in its dual estimator,
            #     so a NaN actor poisons cost critic too.
            #   a_*: same logic, but actor params poison the next rollout
            #     (collect_step → tanh(means + std·N) → action NaN → buffer NaN)
            #   cc_*: same logic; cost critic NaN doesn't poison rollout but
            #     poisons λ̃ via dual estimator.
            "c_grad_nan":     c_grad_nan,
            "c_grad_norm":    c_grad_norm,
            "c_params_nan":   c_params_nan,
            "a_grad_nan":     a_grad_nan,
            "a_grad_norm":    a_grad_norm,
            "a_params_nan":   a_params_nan,
            "cc_grad_nan":    cc_grad_nan,
            "cc_grad_norm":   cc_grad_norm,
            "cc_params_nan":  cc_params_nan,
            # Phase-1g actor-loss component forensics. Each is a per-sgd-step
            # mean. Reading these together at epoch 1 vs epoch 50 of a run
            # whose actor_loss diverges to −484K decomposes the divergence
            # into a definite causal source (entropy temperature α, policy
            # log-density, tanh-saturation Jacobian, or Gaussian density).
            "alpha_actor":           alpha_metric_a,
            "log_prob_mean_actor":   log_prob_mean_a,
            "alpha_logprob_actor":   alpha_logprob_mean_a,
            "gaussian_logp_actor":   gaussian_logp_mean_a,
            "sat_correction_actor":  sat_correction_mean_a,
            "log_std_mean_actor":    log_std_mean_a,
            "f_term_mean_actor":     f_term_mean_a,
            # Phase-1g α-cap diagnostic. 1.0 means the cap bound this step
            # (auto-α wanted to push α above alpha_max but was clipped).
            "alpha_clip_active":     alpha_clip_active,
            "cost_critic_loss": cc_m["cost_critic_loss"],
            "mean_step_cost":   cc_m["mean_step_cost"],
            "mean_d_wall":      cc_m["mean_d_wall"],
            "hard_violation_rate": cc_m["hard_violation_rate"],
            "mean_qc":          cc_m["mean_qc"],
            # Phase-1g velocity-quadratic diagnostics
            "mean_v_xy_norm":      cc_m["mean_v_xy_norm"],
            "mean_vel_cost_mult":  cc_m["mean_vel_cost_mult"],
            "lambda_tilde":     lambda_tilde_new,
            "j_c_hat":          j_c_hat_metric,
            "mean_qc_pi":       mean_qc_pi,
        }
        return (new_ts, key), metrics

    # ──────────────────────────────────────────────────────────
    #  Env collection step (no buffer write — done in bulk below)
    # ──────────────────────────────────────────────────────────

    def collect_step(carry, unused_t):
        """One environment step.  Returns transition; buffer insert is outside."""
        env_state, actor_params, key = carry
        key, action_key = jax.random.split(key)

        obs   = env_state.obs
        state = obs[:, :args.obs_dim]
        goal  = obs[:, args.obs_dim:]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = actor.apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(action_key, shape=means.shape)
        action = nn.tanh(x_ts)

        # Safety-cost scalars at s_t (CURRENT state) — read BEFORE stepping.
        # env_state.info["cost"] was populated by reset() (at t=0) or by the
        # previous step() (for t>0), so it always refers to s_t.  This matches
        # the Bellman target y = c(s_t) + γ_c · Q_c(s_{t+1}, a'), which is the
        # convention the existing cost_critic_loss_fn uses.
        _prev_info = env_state.info
        _zeros = jnp.zeros(args.num_envs)
        _cost  = _prev_info["cost"]           if "cost"           in _prev_info else _zeros
        _dwall = _prev_info["d_wall"]         if "d_wall"         in _prev_info else _zeros
        _hviol = _prev_info["hard_violation"] if "hard_violation" in _prev_info else _zeros
        # Phase-1g velocity-quadratic diagnostics. Same positional convention
        # as `cost` / `d_wall` / `hard_violation` — read from _prev_info so
        # the emitted scalar is paired with c(s_t). For non-velocity cost
        # types the env still emits these as constants (v_xy_norm = ‖v_xy‖,
        # vel_cost_mult = 0), so the schema is uniform.
        _vnorm = _prev_info["v_xy_norm"]      if "v_xy_norm"      in _prev_info else _zeros
        _vmult = _prev_info["vel_cost_mult"]  if "vel_cost_mult"  in _prev_info else _zeros

        next_env_state = env.step(env_state, action)
        _info = next_env_state.info
        # Our maze envs store `seed` directly in info; brax's EpisodeWrapper
        # adds `truncation`.  There is no "state_extras" sub-dict — reading it
        # silently fell back to zeros and broke goal relabeling (all trajectories
        # looked like one episode).  Access the keys directly.
        _seed  = _info["seed"]       if "seed"       in _info else jnp.zeros(args.num_envs)
        _trunc = _info["truncation"] if "truncation" in _info else jnp.zeros(args.num_envs)

        transition = Transition(
            observation=obs,
            action=action,
            reward=next_env_state.reward,
            discount=1.0 - next_env_state.done,
            extras={
                "state_extras": {
                    # seed tracks episode ID; incremented by brax's EpisodeWrapper
                    "seed":       _seed,
                    "truncation": _trunc,
                },
                "state":      obs[:, :args.obs_dim],                  # s_t
                "next_state": next_env_state.obs[:, :args.obs_dim],   # s_{t+1} — needed
                                                                      # by cost_critic_loss_fn
                                                                      # Bellman backup
                # Safety-cost scalars evaluated at s_{t+1} (i.e., the state
                # reached by taking `action` in s_t).  cost_critic_loss_fn
                # treats this as c(s_t, a_t, s_{t+1}) — the reward/cost
                # returned by the env for the transition.
                "cost":            _cost,
                "d_wall":          _dwall,
                "hard_violation":  _hviol,
                # Phase-1g velocity-quadratic diagnostics (env-level scalars,
                # NOT used in the Bellman target — just for logging).
                "v_xy_norm":       _vnorm,
                "vel_cost_mult":   _vmult,
            },
        )
        return (next_env_state, actor_params, key), transition

    # ──────────────────────────────────────────────────────────
    #  Training step
    # ──────────────────────────────────────────────────────────

    def training_step(carry, unused_t):
        training_state, env_state, buffer_state, key = carry
        key, collect_key, sample_key, perm_key = jax.random.split(key, 4)

        # 1. Collect unroll_length steps → (unroll_length, num_envs, ...)
        (env_state, _, _), transitions = jax.lax.scan(
            collect_step,
            (env_state, training_state.actor_state.params, collect_key),
            (),
            length=args.unroll_length,
        )

        # 2. Bulk insert into buffer (insert_internal is pure JAX, safe in scan)
        buffer_state = replay_buffer.insert_internal(buffer_state, transitions)

        training_state = training_state.replace(
            env_steps=training_state.env_steps
            + args.unroll_length * args.num_envs * args.action_repeat,
        )

        # 3. Sample from buffer → (num_envs, episode_length, ...)
        buffer_state, sampled = replay_buffer.sample(buffer_state)

        # 4. Apply flatten_crl_fn per trajectory (vmap over num_envs)
        #    Signature: flatten_crl_fn(buffer_config_tuple, transition, sample_key)
        buffer_config = (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx)
        flat_fn = functools.partial(
            TrajectoryUniformSamplingQueue.flatten_crl_fn, buffer_config)
        flat_keys = jax.random.split(sample_key, args.num_envs)
        sampled = jax.vmap(flat_fn)(sampled, flat_keys)
        # → (num_envs, episode_length-1, ...)

        # 5. Flatten to (N, ...), shuffle, take first batch_total, reshape
        batch_total = args.num_update_epochs * args.num_minibatches * args.batch_size
        sampled = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), sampled)
        n = sampled.observation.shape[0]
        perm = jax.random.permutation(perm_key, n)
        sampled = jax.tree_util.tree_map(lambda x: x[perm[:batch_total]], sampled)
        # → (batch_total, ...)

        # 6. Reshape into (num_update_epochs * num_minibatches, batch_size, ...)
        sampled = jax.tree_util.tree_map(
            lambda x: x.reshape(
                args.num_update_epochs * args.num_minibatches,
                args.batch_size,
                *x.shape[1:],
            ),
            sampled,
        )

        # 7. Run gradient updates
        (training_state, _), sgd_metrics = jax.lax.scan(
            sgd_step,
            (training_state, sample_key),
            sampled,
        )

        return (training_state, env_state, buffer_state, key), sgd_metrics

    # ──────────────────────────────────────────────────────────
    #  Training epoch
    # ──────────────────────────────────────────────────────────

    def training_epoch(carry, unused_t):
        steps_per_epoch = max(1,
            args.total_env_steps // (
                args.num_epochs * args.unroll_length
                * args.num_envs * args.action_repeat
            )
        )
        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            training_step,
            carry,
            (),
            length=steps_per_epoch,
        )
        return (training_state, env_state, buffer_state, key), metrics

    # ──────────────────────────────────────────────────────────
    #  Initial env reset and buffer prefill
    # ──────────────────────────────────────────────────────────

    rng, reset_key = jax.random.split(rng)
    env_keys = jax.random.split(reset_key, args.num_envs)
    env_state = env.reset(env_keys)

    # Prefill buffer with random actions (outside JIT for simplicity)
    prefill_steps = args.min_replay_size // args.num_envs + 1
    rng, prefill_key = jax.random.split(rng)

    @jax.jit
    def prefill_one(carry, unused):
        env_state, buffer_state, key = carry
        key, ak = jax.random.split(key)
        action = jax.random.uniform(ak, (args.num_envs, action_size), minval=-1.0, maxval=1.0)
        next_env_state = env.step(env_state, action)
        _info = next_env_state.info
        _seed  = _info["seed"]       if "seed"       in _info else jnp.zeros(args.num_envs)
        _trunc = _info["truncation"] if "truncation" in _info else jnp.zeros(args.num_envs)
        # Phase-0 safety-cost scalars — must be present here so the prefill
        # transitions share the same pytree structure as collect_step's (the
        # replay buffer's JAX arrays are keyed by tree structure, so mismatch
        # would crash at first sample).
        _zeros = jnp.zeros(args.num_envs)
        _cost  = _info["cost"]           if "cost"           in _info else _zeros
        _dwall = _info["d_wall"]         if "d_wall"         in _info else _zeros
        _hviol = _info["hard_violation"] if "hard_violation" in _info else _zeros
        # Phase-1g velocity-quadratic diagnostics — populate here too so the
        # buffer's tree structure matches collect_step's. Velocity at prefill
        # time comes from the random-action env step (post-step info).
        _vnorm = _info["v_xy_norm"]      if "v_xy_norm"      in _info else _zeros
        _vmult = _info["vel_cost_mult"]  if "vel_cost_mult"  in _info else _zeros
        t = Transition(
            observation=env_state.obs,
            action=action,
            reward=next_env_state.reward,
            discount=1.0 - next_env_state.done,
            extras={
                "state_extras": {
                    "seed":       _seed,
                    "truncation": _trunc,
                },
                "state":      env_state.obs[:, :args.obs_dim],       # s_t
                "next_state": next_env_state.obs[:, :args.obs_dim],  # s_{t+1}
                "cost":            _cost,
                "d_wall":          _dwall,
                "hard_violation":  _hviol,
                "v_xy_norm":       _vnorm,
                "vel_cost_mult":   _vmult,
            },
        )
        t1 = jax.tree_util.tree_map(lambda x: x[None], t)  # add unroll dim=1
        buffer_state = replay_buffer.insert_internal(buffer_state, t1)
        return (next_env_state, buffer_state, key), ()

    (env_state, buffer_state, _), _ = jax.lax.scan(
        prefill_one, (env_state, buffer_state, prefill_key), (), length=prefill_steps)
    print(f"Buffer prefilled with ~{prefill_steps * args.num_envs} transitions.")

    # Phase-1f NaN-forensics: post-prefill probe. Runs ONCE, before any
    # training step. Distinguishes:
    #   buffer NaN here          → env emit under random uniform action is the
    #                              source. Fix is in env (MJX integrator) or
    #                              prefill action sampler.
    #   buffer clean, env NaN    → final prefill env_state was corrupted but
    #                              buffer rows so far are OK; first post-prefill
    #                              rollout will read NaN obs.
    #   both clean               → NaN enters in first training step (gradient
    #                              poisons actor params → next rollout emits
    #                              NaN action → next buffer row contaminates).
    _buf_data = buffer_state.data
    _buf_step_nan = jnp.any(jnp.isnan(_buf_data), axis=tuple(range(1, _buf_data.ndim)))
    _buf_any_nan  = bool(jnp.any(_buf_step_nan))
    _buf_first_nan = int(jnp.argmax(_buf_step_nan.astype(jnp.int32))) if _buf_any_nan else -1
    _env_obs_nan = bool(jnp.any(jnp.isnan(env_state.obs)))
    print(f"[prefill probe] buffer.data shape={tuple(_buf_data.shape)}")
    print(f"[prefill probe] buffer NaN anywhere: {_buf_any_nan}"
          + (f"  first NaN row: {_buf_first_nan} / {prefill_steps} written"
             if _buf_any_nan else ""))
    print(f"[prefill probe] env_state.obs NaN: {_env_obs_nan}  "
          f"obs[:, 0:3] = {env_state.obs[:1, 0:3]}")

    print(f"Starting {args.num_epochs} epochs.  Constraints: {args.use_constraints}")
    if args.use_constraints:
        if args.cost_type == "quadratic":
            print(f"  cost_type=quadratic  budget={args.cost_budget_d}  "
                  f"ε_train={args.cost_epsilon}  ε_hard={args.cost_epsilon_hard}  "
                  f"PID=({args.pid_kp},{args.pid_ki},{args.pid_kd})")
        else:
            print(f"  cost_type=sigmoid  budget={args.cost_budget_d}  "
                  f"ε={args.cost_epsilon}  τ={args.cost_tau}  "
                  f"PID=({args.pid_kp},{args.pid_ki},{args.pid_kd})")

    # ──────────────────────────────────────────────────────────
    #  Main training loop
    # ──────────────────────────────────────────────────────────

    jit_epoch = jax.jit(training_epoch)

    for epoch in range(args.num_epochs):
        t0 = time.time()

        (training_state, env_state, buffer_state, rng), epoch_metrics = jit_epoch(
            (training_state, env_state, buffer_state, rng), ()
        )

        # Phase-1f epoch-1 time-series forensics (one-shot, host-side).
        # epoch_metrics[k] has shape (steps_per_epoch, num_update_epochs *
        # num_minibatches). Per-epoch jnp.max collapsed both axes and only
        # told us "everything was NaN by end of epoch 1". This dump finds
        # the FIRST sgd-step where each NaN flag fired, plus the first few
        # grad-norm values, so we can localize the origin within epoch 1.
        #
        # Decision tree on the output:
        #   c_grad_nan first at flat_idx = 0   →  critic-loss autograd produces
        #                                          NaN from clean init params on
        #                                          a clean prefill batch. The
        #                                          loss has an analytical
        #                                          singularity (most likely
        #                                          cosine logits / τ overflow
        #                                          → softmax saturation → log 0).
        #   c_grad_nan first at flat_idx > 0  →  the optimizer apply at the
        #                                          previous step produced NaN
        #                                          params; check c_params_nan
        #                                          at flat_idx − 1.
        #   c_params_nan first at flat_idx with c_grad_nan first at flat_idx + 1
        #                                       →  Adam's update poisoned params
        #                                          even from a finite gradient.
        #   nan_obs_critic first at flat_idx ≫ first-grad-NaN
        #                                       →  buffer was clean while early
        #                                          grad-NaN built up; later
        #                                          rollout under NaN actor
        #                                          contaminated the buffer.
        if epoch == 0:
            sgd_per_step = args.num_update_epochs * args.num_minibatches
            def _first_one_idx(arr):
                """Return flat index of first 1.0 (or NaN treated as 1) entry,
                or -1 if the array is all-zero. Works for the per-step boolean
                flags we inserted in metrics."""
                a = jnp.asarray(arr).reshape(-1)
                # Flag is 1.0 / 0.0 float; treat any positive entry as a hit.
                hits = a > 0.0
                if not bool(jnp.any(hits)):
                    return -1
                return int(jnp.argmax(hits.astype(jnp.int32)))

            def _decode(flat_idx):
                if flat_idx < 0:
                    return "(never)"
                it, sg = divmod(flat_idx, sgd_per_step)
                return f"flat={flat_idx}  iter={it}  sgd={sg}"

            print("[epoch1 forensics] sgd_per_step=", sgd_per_step,
                  "steps_per_epoch=", epoch_metrics["c_grad_nan"].shape[0])
            for label, key in [
                ("c_grad_nan",       "c_grad_nan"),
                ("c_params_nan",     "c_params_nan"),
                ("a_grad_nan",       "a_grad_nan"),
                ("a_params_nan",     "a_params_nan"),
                ("cc_grad_nan",      "cc_grad_nan"),
                ("cc_params_nan",    "cc_params_nan"),
                ("nan_obs_critic",   "nan_obs_critic"),
                ("nan_sa_critic",    "nan_sa_critic"),
                ("nan_g_critic",     "nan_g_critic"),
                ("nan_logits_critic","nan_logits_critic"),
                ("nan_sa_actor",     "nan_sa_actor"),
                ("nan_action_actor", "nan_action_actor"),
                ("nan_f_actor",      "nan_f_actor"),
            ]:
                idx = _first_one_idx(epoch_metrics[key])
                print(f"  first {label:<20s}: {_decode(idx)}")

            # First 5 grad-norms and first 5 params-NaN flags. If
            # c_grad_norm[0] is finite, the very first backward on clean
            # init+clean batch is finite — so the cascade starts at the
            # optimizer apply, not in the loss autograd.
            for label, key in [
                ("c_grad_norm",   "c_grad_norm"),
                ("a_grad_norm",   "a_grad_norm"),
                ("cc_grad_norm",  "cc_grad_norm"),
            ]:
                v = jnp.asarray(epoch_metrics[key]).reshape(-1)
                head = [float(x) for x in v[:5]]
                print(f"  {label}[:5] = {head}")
            for label, key in [
                ("c_params_nan",  "c_params_nan"),
                ("a_params_nan",  "a_params_nan"),
                ("cc_params_nan", "cc_params_nan"),
            ]:
                v = jnp.asarray(epoch_metrics[key]).reshape(-1)
                head = [float(x) for x in v[:5]]
                print(f"  {label}[:5] = {head}")

        dur = time.time() - t0
        env_steps  = int(training_state.env_steps)
        grad_steps = int(training_state.gradient_steps)

        # Evaluation every eval_every epochs
        eval_metrics = {}
        if (epoch + 1) % args.eval_every == 0:
            eval_metrics = evaluator.run_evaluation(training_state, {})

        log_dict = {
            "epoch": epoch,
            "env_steps": env_steps,
            "gradient_steps": grad_steps,
            "epoch_time_s": dur,
            # Critic
            "critic_loss": float(jnp.mean(epoch_metrics["critic_loss"])),
            "accuracy":    float(jnp.mean(epoch_metrics["accuracy"])),
            "logits_pos":  float(jnp.mean(epoch_metrics["logits_pos"])),
            "logits_neg":  float(jnp.mean(epoch_metrics["logits_neg"])),
            "logsumexp":   float(jnp.mean(epoch_metrics["logsumexp"])),
            # Actor / entropy
            "actor_loss":  float(jnp.mean(epoch_metrics["actor_loss"])),
            "alpha":       float(jnp.mean(epoch_metrics["alpha"])),
            "alpha_loss":  float(jnp.mean(epoch_metrics["alpha_loss"])),
            # Phase-1f representation-scaling diagnostics
            #   _critic: ‖φ‖, ‖ψ‖ (pre-normalize) seen in the InfoNCE pass
            #   _actor : ‖φ‖, ‖ψ‖ (pre-normalize) seen by the actor pass
            # All four should stay bounded and roughly stationary. Monotone
            # growth here is the direct signature of the Phase-1d failure
            # mode we are guarding against.
            "sa_repr_norm_critic": float(jnp.mean(epoch_metrics["sa_repr_norm_critic"])),
            "g_repr_norm_critic":  float(jnp.mean(epoch_metrics["g_repr_norm_critic"])),
            "sa_repr_norm_actor":  float(jnp.mean(epoch_metrics["sa_repr_norm_actor"])),
            "g_repr_norm_actor":   float(jnp.mean(epoch_metrics["g_repr_norm_actor"])),
            # Phase-1f NaN-forensics (max over grad-steps so a single NaN event
            # flips the flag; min over norms so zero-norm shows up even if mean
            # is finite). Values live in [0,1] for flags, ≥0 for mins/max.
            "nan_obs_critic":     float(jnp.max(epoch_metrics["nan_obs_critic"])),
            "nan_sa_critic":      float(jnp.max(epoch_metrics["nan_sa_critic"])),
            "nan_g_critic":       float(jnp.max(epoch_metrics["nan_g_critic"])),
            "nan_logits_critic":  float(jnp.max(epoch_metrics["nan_logits_critic"])),
            "sa_norm_min_critic": float(jnp.min(epoch_metrics["sa_norm_min_critic"])),
            "g_norm_min_critic":  float(jnp.min(epoch_metrics["g_norm_min_critic"])),
            "nan_sa_actor":       float(jnp.max(epoch_metrics["nan_sa_actor"])),
            "nan_g_actor":        float(jnp.max(epoch_metrics["nan_g_actor"])),
            "nan_action_actor":   float(jnp.max(epoch_metrics["nan_action_actor"])),
            "nan_f_actor":        float(jnp.max(epoch_metrics["nan_f_actor"])),
            "sa_norm_min_actor":  float(jnp.min(epoch_metrics["sa_norm_min_actor"])),
            "g_norm_min_actor":   float(jnp.min(epoch_metrics["g_norm_min_actor"])),
            "action_max_actor":   float(jnp.max(epoch_metrics["action_max_actor"])),
            # Phase-1f grad/param NaN-forensics — max over grad-steps so a
            # single NaN flips the flag; max over grad norms so runaway
            # spikes show up even if mean is finite.
            "c_grad_nan":     float(jnp.max(epoch_metrics["c_grad_nan"])),
            "c_grad_norm":    float(jnp.max(epoch_metrics["c_grad_norm"])),
            "c_params_nan":   float(jnp.max(epoch_metrics["c_params_nan"])),
            "a_grad_nan":     float(jnp.max(epoch_metrics["a_grad_nan"])),
            "a_grad_norm":    float(jnp.max(epoch_metrics["a_grad_norm"])),
            "a_params_nan":   float(jnp.max(epoch_metrics["a_params_nan"])),
            "cc_grad_nan":    float(jnp.max(epoch_metrics["cc_grad_nan"])),
            "cc_grad_norm":   float(jnp.max(epoch_metrics["cc_grad_norm"])),
            "cc_params_nan":  float(jnp.max(epoch_metrics["cc_params_nan"])),
            # Phase-1g actor-loss component forensics — mean across all sgd-
            # steps in the epoch. None of these are flag-like, so use mean
            # not max. abs-max useful for log_prob_mean/alpha_logprob since
            # they can be large negative or positive.
            "alpha_actor":          float(jnp.mean(epoch_metrics["alpha_actor"])),
            "log_prob_mean_actor":  float(jnp.mean(epoch_metrics["log_prob_mean_actor"])),
            "alpha_logprob_actor":  float(jnp.mean(epoch_metrics["alpha_logprob_actor"])),
            "gaussian_logp_actor":  float(jnp.mean(epoch_metrics["gaussian_logp_actor"])),
            "sat_correction_actor": float(jnp.mean(epoch_metrics["sat_correction_actor"])),
            "log_std_mean_actor":   float(jnp.mean(epoch_metrics["log_std_mean_actor"])),
            "f_term_mean_actor":    float(jnp.mean(epoch_metrics["f_term_mean_actor"])),
            # Phase-1g α-cap binding rate. mean over sgd-steps gives fraction
            # of steps where the cap was active. 0.0 = auto-α naturally below
            # cap (α_max chosen too loose); 1.0 = always clipped (auto-α wants
            # higher α than the cap allows — expected once α_max < equilibrium).
            "alpha_clip_active":    float(jnp.mean(epoch_metrics["alpha_clip_active"])),
            **eval_metrics,
        }

        if args.use_constraints:
            _mean_cost = float(jnp.mean(epoch_metrics["mean_step_cost"]))
            _j_c_hat = float(jnp.mean(epoch_metrics["j_c_hat"]))
            log_dict.update({
                "cost_critic_loss":     float(jnp.mean(epoch_metrics["cost_critic_loss"])),
                "mean_step_cost":       _mean_cost,
                "mean_d_wall":          float(jnp.mean(epoch_metrics["mean_d_wall"])),
                "hard_violation_rate":  float(jnp.mean(epoch_metrics["hard_violation_rate"])),
                "mean_qc":              float(jnp.mean(epoch_metrics["mean_qc"])),
                "mean_qc_pi":           float(jnp.mean(epoch_metrics["mean_qc_pi"])),
                "lambda_tilde":         float(jnp.mean(epoch_metrics["lambda_tilde"])),
                "j_c_hat":              _j_c_hat,
                # Constraint violation from critic-based estimator
                "constraint_violation": max(0.0, _j_c_hat - args.cost_budget_d),
                # Phase-1g velocity-quadratic diagnostics. mean_v_xy_norm
                # is the typical torso speed; mean_vel_cost_mult is the
                # velocity multiplier on the geometric cost shell (= 0
                # for non-velocity cost types). Together they decompose
                # the velocity-coupled cost into its motion and geometric
                # components.
                "mean_v_xy_norm":       float(jnp.mean(epoch_metrics["mean_v_xy_norm"])),
                "mean_vel_cost_mult":   float(jnp.mean(epoch_metrics["mean_vel_cost_mult"])),
            })

        print(
            f"[{epoch+1:3d}/{args.num_epochs}] steps={env_steps:,} | "
            f"c_loss={log_dict['critic_loss']:.4f} acc={log_dict['accuracy']:.3f} | "
            f"a_loss={log_dict['actor_loss']:.4f} | "
            f"t={dur:.1f}s"
        )
        if args.use_constraints:
            print(
                f"         hard_viol={log_dict['hard_violation_rate']:.4f} "
                f"cost={log_dict['mean_step_cost']:.4f} "
                f"λ̃={log_dict['lambda_tilde']:.4f} "
                f"Ĵ_c={log_dict['j_c_hat']:.4f} "
                f"Qc={log_dict['mean_qc']:.4f} "
                # Phase-1g velocity-quadratic diagnostics. v_xy_norm tells
                # us the typical torso speed; vel_mult is the velocity
                # multiplier (= 0 for non-velocity cost types).
                f"|v_xy|={log_dict['mean_v_xy_norm']:.3f} "
                f"vel_mult={log_dict['mean_vel_cost_mult']:.3f}"
            )
        # Phase-1f NaN-forensics one-liner. Always printed so the probe values
        # appear next to c_loss/a_loss. Flags are 0/1 aggregated across the
        # epoch's grad-steps (max); norms are per-epoch min; action_max is max.
        print(
            f"         nan[obs_c={log_dict['nan_obs_critic']:.0f} "
            f"sa_c={log_dict['nan_sa_critic']:.0f} "
            f"g_c={log_dict['nan_g_critic']:.0f} "
            f"lg_c={log_dict['nan_logits_critic']:.0f} "
            f"sa_a={log_dict['nan_sa_actor']:.0f} "
            f"g_a={log_dict['nan_g_actor']:.0f} "
            f"act_a={log_dict['nan_action_actor']:.0f} "
            f"f_a={log_dict['nan_f_actor']:.0f}] "
            f"‖φ‖min_c={log_dict['sa_norm_min_critic']:.3f} "
            f"‖ψ‖min_c={log_dict['g_norm_min_critic']:.3f} "
            f"‖φ‖min_a={log_dict['sa_norm_min_actor']:.3f} "
            f"‖ψ‖min_a={log_dict['g_norm_min_actor']:.3f} "
            f"|a|max={log_dict['action_max_actor']:.3f}"
        )
        # Phase-1f grad/param NaN one-liner. Reading order at first epoch:
        #   if all *_grad_nan = 0 and *_params_nan = 0 but forward NaN flags
        #   were set, the corruption came in via the buffer (next rollout
        #   reads NaN params or NaN obs).  If c_grad_nan = 1 first, NaN
        #   originates in the autograd path of the critic loss.  Norms in
        #   scientific notation so blowups remain readable.
        print(
            f"         grad[c={log_dict['c_grad_nan']:.0f}/"
            f"{log_dict['c_grad_norm']:.2e} "
            f"a={log_dict['a_grad_nan']:.0f}/"
            f"{log_dict['a_grad_norm']:.2e} "
            f"cc={log_dict['cc_grad_nan']:.0f}/"
            f"{log_dict['cc_grad_norm']:.2e}] "
            f"params[c={log_dict['c_params_nan']:.0f} "
            f"a={log_dict['a_params_nan']:.0f} "
            f"cc={log_dict['cc_params_nan']:.0f}]"
        )
        # Phase-1g actor-loss component one-liner. Reading order:
        #   a_loss should equal α·log_p − f_term within rounding (no λ̃ at d=0.15)
        #     i.e., log_dict['actor_loss'] ≈ alpha_logprob_actor − f_term_mean_actor.
        #   If α grows monotonically and α·log_p stays large negative → SAC
        #     entropy temperature is unbounded; fix is to clip α or constrain it.
        #   If sat_corr_a is large positive (→ +110 = 8 dim × +13.8 at saturation),
        #     and Gaussian_lp_a is small, the tanh-saturation Jacobian dominates
        #     log_prob; fix is numerically-stable softplus form or action smoothing.
        #   If log_std_a → −5 (lower clip, σ ≈ 0.0067), policy has collapsed to
        #     a near-deterministic peak; entropy collapse is the source.
        #   If f_term_a |·| > 10 despite row-L2, the row-L2 contract is broken
        #     somewhere. Should never happen.
        print(
            f"         actor[α={log_dict['alpha_actor']:.4g} "
            f"log_p={log_dict['log_prob_mean_actor']:.3g} "
            f"α·log_p={log_dict['alpha_logprob_actor']:.3g} "
            f"gauss_lp={log_dict['gaussian_logp_actor']:.3g} "
            f"sat_corr={log_dict['sat_correction_actor']:.3g} "
            f"log_std={log_dict['log_std_mean_actor']:.3g} "
            f"f_term={log_dict['f_term_mean_actor']:.3g} "
            f"α_clip={log_dict['alpha_clip_active']:.2f}]"
        )

        if args.track:
            wandb.log(log_dict, step=env_steps)

    # ── Checkpoint ─────────────────────────────────────────────
    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "actor_params":  training_state.actor_state.params,
        "critic_params": training_state.critic_state.params,
        "alpha_params":  training_state.alpha_state.params,
    }
    if args.use_constraints:
        ckpt.update({
            "cost_critic_params":        training_state.cost_critic_state.params,
            "cost_critic_target_params": training_state.cost_critic_target_params,
            "lambda_tilde":              training_state.lambda_tilde,
            "pid_e_prev":                training_state.pid_e_prev,
            "pid_sum_e":                 training_state.pid_sum_e,
        })
    with open(f"{ckpt_dir}/params.pkl", "wb") as f:
        pickle.dump(ckpt, f)
    print(f"Checkpoint saved → {ckpt_dir}/params.pkl")

    if args.save_buffer:
        with open(f"{ckpt_dir}/buffer.pkl", "wb") as f:
            pickle.dump(buffer_state, f)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    # FlagConversionOff: let booleans accept `--use_constraints True/False`
    # instead of tyro's default `--use-constraints`/`--no-use-constraints`.
    # This matches the syntax used in every SLURM script in this repo.
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))
    main(args)
