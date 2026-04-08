"""
train.py — Constrained Contrastive RL (C-CRL)

Drop-in replacement for scaling-crl/train.py that adds:
  • CostCritic  Q_c(s,a,g)  trained with Bellman backups
  • Hybrid step cost  c = α·1{collision} + (1-α)·exp(-d/σ)
  • Lagrangian actor loss  L = E[α_sac·log π - Q + λ·Q_c]
  • Dual ascent  λ ← clip(λ + lr_λ·(Ĵ_c - d), 0, λ_max)
  • Full wandb logging of cost metrics

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
from cost_utils import get_wall_centers, hybrid_cost


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
    normalize_observations: bool = False
    max_replay_size: int = 10_000
    min_replay_size: int = 8_192
    deterministic_eval: bool = True
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
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

    # ── Constrained CRL additions ──────────────────────────────
    use_constraints: bool = True
    # Hybrid cost: c = α·1{collision} + (1-α)·exp(-d/σ)
    alpha_cost: float = 0.5
    sigma_wall: float = 1.0
    contact_threshold: float = 0.1
    # CMDP budget & Lagrange multiplier
    cost_budget_d: float = 0.1
    lambda_init: float = 0.0
    lambda_lr: float = 1e-3
    lambda_max: float = 100.0
    # Cost critic architecture
    cost_critic_lr: float = 3e-4
    cost_critic_width: int = 256
    cost_critic_depth: int = 4
    cost_critic_skip_connections: int = 0
    cost_critic_tau: float = 0.005   # Polyak
    cost_discount: float = 0.99      # γ_c for Bellman backup
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
    # Constraint additions
    cost_critic_state: TrainState
    cost_critic_target_params: dict
    log_lambda: jnp.ndarray


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
    raw_env = envs.get_environment(args.env_id)
    env = envs.training.wrap(
        raw_env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
    )
    eval_env_id = args.eval_env_id if args.eval_env_id else args.env_id
    eval_env = envs.get_environment(eval_env_id)

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

    log_lambda = jnp.log(jnp.maximum(args.lambda_init, 1e-8))

    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        cost_critic_state=cost_critic_state,
        cost_critic_target_params=cost_critic_target_params,
        log_lambda=log_lambda,
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
            "state": jnp.zeros(args.obs_dim),
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

    # Eval-time cost function — closed over wall geometry and cost params.
    # Returns scalar cost for each xy position: (xy: [...,2]) -> [...].
    if args.use_constraints:
        def eval_cost_fn(xy: jnp.ndarray) -> jnp.ndarray:
            cost, _, _ = hybrid_cost(
                xy, wall_centers, half_wall_size,
                args.alpha_cost, args.sigma_wall, args.contact_threshold,
            )
            return cost
    else:
        eval_cost_fn = None

    evaluator = CrlEvaluator(
        actor_step=actor_step,
        eval_env=eval_env,
        num_eval_envs=args.eval_envs,
        episode_length=args.episode_length,
        key=eval_key,
        cost_fn=eval_cost_fn,
        cost_budget_d=args.cost_budget_d,
        obs_dim=args.obs_dim,
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

        logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
        logsumexp_val = jax.nn.logsumexp(logits, axis=1)
        loss = -jnp.mean(jnp.diag(logits) - logsumexp_val)
        loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp_val ** 2)

        I = jnp.eye(logits.shape[0])
        correct  = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I)       / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        return loss, (logits_pos, logits_neg, jnp.mean(correct), jnp.mean(logsumexp_val))

    def actor_loss_fn(actor_params, critic_params, log_alpha,
                      cost_critic_params, log_lam, transitions, key):
        """Lagrangian actor loss: E[α_sac·log π  -  Q  +  λ·Q_c]."""
        obs          = transitions.observation
        future_state = transitions.extras["future_state"]
        state        = obs[:, :args.obs_dim]
        goal         = future_state[:, args.goal_start_idx:args.goal_end_idx]
        observation  = jnp.concatenate([state, goal], axis=1)

        means, log_stds = actor.apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)

        sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, action)
        g_repr  = g_encoder.apply(critic_params["g_encoder"],  goal)
        qf_pi   = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

        alpha      = jnp.exp(log_alpha)
        actor_loss = jnp.mean(alpha * log_prob - qf_pi)

        if args.use_constraints:
            lam    = jnp.exp(log_lam)
            qc_pi  = cost_critic.apply(cost_critic_params, state, action, goal)
            actor_loss = actor_loss + jnp.mean(lam * qc_pi)
        else:
            qc_pi = jnp.zeros_like(qf_pi)

        return actor_loss, (log_prob, jnp.mean(qc_pi))

    def alpha_loss_fn(log_alpha, log_prob):
        return -jnp.mean(log_alpha * (log_prob + action_size))

    def cost_critic_loss_fn(cost_critic_params, cost_critic_target_params,
                            actor_params, transitions, key):
        """TD(0) Bellman loss for Q_c.

        FIX vs. old version: actor_params is now an explicit argument so the
        loss always uses the *current* (just-updated) actor, not stale closure.
        """
        obs          = transitions.observation
        action       = transitions.action
        future_state = transitions.extras["future_state"]
        state        = obs[:, :args.obs_dim]
        next_state   = transitions.extras["state"]   # raw s stored at collection
        goal         = future_state[:, args.goal_start_idx:args.goal_end_idx]

        # Step cost from maze geometry (computed on current state)
        agent_xy = state[:, :2]
        cost, d_wall, collision = hybrid_cost(
            agent_xy, wall_centers, half_wall_size,
            alpha_cost=args.alpha_cost,
            sigma_wall=args.sigma_wall,
            contact_threshold=args.contact_threshold,
        )

        # Next-state action from current policy
        next_obs = jnp.concatenate([next_state, goal], axis=1)
        means_next, log_stds_next = actor.apply(actor_params, next_obs)
        key, subkey = jax.random.split(key)
        x_next = means_next + jnp.exp(log_stds_next) * jax.random.normal(
            subkey, shape=means_next.shape)
        action_next = nn.tanh(x_next)

        qc_target = cost_critic.apply(cost_critic_target_params, next_state, action_next, goal)
        td_target  = jax.lax.stop_gradient(cost + args.cost_discount * qc_target)

        qc_online = cost_critic.apply(cost_critic_params, state, action, goal)
        loss = jnp.mean((qc_online - td_target) ** 2)

        return loss, {
            "cost_critic_loss": loss,
            "mean_step_cost":   jnp.mean(cost),
            "mean_d_wall":      jnp.mean(d_wall),
            "collision_rate":   jnp.mean(collision),
            "mean_qc":          jnp.mean(qc_online),
        }

    # ──────────────────────────────────────────────────────────
    #  SGD step — NO @jax.jit; it lives inside jax.lax.scan
    # ──────────────────────────────────────────────────────────

    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key, cost_key = jax.random.split(key, 4)

        # Critic
        (c_loss, (lp, ln, acc, lse)), c_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True)(
            training_state.critic_state.params, transitions, critic_key)
        critic_state = training_state.critic_state.apply_gradients(grads=c_grads)

        # Actor (Lagrangian)
        (a_loss, (log_prob, mean_qc_pi)), a_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True)(
            training_state.actor_state.params,
            critic_state.params,
            training_state.alpha_state.params,
            training_state.cost_critic_state.params,
            training_state.log_lambda,
            transitions,
            actor_key,
        )
        actor_state = training_state.actor_state.apply_gradients(grads=a_grads)

        # Alpha
        al_loss, al_grads = jax.value_and_grad(alpha_loss_fn)(
            training_state.alpha_state.params, log_prob)
        alpha_state = training_state.alpha_state.apply_gradients(grads=al_grads)

        # Cost critic — FIX: pass actor_state.params (updated above)
        if args.use_constraints:
            (cc_loss, cc_m), cc_grads = jax.value_and_grad(
                cost_critic_loss_fn, has_aux=True)(
                training_state.cost_critic_state.params,
                training_state.cost_critic_target_params,
                actor_state.params,           # ← updated actor params
                transitions,
                cost_key,
            )
            cost_critic_state_new = training_state.cost_critic_state.apply_gradients(
                grads=cc_grads)

            tau = args.cost_critic_tau
            target_params_new = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1.0 - tau) * tp,
                cost_critic_state_new.params,
                training_state.cost_critic_target_params,
            )

            lam = jnp.exp(training_state.log_lambda)
            lam_new = jnp.clip(
                lam + args.lambda_lr * (cc_m["mean_step_cost"] - args.cost_budget_d),
                1e-8, args.lambda_max,
            )
            log_lambda_new = jnp.log(lam_new)
        else:
            cost_critic_state_new = training_state.cost_critic_state
            target_params_new     = training_state.cost_critic_target_params
            log_lambda_new        = training_state.log_lambda
            cc_loss               = jnp.zeros(())
            cc_m = {k: jnp.zeros(()) for k in
                    ["cost_critic_loss", "mean_step_cost", "mean_d_wall",
                     "collision_rate", "mean_qc"]}

        new_ts = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
            cost_critic_state=cost_critic_state_new,
            cost_critic_target_params=target_params_new,
            log_lambda=log_lambda_new,
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
            "cost_critic_loss": cc_m["cost_critic_loss"],
            "mean_step_cost":   cc_m["mean_step_cost"],
            "mean_d_wall":      cc_m["mean_d_wall"],
            "collision_rate":   cc_m["collision_rate"],
            "mean_qc":          cc_m["mean_qc"],
            "lambda":           jnp.exp(log_lambda_new),
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

        next_env_state = env.step(env_state, action)
        _info = next_env_state.info
        _seed = (_info["state_extras"]["seed"]
                 if "state_extras" in _info else jnp.zeros(args.num_envs))
        _trunc = (_info["state_extras"]["truncation"]
                  if "state_extras" in _info else jnp.zeros(args.num_envs))

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
                "state": obs[:, :args.obs_dim],  # raw state for cost critic next-step
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
        _seed = (_info["state_extras"]["seed"]
                 if "state_extras" in _info else jnp.zeros(args.num_envs))
        _trunc = (_info["state_extras"]["truncation"]
                  if "state_extras" in _info else jnp.zeros(args.num_envs))
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
                "state": env_state.obs[:, :args.obs_dim],
            },
        )
        t1 = jax.tree_util.tree_map(lambda x: x[None], t)  # add unroll dim=1
        buffer_state = replay_buffer.insert_internal(buffer_state, t1)
        return (next_env_state, buffer_state, key), ()

    (env_state, buffer_state, _), _ = jax.lax.scan(
        prefill_one, (env_state, buffer_state, prefill_key), (), length=prefill_steps)
    print(f"Buffer prefilled with ~{prefill_steps * args.num_envs} transitions.")
    print(f"Starting {args.num_epochs} epochs.  Constraints: {args.use_constraints}")
    if args.use_constraints:
        print(f"  budget={args.cost_budget_d}  α_cost={args.alpha_cost}  σ={args.sigma_wall}")

    # ──────────────────────────────────────────────────────────
    #  Main training loop
    # ──────────────────────────────────────────────────────────

    jit_epoch = jax.jit(training_epoch)

    for epoch in range(args.num_epochs):
        t0 = time.time()

        (training_state, env_state, buffer_state, rng), epoch_metrics = jit_epoch(
            (training_state, env_state, buffer_state, rng), ()
        )

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
            **eval_metrics,
        }

        if args.use_constraints:
            _mean_cost = float(jnp.mean(epoch_metrics["mean_step_cost"]))
            log_dict.update({
                "cost_critic_loss":    float(jnp.mean(epoch_metrics["cost_critic_loss"])),
                "mean_step_cost":      _mean_cost,
                "mean_d_wall":         float(jnp.mean(epoch_metrics["mean_d_wall"])),
                "collision_rate":      float(jnp.mean(epoch_metrics["collision_rate"])),
                "mean_qc":             float(jnp.mean(epoch_metrics["mean_qc"])),
                "mean_qc_pi":          float(jnp.mean(epoch_metrics["mean_qc_pi"])),
                "lambda":              float(jnp.mean(epoch_metrics["lambda"])),
                # Explicit CMDP violation: max(0, Ĵ_c - d).
                # Positive values mean the constraint is currently violated.
                "constraint_violation": max(0.0, _mean_cost - args.cost_budget_d),
            })

        print(
            f"[{epoch+1:3d}/{args.num_epochs}] steps={env_steps:,} | "
            f"c_loss={log_dict['critic_loss']:.4f} acc={log_dict['accuracy']:.3f} | "
            f"a_loss={log_dict['actor_loss']:.4f} | "
            f"t={dur:.1f}s"
        )
        if args.use_constraints:
            print(
                f"         collisions={log_dict['collision_rate']:.4f} "
                f"cost={log_dict['mean_step_cost']:.4f} "
                f"λ={log_dict['lambda']:.4f} "
                f"Qc={log_dict['mean_qc']:.4f}"
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
            "log_lambda":                training_state.log_lambda,
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
    args = tyro.cli(Args)
    main(args)
