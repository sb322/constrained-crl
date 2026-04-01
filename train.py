"""
train.py — Constrained Contrastive RL (C-CRL).

Drop-in replacement for scaling-crl/train.py that adds:
  • CostCritic network (Bellman-backed Q_c)
  • Hybrid step cost from maze geometry
  • Lagrangian actor loss:  L_actor = E[α·log π - Q + λ·Q_c]
  • Dual ascent on λ with cost budget
  • Full cost metric logging to wandb

All original CRL logic is preserved verbatim.
"""

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

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

from buffer import TrajectoryUniformSamplingQueue, flatten_crl_fn
from evaluator import CrlEvaluator
from cost_utils import get_wall_centers, hybrid_cost

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class Args:
    # ── Original CRL args ──────────────────────────────────────
    env_id: str = "humanoid"
    eval_env_id: str = ""
    episode_length: int = 1000
    total_env_steps: int = 100000000
    num_epochs: int = 100
    num_envs: int = 512
    eval_envs: int = 128
    eval_every: int = 1
    action_repeat: int = 1
    unroll_length: int = 62
    num_minibatches: int = 32
    num_update_epochs: int = 4
    seed: int = 0
    batch_size: int = 256
    num_evals: int = 10
    normalize_observations: bool = False
    max_replay_size: int = 10000
    min_replay_size: int = 8192
    grad_updates_per_step: float = 1.0
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
    obs_dim: int = 29
    goal_start_idx: int = 0
    goal_end_idx: int = 2
    vis_length: int = 200
    save_buffer: int = 0
    track: bool = True
    wandb_project: str = "scaling-crl"
    wandb_entity: str = ""
    wandb_group: str = ""

    # ── Constrained CRL additions ──────────────────────────────
    # Toggle the constraint machinery on/off
    use_constraints: bool = True
    # Hybrid cost parameters: c = α·1{contact} + (1-α)·exp(-d/σ)
    alpha_cost: float = 0.5
    sigma_wall: float = 1.0
    contact_threshold: float = 0.1
    # CMDP budget & Lagrange multiplier
    cost_budget_d: float = 0.1        # per-step cost budget d
    lambda_init: float = 0.0          # initial Lagrange multiplier
    lambda_lr: float = 1e-3           # dual ascent step size
    lambda_max: float = 100.0         # clamp to prevent blow-up
    # Cost critic architecture
    cost_critic_lr: float = 3e-4
    cost_critic_width: int = 256
    cost_critic_depth: int = 4
    cost_critic_skip_connections: int = 0
    cost_critic_tau: float = 0.005    # Polyak averaging coefficient
    cost_discount: float = 0.99       # γ_c for cost Bellman backup
    # Whether to use the contrastive critic's repr or raw (s,a) for cost critic
    cost_critic_use_repr: bool = False


# ═══════════════════════════════════════════════════════════════
#  Network building blocks
# ═══════════════════════════════════════════════════════════════

lecun_unfirom = nn.initializers.lecun_normal()  # sic — matches upstream typo
bias_init = nn.initializers.zeros


def residual_block(x, width, normalize, activation):
    """4-Dense residual block with LayerNorm + Swish, matching upstream."""
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class SA_encoder(nn.Module):
    """Contrastive (s,a) encoder — unchanged from upstream."""
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, s, a):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                      bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (
                    self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    """Goal encoder — unchanged from upstream."""
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, g):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = g
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                      bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (
                    self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    """SAC-style stochastic actor — unchanged from upstream."""
    action_size: int = 8
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, x):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                      bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (
                    self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
        means = nn.Dense(self.action_size, kernel_init=lecun_unfirom,
                         bias_init=bias_init)(x)
        log_stds = nn.Dense(self.action_size, kernel_init=lecun_unfirom,
                            bias_init=bias_init)(x)
        log_stds = jnp.clip(log_stds, -5, 2)
        return means, log_stds


# ── NEW: Cost Critic ──────────────────────────────────────────

class CostCritic(nn.Module):
    """Q_c(s, a, g) → scalar cost-to-go estimate.

    Same residual architecture as SA_encoder but takes (s, a, g) and
    outputs a single scalar (no contrastive structure).
    """
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 0

    @nn.compact
    def __call__(self, s, a, g):
        normalize = nn.LayerNorm()
        activation = nn.swish
        x = jnp.concatenate([s, a, g], axis=-1)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                      bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        for i in range(self.network_depth // 4):
            if self.skip_connections > 0 and i >= (
                    self.network_depth // 4 - self.skip_connections):
                x = residual_block(x, self.network_width, normalize, activation)
            else:
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
                x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                              bias_init=bias_init)(x)
                x = normalize(x)
                x = activation(x)
        # Single scalar output
        x = nn.Dense(1, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x.squeeze(-1)


# ═══════════════════════════════════════════════════════════════
#  Transition & TrainingState
# ═══════════════════════════════════════════════════════════════

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@flax.struct.dataclass
class TrainingState:
    """Extended training state with cost critic and Lagrange multiplier."""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    # ── New fields ────────────────────────────────────────────
    cost_critic_state: TrainState          # online cost critic
    cost_critic_target_params: dict        # Polyak-averaged target params
    log_lambda: jnp.ndarray               # log(λ) for dual ascent


# ═══════════════════════════════════════════════════════════════
#  Main training function
# ═══════════════════════════════════════════════════════════════

def main(args: Args):
    # ── Wandb ──────────────────────────────────────────────────
    run_name = f"{args.env_id}_constrained_{args.seed}_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            group=args.wandb_group if args.wandb_group else None,
            config=vars(args),
            name=run_name,
        )

    # ── RNG ────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    rng, env_key, eval_key, buf_key = jax.random.split(rng, 4)

    # ── Environment ────────────────────────────────────────────
    env = envs.get_environment(args.env_id)
    eval_env_id = args.eval_env_id if args.eval_env_id else args.env_id
    eval_env = envs.get_environment(eval_env_id)

    action_size = env.action_size
    obs_size = args.obs_dim + (args.goal_end_idx - args.goal_start_idx)

    # ── Cost geometry (walls) ──────────────────────────────────
    if args.use_constraints:
        wall_centers, half_wall_size = get_wall_centers(args.env_id)
    else:
        wall_centers = jnp.zeros((1, 2))
        half_wall_size = 1.0

    # ── Network init ───────────────────────────────────────────
    rng, actor_key, sa_key, g_key, cost_key = jax.random.split(rng, 5)

    # Contrastive critic
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

    sa_encoder_params = sa_encoder.init(
        sa_key,
        np.ones([1, args.obs_dim]),
        np.ones([1, action_size]),
    )
    g_encoder_params = g_encoder.init(
        g_key,
        np.ones([1, args.goal_end_idx - args.goal_start_idx]),
    )

    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Actor
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

    # SAC temperature
    log_alpha_init = jnp.array(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params=log_alpha_init,
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    # ── Cost critic ────────────────────────────────────────────
    cost_critic = CostCritic(
        network_width=args.cost_critic_width,
        network_depth=args.cost_critic_depth,
        skip_connections=args.cost_critic_skip_connections,
    )
    goal_dim = args.goal_end_idx - args.goal_start_idx
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
    cost_critic_target_params = jax.tree.map(jnp.copy, cost_critic_params)

    # ── Lagrange multiplier (log-space) ────────────────────────
    log_lambda = jnp.log(jnp.maximum(args.lambda_init, 1e-8))

    # ── Training state ─────────────────────────────────────────
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
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        extras={
            "state": jnp.zeros((args.obs_dim,)),
            "future_state": jnp.zeros((args.obs_dim,)),
            "future_action": jnp.zeros((action_size,)),
        },
    )

    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=args.max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=args.batch_size * args.num_minibatches
                          * args.num_update_epochs,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
    )
    buffer_state = replay_buffer.init(buf_key)

    # ── Evaluator ──────────────────────────────────────────────
    evaluator = CrlEvaluator(
        eval_env,
        lambda params, obs: {
            "sa_encoder": sa_encoder,
            "g_encoder": g_encoder,
            "actor": actor,
            "params": params,
            "obs_dim": args.obs_dim,
            "goal_start_idx": args.goal_start_idx,
            "goal_end_idx": args.goal_end_idx,
            "deterministic": args.deterministic_eval,
        },
        num_eval_envs=args.eval_envs,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        key=eval_key,
    )

    # ──────────────────────────────────────────────────────────
    #  Loss functions
    # ──────────────────────────────────────────────────────────

    def critic_loss_fn(critic_params, transitions, key):
        """InfoNCE contrastive loss — unchanged from upstream."""
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        obs = transitions.observation
        action = transitions.action
        future_state = transitions.extras["future_state"]
        goal = future_state[:, args.goal_start_idx:args.goal_end_idx]

        sa_repr = sa_encoder.apply(sa_encoder_params, obs[:, :args.obs_dim], action)
        g_repr = g_encoder.apply(g_encoder_params, goal)

        logits = -jnp.sqrt(
            jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)
        )
        # InfoNCE
        logsumexp = jax.nn.logsumexp(logits, axis=1)
        critic_loss = -jnp.mean(jnp.diag(logits) - logsumexp)
        # Logsumexp penalty
        critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp ** 2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logits_pos, logits_neg, jnp.mean(correct),
                             jnp.mean(logsumexp))

    def actor_loss_fn(actor_params, critic_params, log_alpha,
                      cost_critic_params, log_lam, transitions, key):
        """Lagrangian actor loss: E[α·log π - Q + λ·Q_c].

        When args.use_constraints is False the λ·Q_c term is zero.
        """
        obs = transitions.observation
        future_state = transitions.extras["future_state"]
        state = obs[:, :args.obs_dim]
        goal = future_state[:, args.goal_start_idx:args.goal_end_idx]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = actor.apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)

        # Contrastive Q-value
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]
        sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
        g_repr = g_encoder.apply(g_encoder_params, goal)
        qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

        # Base SAC actor loss
        alpha = jnp.exp(log_alpha)
        actor_loss = jnp.mean(alpha * log_prob - qf_pi)

        # ── Lagrangian penalty ─────────────────────────────────
        if args.use_constraints:
            lam = jnp.exp(log_lam)
            qc_pi = cost_critic.apply(cost_critic_params, state, action, goal)
            actor_loss = actor_loss + jnp.mean(lam * qc_pi)
        else:
            qc_pi = jnp.zeros_like(qf_pi)

        return actor_loss, (log_prob, jnp.mean(qc_pi))

    def alpha_loss_fn(log_alpha, log_prob):
        """SAC entropy temperature loss — unchanged."""
        alpha_loss = -jnp.mean(log_alpha * (log_prob + action_size))
        return alpha_loss

    # ── Cost critic loss ───────────────────────────────────────

    def cost_critic_loss_fn(cost_critic_params, cost_critic_target_params,
                            transitions, key):
        """TD(0) Bellman loss for Q_c.

        target = c(s,a,s') + γ_c · Q_c^{targ}(s', a', g)
        where a' ~ π(·|s',g) and c is the hybrid cost from maze geometry.
        """
        obs = transitions.observation
        action = transitions.action
        future_state = transitions.extras["future_state"]
        state = obs[:, :args.obs_dim]
        next_state_full = transitions.extras.get(
            "state", obs[:, :args.obs_dim]
        )  # s' approximated from replay; see note below
        goal = future_state[:, args.goal_start_idx:args.goal_end_idx]

        # Current-state agent xy for cost computation
        agent_xy = state[:, :2]

        # Compute step cost from maze geometry
        cost, d_wall, collision = hybrid_cost(
            agent_xy, wall_centers, half_wall_size,
            alpha_cost=args.alpha_cost,
            sigma_wall=args.sigma_wall,
            contact_threshold=args.contact_threshold,
        )

        # Next-state action from current policy (for target)
        next_obs = jnp.concatenate([next_state_full, goal], axis=1)
        means_next, log_stds_next = actor.apply(
            training_state.actor_state.params, next_obs
        )
        stds_next = jnp.exp(log_stds_next)
        key, subkey = jax.random.split(key)
        x_next = means_next + stds_next * jax.random.normal(
            subkey, shape=means_next.shape
        )
        action_next = nn.tanh(x_next)

        # Target Q_c
        qc_target = cost_critic.apply(
            cost_critic_target_params,
            next_state_full, action_next, goal,
        )
        td_target = cost + args.cost_discount * qc_target
        td_target = jax.lax.stop_gradient(td_target)

        # Online Q_c
        qc_online = cost_critic.apply(
            cost_critic_params, state, action, goal,
        )
        loss = jnp.mean((qc_online - td_target) ** 2)

        return loss, {
            "cost_critic_loss": loss,
            "mean_step_cost": jnp.mean(cost),
            "mean_d_wall": jnp.mean(d_wall),
            "collision_rate": jnp.mean(collision),
            "mean_qc": jnp.mean(qc_online),
        }

    # ──────────────────────────────────────────────────────────
    #  SGD step  (one gradient update on one minibatch)
    # ──────────────────────────────────────────────────────────

    @jax.jit
    def sgd_step(carry, transitions):
        (training_state, key) = carry
        key, critic_key, actor_key, cost_key = jax.random.split(key, 4)

        # ── Critic update ──────────────────────────────────────
        critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (critic_loss, (logits_pos, logits_neg, accuracy, logsumexp)), \
            critic_grads = critic_grad_fn(
                training_state.critic_state.params, transitions, critic_key)
        critic_state = training_state.critic_state.apply_gradients(
            grads=critic_grads)

        # ── Actor update (Lagrangian) ──────────────────────────
        actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, (log_prob, mean_qc_pi)), actor_grads = actor_grad_fn(
            training_state.actor_state.params,
            critic_state.params,
            training_state.alpha_state.params,
            training_state.cost_critic_state.params,
            training_state.log_lambda,
            transitions,
            actor_key,
        )
        actor_state = training_state.actor_state.apply_gradients(
            grads=actor_grads)

        # ── Alpha (temperature) update ─────────────────────────
        alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
        alpha_loss, alpha_grads = alpha_grad_fn(
            training_state.alpha_state.params, log_prob)
        alpha_state = training_state.alpha_state.apply_gradients(
            grads=alpha_grads)

        # ── Cost critic update ─────────────────────────────────
        if args.use_constraints:
            cost_critic_grad_fn = jax.value_and_grad(
                cost_critic_loss_fn, has_aux=True)
            (cc_loss, cc_metrics), cc_grads = cost_critic_grad_fn(
                training_state.cost_critic_state.params,
                training_state.cost_critic_target_params,
                transitions,
                cost_key,
            )
            cost_critic_state_new = training_state.cost_critic_state \
                .apply_gradients(grads=cc_grads)

            # Polyak average target params
            tau = args.cost_critic_tau
            target_params_new = jax.tree.map(
                lambda p, tp: tau * p + (1 - tau) * tp,
                cost_critic_state_new.params,
                training_state.cost_critic_target_params,
            )

            # ── Dual (lambda) update ───────────────────────────
            # λ ← max(0, λ + lr_λ · (Ĵ_c - d))
            # In log-space: log_λ' = log(max(ε, exp(log_λ) + lr·(Ĵ_c - d)))
            lam = jnp.exp(training_state.log_lambda)
            constraint_violation = cc_metrics["mean_step_cost"] - args.cost_budget_d
            lam_new = jnp.clip(
                lam + args.lambda_lr * constraint_violation,
                1e-8, args.lambda_max,
            )
            log_lambda_new = jnp.log(lam_new)
        else:
            cost_critic_state_new = training_state.cost_critic_state
            target_params_new = training_state.cost_critic_target_params
            log_lambda_new = training_state.log_lambda
            cc_metrics = {
                "cost_critic_loss": 0.0,
                "mean_step_cost": 0.0,
                "mean_d_wall": 0.0,
                "collision_rate": 0.0,
                "mean_qc": 0.0,
            }

        # ── Assemble new state ─────────────────────────────────
        new_training_state = TrainingState(
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
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_state.params),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "accuracy": accuracy,
            "logsumexp": logsumexp,
            # Cost metrics
            "cost_critic_loss": cc_metrics["cost_critic_loss"],
            "mean_step_cost": cc_metrics["mean_step_cost"],
            "mean_d_wall": cc_metrics["mean_d_wall"],
            "collision_rate": cc_metrics["collision_rate"],
            "mean_qc": cc_metrics["mean_qc"],
            "lambda": jnp.exp(log_lambda_new),
            "mean_qc_pi": mean_qc_pi,
        }

        return (new_training_state, key), metrics

    # ──────────────────────────────────────────────────────────
    #  Env step  (collect rollouts → add to buffer)
    # ──────────────────────────────────────────────────────────

    def env_step(carry, unused_t):
        """One unroll step in the environment."""
        (env_state, training_state, buffer_state, key) = carry
        key, action_key, step_key = jax.random.split(key, 3)

        obs = env_state.obs
        state = obs[:, :args.obs_dim]
        # During collection, use a random goal or the env-provided goal
        goal = obs[:, args.obs_dim:]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = actor.apply(
            training_state.actor_state.params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(
            action_key, shape=means.shape)
        action = nn.tanh(x_ts)

        next_env_state = env.step(env_state, action)

        transition = Transition(
            observation=obs,
            action=action,
            reward=next_env_state.reward,
            discount=1.0 - next_env_state.done,
            extras={
                "state": obs[:, :args.obs_dim],
            },
        )
        buffer_state = replay_buffer.insert(buffer_state, transition)

        return (next_env_state, training_state, buffer_state, key), transition

    # ──────────────────────────────────────────────────────────
    #  Training step  (sample from buffer → multiple SGD steps)
    # ──────────────────────────────────────────────────────────

    def training_step(carry, unused_t):
        (training_state, env_state, buffer_state, key) = carry
        key, sample_key, unroll_key = jax.random.split(key, 3)

        # Collect new experience
        (env_state, training_state, buffer_state, _), _ = jax.lax.scan(
            env_step,
            (env_state, training_state, buffer_state, unroll_key),
            (),
            length=args.unroll_length,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps
            + args.unroll_length * args.num_envs * args.action_repeat,
        )

        # Sample and reshape for minibatches
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        transitions = flatten_crl_fn(transitions, args.gamma)

        # Reshape into (num_update_epochs * num_minibatches, batch_size, ...)
        transitions = jax.tree.map(
            lambda x: x.reshape(
                (args.num_update_epochs * args.num_minibatches,
                 args.batch_size) + x.shape[1:]
            ),
            transitions,
        )

        (training_state, key), sgd_metrics = jax.lax.scan(
            sgd_step,
            (training_state, sample_key),
            transitions,
        )

        return (training_state, env_state, buffer_state, key), sgd_metrics

    # ──────────────────────────────────────────────────────────
    #  Training epoch  (multiple training steps)
    # ──────────────────────────────────────────────────────────

    def training_epoch(carry, unused_t):
        (training_state, env_state, buffer_state, key) = carry

        # How many training_step calls per epoch
        steps_per_epoch = (
            args.total_env_steps
            // (args.num_epochs
                 * args.unroll_length
                 * args.num_envs
                 * args.action_repeat)
        )

        (training_state, env_state, buffer_state, key), metrics = \
            jax.lax.scan(
                training_step,
                (training_state, env_state, buffer_state, key),
                (),
                length=steps_per_epoch,
            )

        return (training_state, env_state, buffer_state, key), metrics

    # ──────────────────────────────────────────────────────────
    #  Main training loop
    # ──────────────────────────────────────────────────────────

    # Reset environment
    rng, env_reset_key = jax.random.split(rng)
    env_keys = jax.random.split(env_reset_key, args.num_envs)
    env_state = jax.vmap(env.reset)(env_keys)

    # Prefill buffer with random actions
    rng, prefill_key = jax.random.split(rng)

    def prefill_step(carry, unused):
        env_state, buffer_state, key = carry
        key, action_key, step_key = jax.random.split(key, 3)
        action = jax.random.uniform(
            action_key, (args.num_envs, action_size),
            minval=-1.0, maxval=1.0,
        )
        next_env_state = env.step(env_state, action)
        transition = Transition(
            observation=env_state.obs,
            action=action,
            reward=next_env_state.reward,
            discount=1.0 - next_env_state.done,
            extras={"state": env_state.obs[:, :args.obs_dim]},
        )
        buffer_state = replay_buffer.insert(buffer_state, transition)
        return (next_env_state, buffer_state, key), ()

    prefill_length = args.min_replay_size // args.num_envs + 1
    (env_state, buffer_state, _), _ = jax.lax.scan(
        prefill_step,
        (env_state, buffer_state, prefill_key),
        (),
        length=prefill_length,
    )

    print(f"Prefilled buffer with {prefill_length * args.num_envs} transitions")
    print(f"Starting training for {args.num_epochs} epochs...")
    if args.use_constraints:
        print(f"  Constraints ON: budget={args.cost_budget_d}, "
              f"α_cost={args.alpha_cost}, σ_wall={args.sigma_wall}")

    # JIT-compile the training epoch
    jit_training_epoch = jax.jit(training_epoch)

    for epoch in range(args.num_epochs):
        t0 = time.time()

        (training_state, env_state, buffer_state, rng), epoch_metrics = \
            jit_training_epoch(
                (training_state, env_state, buffer_state, rng), ()
            )

        epoch_time = time.time() - t0
        env_steps = int(training_state.env_steps)
        grad_steps = int(training_state.gradient_steps)

        # ── Evaluation ─────────────────────────────────────────
        if (epoch + 1) % args.eval_every == 0:
            eval_params = {
                "actor": training_state.actor_state.params,
                "sa_encoder": training_state.critic_state.params["sa_encoder"],
                "g_encoder": training_state.critic_state.params["g_encoder"],
            }
            # Note: evaluator interface depends on the CrlEvaluator implementation
            # This is a placeholder — adapt to the actual evaluator API
            # eval_metrics = evaluator.run_evaluation(eval_params, {})

        # ── Logging ────────────────────────────────────────────
        # Average metrics over the epoch
        log_dict = {
            "epoch": epoch,
            "env_steps": env_steps,
            "gradient_steps": grad_steps,
            "epoch_time": epoch_time,
            "steps_per_sec": (args.total_env_steps / args.num_epochs) / max(epoch_time, 1e-6),
            # Critic
            "critic_loss": float(jnp.mean(epoch_metrics["critic_loss"])),
            "accuracy": float(jnp.mean(epoch_metrics["accuracy"])),
            "logits_pos": float(jnp.mean(epoch_metrics["logits_pos"])),
            "logits_neg": float(jnp.mean(epoch_metrics["logits_neg"])),
            "logsumexp": float(jnp.mean(epoch_metrics["logsumexp"])),
            # Actor
            "actor_loss": float(jnp.mean(epoch_metrics["actor_loss"])),
            "alpha": float(jnp.mean(epoch_metrics["alpha"])),
            "alpha_loss": float(jnp.mean(epoch_metrics["alpha_loss"])),
        }

        if args.use_constraints:
            log_dict.update({
                # Cost
                "cost_critic_loss": float(jnp.mean(epoch_metrics["cost_critic_loss"])),
                "mean_step_cost": float(jnp.mean(epoch_metrics["mean_step_cost"])),
                "mean_d_wall": float(jnp.mean(epoch_metrics["mean_d_wall"])),
                "collision_rate": float(jnp.mean(epoch_metrics["collision_rate"])),
                "mean_qc": float(jnp.mean(epoch_metrics["mean_qc"])),
                "mean_qc_pi": float(jnp.mean(epoch_metrics["mean_qc_pi"])),
                "lambda": float(jnp.mean(epoch_metrics["lambda"])),
            })

        print(f"Epoch {epoch+1}/{args.num_epochs}  |  "
              f"steps={env_steps:,}  |  "
              f"critic_loss={log_dict['critic_loss']:.4f}  |  "
              f"actor_loss={log_dict['actor_loss']:.4f}  |  "
              f"accuracy={log_dict['accuracy']:.3f}  |  "
              f"time={epoch_time:.1f}s")
        if args.use_constraints:
            print(f"  COST: collision_rate={log_dict['collision_rate']:.4f}  |  "
                  f"mean_cost={log_dict['mean_step_cost']:.4f}  |  "
                  f"lambda={log_dict['lambda']:.4f}  |  "
                  f"qc={log_dict['mean_qc']:.4f}")

        if args.track:
            wandb.log(log_dict, step=env_steps)

    # ── Save checkpoint ────────────────────────────────────────
    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "actor_params": training_state.actor_state.params,
        "critic_params": training_state.critic_state.params,
        "alpha_params": training_state.alpha_state.params,
    }
    if args.use_constraints:
        ckpt.update({
            "cost_critic_params": training_state.cost_critic_state.params,
            "cost_critic_target_params": training_state.cost_critic_target_params,
            "log_lambda": training_state.log_lambda,
        })
    with open(f"{ckpt_dir}/params.pkl", "wb") as f:
        pickle.dump(ckpt, f)
    print(f"Checkpoint saved to {ckpt_dir}/params.pkl")

    # ── Save buffer (optional) ─────────────────────────────────
    if args.save_buffer:
        with open(f"{ckpt_dir}/buffer.pkl", "wb") as f:
            pickle.dump(buffer_state, f)
        print(f"Buffer saved to {ckpt_dir}/buffer.pkl")

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
