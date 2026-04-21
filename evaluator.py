import jax
import time
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from brax import envs
from envs.ant import Ant
from typing import NamedTuple, Optional, Callable
from collections import namedtuple


def generate_unroll(actor_step, training_state, env, env_state,
                    unroll_length, extra_fields=()):
    """Collect trajectories of given unroll_length.

    Returns:
        final_state: env state after the last step.
        data: Transition pytree stacked over [unroll_length, num_envs, ...].
    """
    @jax.jit
    def f(carry, unused_t):
        state = carry
        nstate, transition = actor_step(training_state, env, state,
                                        extra_fields=extra_fields)
        return nstate, transition

    final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
    return final_state, data


class CrlEvaluator():
    def __init__(self, actor_step, eval_env, num_eval_envs,
                 episode_length, key,
                 cost_fn: Optional[Callable] = None,
                 cost_budget_d: Optional[float] = None,
                 obs_dim: Optional[int] = None,
                 smooth_cost_fn: Optional[Callable] = None):
        """
        Args:
            actor_step: deterministic policy step function.
            eval_env: Brax environment for evaluation.
            num_eval_envs: number of parallel eval environments.
            episode_length: steps per episode.
            key: JAX PRNGKey.
            cost_fn: hard indicator cost function (agent_xy: [...,2]) -> [...].
                     Used for interpretable constraint violation reporting.
            cost_budget_d: CMDP budget threshold d.  Required when cost_fn is set.
            obs_dim: state dimension (before goal concat) in the observation.
                     The first 2 dims of the state are assumed to be (x, y).
                     Required when cost_fn is set.
            smooth_cost_fn: optional smooth sigmoid cost function for calibration
                     comparison between training and eval costs.
        """
        self._key = key
        self._eval_walltime = 0.
        self._cost_fn = cost_fn
        self._cost_budget_d = cost_budget_d
        self._obs_dim = obs_dim
        self._smooth_cost_fn = smooth_cost_fn

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            # Return both final_state and trajectory data (for cost computation).
            return generate_unroll(
                actor_step,
                training_state,
                eval_env,
                eval_first_state,
                unroll_length=episode_length)

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, training_state, training_metrics,
                       aggregate_episodes=True):
        """Run one epoch of evaluation.

        Returns a dict of eval/* metrics plus all training_metrics.
        When cost_fn was provided at construction, also logs:
            eval/mean_hard_cost      — mean per-step hard indicator cost
            eval/constraint_violation — max(0, mean_hard_cost - budget)
            eval/mean_smooth_cost    — mean per-step smooth sigmoid cost (if provided)
        """
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        # generate_eval_unroll now returns (final_state, trajectory_data)
        eval_state, eval_data = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}

        aggregating_fns = [
            (np.mean, ""),
        ]

        print("Available keys in episode_metrics:",
              eval_metrics.episode_metrics.keys())

        for (fn, suffix) in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(eval_metrics.episode_metrics[name])
                        if aggregate_episodes
                        else eval_metrics.episode_metrics[name]
                    )
                    for name in ['reward', 'success', 'success_easy',
                                 'success_hard', 'dist',
                                 'distance_from_origin']
                    if name in eval_metrics.episode_metrics
                }
            )

        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(
                eval_metrics.episode_metrics["success"] > 0.0
            )

        # ── Eval-time constraint metrics ───────────────────────
        # PHASE-0 FIX: previously we reconstructed cost from
        # `obs_np[:, :, :2]`, which is [z, quat_w] (not (x, y)) under
        # `exclude_current_positions_from_observation=True`.  That produced
        # a constant eval/mean_hard_cost identical to the broken training
        # signal.  We now pull the authoritative per-step scalars straight
        # from `eval_metrics.episode_metrics`, which are sums over each
        # episode of the scalars the env wrote into `state.metrics`
        # (computed from `pipeline_state.x.pos[0, :2]`).
        if self._cost_fn is not None:
            ep_steps = np.asarray(eval_metrics.episode_steps)        # [num_eval_envs]
            # Avoid divide-by-zero on pathological envs that never stepped.
            ep_steps_safe = np.maximum(ep_steps, 1.0)

            if "hard_violation" in eval_metrics.episode_metrics:
                # Per-episode mean hard-violation rate = (sum over steps) / steps.
                hv_sum_per_ep = np.asarray(
                    eval_metrics.episode_metrics["hard_violation"])  # [num_eval_envs]
                hv_rate_per_ep = hv_sum_per_ep / ep_steps_safe
                mean_hard_cost = float(np.mean(hv_rate_per_ep))
                constraint_violation = float(
                    max(0.0, mean_hard_cost - self._cost_budget_d)
                ) if self._cost_budget_d is not None else 0.0
                metrics["eval/mean_hard_cost"]       = mean_hard_cost
                metrics["eval/constraint_violation"] = constraint_violation

            if "cost" in eval_metrics.episode_metrics:
                cost_sum_per_ep = np.asarray(
                    eval_metrics.episode_metrics["cost"])            # [num_eval_envs]
                cost_rate_per_ep = cost_sum_per_ep / ep_steps_safe
                metrics["eval/mean_smooth_cost"] = float(np.mean(cost_rate_per_ep))

            if "d_wall" in eval_metrics.episode_metrics:
                dwall_sum_per_ep = np.asarray(
                    eval_metrics.episode_metrics["d_wall"])
                dwall_rate_per_ep = dwall_sum_per_ep / ep_steps_safe
                metrics["eval/mean_d_wall"] = float(np.mean(dwall_rate_per_ep))

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime,
                   **training_metrics, **metrics}
        return metrics
