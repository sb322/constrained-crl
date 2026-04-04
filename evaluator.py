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
                 obs_dim: Optional[int] = None):
        """
        Args:
            actor_step: deterministic policy step function.
            eval_env: Brax environment for evaluation.
            num_eval_envs: number of parallel eval environments.
            episode_length: steps per episode.
            key: JAX PRNGKey.
            cost_fn: optional JAX-compatible function (agent_xy: [...,2]) ->
                     cost: [...].  When provided, eval-time constraint metrics
                     are computed: eval/mean_step_cost, eval/constraint_violation.
            cost_budget_d: CMDP budget threshold d.  Required when cost_fn is set.
            obs_dim: state dimension (before goal concat) in the observation.
                     The first 2 dims of the state are assumed to be (x, y).
                     Required when cost_fn is set.
        """
        self._key = key
        self._eval_walltime = 0.
        self._cost_fn = cost_fn
        self._cost_budget_d = cost_budget_d
        self._obs_dim = obs_dim

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
            eval/mean_step_cost      — mean per-step cost across trajectory
            eval/constraint_violation — max(0, mean_step_cost - budget)
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
        # eval_data.observation: [episode_length, num_eval_envs, obs_size]
        # First 2 dims of the state portion are the agent (x, y) position.
        if self._cost_fn is not None and self._obs_dim is not None:
            # Convert trajectory observations to numpy for cost computation.
            # Shape: [T, N, obs_size] — the first 2 channels are (x, y).
            obs_np = np.array(eval_data.observation)          # [T, N, obs_size]
            xy_flat = obs_np[:, :, :2].reshape(-1, 2)         # [T*N, 2]
            xy_jax = jnp.array(xy_flat)
            # cost_fn handles batched input [..., 2] -> [...]
            step_costs = np.array(self._cost_fn(xy_jax))      # [T*N]
            mean_step_cost = float(np.mean(step_costs))
            constraint_violation = float(
                max(0.0, mean_step_cost - self._cost_budget_d)
            )
            metrics["eval/mean_step_cost"]      = mean_step_cost
            metrics["eval/constraint_violation"] = constraint_violation

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime,
                   **training_metrics, **metrics}
        return metrics
