# SR-CPO Depth Ablation — v2 Interpretation Package

Self-contained briefing for a second-opinion reviewer.  Contains the
research question, algorithm, environment, every hyperparameter, the
depth × seed grid, the v1 and v2 raw endpoint numbers, the qualitative
shape of each of the 6 monitored metrics, and the author's
interpretation.  A reader with no prior context should be able to form
their own verdict from this document alone.

---

## 1. Research question

Does network depth (actor + reward critic) affect CMDP feasibility and
constraint-violation dynamics of **SR-CPO** (Surrogate-Reward Constrained
Policy Optimization) on Brax/MJX **Ant Big Maze**?  Specifically, does
deeper actor/critic accelerate convergence of the PID-Lagrangian dual
variable λ and of the cost-critic estimate Ĵ_c toward the budget d, or
does it only increase per-step compute?

The ablation is **one variable (depth) × 2 seeds** over
depth ∈ {2, 4, 6, 8}, seed ∈ {0, 42}.  Everything else is held fixed.

---

## 2. Algorithm (SR-CPO, one-screen summary)

**Optimization problem (CMDP).**
$$
\max_{\pi}\; J_r(\pi) = \mathbb{E}_\pi\!\left[\sum_{t\ge 0}\gamma^{t} r(s_t,a_t)\right]
\quad\text{s.t.}\quad
J_c(\pi) = \mathbb{E}_\pi\!\left[\sum_{t\ge 0}\gamma_c^{t} c(s_t,a_t)\right] \le d.
$$

**Components.**
* Reward critic — InfoNCE goal-conditioned contrastive network
  (architecture varied: `actor_depth = critic_depth ∈ {2,4,6,8}`).
* Cost critic — TD3-style scalar value network with Polyak target
  (**architecture held fixed**: width 256, depth 4).
* Actor — SAC-style squashed Gaussian policy with augmented loss
  $\mathcal{L}_\text{actor}(\pi) = \mathcal{L}_\text{SAC}(\pi) - \lambda\, Q_c(s,\pi(s))$.

**Dual update (PID-Lagrangian).**
$$
e_t = \hat{J}_c(\pi_t) - d, \qquad
\lambda_{t+1} = \mathrm{clip}\!\bigl(\lambda_t + k_p e_t + k_i \textstyle\sum_{s\le t} e_s + k_d(e_t - e_{t-1}),\;0,\;\lambda_\max\bigr).
$$
$\hat{J}_c$ is a Monte-Carlo rollout estimate on 64 randomly sampled goals.

**Pre-activation regime.**  While $\hat{J}_c < d$, the error $e_t$ is
negative, so (i) the proportional term pushes down against the zero
clip and (ii) the integral accumulator stays at zero.  λ is identically
0 and the actor's cost-pressure term vanishes — SR-CPO then reduces to
vanilla CRL with an idle cost head.  This regime is **the central
obstacle** of the current runs.

---

## 3. Environment

Brax/MJX goal-conditioned **Ant Big Maze**.
* Training env id: `ant_big_maze`
* Evaluation env id: `ant_big_maze_eval`
* Observation: 29-dim proprioceptive + 3-dim goal = 32-dim
* Action: 8-dim continuous
* Horizon: 1000 steps per episode
* Cost signal: smooth sigmoid of distance-to-nearest-wall,
  $c(s,a) = \sigma\!\bigl((\varepsilon - \mathrm{dist}(s))/\tau\bigr)$,
  with $\varepsilon = 0.1$, $\tau = 0.05$, giving $c \in [0, 1]$.
* Hard indicator $\mathbf{1}\{\text{wall-contact}\}$ is tracked only
  for monitoring — it does NOT enter the training objective.

---

## 4. Hyperparameters (complete list)

### 4.1 v2 configuration (currently analysing)
| Parameter | Value | Role |
|---|---|---|
| `env_id` | `ant_big_maze` | training env |
| `eval_env_id` | `ant_big_maze_eval` | held-out eval env |
| `num_epochs` | 200 | outer Python loop (epoch = one jit call) |
| `total_env_steps` | 20 000 000 | target env-step budget per run |
| `num_envs` | 512 | parallel Brax envs per training step |
| `unroll_length` | 62 (default) | env-steps per training-step per env |
| `batch_size` | 512 | SGD minibatch |
| `num_minibatches` | 8 | minibatch partitions per update-epoch |
| `num_update_epochs` | **2**  (v2 change; was 4 in v1) | SGD passes per training step |
| γ, γ_c | 0.99 | reward discount, cost discount |
| `cost_budget_d` | **0.1** | CMDP threshold |
| `cost_epsilon` (ε) | 0.1 | sigmoid shift in smooth cost |
| `cost_tau` (τ) | 0.05 | sigmoid temperature in smooth cost |
| PID gains (k_p, k_i, k_d) | 0.1, 0.003, 0.001 | Lagrangian controller |
| `lambda_max` | 100.0 | PID upper clip |
| `cost_critic_width` | 256 | cost critic MLP width (fixed) |
| `cost_critic_depth` | 4 | cost critic depth (NOT varied) |
| `cost_critic_tau` | 0.005 | cost critic Polyak rate |
| `cost_critic_lr` | 3 × 10⁻⁴ | cost critic Adam lr |
| `cost_discount` | 0.99 | γ_c |
| `dual_estimator_goals` | 64 | MC goals for Ĵ_c(π) |
| `actor_depth` (varied) | 2, 4, 6, 8 | ablation axis |
| `critic_depth` (varied) | 2, 4, 6, 8 | ablation axis (= actor_depth) |
| `actor_skip_connections` | depth / 2 | 1, 2, 3, 4 |
| `critic_skip_connections` | depth / 2 | 1, 2, 3, 4 |
| `actor_width` / `critic_width` | 256 (default) | MLP width |
| Activation | silu (flax default) | ... |
| seed | 0, 42 | 2 seeds per depth |

### 4.2 Infrastructure
* NJIT Wulver HPC, GPU partition, QOS = `standard`
* 32 GB host RAM, A100-SXM4-80 GB per task
* JAX 0.4.x with CUDA 12.8 / 12.9 hybrid stack
* XLA compile cache on disk (`.jax_cache/`) amortises first-trace cost
* SLURM array: 8 tasks = 4 depths × 2 seeds
* Walltime per task: v2 is **8 h** (was 5 h in v1)

---

## 5. Depth × seed grid

SLURM array job 958877.

| task_id | depth | seed | skip-conn. |
|:---:|:---:|:---:|:---:|
| 0 | 2 | 0  | 1 |
| 1 | 2 | 42 | 1 |
| 2 | 4 | 0  | 2 |
| 3 | 4 | 42 | 2 |
| 4 | 6 | 0  | 3 |
| 5 | 6 | 42 | 3 |
| 6 | 8 | 0  | 4 |
| 7 | 8 | 42 | 4 |

---

## 6. Run outcomes

**v1 (job 949311, 5 h walltime, `num_update_epochs`=4):** all 8 tasks
hit SLURM TIMEOUT at ≈ 5:00:01, completing **14 training epochs ≈
1.33 × 10⁶ env-steps** per run.

**v2 (job 958877, 8 h walltime, `num_update_epochs`=2):** all 8 tasks
hit SLURM TIMEOUT at 08:00:22–24 (exit 0:15), completing **24 training
epochs = 2 190 336 env-steps** per run. No OOMs, no tracebacks.
Measured per-epoch wall ≈ **718 s ≈ 12 min** (the v1 projection of
"10 min/epoch after halving SGD" was optimistic — not all overhead is
SGD).

Steps per epoch ≈ 95 232 in both v1 and v2 (set by
`num_envs × unroll_length × num_training_steps_per_epoch`).

---

## 7. v2 observed metrics — final epoch (epoch 23, 2 190 336 env-steps)

Last-line summary for each task, verbatim from the logs:

| task | depth | seed | c_loss | acc | a_loss | hard_viol | cost | λ | Ĵ_c | Q_c |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 2 | 0  | 5.8373 | 0.016 | −6.5064 | 1.0000 | 0.8806 | 0.0000 | 0.0532 | 5.3480 |
| 1 | 2 | 42 | 5.8533 | 0.014 | −6.4995 | 1.0000 | 0.8806 | 0.0000 | 0.0533 | 5.3217 |
| 2 | 4 | 0  | 5.7021 | 0.024 | −6.4729 | 1.0000 | 0.8806 | 0.0000 | 0.0535 | 5.3702 |
| 3 | 4 | 42 | 5.6926 | 0.023 | −6.4415 | 1.0000 | 0.8806 | 0.0000 | 0.0527 | 5.2626 |
| 4 | 6 | 0  | 5.7228 | 0.021 | −6.4909 | 1.0000 | 0.8806 | 0.0000 | 0.0538 | 5.3906 |
| 5 | 6 | 42 | 5.6906 | 0.022 | −6.4386 | 1.0000 | 0.8806 | 0.0000 | 0.0524 | 5.2583 |
| 6 | 8 | 0  | 5.6805 | 0.023 | −6.4447 | 1.0000 | 0.8806 | 0.0000 | 0.0537 | 5.3910 |
| 7 | 8 | 42 | 5.6791 | 0.023 | −6.3574 | 1.0000 | 0.8806 | 0.0000 | 0.0528 | 5.2551 |

**Max Ĵ_c across all 8 runs in v2:** 0.0538 (task 4, depth=6, seed=0).
No run crossed the budget d = 0.1.  λ is identically zero in every run.

---

## 8. v1 vs v2 comparison

| quantity | v1 (epoch 14, 1.33 M steps) | v2 (epoch 23, 2.19 M steps) |
|:---|:---:|:---:|
| hard_viol | 1.0000 | 1.0000 |
| cost      | 0.8806 | 0.8806 |
| λ         | 0.0000 | 0.0000 |
| Ĵ_c       | ≈ 0.06 | ≈ 0.053 |
| c_loss (depth=2, seed=0) | ≈ 5.82 | 5.8373 |
| c_loss (depth=8, seed=0) | ≈ 5.67 | 5.6805 |
| a_loss    | ≈ −7.0 | ≈ −6.5 |

**Important non-monotonicity:** Ĵ_c was higher at epoch 14 of v1 (≈ 0.06)
than at epoch 23 of v2 (≈ 0.053), despite v2 having more env-steps.
The most plausible explanation is that `num_update_epochs` 4 → 2 slows
the actor's state-distribution drift, so the cost critic's Monte-Carlo
estimate of long-run policy cost grows more slowly in env-step terms
under v2 than v1.

---

## 9. Qualitative shape of each of the 6 plotted metrics

The six-panel figure is produced by `plot_depth_ablation.py` and plots
metric vs environment steps, with color encoding depth (viridis) and
line-style encoding seed (solid = 0, dashed = 42).

1. **hard_viol.** Flat at 1.0 across all 8 runs for the entire window.
2. **cost.** Flat at 0.8806 across all 8 runs.
3. **Ĵ_c.** Monotonic concave rise, all 8 runs nearly coincident; red
   dotted line at the budget d = 0.1 is well above every curve.
   **No crossing.**
4. **λ.** Flat at 0 across all 8 runs.
5. **c_loss** (log-scaled y-axis). Three-phase trajectory that is
   reproducible across seeds:
   * **Phase A — rapid descent (epochs 1 → 3).** All runs fall from
     ≈ 5.94 to a local minimum at ≈ 5.52 (depth=2) or ≈ 5.42 (depth ≥ 4).
     Contrastive critic fitting easy negatives from the near-random
     initial policy.
   * **Phase B — rebound (epochs 3 → 6).** c_loss climbs back up to a
     local maximum at ≈ 5.99 (depth=2) or ≈ 5.82 (depth ≥ 4).
     Distributional shift: the actor's policy is improving faster than
     the InfoNCE critic can track, so on-policy pairs that the critic
     had seen as negatives are now positives.
   * **Phase C — slow decay (epochs 6 → end).**  Monotonic decrease;
     depth ordering is stable through this phase.
6. **a_loss.** Near-linear decrease from ≈ −3.72 at epoch 1 to ≈ −6.5
   at epoch 23, with all 8 runs tightly overlapping.

---

## 10. Author's interpretation

### 10.1 What v1 + v2 jointly support

A. **Pre-activation regime characterisation.**  For at least
2.2 × 10⁶ env-steps at the paper's config (budget d = 0.1, PID =
(0.1, 0.003, 0.001)), SR-CPO runs with λ ≡ 0 in a goal-conditioned
Ant Big Maze.  The constraint is not binding; the algorithm is
effectively vanilla CRL with an idle cost head.  The PID integrator
does not accumulate because $\hat{J}_c < d$ everywhere.

B. **Depth saturation on the contrastive reward critic.**  Across both
v1 (1.4 M steps) and v2 (2.2 M steps), the ordering
$\mathtt{c\_loss}(\text{depth}=2) > \mathtt{c\_loss}(\text{depth}\ge 4)$
is robust, and the spread within $\{4, 6, 8\}$ is at or below the
seed-jitter noise floor.  Reading: depth matters for goal-conditioned
InfoNCE value fitting, but the benefit saturates around depth ≈ 4 in
the pre-activation regime with `skip_connections = depth/2`.

C. **Infrastructure validity.**  8/8 tasks survive the CUDA 12.8/12.9
hybrid stack, the XLA compile cache, the SLURM array, and produce
seed-reproducible curves.  Per-epoch wall = 12 min at depth 8.

D. **Linear extrapolation from v1 is falsified.**  v1 predicted a
$\hat{J}_c = d$ crossing at 2.5–3 M env-steps.  v2 reached 2.19 M
env-steps with $\hat{J}_c \approx 0.053$ — growth is sub-linear and
decelerating, so the true crossing (if it exists at all at this
budget) is much further out (estimated 8–15 M env-steps).

### 10.2 What v1 + v2 do NOT support

* Any claim about depth effects on feasibility, hard-violation rate, or
  steady-state $\hat{J}_c - d$ gap — panels 1, 2, 4 are flat.
* Any claim about dual dynamics — λ never left zero.
* Any claim about depth effects on the cost critic itself — its
  architecture was held fixed.

### 10.3 Verdict

Useful, but not as a "headline result on depth-vs-feasibility."  The
runs establish the pre-activation regime as a distinct object of study
and the depth saturation on the contrastive critic, and they
quantitatively falsify the linear extrapolation used in v1.  They do
not answer the active-constraint question the project is ultimately
after.

### 10.4 Recommended next step

Reduce the budget to **d = 0.02 or 0.03** (so Ĵ_c crosses d in epoch
5–10 at current growth rates) and re-run the depth × seed grid at the
v2 walltime.  That gives the active-constraint regime inside a single
task's walltime and preserves the depth axis.  Everything else in the
config stays.

Alternatives considered and deferred:
* Warm-start λ at a non-zero value — contaminates the ablation.
* Go longer at d = 0.1 — would need ≈ 100 h/run, infeasible under a
  24 h walltime.
* Harder initial policy — changes the env, contaminates comparability
  with the paper.

---

## 11. Questions I'd like a second opinion on

1. Is the v1 → v2 deceleration of Ĵ_c actually explained by
   `num_update_epochs` 4 → 2, or is there another mechanism I'm missing
   (e.g. cost-critic Polyak-target lag, replay freshness, Monte-Carlo
   estimator variance)?
2. Is lowering d to 0.02 the cleanest move, or is there a more
   principled choice — e.g. rescale the sigmoid cost so its initial-
   policy mean is closer to d by construction?
3. Given the depth saturation at depth ≈ 4 in the pre-activation
   regime, should the v3 ablation drop depth=2 and add depth=12 or
   depth=16 on the upper end?  Or add a width axis alongside depth?
4. The depth=2 c_loss rebound in Phase B is larger than depth ≥ 4's
   rebound (Δ ≈ 0.47 vs Δ ≈ 0.17).  Is this just parameter count, or a
   statement about contrastive critic capacity to track a moving
   policy distribution?
5. For an ICML/NeurIPS submission, do v1 + v2 stand as "Section 5.1:
   characterisation of the pre-activation regime"?  Or are they best
   relegated to an appendix in support of a v3+ headline plot?
