# Phase-1 Smoke-Run — Pass / Fail Criteria  (v1b — compact-support cost)

Scope: a single SR-CPO training run on `ant_big_maze`, seed 0, **5 M env
steps, 50 epochs**, using the **compact-support quadratic training cost**
introduced after the v1a post-mortem.  The question Phase 1 answers is
whether the full pipeline (Phase-0 cost-wiring fix + Phase-1 CRL-goal fix
+ Phase-1b cost-shape fix) produces a training loop that is *internally
consistent and responsive to the constraint*.  Phase 1 does **not**
evaluate task performance, sample efficiency, or final safety rate;
those are Phase-2 concerns.

Training cost (v1b):

    c_train(s) = (1 − d_wall(s)/ε_train)²   for d_wall(s) ≤ ε_train,
               = 0                           otherwise,
    with ε_train = 2.0, ε_hard = 0.1, budget d = 0.15.

ε_train = 2.0 is *not* the originally-sketched 0.3 — it was rescaled to
match the ant_big_maze corridor geometry.  The maze uses
`maze_size_scaling = 4.0`, so each wall is a 4×4 box and the open-cell
interior has d_wall ∈ [0, 2.0].  A dense-grid sweep of the open interior
(see `outputs/scale_check.py`) gives the following d_wall percentiles:

    p5  0.15   p10 0.35   p25 0.69   p50 1.20
    p75 1.70   p90 2.08   p95 2.47   p99 3.25

Under ε_train = 0.3, only the points below p10 would see nonzero cost —
i.e. the agent would have to be essentially against a wall for the PID to
notice.  Under ε_train = 2.0, 87 % of interior positions are inside the
support, and the quadratic gives a smooth gradient field throughout the
corridor interior (u = 1 − d/ε_train is meaningful for every d < 2.0).

Hard violation (logged, used for C4 calibration and CMDP accounting):

    c_hard (s) = 1{d_wall(s) < ε_hard}    with ε_hard = 0.1.

The hard threshold is *independent* of ε_train; this is the same
shaping-vs-accounting decoupling that CBF safety filters use.

Why this replaces the sigmoid (the v1a cost):  the sigmoid
`σ((ε − d_wall)/τ)` with ε=0.1, τ=0.05 has infinite support but collapses
to ~10⁻¹⁵ over the interior d_wall ≈ 1.75 that the ant actually occupies.
Under that cost field, run 962826 ran 50 epochs with `train/mean_step_cost
≡ 0`, `λ̃ ≡ 0`, and `train/actor_loss → −485,495` — vacuous safety and
unbounded CRL InfoNCE divergence.  The compact quadratic has matched
support to the danger band, bounded range ∈ [0, 1], smooth transition at
d_wall = ε_train (C¹ at the boundary), and matches the geometric CBF
semantics of distance-to-wall.

Runtime expectation (unchanged from v1a): on one A100 at 256 envs × 62
unroll with `eval_every=5`, SR-CPO on ant sustains ~270–300 k env-steps/h
after JAX cache warm-up.  Wall time for 5 M steps is therefore **~15–18 h**,
inside the 18 h walltime cap.

---

## Hard pass criteria — all must hold

Watch these in the **wandb panel for group `phase1b_smoke`**.

### C1. Cost signal is state-dependent and non-vacuous
- Panel: `train/mean_step_cost` (logged per sgd step inside epoch).
- Targets (v1b, compact-support with ε_train=2.0):
  - **Early-run (epoch 0–5): mean ∈ [0.10, 0.35]**.  Baseline estimate
    under a uniform/near-random policy over the open interior is
    E[c] ≈ 0.26 (from the dense-grid sweep).  A mean below 0.1 would
    mean the initial policy hugs the centerline too tightly — unlikely
    with random exploration; below 0.02 means the cost didn't
    propagate at all.
  - **std across steps within any epoch > 0.05** — cost varies with
    agent position by construction (0 at d≥2.0, (1-d/2)² below).
- Bug signature if violated: `mean_step_cost ≡ 0` across all epochs.
  Means either (a) env_kwargs did not reach ant_maze (check v1b
  `cost_type="quadratic"` / `cost_epsilon=2.0` passthrough in
  `train.py::env_kwargs`), or (b) `d_wall` reported as > ε_train for every
  step — check `get_wall_centers("ant_big_maze")` returns 38 walls with
  `half_size=2.0`.

### C2. Cost descends
- Panel: `train/mean_step_cost` aggregated per epoch.
- Target: **mean over epoch 40–50 strictly less than mean over epoch 0–5**
  by at least 10 % relative.  Absolute target: epoch-50 mean < 0.20
  (this corresponds to u² = 0.20 ⇒ u ≈ 0.447 ⇒ d_wall ≈ 1.11 — i.e.,
  the agent typically stays >1 unit from the nearest wall, a mild but
  nontrivial safety improvement).
- Bug signature if violated: flat or rising trajectory.  Means PID is
  not exerting downward pressure — check C3 first.

### C3. Lagrangian multiplier responds
- Panels: `train/lambda_tilde`, `train/jhat_c`, `train/constraint_gap`.
- Targets:
  - `lambda_tilde` ∈ (0, 50) for at least **some** epochs in the first 30.
    A brief transient `λ̃ > 0` followed by decay once `Ĵ_c < d` is the
    expected PID signature.
  - When `jhat_c > cost_budget_d` (= 0.15), `lambda_tilde` must become
    strictly positive within ≤ 3 epochs of the violation.
  - When `jhat_c < cost_budget_d`, `lambda_tilde` decreases (up to PID
    derivative-channel noise).
- Bug signature if violated: `lambda_tilde ≡ 0` for all 50 epochs.  Under
  ε_train=2.0, the shaping cost is nontrivial *everywhere the ant goes*,
  so a pinned-at-zero λ̃ with nonzero mean_step_cost means the **hard**
  Ĵ_c (over the 1{d<ε_hard=0.1} indicator) stays below d=0.15 — i.e. the
  CMDP is feasible for the random policy.  If that's the case, the PID
  has nothing to fight, which is actually correct behavior.  Only flag
  it as a bug if λ̃ ≡ 0 *and* `train/jhat_c` > d: inspect
  `_critic_based_dual_estimator` and the PID integrator update.

### C4. Calibration consistency
- Panels: `train/mean_step_cost`, `eval/mean_hard_cost`,
  and (if available) `eval/mean_quad_cost` (smooth eval cost, under v1b
  it uses the same quadratic shape as training).
- Targets:
  - At every logged epoch,
    `| train/mean_step_cost − eval/mean_quad_cost |  ≤ 3×`
    ratio (not absolute — they should be the *same* functional of
    d_wall under v1b, so this should be tight).
  - `eval/mean_hard_cost  ≤  train/mean_step_cost` — the hard indicator
    is *narrower* than the quadratic (ε_hard = 0.1 vs ε_train = 0.3),
    so the hard count is a lower bound on the quadratic integral over
    the same trajectories.
- Bug signature if violated: the train-side and eval-side quadratic
  disagree by > 3×.  Means the eval env is not receiving matched
  `cost_type`/`cost_epsilon` env_kwargs (check `eval_env_kwargs`
  passthrough in `train.py`).

### C5. No NaNs and bounded actor loss
- Panel: any loss curve (`train/actor_loss`, `train/critic_loss`,
  `train/cost_critic_loss`, `train/alpha_loss`).
- Targets:
  - **Finite throughout**.  Early transient spikes (epoch < 3) are OK;
    mid-run NaN is not.
  - **actor_loss stays within ±10⁴** — the v1a run drifted to −4.85 × 10⁵
    because `λ̃ ≡ 0` removed the cost-penalty term, leaving the
    unconstrained CRL InfoNCE contrast to diverge on off-support goals.
    Under v1b with a responsive PID, the cost-penalty should clamp
    actor_loss into a tight band well below |10⁴|.
- Bug signature if violated: gradient flow through `lambda_tilde * qc_pi / nu_c`
  exploding because `qc_pi` is unbounded.  With quadratic cost ∈ [0, 1]
  and γ_c = 0.99, |Q_c| ≤ 100, so `qc_pi / nu_c` ≤ 1 — this should NOT
  blow up.  If it does, inspect `nu_c` auto-compute.

---

## Soft pass (informational, don't block Phase 2)

### S1. Task reward descending
- Panel: `train/reward` or `eval/episode_reward`.
- Target: any downward drift in `dist` (or upward in `reward`).  Not expected
  to be strong at 5 M steps — SR-CPO on ant typically needs 20–50 M steps
  for the goal-reach policy to emerge.
- If violated: Phase-2 representation limits may be severe, or the ant is
  still tangled in the CRL-goal quirk even after Phase-1 fix.  Not a Phase-1
  blocker; flag for Phase-2 investigation.

### S2. `eval/mean_d_wall` reasonable
- Panel: `eval/mean_d_wall`.
- Target: in the range `[0, 5]`.  Starts around 2 (ant spawn at `R` cell,
  which is 2 units from any wall) and may drift depending on exploration.
- If violated: env or cost geometry is broken.

### S3. wandb lineage is clean
- Run name format: `ant_big_maze_ccrl_d4_s0_<timestamp>`.
- Group: `phase1b_smoke`.  **Distinct from `phase1_smoke`** so v1a (sigmoid
  cost) and v1b (quadratic cost) plots don't co-mingle.

---

## What to do on failure

| Failed criterion | First diagnostic action | Escalation |
|------------------|-------------------------|------------|
| C1               | Rerun with `--steps 10` logging to print `transitions.extras` contents | Unit-test `flatten_crl_fn` on a synthetic trajectory |
| C2               | Check C3 first; if C3 passes, C2 failure means task is too hard at 5M steps (reclassify as S1) | Relax budget to `d=0.3` and rerun |
| C3               | Log `training/lambda_pid_p`, `training/lambda_pid_i`, `training/lambda_pid_d` separately to see which channel is stuck | Halve all PID gains |
| C4               | Compare `state.metrics['cost']` (env-emitted) with `transitions.extras['cost']` (buffer-stored) in a minute-of-training dump | Same instrumentation as C1 |
| C5               | Bisect: set `use_constraints False` to isolate unconstrained CRL; if that's also NaN, the issue is CRL, not constraints | Gradient clipping + smaller LR |

## Wandb query recipe

From wandb UI or `wandb.Api`:

```python
import wandb
api = wandb.Api()
runs = api.runs("constrained-crl", filters={"group": "phase1b_smoke"})
for run in runs:
    h = run.history(keys=[
        "epoch",
        "train/mean_step_cost",
        "train/hard_violation_rate",
        "train/lambda_tilde",
        "train/jhat_c",
        "train/cost_critic_loss",
        "eval/mean_hard_cost",
        "eval/mean_d_wall",
        "eval/constraint_violation",
    ])
    print(run.name)
    print(h.describe())
```

If this table shows the correct shapes (not all-NaN, not all-constant), C1–C4
are likely satisfied.
