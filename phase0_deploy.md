# Phase-0 Deployment — SR-CPO safety-cost wiring fix

This note covers the **minimal deployment** required to verify that the
Phase-0 bug fix is correct before burning any GPU-hours on the depth
ablation.  Nothing here launches a full SR-CPO run; it only proves that the
env now emits a **state-dependent** cost signal and that the training loop
threads it through to `cost_critic_loss_fn` without loss.

## What changed

| File | Change |
|------|--------|
| `envs/ant_maze.py` | `make_maze` returns wall centers. Constructor stores `_wall_centers`, `_half_wall_size`, `_enable_cost`, `_cost_epsilon`, `_cost_tau`. `reset()` / `step()` compute `(cost, d_wall, hard_violation)` from the authoritative torso xy (`pipeline_state.x.pos[0, :2]`) and emit them into both `state.info` and `state.metrics`. |
| `train.py` — `dummy_transition` | Added `cost`, `d_wall`, `hard_violation` keys to `extras` (scalar shapes). |
| `train.py` — `collect_step` / `prefill_one` | Lift the scalars from `env_state.info` (with zero fallbacks) into `transition.extras`. |
| `buffer.py` — `flatten_crl_fn` | Preserves the new extras fields through the trajectory flattening; drops last seq_len-1 to match state/next_state slicing. |
| `train.py` — `cost_critic_loss_fn` | Reads `cost`, `d_wall`, `hard_indicator` from `transitions.extras`. No more `agent_xy = state[:, :2]` slicing. |
| `evaluator.py` — `run_evaluation` | Reads `eval/mean_hard_cost`, `eval/mean_smooth_cost`, `eval/mean_d_wall` from `eval_metrics.episode_metrics[...] / episode_steps`. The old `obs_np[:, :, :2]` reconstruction path is gone — it had the same bug as the training loss (wrong two dims). |

Why this matters: with `exclude_current_positions_from_observation=True`
(the Brax default), `obs[:, :2]` is `[z, quat_w]`, not `(x, y)`.  The fake
"agent" sat pinned at `(z≈0.5, quat≈0)` inside the wall box at the origin,
giving `d_wall ≡ 0` and `c(s) ≡ σ((0.1 − 0)/0.05) = 0.8807585` — which is
exactly the flat `mean_cost = 0.8806` we saw in every prior run.

## 1. Sync local edits to Wulver

All edits live inside `constrained_crl/`.  On the local machine:

```bash
cd "/Users/cyamac/Documents/Claude/Projects/ICML-Level Constrained RL Design & Audit (CMDP / Risk-Sensitive PG)"
rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude '*.pyc' \
  constrained_crl/ \
  sb3222@wulver.njit.edu:/mmfs1/home/sb3222/projects/constrained-crl/constrained_crl/
```

On Wulver (HPC working dir is `/mmfs1/home/sb3222/projects/constrained-crl`,
which is the default cwd per saved memory — so no absolute-path prefix is
needed on any command below):

```bash
cd constrained_crl
ls phase0_deploy.md probe_cost_wiring.py   # sanity: files present
```

## 2. Run the cost-wiring probe

The probe is a **non-training** sanity check.  It constructs the env with
`enable_cost=True`, rolls out a uniform-random policy for 200 steps × 8
envs, and verifies six invariants (P1–P6) described in the script header.

```bash
# Interactive test (CPU is fine; this is <1 min)
module load python/3.11  anaconda3       # or whatever your env module is
conda activate crl-jax                   # or the env that has brax + jax
cd constrained_crl
python probe_cost_wiring.py --env_id ant_big_maze --steps 200 --num_envs 8
```

Artifacts produced in `constrained_crl/`:

- `probe_cost_wiring_report.txt` — line-by-line pass/fail per check
- `probe_cost_wiring_scatter.png` — (x, y) trajectory colored by d_wall +
  per-step cost time series vs. σ(2) reference
- `probe_cost_wiring_metrics.npz` — raw arrays for post-hoc analysis

Exit code `0` = all checks pass, `1` = at least one fails (CI-friendly).

### Pass criteria (all must hold)

| ID | Assertion | Bug value | Fixed value (expected) |
|----|-----------|-----------|------------------------|
| P1 | `std(d_wall) > 0.05` | 0 (constant) | ≳ 0.3 |
| P2 | `max − min d_wall > 0.5` | 0 | several units |
| P3 | `|mean(cost) − σ(2)| > 0.05` | 0 (exactly σ(2) = 0.8807585) | depends on rollout, typically 0.2–0.7 |
| P4a | `mean(hard_viol) < 0.999` | 1.0 | < 0.5 for most random rollouts |
| P5 | `max(x_range, y_range) > 0.5` | 0 (torso pinned) | ≳ 1 unit of maze-scale motion |
| P6 | `max|info − metrics| < 1e-5` | n/a (new) | must be exactly identical |

The weakest assumption is **P4a**: if τ is very small relative to ε, the
sigmoid is almost a step function, and a random policy that happens to
spend most of its time near walls could still register `hard_viol ≈ 1`.
The probe flags this but does not fail on it (P4b is informational only).

### If a check fails

- **P1/P2 fail** → env is not actually computing d_wall.  Check
  `ant_maze.py` step() log: the `agent_xy` line should use
  `pipeline_state.x.pos[0, :2]`, not `obs[:2]`.
- **P3 fails with mean = 0.8806 exactly** → the old bug is still present
  somewhere.  grep for `state[:, :2]` and `smooth_sigmoid_cost(agent_xy`
  in `train.py`.
- **P5 fails** → physics isn't stepping.  Likely VmapWrapper/EpisodeWrapper
  ordering wrong, or env is hitting NaN immediately.  Check
  `env_state.pipeline_state` contents.
- **P6 fails** → `state.info` and `state.metrics` disagree.  This means
  `step()` wrote different values to the two dicts — look at lines
  `info["cost"] = ...` vs `state.metrics.update(cost=...)` in
  `ant_maze.py` (they must use the same variable).

## 3. Next action after probe passes

Do **not** launch the full depth ablation yet.  Phase 0 ends here.  The
next step is Phase 1 (short smoke run of SR-CPO with the fixed pipeline,
single seed, single depth, ~5M env steps) to confirm:

- `mean_cost` now varies with epoch (not constant at 0.88),
- `hard_viol` decreases below 1.0,
- `λ` starts moving (PID detects constraint violation, pushes actor off the
  unsafe region).

That Phase 1 configuration and SLURM template will be produced in a
follow-up note; this file is strictly for Phase-0 verification.

## Quick rollback

If any of this breaks the existing unconstrained baseline:

```bash
cd constrained_crl
git diff HEAD~1 envs/ant_maze.py train.py buffer.py > /tmp/phase0.patch
git checkout HEAD~1 -- envs/ant_maze.py train.py buffer.py
# restore manually later with: git apply /tmp/phase0.patch
```

(Replace `HEAD~1` with whatever commit represents the pre-Phase-0 state in
your local history.)
