# Obs-Indexing Audit — Findings & Proposed Patches

**Date:** 2026-04-21
**Scope:** Proactive audit for latent bugs of the same class as the v1a
cost-wiring trap (reading state off the observation vector, whose layout
depends on `exclude_current_positions_from_observation`).
**Verdict:** 2 latent bugs found. Neither is live in the current Phase-1b
pipeline because `train.py` forces `exclude_current_positions_from_observation=False`,
but both would silently produce wrong reward / wrong termination if that
override were ever removed, or if these envs were used elsewhere. Fixes are
ready; recommend applying **after** Phase-1b overnight completes to avoid
mid-run code divergence.

---

## 1. Class of bug

Brax MuJoCo envs expose `obs` whose first dims change with
`exclude_current_positions_from_observation`:

| flag  | obs[:2] (Ant) | obs[:3] (Humanoid) |
|-------|---------------|--------------------|
| True (default)  | `[z, qw]`       | `[z, qw, qx]`      |
| False (Phase-1 override) | `[x, y]` | `[x, y, z]`    |

Any code that reads `obs[:k]` and treats it as a Cartesian position is
load-bearing on the flag being `False`. The v1a bug had exactly this
shape (`obs[:, :2]` used for safety-cost, giving `σ((ε − z)/τ) ≈ σ(2) ≈
0.88` — a flat cost that the critic trivially fit and the PID dual
ignored).

## 2. Bugs found

### 2.1 `envs/ant_maze.py:550` — reward, success, success_easy

```python
# current
dist = jp.linalg.norm(obs[:2] - obs[-2:])
success       = jp.array(dist < 0.5, dtype=float)
success_easy  = jp.array(dist < 2.,  dtype=float)
reward        = -dist + healthy_reward - ctrl_cost - contact_cost
```

`obs[:2]` is `[x, y]` only when `exclude_current_positions_from_observation=False`.
With the Brax default, `obs[:2] = [z, qw]`, so:

- `dist = ‖[z, qw] − target_xy‖` — geometrically meaningless.
- `reward = -dist + ...` — the actor is optimizing a bogus distance.
- `success`, `success_easy`, and the logged `metrics.dist` are all wrong.

Status in current Phase-1b run: **correct by accident** — `train.py:328,
346` force `exclude_current_positions_from_observation=False`, so
`obs[:2]` really is xy. But the dependency is implicit and untested.

### 2.2 `envs/humanoid_maze.py:255` — reward, distance_to_target

```python
# current
distance_to_target = jnp.linalg.norm(obs[:3] - obs[-3:])
reward             = -distance_to_target + healthy_reward - ctrl_cost
```

`obs[:3]` is `[x, y, z]` only when the flag is False; otherwise `[z, qw,
qx]`. Same class of bug. Humanoid is not used in the active Phase-1b
run, but the env is imported by `train.py` when `args.env="humanoid_maze"`.

### 2.3 Cost computation (negative control)

`envs/ant_maze.py:522` is already correct:

```python
agent_xy = pipeline_state.x.pos[0, :2]
cost, d_wall, hard_violation = self._compute_safety_cost(agent_xy)
```

This is what we fixed in Phase-0. The present audit confirms the cost
path is clean; the bugs are in the **reward / success** path and in the
**humanoid** env, both of which were never touched by the Phase-0 sweep.

## 3. Grep evidence

```
$ grep -nE "obs\[\s*:\s*[0-9]+\s*\]|obs\[\s*-\s*[0-9]+\s*:\s*\]" \
    constrained_crl/envs/*.py constrained_crl/train.py \
    constrained_crl/buffer.py constrained_crl/evaluator.py
envs/ant_maze.py:550:  dist = jp.linalg.norm(obs[:2] - obs[-2:])
envs/humanoid_maze.py:255: distance_to_target = jnp.linalg.norm(obs[:3] - obs[-3:])
```

Buffer, evaluator, train.py slice obs only via `args.goal_start_idx` /
`args.goal_end_idx` (CRL goal block) or on the CRL-specific
`obs[..., obs_dim:]` tail — both parametric, not position-assumed.

## 4. Proposed patches

Both follow the same pattern the Phase-0 cost-wiring fix used:
**read position off `pipeline_state.x.pos`, not off `obs`.**

### 4.1 `envs/ant_maze.py`

```python
# replace line 550
agent_xy  = pipeline_state.x.pos[0, :2]     # authoritative torso xy
target_xy = pipeline_state.x.pos[-1, :2]    # target body is last
dist = jp.linalg.norm(agent_xy - target_xy)
```

The `pos[-1]` assumption (target is the last body) matches the env's own
`_get_obs`, which uses `pipeline_state.x.pos[-1][:2]` as `target_pos`.

### 4.2 `envs/humanoid_maze.py`

Humanoid bakes a fixed `TARGET_Z_COORD` into `obs[-3:]`, so the patch
must mirror that to keep `-distance` bounded identically:

```python
# replace line 255
agent_xyz      = pipeline_state.x.pos[0]                 # torso
target_xy      = pipeline_state.x.pos[-1][:2]
target_xyz     = jnp.concatenate([target_xy,
                                  jnp.array([TARGET_Z_COORD])])
distance_to_target = jnp.linalg.norm(agent_xyz - target_xyz)
```

(If `TARGET_Z_COORD` is not defined in humanoid_maze.py, hoist it from
the constants block — do **not** reuse `pipeline_state.x.pos[-1, 2]`,
since the target has a physical z that drifts with the mocap and would
break previous baselines.)

## 5. Risk assessment

| Bug | Live impact now | Silent-break scenario |
|-----|-----------------|-----------------------|
| ant_maze.py:550  | None — train.py overrides the flag. | Remove the override → reward collapses, CRL contrastive loss degenerates. |
| humanoid_maze.py:255 | None — humanoid not in Phase-1b. | Any humanoid run inherits the bug. |

Both are 1-line, purely local fixes. Neither touches the CMDP/dual/cost
pipeline. They do not invalidate the overnight run — the overnight run
does not exercise these code paths in a way that could be affected.

## 6. Recommendation

1. **Do NOT patch during the 18h Phase-1b overnight.** Code changes
   mid-run create attribution ambiguity (was the result from pre- or
   post-patch code?) even though the specific paths are inert.
2. After the overnight completes and we have the C1–C5 verdict, apply
   both patches in a single commit titled
   `fix(envs): read agent/target xy from pipeline_state, not obs`, plus
   a short note in the commit body referencing the v1a class of bug.
3. Add a regression guard: an assertion in env `__init__` of the form
   `assert not self._exclude_current_positions_from_observation, ...`
   for configurations we have validated, or — better — refactor to make
   the reward path independent of the obs layout entirely (the patches
   above already achieve this; the assertion just documents intent).
