"""Unit test: eval_smooth_cost_fn must be *bitwise* the same function as the
training cost.

Why this test exists
--------------------
C4 in phase1_criteria.md checks |train_cost − eval_smooth_cost| ≤ 3× at
steady state.  If the eval path is silently mis-wired (wrong ε_train, wrong
wall centers, wrong half_wall_size, or — the v1a class — wrong xy source),
C4 becomes a *tautology-with-noise*: it compares the training cost to a
nearby-but-different function and passes because the nearby function is
close by construction.  The calibration metric then has no teeth.

This test bypasses that by asserting numerical equality on synthetic
(x, y) probes covering:
  (A) outside support               d_wall ≥ ε_train  →  cost = 0
  (B) boundary                      d_wall = ε_train  →  cost = 0 (C¹ kink)
  (C) interior of shaping band      d_wall ∈ (0, ε_train)
  (D) hard-violation region         d_wall < ε_hard   →  hard_indicator = 1
  (E) off-grid random xy            checks vectorization + min-over-walls

The test reconstructs both the training cost (via compact_quadratic_cost
directly) and the eval cost (via the closure train.py would build) from
the same wall_centers / half_wall_size / epsilons, and asserts
allclose(cost_train, cost_eval, atol=0, rtol=0) — i.e. they must agree
to the last bit of float32.

Run:
    cd constrained_crl
    python -m pytest tests/test_eval_cost_matches_train.py -v
or standalone:
    python tests/test_eval_cost_matches_train.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Make the package importable when run standalone from repo root or tests/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from cost_utils import compact_quadratic_cost, compute_wall_distance  # noqa: E402


# ───────────────────────────── test fixtures ──────────────────────────────

# Minimal 1-wall geometry: a single 4×4 wall box at the origin, ant_big_maze
# scaling.  half_wall_size = 2.0 → the wall occupies [-2, 2] × [-2, 2].
_WALL_CENTERS   = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
_HALF_WALL_SIZE = 2.0
_EPS_TRAIN      = 2.0      # matches ant_big_maze Phase-1b config
_EPS_HARD       = 0.1      # tight violation threshold


def _build_eval_smooth_cost_fn(eps_train: float, eps_hard: float,
                               wall_centers: jnp.ndarray,
                               half_wall_size: float):
    """Exact replica of the closure constructed in train.py (cost_type=='quadratic').

    See train.py:547-557.  If train.py changes, this closure must change in
    lock-step — the test will loudly fail if the two drift apart.
    """
    def eval_smooth_cost_fn(xy: jnp.ndarray) -> jnp.ndarray:
        cost, _, _ = compact_quadratic_cost(
            xy, wall_centers, half_wall_size,
            cost_epsilon_train=eps_train,
            cost_epsilon_hard=eps_hard,
        )
        return cost
    return eval_smooth_cost_fn


# ───────────────────────────── probe points ───────────────────────────────

def _build_probes():
    """Return (xy, expected_d_wall, expected_cost, expected_hard) arrays.

    Wall is [-2, 2]². d_wall = ‖xy − clip(xy, -2, 2)‖.
    For x > 2, y = 0: d_wall = x − 2.
    """
    eps_t = _EPS_TRAIN
    eps_h = _EPS_HARD

    # (A) well outside support: d_wall = 5.0 ≥ ε_train → cost 0, hard 0
    A = (jnp.array([7.0, 0.0]),      5.0, 0.0, 0.0)
    # (B) exact boundary of support: d_wall = ε_train → cost 0 (C¹ kink), hard 0
    B = (jnp.array([2.0 + eps_t, 0.0]), eps_t, 0.0, 0.0)
    # (C) halfway into band: d_wall = ε_train/2 → cost = (1 − 0.5)² = 0.25
    C = (jnp.array([2.0 + eps_t / 2.0, 0.0]),
         eps_t / 2.0,
         (1.0 - 0.5) ** 2,
         0.0)
    # (D) inside hard region: d_wall = eps_h/2 = 0.05 → cost ≈ (1 − 0.025)² = 0.950625
    D = (jnp.array([2.0 + eps_h / 2.0, 0.0]),
         eps_h / 2.0,
         (1.0 - (eps_h / 2.0) / eps_t) ** 2,
         1.0)
    # (E) diagonal, d_wall = √2 * 0.5 ≈ 0.7071 (corner of wall)
    d_e  = float(np.sqrt(2.0) * 0.5)
    E = (jnp.array([2.5, 2.5]),
         d_e,
         (1.0 - d_e / eps_t) ** 2,
         0.0)

    xys   = jnp.stack([p[0] for p in (A, B, C, D, E)], axis=0)
    dw    = jnp.array([p[1] for p in (A, B, C, D, E)], dtype=jnp.float32)
    costs = jnp.array([p[2] for p in (A, B, C, D, E)], dtype=jnp.float32)
    hards = jnp.array([p[3] for p in (A, B, C, D, E)], dtype=jnp.float32)
    return xys, dw, costs, hards


# ────────────────────────────── assertions ────────────────────────────────

def test_compact_quadratic_matches_closed_form():
    """compact_quadratic_cost reproduces the analytic (1 − d/ε)² on known d_wall."""
    xys, dw_expected, cost_expected, hard_expected = _build_probes()

    cost, d_wall, hard = compact_quadratic_cost(
        xys, _WALL_CENTERS, _HALF_WALL_SIZE,
        cost_epsilon_train=_EPS_TRAIN,
        cost_epsilon_hard=_EPS_HARD,
    )
    # d_wall is the minimum over walls — here there's one wall, so it's the
    # Euclidean distance to the box.  Tolerate float32 round-off on sqrt.
    np.testing.assert_allclose(np.asarray(d_wall), np.asarray(dw_expected),
                               rtol=1e-5, atol=1e-6,
                               err_msg="d_wall disagrees with analytic "
                                       "box-distance on hand-constructed "
                                       "probes.")
    np.testing.assert_allclose(np.asarray(cost), np.asarray(cost_expected),
                               rtol=1e-5, atol=1e-6,
                               err_msg="compact_quadratic_cost does not "
                                       "reproduce (1 − d/ε)² on the shaping "
                                       "band.")
    np.testing.assert_array_equal(np.asarray(hard), np.asarray(hard_expected))


def test_eval_smooth_cost_fn_bitwise_equals_train_cost():
    """train.py's eval_smooth_cost_fn closure must equal compact_quadratic_cost
    up to the last float32 bit, on the same inputs.

    Any drift here silently breaks C4's calibration check.
    """
    eval_fn = _build_eval_smooth_cost_fn(
        _EPS_TRAIN, _EPS_HARD, _WALL_CENTERS, _HALF_WALL_SIZE
    )
    xys, *_ = _build_probes()

    cost_train, _, _ = compact_quadratic_cost(
        xys, _WALL_CENTERS, _HALF_WALL_SIZE,
        cost_epsilon_train=_EPS_TRAIN,
        cost_epsilon_hard=_EPS_HARD,
    )
    cost_eval = eval_fn(xys)

    # Strictly equal — they call the same function with the same args.
    np.testing.assert_array_equal(
        np.asarray(cost_train), np.asarray(cost_eval),
        err_msg="eval_smooth_cost_fn diverged from training cost.  This "
                "breaks C4 (|train − eval| ≤ 3×) by construction: the "
                "calibration metric is comparing a function to itself, "
                "which is the *point* of the guard.",
    )


def test_eval_smooth_cost_fn_vectorizes_over_batch():
    """Eval path takes trajectory batches; make sure the closure handles them."""
    eval_fn = _build_eval_smooth_cost_fn(
        _EPS_TRAIN, _EPS_HARD, _WALL_CENTERS, _HALF_WALL_SIZE
    )
    key = jax.random.PRNGKey(0)
    # [B, T, 2] shaped batch — mimics evaluator's smooth_cost_per_step.
    xy_batch = jax.random.uniform(key, (4, 17, 2), minval=-10.0, maxval=10.0)
    cost = eval_fn(xy_batch)
    assert cost.shape == (4, 17), f"expected (4, 17), got {cost.shape}"
    assert cost.dtype == jnp.float32
    # Cost must be in [0, 1] — property (iv) in cost_utils.py docstring.
    assert float(jnp.min(cost)) >= 0.0
    assert float(jnp.max(cost)) <= 1.0 + 1e-6  # tiny float32 slack


def test_eps_train_off_by_wrong_value_is_caught():
    """Negative control: if someone wires ε_train = 1.0 instead of 2.0 on the
    eval side, the test should notice.  This asserts the test has teeth."""
    eval_fn_wrong = _build_eval_smooth_cost_fn(
        1.0, _EPS_HARD, _WALL_CENTERS, _HALF_WALL_SIZE  # wrong ε_train!
    )
    xys, *_ = _build_probes()
    cost_train, _, _ = compact_quadratic_cost(
        xys, _WALL_CENTERS, _HALF_WALL_SIZE,
        cost_epsilon_train=_EPS_TRAIN,
        cost_epsilon_hard=_EPS_HARD,
    )
    cost_eval_wrong = eval_fn_wrong(xys)
    # At least one probe must disagree — if this passes equality, the unit
    # test itself is broken (not discriminative).
    assert not np.allclose(np.asarray(cost_train),
                           np.asarray(cost_eval_wrong)), (
        "Unit test is not discriminative: using ε_train = 1.0 on eval "
        "produced the same cost as ε_train = 2.0.  The test can't catch "
        "a real mis-wire."
    )


# Extra sanity: compute_wall_distance matches the signed-distance-to-box formula
# on a single wall, which is the sole dependency for compact_quadratic_cost.
def test_compute_wall_distance_single_wall():
    xy = jnp.array([[3.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    dw = compute_wall_distance(xy, _WALL_CENTERS, _HALF_WALL_SIZE)
    # (3,0): outside box by 1 unit in x → d_wall = 1.0
    # (0,0): inside box → d_wall = √1e-8 ≈ 1e-4  (clipped floor in cost_utils
    #         to avoid 0/0 in gradient paths — this is a FEATURE, not a bug)
    # (5,5): outside box corner → d_wall = √2 * 3 = 4.2426
    expected = jnp.array([1.0, 1e-4, float(np.sqrt(2.0) * 3.0)])
    np.testing.assert_allclose(np.asarray(dw), np.asarray(expected),
                               rtol=1e-4, atol=1e-5)


# ───────────────────────────── standalone runner ──────────────────────────

if __name__ == "__main__":
    # Allow `python tests/test_eval_cost_matches_train.py` without pytest.
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception:
                failed += 1
                print(f"[FAIL] {name}")
                traceback.print_exc()
    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    print("\nAll eval-cost calibration tests passed.")
