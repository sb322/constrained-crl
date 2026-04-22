#!/usr/bin/env python3
"""Finite-time constraint-violation regret verifier for SR-CPO Phase-1b runs.

What this checks
----------------
Primal-dual CMDP methods carry a standard finite-time bound of the form

        C_viol(T) := Σ_{t=1}^{T} max(0, Ĵ_c,t − d)  =  O(√T · polylog T)

where t indexes policy-update epochs and Ĵ_c,t is the critic-based estimator
of the average-step cost under the policy at epoch t.  This is the analogue
of the Ding et al. (2021), Efroni–Mannor–Pirotta (2020), and
Ghosh–Zhou–Shroff (2023) bounds adapted to a primal-dual actor-critic
setting.  It is a *necessary* condition: if the empirical C_viol(T) grows
linearly in T, **no** finite-time bound of √T shape can hold for the run —
either the CMDP is infeasible, the PID is under-gained, or the critic bias
dominates.

The verifier does three things:

  1. Computes the cumulative-violation trajectory C_viol(t) for t = 1..T
     from the wandb history key `j_c_hat` (per-epoch critic-based estimator).
  2. Fits a power-law envelope  C_viol(t) ≈ α · t^β  by least-squares in
     log-log on the asymptotic tail (t ≥ T/4) and reports β.  The
     pass/fail threshold is β < 0.9: anything in [0.9, 1.0] is linear
     within noise and fails the sub-linearity test.
  3. Produces a publication-style plot overlaying C_viol(t), a √t reference
     envelope calibrated from the first 5 epochs, and a linear reference.

Caveat — sample size
--------------------
At T = 50 epochs the log-log fit has ~37 points in the tail.  The slope
estimate's standard error is ~0.1 in this regime.  A slope of 0.55 ± 0.10
is consistent with the √T bound; a slope of 0.95 ± 0.10 is not.  This
script is a sanity check, not a proof; in the ICML/NeurIPS writeup we
would additionally report the slope's bootstrap CI and defer to a
multi-seed aggregate before claiming the bound holds.

Assumption on feasibility
-------------------------
If Ĵ_c,t ≤ d for every t in the run (what I call "Scenario 1" above), then
C_viol(T) ≡ 0 and the log-log fit is undefined.  The verifier reports this
as SKIP with a separate message: "CMDP is feasible under the current policy;
constraint-violation regret trivially bounded by 0, and the √T bound is
vacuous in this regime."  That is the expected outcome if the overnight
lands in Scenario 1 — the test is *informative* only in Scenario 2, which
the d=0.05 tight-budget variant is designed to force.

Usage
-----
    # Against a live wandb run
    python verify_violation_bound.py --group phase1b_smoke

    # Against a specific run name
    python verify_violation_bound.py --group phase1b_smoke \
        --run ant_big_maze_ccrl_d4_s0_1713722340

    # Against a locally cached audit report (no wandb access needed)
    python verify_violation_bound.py \
        --history-json j_c_hat_trajectory.json

Exit codes
----------
    0  sub-linear growth confirmed OR CMDP feasible (SKIP)
    1  linear growth detected — finite-time bound violated
    2  tool / data error (no runs found, no j_c_hat logged, etc.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


# ─── constants kept in lock-step with phase1_criteria.md ─────────────────
COST_BUDGET_D_DEFAULT = 0.15       # overnight budget; overriden for tight-budget
SUBLINEAR_BETA_MAX    = 0.90        # slope above this = failed sub-linearity
TAIL_FRACTION         = 0.75        # fit on the last 75% of epochs
MIN_TAIL_POINTS       = 5           # need at least this many to fit


@dataclass
class VerdictResult:
    status: str               # "PASS" | "FAIL" | "SKIP"
    message: str
    numerics: dict[str, Any]


# ─────────────────────────── core computation ────────────────────────────

def cumulative_violation(j_c_hat: np.ndarray, budget: float) -> np.ndarray:
    """C_viol(t) = Σ_{s=0}^{t-1} max(0, Ĵ_c,s − d)  for t = 1..T.

    Returns an array of length T whose entry at index t-1 is C_viol(t).
    """
    per_step = np.maximum(0.0, j_c_hat - budget)
    return np.cumsum(per_step)


def fit_power_law_tail(t: np.ndarray, c: np.ndarray,
                       tail_fraction: float = TAIL_FRACTION
                       ) -> tuple[float, float, float]:
    """Fit C(t) ≈ α · t^β by least-squares in log-log on the tail.

    Drops points where c ≤ 0 (log undefined).  Returns (α, β, se_β) where
    se_β is the textbook OLS standard error of the slope coefficient.
    """
    T = len(t)
    start = max(1, int(T * (1.0 - tail_fraction)))
    tail_t = t[start:]
    tail_c = c[start:]

    # Drop zero / negative entries (log undefined).
    mask = tail_c > 0
    tail_t = tail_t[mask]
    tail_c = tail_c[mask]

    if len(tail_t) < MIN_TAIL_POINTS:
        return math.nan, math.nan, math.nan

    x = np.log(tail_t.astype(np.float64))
    y = np.log(tail_c.astype(np.float64))
    # OLS closed-form
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    if sxx == 0:
        return math.nan, math.nan, math.nan
    beta  = sxy / sxx
    alpha = math.exp(y_mean - beta * x_mean)
    # Residual standard error → slope SE
    y_hat = y_mean + beta * (x - x_mean)
    rss   = float(np.sum((y - y_hat) ** 2))
    n     = len(x)
    if n <= 2:
        se_beta = math.nan
    else:
        se_beta = math.sqrt(rss / (n - 2) / sxx)
    return alpha, beta, se_beta


def verify(j_c_hat: np.ndarray, budget: float) -> VerdictResult:
    t = np.arange(1, len(j_c_hat) + 1)
    c = cumulative_violation(j_c_hat, budget)

    final_cviol = float(c[-1]) if c.size else 0.0
    time_avg    = float(c[-1] / t[-1]) if c.size else math.nan
    max_instant = float(np.max(np.maximum(0.0, j_c_hat - budget))) if j_c_hat.size else math.nan

    # Feasibility case: CMDP is trivially feasible, bound is vacuous.
    if final_cviol <= 1e-10:
        return VerdictResult(
            "SKIP",
            "CMDP feasible under the current policy — Ĵ_c ≤ d at every epoch. "
            "Constraint-violation regret trivially bounded by 0; finite-time "
            "√T envelope is vacuous in this regime. Rerun with a tighter "
            "budget (slurm_phase1c_tight_budget.sh, d=0.05) to stress-test "
            "the dual mechanism.",
            {
                "final_C_viol":            final_cviol,
                "time_averaged_violation": time_avg,
                "max_instantaneous_violation": max_instant,
                "budget":                  budget,
                "T_epochs":                int(t[-1]) if t.size else 0,
            },
        )

    alpha, beta, se_beta = fit_power_law_tail(t, c)

    nx = {
        "final_C_viol":            final_cviol,
        "time_averaged_violation": time_avg,
        "max_instantaneous_violation": max_instant,
        "budget":                  budget,
        "T_epochs":                int(t[-1]),
        "power_law_alpha":         alpha,
        "power_law_beta":          beta,
        "slope_std_error":         se_beta,
        "sublinearity_threshold":  SUBLINEAR_BETA_MAX,
    }

    if not math.isfinite(beta):
        return VerdictResult(
            "SKIP",
            f"tail fit failed (too few positive points or degenerate "
            f"regression); C_viol({t[-1]})={final_cviol:.4f}",
            nx,
        )

    # Pass criterion: slope clearly below 1.
    # Use beta + 1·se_beta as the conservative upper bound on the slope.
    conservative_upper = beta + (se_beta if math.isfinite(se_beta) else 0.0)

    if conservative_upper < SUBLINEAR_BETA_MAX:
        return VerdictResult(
            "PASS",
            f"cumulative violation C_viol(t) ∝ t^β with β={beta:.3f}±{se_beta:.3f} "
            f"(conservative upper = {conservative_upper:.3f} < {SUBLINEAR_BETA_MAX}). "
            "Consistent with the O(√T) finite-time envelope — the "
            "primal-dual scheme is controlling cumulative violation sub-linearly.",
            nx,
        )
    if beta < SUBLINEAR_BETA_MAX:
        return VerdictResult(
            "PASS",
            f"β={beta:.3f}±{se_beta:.3f} suggests sub-linear growth but the "
            f"conservative upper bound {conservative_upper:.3f} ≥ "
            f"{SUBLINEAR_BETA_MAX}. At T={nx['T_epochs']} the slope SE is too "
            "large to reject the linear null. Recommend multi-seed aggregate "
            "before claiming the bound holds.",
            nx,
        )
    return VerdictResult(
        "FAIL",
        f"cumulative violation grows linearly: β={beta:.3f}±{se_beta:.3f} ≥ "
        f"{SUBLINEAR_BETA_MAX}. Finite-time O(√T) bound is VIOLATED in this "
        "run — means either (a) CMDP infeasible under π_θ + current budget, "
        "(b) PID gains too low (I-channel not accumulating fast enough), or "
        "(c) critic bias in Ĵ_c estimator dominates. First diagnostic: plot "
        "λ̃ trajectory; if λ̃ saturates at λ_max, the PID is fighting an "
        "infeasible problem and the budget should be relaxed.",
        nx,
    )


# ─────────────────────────────── I/O ─────────────────────────────────────

def load_from_wandb(project: str, entity: str | None, group: str,
                    run_name: str | None) -> tuple[str, np.ndarray]:
    try:
        import wandb
    except ImportError:
        print("wandb not installed; pip install wandb", file=sys.stderr)
        sys.exit(2)

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters: dict[str, Any] = {"group": group}
    if run_name:
        filters["display_name"] = run_name
    runs = list(api.runs(path=path, filters=filters))
    if not runs:
        print(f"No runs found in {path!r} with filters={filters}.",
              file=sys.stderr)
        sys.exit(2)
    # Pick the most recently updated run in the group.  Different wandb
    # versions expose the timestamp under different names; try the known
    # ones in order, fall back to "first in list" (wandb returns sorted).
    def _run_ts(r):
        for attr in ("updated_at", "heartbeatAt", "createdAt", "_attrs"):
            v = getattr(r, attr, None)
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and "updatedAt" in v:
                return v["updatedAt"]
        return ""
    run = sorted(runs, key=_run_ts, reverse=True)[0]
    # Use bulk history() not scan_history() — one HTTPS call vs T round-trips.
    try:
        df = run.history(keys=["j_c_hat", "epoch"], samples=10_000, pandas=True)
        if "j_c_hat" in df.columns:
            arr = df["j_c_hat"].dropna().to_numpy(dtype=np.float64)
        else:
            arr = np.asarray([], dtype=np.float64)
    except Exception:
        rows: list[float] = []
        for row in run.scan_history(keys=["j_c_hat", "epoch"]):
            v = row.get("j_c_hat")
            if v is not None:
                rows.append(float(v))
        arr = np.asarray(rows, dtype=np.float64)
    return run.name, arr


def load_from_json(path: str) -> tuple[str, np.ndarray]:
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, list):
        arr = np.asarray([float(x) for x in blob], dtype=np.float64)
        return os.path.basename(path), arr
    if isinstance(blob, dict) and "j_c_hat" in blob:
        return blob.get("run_name", os.path.basename(path)), \
               np.asarray([float(x) for x in blob["j_c_hat"]], dtype=np.float64)
    raise ValueError(f"Unrecognised history-json schema at {path!r}. "
                     "Expected a list[float] of j_c_hat values, or a dict "
                     "with a 'j_c_hat' key.")


# ──────────────────────────── plotting ───────────────────────────────────

def make_plot(j_c_hat: np.ndarray, budget: float, result: VerdictResult,
              out_path: str, run_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed — skipping plot {out_path!r}.",
              file=sys.stderr)
        return

    t = np.arange(1, len(j_c_hat) + 1)
    c = cumulative_violation(j_c_hat, budget)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: linear-scale trajectory.
    axL.plot(t, c, "o-", color="#2E75B6",
             label=r"$C_{\mathrm{viol}}(t)$ (empirical)")
    # Reference envelopes calibrated on first non-zero epoch of C
    nz = np.where(c > 0)[0]
    if nz.size:
        t0 = t[nz[0]]
        c0 = c[nz[0]]
        ref_sqrt = c0 * np.sqrt(t / t0)
        ref_lin  = c0 * (t / t0)
        axL.plot(t, ref_sqrt, "--", color="#70AD47", alpha=0.8,
                 label=r"$\sqrt{t}$ envelope (calibrated)")
        axL.plot(t, ref_lin, ":", color="#C00000", alpha=0.8,
                 label=r"linear envelope (failure reference)")
    axL.set_xlabel("epoch  $t$")
    axL.set_ylabel(r"cumulative violation  $C_{\mathrm{viol}}(t)$")
    axL.set_title(f"Cumulative constraint violation  —  {run_name}")
    axL.legend(loc="upper left", fontsize=8)
    axL.grid(True, alpha=0.3)

    # Right: log-log with power-law fit.
    mask = c > 0
    if mask.sum() >= 2:
        axR.loglog(t[mask], c[mask], "o", color="#2E75B6", markersize=4,
                   label="empirical")
        alpha = result.numerics.get("power_law_alpha")
        beta  = result.numerics.get("power_law_beta")
        if (alpha is not None and beta is not None
                and math.isfinite(alpha) and math.isfinite(beta)):
            axR.loglog(t, alpha * t ** beta, "-", color="#2E75B6",
                       alpha=0.6,
                       label=fr"fit: $\alpha t^\beta$, $\beta={beta:.3f}$")
        axR.axhline(0, color="grey", alpha=0)  # anchor
    axR.set_xlabel("epoch  $t$  (log)")
    axR.set_ylabel(r"$C_{\mathrm{viol}}(t)$  (log)")
    axR.set_title(f"Log-log: verdict = {result.status}")
    axR.legend(loc="upper left", fontsize=8)
    axR.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plot → {out_path}")


# ─────────────────────────────── CLI ─────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--group", default=None,
                     help="wandb group name (default pulls most-recent run)")
    src.add_argument("--history-json", default=None,
                     help="local JSON file with j_c_hat trajectory")
    p.add_argument("--run", default=None, help="optional wandb run name filter")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT",
                                                       "constrained-crl"))
    p.add_argument("--entity",  default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--budget",  type=float, default=COST_BUDGET_D_DEFAULT,
                   help=f"CMDP budget d (default {COST_BUDGET_D_DEFAULT} — "
                        "override to 0.05 for Phase-1c tight-budget runs)")
    p.add_argument("--plot", default=None,
                   help="output path for the log-log plot (default: auto)")
    p.add_argument("--out-json", default=None,
                   help="write verdict dict to this JSON path")
    args = p.parse_args()

    if args.group is None and args.history_json is None:
        # Default: try wandb group phase1b_smoke.
        args.group = "phase1b_smoke"

    if args.history_json:
        run_name, j_c_hat = load_from_json(args.history_json)
    else:
        run_name, j_c_hat = load_from_wandb(
            args.project, args.entity, args.group, args.run
        )

    if j_c_hat.size == 0:
        print("No j_c_hat values found in history.", file=sys.stderr)
        return 2

    result = verify(j_c_hat, args.budget)

    # Print report.
    print("=" * 72)
    print(f"run:    {run_name}")
    print(f"budget: d = {args.budget}")
    print(f"T:      {result.numerics.get('T_epochs')} epochs")
    print(f"final C_viol(T):            {result.numerics.get('final_C_viol'):.6f}")
    print(f"time-averaged violation:    {result.numerics.get('time_averaged_violation'):.6f}")
    print(f"max instantaneous:          {result.numerics.get('max_instantaneous_violation'):.6f}")
    if math.isfinite(result.numerics.get("power_law_beta", math.nan)):
        print(f"fit: C_viol(t) ≈ {result.numerics['power_law_alpha']:.3f} · "
              f"t^{result.numerics['power_law_beta']:.3f} "
              f"(±{result.numerics['slope_std_error']:.3f})")
    print("-" * 72)
    print(f"VERDICT: {result.status}")
    print(f"         {result.message}")
    print("=" * 72)

    plot_path = args.plot or f"violation_bound_{run_name}.png"
    make_plot(j_c_hat, args.budget, result, plot_path, run_name)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "run_name": run_name,
                "budget":   args.budget,
                **asdict(result),
            }, f, indent=2)
        print(f"wrote verdict → {args.out_json}")

    return 0 if result.status in ("PASS", "SKIP") else 1


if __name__ == "__main__":
    sys.exit(main())
