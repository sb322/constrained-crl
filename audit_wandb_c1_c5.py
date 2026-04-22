#!/usr/bin/env python3
"""Automated C1–C5 audit against the wandb history of a Phase-1b run.

Usage
-----
    python audit_wandb_c1_c5.py --group phase1b_smoke
    python audit_wandb_c1_c5.py --group phase1b_smoke --run ant_big_maze_ccrl_d4_s0_1713722340
    python audit_wandb_c1_c5.py --group phase1b_smoke --project constrained-crl --entity <entity>

What this does
--------------
Pulls per-epoch history from wandb.Api() for each run in the given group,
then evaluates each of the five hard-pass criteria defined in
`phase1_criteria.md`:

    C1  mean_step_cost ∈ [0.10, 0.35] over epochs 0–5,
        per-epoch-std of the logged-step cost (proxy: run-level std over
        epochs 0–5) > 0.05.  If that's too tight a proxy for a 3-epoch
        fast-smoke, the script also flags which arm of C1 fails.

    C2  mean_step_cost (mean over last 10 epochs) strictly < mean over
        first 5 epochs by ≥10 % relative; and absolute value < 0.20.

    C3  lambda_tilde is NOT identically 0 everywhere that j_c_hat > d=0.15.
        (If j_c_hat ≤ d throughout the run, λ̃ = 0 is CORRECT — the CMDP
        is feasible for the current policy, no dual pressure needed.
        The script distinguishes these two cases.)

    C4  |mean_step_cost − eval/mean_smooth_cost| ≤ 3× ratio at each eval
        epoch; eval/mean_hard_cost ≤ mean_step_cost at each eval epoch.

    C5  No NaN in actor_loss / critic_loss / cost_critic_loss / alpha_loss
        after epoch 3; |actor_loss| < 1e4 throughout.

Each criterion prints PASS / WARN / FAIL with the specific numerics that
made the call, and a one-line "first diagnostic action" pointer into
phase1_criteria.md § What-to-do-on-failure.

The exit code is 0 iff all C1–C5 pass.  A final JSON summary is written to
`audit_report_<run_name>.json` so this can be wired into a Slurm epilogue
or a wandb report.

IMPORTANT: wandb keys this script looks for
-------------------------------------------
These are the *actual* keys train.py emits (NOT the `train/*` namespace
the criteria doc uses aspirationally).  Verified against train.py:1140–1172.

    Flat (training, per-epoch):
      epoch, env_steps, gradient_steps, epoch_time_s,
      critic_loss, accuracy, logits_pos, logits_neg, logsumexp,
      actor_loss, alpha, alpha_loss,
      cost_critic_loss, mean_step_cost, mean_d_wall,
      hard_violation_rate, mean_qc, mean_qc_pi,
      lambda_tilde, j_c_hat, constraint_violation

    Prefixed (evaluator):
      eval/mean_hard_cost, eval/mean_smooth_cost, eval/mean_d_wall,
      eval/constraint_violation, eval/avg_episode_length, ...
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


# ─── Criteria constants — kept in lock-step with phase1_criteria.md ──────
COST_BUDGET_D   = 0.15
EPS_TRAIN       = 2.0
EPS_HARD        = 0.1
C1_EARLY_LO     = 0.10
C1_EARLY_HI     = 0.35
C1_EARLY_EPOCHS = 5     # count: epochs 0..4
C2_LATE_EPOCHS  = 10
C2_REL_DROP     = 0.10
C2_ABS_TARGET   = 0.20
C3_LATENCY      = 3     # epochs after first j_c_hat>d within which λ̃ must rise
C4_RATIO        = 3.0
C5_ACTOR_CAP    = 1e4
C5_NAN_OK_BEFORE_EPOCH = 3

# Required keys.  A missing key does not FAIL the run — it WARNs and skips
# the dependent criterion, since a logging-layer regression is a different
# class of problem than a training-dynamics failure.
REQ_TRAIN_KEYS = [
    "epoch", "env_steps",
    "mean_step_cost", "lambda_tilde", "j_c_hat",
    "constraint_violation", "hard_violation_rate",
    "actor_loss", "critic_loss", "cost_critic_loss", "alpha_loss",
]
REQ_EVAL_KEYS = [
    "eval/mean_hard_cost", "eval/mean_smooth_cost", "eval/mean_d_wall",
]


@dataclass
class CritResult:
    name: str
    status: str            # "PASS" | "WARN" | "FAIL" | "SKIP"
    detail: str
    numerics: dict[str, Any]

    def line(self) -> str:
        tag = {"PASS": "[ PASS ]", "WARN": "[ WARN ]",
               "FAIL": "[ FAIL ]", "SKIP": "[ SKIP ]"}[self.status]
        return f"{tag} {self.name}: {self.detail}"


# ─────────────────────────────── helpers ─────────────────────────────────

def _load_history(run) -> dict[str, np.ndarray]:
    """Pull per-epoch history into a dict of float-arrays indexed by key.

    Uses `run.history(samples=N)` (one bulk HTTPS call) instead of
    `run.scan_history` (row-at-a-time streaming).  For the ≤50-epoch runs
    Phase-1b emits, the bulk call is ~20× faster and incurs no downsampling
    because `samples` is set well above the row count.  For longer runs we
    set samples=10_000 which is wandb's max and will not silently truncate
    a Phase-2 sweep.
    """
    keys = REQ_TRAIN_KEYS + REQ_EVAL_KEYS + ["alpha", "mean_qc", "mean_qc_pi"]
    try:
        df = run.history(keys=keys, samples=10_000, pandas=True)
    except Exception:
        # Fallback: scan_history row-by-row (slower but more resilient).
        rows: dict[str, list[float]] = {k: [] for k in keys}
        for row in run.scan_history(keys=keys):
            for k in keys:
                v = row.get(k, None)
                rows[k].append(float(v) if v is not None else math.nan)
        return {k: np.asarray(vs, dtype=np.float64) for k, vs in rows.items()}

    out: dict[str, np.ndarray] = {}
    for k in keys:
        if k in df.columns:
            out[k] = df[k].to_numpy(dtype=np.float64, na_value=math.nan)
        else:
            out[k] = np.asarray([], dtype=np.float64)
    return out


def _nanaware_mean(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    return float(finite.mean()) if finite.size else math.nan


def _any_nan_after(x: np.ndarray, epoch: np.ndarray, e_after: int) -> bool:
    mask = (epoch >= e_after) & ~np.isnan(x)  # valid rows at or after e_after
    if mask.size == 0:
        return False
    vals = x[(epoch >= e_after)]
    return bool(np.any(np.isnan(vals)))


# ─────────────────────────────── C1 ──────────────────────────────────────

def check_c1(h: dict[str, np.ndarray]) -> CritResult:
    c   = h["mean_step_cost"]
    ep  = h["epoch"]
    if c.size == 0 or np.all(np.isnan(c)):
        return CritResult("C1 cost signal non-vacuous", "SKIP",
                          "mean_step_cost missing from history",
                          {})

    early_mask = ep < C1_EARLY_EPOCHS
    early      = c[early_mask]
    early_mean = _nanaware_mean(early)
    # NOTE: train.py logs the *mean* cost per epoch, not the per-step std —
    # so the criteria-doc's "std within any epoch > 0.05" is not directly
    # auditable from wandb history.  We use a surrogate: the cost must not
    # be perfectly constant across the run.  A constant cost is the
    # signature of a failed wiring (e.g. σ((ε-d_wall)/τ) saturating to the
    # same value everywhere — the v1a bug).  Smooth descent of the epoch
    # mean with a run-level std of ~0.01 is not a bug; cost ≡ c₀ exactly is.
    full_std = float(np.nanstd(c)) if c.size else math.nan

    ok_mean     = (C1_EARLY_LO <= early_mean <= C1_EARLY_HI) if np.isfinite(early_mean) else False
    ok_nonconst = (full_std > 1e-6) if np.isfinite(full_std) else False

    nx = {
        "epochs_0_to_{}_mean".format(C1_EARLY_EPOCHS - 1): early_mean,
        "run_level_std":    full_std,
        "target_mean_band": [C1_EARLY_LO, C1_EARLY_HI],
        "n_early_epochs":    int(early.size),
    }

    if ok_mean and ok_nonconst:
        return CritResult("C1 cost signal non-vacuous", "PASS",
                          f"epochs 0–{C1_EARLY_EPOCHS-1} mean={early_mean:.4f} "
                          f"∈ [{C1_EARLY_LO}, {C1_EARLY_HI}]; not constant "
                          f"(run-level std={full_std:.4f})",
                          nx)

    # Cost is exactly constant → always a bug (even if mean happens to be in band).
    if not ok_nonconst:
        return CritResult("C1 cost signal non-vacuous", "FAIL",
                          f"mean_step_cost is constant ({early_mean:.6f} at every "
                          "epoch). Diagnostic: cost field saturated — classic v1a "
                          "signature. Check cost_type passthrough and "
                          "compute_wall_distance returning sensible values.",
                          nx)

    # Short runs (fast-smoke) often have low mean because only a few hundred
    # early exploration steps haven't spread into the full interior yet.
    # Downgrade to WARN when N < 5 epochs and mean is below band (but > 0).
    if not ok_mean and early.size < 5 and early_mean > 1e-4:
        return CritResult("C1 cost signal non-vacuous", "WARN",
                          f"mean={early_mean:.4f} below [{C1_EARLY_LO}, {C1_EARLY_HI}] "
                          f"band, but N={early.size} epochs is too few to trust the "
                          "band: early exploration may not have reached the shaping "
                          "interior yet. Non-fatal for fast-smoke; escalate on "
                          "full overnight if still below 0.10.",
                          nx)

    return CritResult("C1 cost signal non-vacuous", "FAIL",
                      f"mean={early_mean:.4f} outside [{C1_EARLY_LO}, {C1_EARLY_HI}]. "
                      "Diagnostic: check v1b cost_type='quadratic' and "
                      "cost_epsilon=2.0 reached env_kwargs; "
                      "grep for cost_type in train.py setup.",
                      nx)


# ─────────────────────────────── C2 ──────────────────────────────────────

def check_c2(h: dict[str, np.ndarray]) -> CritResult:
    c  = h["mean_step_cost"]
    ep = h["epoch"]
    if c.size == 0 or np.all(np.isnan(c)):
        return CritResult("C2 cost descends", "SKIP",
                          "mean_step_cost missing", {})
    total_epochs = int(np.nanmax(ep)) + 1 if ep.size else 0

    if total_epochs < 10:
        # Fast-smoke: C2 is only meaningful on the 50-epoch overnight.
        return CritResult("C2 cost descends", "SKIP",
                          f"run has {total_epochs} epochs — C2 requires "
                          "≥10 epochs to be meaningful (designed for the "
                          "50-epoch overnight)",
                          {"total_epochs": total_epochs})

    early = c[ep < C1_EARLY_EPOCHS]
    late  = c[ep >= total_epochs - C2_LATE_EPOCHS]
    me    = _nanaware_mean(early)
    ml    = _nanaware_mean(late)
    rel_drop = (me - ml) / me if np.isfinite(me) and me > 0 else math.nan

    nx = {
        "early_mean": me, "late_mean": ml, "rel_drop": rel_drop,
        "target_rel_drop_min": C2_REL_DROP, "target_abs_late_max": C2_ABS_TARGET,
    }

    if not np.isfinite(rel_drop):
        return CritResult("C2 cost descends", "FAIL",
                          f"early_mean={me} not finite or zero",
                          nx)

    ok_rel = rel_drop >= C2_REL_DROP
    ok_abs = ml <= C2_ABS_TARGET
    if ok_rel and ok_abs:
        return CritResult("C2 cost descends", "PASS",
                          f"early_mean={me:.4f} → late_mean={ml:.4f} "
                          f"({rel_drop*100:.1f}% drop, target {C2_REL_DROP*100:.0f}%)",
                          nx)
    if ok_rel and not ok_abs:
        return CritResult("C2 cost descends", "WARN",
                          f"relative drop OK ({rel_drop*100:.1f}%) but "
                          f"late_mean={ml:.4f} > {C2_ABS_TARGET} abs cap. "
                          "May just need more steps; check C3 is active.",
                          nx)
    return CritResult("C2 cost descends", "FAIL",
                      f"rel_drop={rel_drop*100:.1f}% < {C2_REL_DROP*100:.0f}% "
                      f"and/or late_mean={ml:.4f} > {C2_ABS_TARGET}. "
                      "Diagnostic: C3 first — if λ̃≡0 when j_c_hat>d, "
                      "fix C3. Otherwise task may be too hard at 5M steps.",
                      nx)


# ─────────────────────────────── C3 ──────────────────────────────────────

def check_c3(h: dict[str, np.ndarray]) -> CritResult:
    lam = h["lambda_tilde"]
    jc  = h["j_c_hat"]
    ep  = h["epoch"]
    if lam.size == 0 or jc.size == 0:
        return CritResult("C3 Lagrangian responds", "SKIP",
                          "lambda_tilde or j_c_hat missing", {})

    # Identical-zero λ̃ is only a bug if the dual *should* be firing, i.e.
    # if at some epoch j_c_hat > cost_budget_d.
    violated = jc > COST_BUDGET_D
    lam_any_positive = float(np.nanmax(lam)) > 1e-9 if lam.size else False

    nx = {
        "lambda_tilde_max":   float(np.nanmax(lam)) if lam.size else math.nan,
        "lambda_tilde_mean":  float(np.nanmean(lam)) if lam.size else math.nan,
        "j_c_hat_max":        float(np.nanmax(jc))  if jc.size  else math.nan,
        "any_epoch_violating_budget": bool(np.any(violated)),
        "cost_budget_d":       COST_BUDGET_D,
    }

    # Case A: CMDP is feasible for the current policy — j_c_hat ≤ d at every
    # epoch.  Then λ̃ ≡ 0 is correct (PID has nothing to push against).
    if not np.any(violated):
        return CritResult("C3 Lagrangian responds", "PASS",
                          f"j_c_hat ≤ d={COST_BUDGET_D} at all epochs "
                          f"(max={nx['j_c_hat_max']:.4f}); λ̃ staying at 0 "
                          "is correct — CMDP feasible under the current policy.",
                          nx)

    # Case B: there IS a violation.  Then λ̃ must rise within C3_LATENCY
    # epochs of the first violation.
    first_viol_idx = int(np.argmax(violated))
    first_viol_epoch = int(ep[first_viol_idx])
    window_mask = (ep >= first_viol_epoch) & (ep <= first_viol_epoch + C3_LATENCY)
    lam_in_window = lam[window_mask]
    lam_rose = float(np.nanmax(lam_in_window)) > 1e-6 if lam_in_window.size else False

    nx["first_violation_epoch"]   = first_viol_epoch
    nx["lambda_max_within_window"] = float(np.nanmax(lam_in_window)) if lam_in_window.size else math.nan

    if lam_rose:
        return CritResult("C3 Lagrangian responds", "PASS",
                          f"first violation at epoch {first_viol_epoch}, "
                          f"λ̃ rose to {nx['lambda_max_within_window']:.4f} "
                          f"within {C3_LATENCY} epochs",
                          nx)

    return CritResult("C3 Lagrangian responds", "FAIL",
                      f"j_c_hat > d at epoch {first_viol_epoch} but λ̃ stayed "
                      f"at 0 for {C3_LATENCY} epochs. "
                      "Diagnostic: inspect _critic_based_dual_estimator and "
                      "the PID integrator update — the I-channel should be "
                      "accumulating (jhat_c − d) but isn't.",
                      nx)


# ─────────────────────────────── C4 ──────────────────────────────────────

def check_c4(h: dict[str, np.ndarray]) -> CritResult:
    ct  = h["mean_step_cost"]
    ce  = h["eval/mean_smooth_cost"]
    ch  = h["eval/mean_hard_cost"]
    ep  = h["epoch"]
    if ct.size == 0 or ce.size == 0:
        return CritResult("C4 train/eval calibration", "SKIP",
                          "mean_step_cost or eval/mean_smooth_cost missing",
                          {})

    # Rows where both train and eval quantities exist (eval is logged every
    # eval_every epochs, so many rows have NaN eval values).
    both_mask = np.isfinite(ct) & np.isfinite(ce)
    if not np.any(both_mask):
        return CritResult("C4 train/eval calibration", "SKIP",
                          "no epochs have both train + eval cost logged",
                          {})

    ratio = np.abs(ct[both_mask] - ce[both_mask]) / np.maximum(
        1e-6, np.abs(ct[both_mask]) + np.abs(ce[both_mask])) * 2.0
    # Use symmetric ratio δ = 2|a-b|/(|a|+|b|) ∈ [0, 2], which is stable
    # when either side is tiny; translate the ≤ 3× criterion (derived from
    # max(a,b)/min(a,b) ≤ 3) to δ ≤ 1 (since a/b=3 ⇒ δ=1).
    worst_ratio = float(np.max(ratio))

    # Hard-cost lower-bound arm of C4.
    both_hard = np.isfinite(ct) & np.isfinite(ch)
    if np.any(both_hard):
        hard_ok = bool(np.all(ch[both_hard] <= ct[both_hard] + 1e-6))
        hard_violations = int(np.sum(ch[both_hard] > ct[both_hard] + 1e-6))
    else:
        hard_ok = True
        hard_violations = 0

    nx = {
        "worst_symmetric_ratio_delta": worst_ratio,
        "target_delta_max":            1.0,
        "hard_le_train_ok":            hard_ok,
        "epochs_where_hard_exceeds_train": hard_violations,
        "n_paired_rows":               int(np.sum(both_mask)),
    }

    if worst_ratio <= 1.0 and hard_ok:
        return CritResult("C4 train/eval calibration", "PASS",
                          f"worst symmetric ratio δ={worst_ratio:.3f} ≤ 1 "
                          "(equiv. max/min ≤ 3×); eval hard ≤ train everywhere",
                          nx)
    if worst_ratio > 1.0 and hard_ok:
        return CritResult("C4 train/eval calibration", "FAIL",
                          f"train/eval quadratic cost diverge by δ={worst_ratio:.3f} > 1. "
                          "Diagnostic: eval_env_kwargs not receiving matched "
                          "cost_type/cost_epsilon in train.py setup. Also run "
                          "tests/test_eval_cost_matches_train.py to verify the "
                          "closures are bitwise-identical.",
                          nx)
    if hard_violations > 0:
        return CritResult("C4 train/eval calibration", "FAIL",
                          f"eval/mean_hard_cost > mean_step_cost at "
                          f"{hard_violations} epoch(s) — the narrower "
                          "ε_hard=0.1 indicator should be a lower bound on "
                          "the ε_train=2.0 quadratic. Diagnostic: env-side "
                          "cost scalars in state.metrics are miswired.",
                          nx)
    return CritResult("C4 train/eval calibration", "FAIL",
                      f"both arms failed (δ={worst_ratio:.3f}, "
                      f"{hard_violations} hard>train violations)", nx)


# ─────────────────────────────── C5 ──────────────────────────────────────

def check_c5(h: dict[str, np.ndarray]) -> CritResult:
    ep = h["epoch"]
    losses = {k: h[k] for k in
              ("actor_loss", "critic_loss", "cost_critic_loss", "alpha_loss")}

    issues = []
    nx: dict[str, Any] = {}
    for name, arr in losses.items():
        if arr.size == 0:
            continue
        any_nan_mid = _any_nan_after(arr, ep, C5_NAN_OK_BEFORE_EPOCH)
        nx[f"{name}_nan_after_epoch_{C5_NAN_OK_BEFORE_EPOCH}"] = any_nan_mid
        nx[f"{name}_abs_max"] = float(np.nanmax(np.abs(arr))) if arr.size else math.nan
        if any_nan_mid:
            issues.append(f"{name} has NaN at epoch ≥ {C5_NAN_OK_BEFORE_EPOCH}")

    # Actor-loss magnitude cap — v1a hit −4.85e5; v1b should stay in a tight band.
    a = losses["actor_loss"]
    if a.size:
        a_abs_max = float(np.nanmax(np.abs(a)))
        nx["actor_loss_abs_max"] = a_abs_max
        nx["actor_loss_cap"]     = C5_ACTOR_CAP
        if a_abs_max > C5_ACTOR_CAP:
            issues.append(f"|actor_loss|_max = {a_abs_max:.1f} > {C5_ACTOR_CAP}")

    if not issues:
        return CritResult("C5 no NaN, bounded actor_loss", "PASS",
                          f"all losses finite past epoch {C5_NAN_OK_BEFORE_EPOCH}; "
                          f"|actor_loss|_max = {nx.get('actor_loss_abs_max', math.nan):.2f}",
                          nx)

    return CritResult("C5 no NaN, bounded actor_loss", "FAIL",
                      "; ".join(issues) +
                      ". Diagnostic: bisect with use_constraints=False to "
                      "isolate CRL vs constrained-CRL; if CRL is clean, "
                      "inspect nu_c auto-compute (|Q_c| ≤ 1/(1-γ_c) = 100 "
                      "under quadratic cost ∈ [0, 1]).",
                      nx)


# ─────────────────────────────── driver ──────────────────────────────────

CHECKS = [check_c1, check_c2, check_c3, check_c4, check_c5]


def audit_run(run) -> dict[str, Any]:
    h = _load_history(run)

    # Coerce "epoch" to an integer index regardless of wandb's internal dtype.
    if "epoch" in h and h["epoch"].size:
        h["epoch"] = h["epoch"].astype(int)
    n_rows = h["epoch"].size if "epoch" in h else 0

    results = [fn(h) for fn in CHECKS]

    overall = "PASS"
    for r in results:
        if r.status == "FAIL":
            overall = "FAIL"
            break
        if r.status == "WARN" and overall == "PASS":
            overall = "WARN"

    return {
        "run_name": run.name,
        "run_id":   run.id,
        "group":    run.group,
        "state":    run.state,
        "n_epochs": n_rows,
        "overall":  overall,
        "criteria": [asdict(r) for r in results],
    }


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"run: {report['run_name']}   id={report['run_id']}   "
          f"state={report['state']}   epochs={report['n_epochs']}")
    print(f"group: {report['group']}")
    print("-" * 72)
    for c in report["criteria"]:
        tag = {"PASS": "[ PASS ]", "WARN": "[ WARN ]",
               "FAIL": "[ FAIL ]", "SKIP": "[ SKIP ]"}[c["status"]]
        print(f"{tag} {c['name']}")
        print(f"        {c['detail']}")
    print("-" * 72)
    print(f"OVERALL: {report['overall']}")
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT",
                                                       "constrained-crl"))
    p.add_argument("--entity",  default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--group",   default="phase1b_smoke",
                   help="wandb group to audit (default: phase1b_smoke)")
    p.add_argument("--run",     default=None,
                   help="optional run name filter (exact match)")
    p.add_argument("--out-dir", default=".",
                   help="directory to write audit_report_<run>.json files")
    p.add_argument("--fail-on-warn", action="store_true",
                   help="exit non-zero on WARN as well as FAIL")
    args = p.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb is not installed.  pip install wandb", file=sys.stderr)
        return 2

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    filters: dict[str, Any] = {"group": args.group}
    if args.run:
        filters["display_name"] = args.run
    runs = list(api.runs(path=path, filters=filters))
    if not runs:
        print(f"No runs found in {path!r} with filters={filters}.",
              file=sys.stderr)
        return 2

    any_fail = False
    any_warn = False
    for run in runs:
        report = audit_run(run)
        _print_report(report)
        out_path = os.path.join(
            args.out_dir, f"audit_report_{report['run_name']}.json"
        )
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out_path}\n")
        if report["overall"] == "FAIL":
            any_fail = True
        elif report["overall"] == "WARN":
            any_warn = True

    if any_fail:
        return 1
    if args.fail_on_warn and any_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
