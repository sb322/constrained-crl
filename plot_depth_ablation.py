#!/usr/bin/env python
"""
Parse SLURM logs from the SR-CPO depth-ablation array job and emit a
supervisor-ready multi-panel figure.

Reads  logs/ccrl_depth_<JOBID>_*.out  files, extracts the per-epoch
metric lines of the form

    [  N/200] steps=X | c_loss=A acc=B | a_loss=C | t=Ds
            hard_viol=E cost=F λ=G Ĵ_c=H Qc=I

groups runs by (depth, seed), and plots six panels on one figure:

    (1) hard_viol_rate vs env_steps     — constraint-violation dynamics
    (2) mean step cost  vs env_steps    — raw cost signal
    (3) Ĵ_c (cost-critic) vs env_steps  — w/ budget line d
    (4) λ (Lagrange mul.) vs env_steps  — dual dynamics
    (5) c_loss (reward critic)          — learning sanity
    (6) a_loss (actor)                  — learning sanity

Seed-0 runs are solid lines, seed-42 runs dashed; color encodes depth.

Usage:
    python plot_depth_ablation.py logs/ccrl_depth_949311_*.out \
        --out   depth_ablation_949311.png \
        --csv   depth_ablation_949311.csv \
        --budget 0.1
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Regexes (Unicode-safe for λ and Ĵ_c) ──────────────────────────────────
LINE1_RE = re.compile(
    r"\[\s*(\d+)\s*/\s*\d+\]\s+steps=([\d,]+)\s+\|\s+"
    r"c_loss=([-\d.eE+]+)\s+acc=([-\d.eE+]+)\s+\|\s+"
    r"a_loss=([-\d.eE+]+)\s+\|\s+t=([-\d.eE+]+)s"
)
# Line 2 has Unicode chars (λ, Ĵ_c) whose exact encoding varies by terminal.
# Detect the line by its reliable prefix and extract numeric fields in order:
#   hard_viol=X cost=Y <unicode>=Z <unicode>=W Qc=V
FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
LINE2_PREFIX_RE = re.compile(r"hard_viol\s*=")
LINE2_NUMS_RE = re.compile(FLOAT)
HDR_RE = re.compile(r"depth\s*=\s*(\d+)\s+skip\s*=\s*\d+\s+seed\s*=\s*(\d+)")


def parse_log(path, debug=False):
    """Return dict(depth, seed, rows=[...], path) or None if unparseable."""
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    hdr = HDR_RE.search(text)
    if not hdr:
        print(f"  [skip] {path}: no 'depth=... seed=...' header",
              file=sys.stderr)
        if debug:
            # Show what headers DO look like
            for ln in text.splitlines()[:40]:
                if "depth" in ln or "seed" in ln:
                    print(f"    candidate header line: {ln!r}",
                          file=sys.stderr)
        return None
    depth = int(hdr.group(1))
    seed = int(hdr.group(2))

    rows = []
    lines = text.splitlines()
    line1_count = 0
    line2_count = 0
    for i, line in enumerate(lines):
        m1 = LINE1_RE.search(line)
        if not m1:
            continue
        line1_count += 1
        # Companion metric line may be the next line or the one after
        m2_line = None
        for j in (1, 2):
            if i + j < len(lines) and LINE2_PREFIX_RE.search(lines[i + j]):
                m2_line = lines[i + j]
                break
        if m2_line is None:
            continue
        line2_count += 1

        nums = LINE2_NUMS_RE.findall(m2_line)
        if len(nums) < 5:
            if debug:
                print(f"    [warn] only {len(nums)} numbers on line-2: "
                      f"{m2_line!r}", file=sys.stderr)
            continue
        hard_viol, cost, lam, j_c, qc = [float(n) for n in nums[:5]]

        rows.append(dict(
            epoch=int(m1.group(1)),
            steps=int(m1.group(2).replace(",", "")),
            c_loss=float(m1.group(3)),
            acc=float(m1.group(4)),
            a_loss=float(m1.group(5)),
            t=float(m1.group(6)),
            hard_viol=hard_viol,
            cost=cost,
            lam=lam,
            j_c=j_c,
            qc=qc,
        ))
    if debug and not rows:
        print(f"  [debug] {os.path.basename(path)}: "
              f"line1_matches={line1_count}  line2_matches={line2_count}",
              file=sys.stderr)
        # Show the first few candidate lines so we can see the actual format
        for ln in lines:
            if "[ " in ln and "/" in ln and "steps=" in ln:
                print(f"    sample line-1: {ln!r}", file=sys.stderr)
                break
        for ln in lines:
            if "hard_viol" in ln:
                print(f"    sample line-2: {ln!r}", file=sys.stderr)
                break
    return dict(depth=depth, seed=seed, rows=rows, path=path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+",
                    help="glob(s) of ccrl_depth_*.out files")
    ap.add_argument("--out", default="depth_ablation.png")
    ap.add_argument("--csv", default="depth_ablation.csv")
    ap.add_argument("--budget", type=float, default=0.1,
                    help="cost budget d (drawn as reference line)")
    ap.add_argument("--debug", action="store_true",
                    help="print parsing diagnostics for failed files")
    args = ap.parse_args()

    runs = []
    for pat in args.logs:
        for path in sorted(glob.glob(pat)):
            r = parse_log(path, debug=args.debug)
            if r is None or not r["rows"]:
                if args.debug and r is not None:
                    print(f"  [skip] {os.path.basename(path)}: "
                          f"0 epoch-rows parsed", file=sys.stderr)
                continue
            runs.append(r)
            print(f"  parsed {os.path.basename(path)}: "
                  f"depth={r['depth']} seed={r['seed']} "
                  f"epochs={len(r['rows'])}")

    if not runs:
        print("FATAL: no runs parsed. Check that log files exist and the regex "
              "matches your log format.", file=sys.stderr)
        sys.exit(1)

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "seed", "epoch", "steps", "c_loss", "acc",
                    "a_loss", "t_elapsed_s", "hard_viol", "cost", "lambda",
                    "j_c", "qc"])
        for r in runs:
            for row in r["rows"]:
                w.writerow([r["depth"], r["seed"], row["epoch"], row["steps"],
                            row["c_loss"], row["acc"], row["a_loss"],
                            row["t"], row["hard_viol"], row["cost"],
                            row["lam"], row["j_c"], row["qc"]])
    print(f"wrote {args.csv}")

    # ── Figure ─────────────────────────────────────────────────────────────
    by_depth = defaultdict(list)
    for r in runs:
        by_depth[r["depth"]].append(r)

    depths = sorted(by_depth.keys())
    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / max(1, len(depths) - 1)) for i, d in enumerate(depths)}

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    panels = [
        ("hard_viol", "Hard violation rate",         None,                 False),
        ("cost",      "Mean step cost",              None,                 False),
        ("j_c",       r"Cost-critic estimate  $\hat{J}_c$", args.budget,    False),
        ("lam",       r"Lagrange multiplier  $\lambda$",    None,           False),
        ("c_loss",    "Reward-critic loss",          None,                 True),
        ("a_loss",    "Actor loss",                  None,                 False),
    ]

    for ax, (key, ylabel, budget, logy) in zip(axes.flat, panels):
        for d in depths:
            for r in by_depth[d]:
                xs = [row["steps"] for row in r["rows"]]
                ys = [row[key] for row in r["rows"]]
                ls = "-" if r["seed"] == 0 else "--"
                lbl = f"depth={d}, seed={r['seed']}"
                ax.plot(xs, ys, ls, color=colors[d], label=lbl,
                        alpha=0.88, linewidth=1.6, marker="o", markersize=3)
        if budget is not None:
            ax.axhline(budget, color="red", linestyle=":", linewidth=1.3,
                       label=f"budget d={budget}", zorder=0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    # One legend in the first panel
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(h, l, fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle(
        "SR-CPO depth ablation — Ant Big Maze "
        f"(budget d={args.budget}, PID=(0.1, 0.003, 0.001), 20M-step target)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # ── Summary table printed to stdout ────────────────────────────────────
    print("\n===== FINAL-EPOCH SUMMARY =====")
    print(f"{'depth':>6} {'seed':>5} {'epochs':>7} "
          f"{'steps':>9} {'hard_v':>8} {'cost':>8} "
          f"{'λ':>8} {'Ĵ_c':>8} {'c_loss':>8}")
    for d in depths:
        for r in sorted(by_depth[d], key=lambda x: x["seed"]):
            last = r["rows"][-1]
            print(f"{d:>6} {r['seed']:>5} {len(r['rows']):>7} "
                  f"{last['steps']:>9d} {last['hard_viol']:>8.4f} "
                  f"{last['cost']:>8.4f} {last['lam']:>8.4f} "
                  f"{last['j_c']:>8.4f} {last['c_loss']:>8.4f}")


if __name__ == "__main__":
    main()
