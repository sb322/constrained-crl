# Progress update — Constrained goal-conditioned contrastive RL

**Date:** 2026-04-23
**Project:** SR-CPO (Surrogate-Reward Constrained Policy Optimization) on Brax MJX ant mazes — depth × feasibility in constrained goal-conditioned contrastive RL.

## 1. Canonical setup (no change this sprint)

- Objective: `max_π J_r(π)   s.t.   J_c(π) ≤ d`
- Lagrangian: `L(θ, λ) = J_r − λ·(J_c − d)`
- Actor loss: `L_actor = −Q_CRL(s,a,g)/ν_f + λ·Q_c(s,a)/ν_c + α·log π(a|s,g)`
- Reward critic: contrastive InfoNCE with `φ(s,a)`, `ψ(g)` encoders (Wang et al. 2025 residual block: 4× Dense→LayerNorm→Swish + skip)
- Cost critic: TD(0) Bellman on `Q_c` with compact-support quadratic cost `c(s) = (1 − d_wall/ε_train)²`, `ε_train=2.0`, hard indicator at `ε_hard=0.1`
- `λ`: PID-Lagrangian (Stooke et al. 2020), `Kp=0.1, Ki=0.003, Kd=0.001`, anti-windup on integral, clipped to `[0, λ_max=100]`
- Env: `ant_big_maze` (Brax MJX), 256 envs, depth 4, budget `d = 0.15`.

## 2. Run 1 — depth ablation (prior work, has plot)

Four depths {2, 4, 8, 16} × two seeds, 20M-step target. See `depth_ablation.png`. Finding: at `d = 0.1` with original sigmoid cost the constraint never fully activated; led to the cost-function refactor (compact-support quadratic) that has been the default since.

## 3. Run 2 — Phase-1b overnight, job 963416 (plotted)

50 epochs, 4.76M env-steps. `slurm_phase1b_fastsmoke.sh`, `d=0.15`, depth 4, seed 0.

| metric | final-epoch value |
|---|---|
| `train/actor_loss` | **−4.86 × 10⁵** (diverging throughout) |
| `train/critic_loss` | 4.26 (InfoNCE, well-behaved) |
| `train/lambda_tilde` | **0.0000** (decayed to 0 within first few epochs) |
| `train/jhat_c` | 0.0189 (≈ 13% of budget; strictly feasible) |
| `train/hard_violation_rate` | 0.0000 |

**Narrative beats, all three confirmed:** CMDP feasible (`Ĵ_c < d`), `λ̃ → 0`, actor destabilized (`|a_loss| > 10³`). Figure: `phase1b_story.png`.

## 4. Run 3 — Phase-1d Option-A, job 976587 (FINISHED, 50/50 epochs)

Identical config to Run 2 except three knobs restored to Wang 2025 / JaxGCRL upstream defaults ("restore baseline CRL" bundle):

1. Critic energy: `−‖φ − ψ‖₂  →  ⟨φ, ψ⟩`
2. `normalize_observations: False → True`
3. `ν_f: log N ≈ 5.5  →  1.0`

No change to architecture, PID update, cost function, env, or budget. Verified at shipped-code level (`train.py:60, 120, 600, 641`) before sbatch.

| metric | Phase-1d final | Phase-1b final |
|---|---|---|
| `train/actor_loss` | **−4.67 × 10⁵** (diverged) | −4.86 × 10⁵ |
| `train/critic_loss` | 4.25 | 4.26 |
| `train/lambda_tilde` | 0.0000 | 0.0000 |
| `train/jhat_c` | 0.0191 | 0.0189 |
| `train/hard_violation_rate` | 0.0000 | 0.0000 |

**Pass criterion failed.** `|a_loss| < 10³` held through epoch ~28 (early-epoch dynamics were bounded and roughly geometric, ~1.28× per epoch), crossed 10³ around epoch 30, and kept going to a terminal value essentially indistinguishable from Phase-1b. Run finished cleanly (exit=0, checkpoint saved, no NaN). Figure: `phase1d_final.png`.

## 5. Interpretation — the memo's earlier "H1 leading" call is withdrawn

The 15-epoch mid-run peek showed `a_loss = −142.58`, which read as "bounded with 3 orders of magnitude margin." That reading was wrong: the run was on the same geometric-growth curve as Phase-1b, just at an earlier point on it. At 50 epochs both runs converge to the same O(10⁵) blow-up.

**Diagnosis. Option-A as implemented is incomplete.** The actual bug is visible on line 597 of the shipped `train.py`:

> "Option-A: dot-product (inner-product) energy — ... LayerNorm inside the encoders anchors ‖φ‖, ‖ψ‖, giving bounded ⟨φ, ψ⟩."

That claim is false. LayerNorm operates on the hidden activations *inside* each residual block, normalizing along the feature axis per-sample. It does **not** bound the final encoder output: the trailing Dense layer can rescale the representation by an arbitrary factor, so `‖φ‖, ‖ψ‖` are unconstrained. The raw inner product `⟨φ, ψ⟩` is therefore still unbounded above — the actor minimizes `L = −⟨φ,ψ⟩/ν_f + α·log π` by growing the representations, exactly the same escape route as `−‖φ−ψ‖₂`. The Option-A fix only changed the *functional form* of the energy, not the *norm bound* that actually controls stability.

**What Wang 2025 / JaxGCRL actually do.** They explicitly L2-normalize the final `φ` and `ψ` vectors before the inner product, typically with a temperature `τ`:

```python
sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=-1, keepdims=True)
g_repr  = g_repr  / jnp.linalg.norm(g_repr,  axis=-1, keepdims=True)
logits  = jnp.einsum("ik,jk->ij", sa_repr, g_repr) / tau
```

This puts `⟨φ, ψ⟩ ∈ [−1, 1]` (cosine similarity) and makes the actor's reward-critic term genuinely bounded. Without this step, nothing in the objective penalizes representation magnitude at `λ = 0`.

**Revised hypothesis state.**
- **H1 as specified (three-knob bundle):** rejected. Option-A did not stabilize the actor at `λ = 0`.
- **H1′ (implementation-driven, but requires ALSO row-L2 normalization + temperature):** open, testable in one run.
- **H2 (intrinsic at `λ = 0`):** still consistent with the data, not yet distinguishable from H1′.

These are distinguishable with a **single-knob** follow-up.

## 6. Next steps (revised)

1. **Phase-1f: add row-L2 normalization + fixed temperature `τ = 1/√D` to `φ, ψ` outputs.** Single-knob change on top of current Option-A. Two lines at line 594 (critic_loss_fn) and two lines at line 640 (actor_loss_fn). Preflight: assert `‖φ‖ = ‖ψ‖ = 1` on a batch. Same SLURM template as Phase-1d, same `d = 0.15`, same depth 4, same seed 0. 5M steps, 50 epochs.
   - **If `|a_loss| < 10³` holds to epoch 50:** H1′ wins — Option-A was incomplete, not wrong; resume depth-feasibility program.
   - **If `|a_loss|` still diverges:** H2 wins. Constraints are a necessary CRL stabilizer; this becomes the paper (reframe around emergent-Lagrangian regularization).
2. **Phase-1e tight-budget `d = 0.05`.** Hold until the Phase-1f result is in. If constraints *are* the stabilizer (H2), Phase-1e will automatically confirm by forcing `λ̃ > 0`. If H1′ wins, run Phase-1e on the fixed code.
3. **Depth sweep** — held until the stabilization question is closed either way. Running it on unstable code would answer the wrong question.

One change at a time; no bundling with unrelated cleanups.

## Artifacts for review
- `phase1b_story.png`, `phase1b_story.csv` — Run 2 (50 epochs)
- `phase1d_final.png`, `phase1d_final.csv` — Run 3 (50 epochs, final)
- `phase1d_midrun.png`, `phase1d_midrun.csv` — Run 3 (15-epoch mid-run, preserved for the record; do not cite as evidence)
- `depth_ablation.png` — Run 1
- `plot_phase1b.py` — single-run plotter
- `crl_formulation_verification.md` — Wang 2025 formulation audit
- `slurm_phase1d_option_a.sh` — finished SLURM (preflight asserted three Option-A knobs in shipped `train.py`; did not check for row-L2 normalization — that is the missing preflight for Phase-1f)

## 7. What to say in the meeting (three sentences)

1. Phase-1d finished 50 epochs cleanly but the pass criterion failed — actor diverged to **−4.67×10⁵**, essentially matching Phase-1b's **−4.86×10⁵** — so the three-knob "Option-A" bundle did **not** stabilize the actor at `λ=0`.
2. Root cause: the shipped code's comment is wrong — internal LayerNorm does not bound the final encoder outputs `φ, ψ`, so the inner product is still unbounded, and the actor escaped the Option-A fix the same way it escaped the negative-L2 energy; the missing piece is **row-L2 normalization + temperature on the final representations**, which is the explicit step in Wang 2025 that we skipped.
3. Plan is a single-knob Phase-1f that adds exactly that normalization; if it stabilizes the actor we're back on the depth-feasibility program, if it doesn't then constraints really are a necessary stabilizer for CRL (H2) and that becomes the paper — either way the answer is one run away.
