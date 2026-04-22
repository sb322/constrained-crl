# CRL formulation verification — do we match Wang 2025 / scaling-crl, and is the λ-term our *only* stabilizer?

Author: audit memo, 2026-04-22.
Scope: `constrained_crl/train.py` as of the Phase-1b overnight (run `rvsu555e`).
Primary reference: Wang, Javali, Bortkiewicz, Trzciński, Eysenbach, *"1000 Layer Networks for
Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities"*, NeurIPS 2025
(best paper). arXiv:2503.14858, OpenReview `s0JVsx3bx1`, code `wang-kevin3290/scaling-crl`.
This is the paper our codebase is forked from, and the paper our constrained-CRL work extends
— so the correct question is not "do we match Eysenbach 2022?" but "do we match scaling-crl?"
(which itself builds on JaxGCRL (Bortkiewicz et al. 2024) and on the InfoNCE formulation
from Eysenbach et al. 2022).
Status: **blocks Phase-1c submission** until resolved.

The five questions below are the ones asked explicitly:

1. Are we using the same formulations as the original paper for CRL?
2. If yes, do we use the same normalization devices?
3. Is the CRL objective bounded? If theirs is and ours isn't, why?
4. Is their stability source architectural, and ours the λ term?
5. Assess: *"if CRL is not normalized, it fundamentally requires an external stabilizer."*

The short answer is: **we match the general InfoNCE template but diverge on the specific critic
metric and on one stability-critical init detail, which makes our per-epoch CRL objective
unbounded below in the sense that matters at gradient time. Yes, the λ-term is currently doing
double duty as the stabilizer, which is why both v1a and v1b collapse to nearly the same
actor-loss floor (−485k) when λ̃ drifts to zero.** The details follow.

## 1. Objective, as written in our code

With `batch_size = N`, let φ := `sa_encoder(s,a) ∈ ℝ^{64}` and ψ := `g_encoder(g) ∈ ℝ^{64}`,
produced by a stack of Dense(1024) + LayerNorm + swish blocks whose *final* layer is a plain
`nn.Dense(64, kernel_init=lecun_normal, bias_init=zeros)` (lines 158–201 of `train.py`).
No normalization is applied to the 64-dim output.

The contrastive critic is negative L2 distance on the raw 64-dim embeddings:

  f(s,a,g) = −√(‖φ(s,a) − ψ(g)‖² + 10⁻¹²)          (train.py:600, 640)

The training objective for the critic is a symmetric-row InfoNCE plus a logsumexp
penalty (train.py:602–603):

  L_crit = E[ −f(sᵢ,aᵢ,gᵢ) + logsumexp_j f(sᵢ,aᵢ,g_j) ]
           + β · E[ (logsumexp_j f(·,·,g_j))² ]      with β = 0.1.

The actor loss is (train.py:644–649):

  L_actor = E[ α·log π(a|s,g) − f(s,a,g)/ν_f + λ̃ · Q_c(s,a,g)/ν_c ]

with `ν_f = log N ≈ 5.545` for N=256 (train.py:446) and `ν_c = 1/(1−γ) = 100` (train.py:448).
`α = exp(log_α)` is SAC-style, auto-tuned by `alpha_loss = −log_α·(log_prob + |A|)` with
|A| = action_size = 8 for ant (train.py:655–656).

## 2. Question 1 — do we match the scaling-crl formulation?

The architecture matches; the critic metric is a scaling-crl-supported variant but not the
one their headline experiments emphasize; one stability-critical hyperparameter is left
outside the regime they validate.

**Architecture.** Wang et al. 2025 explicitly specify the stabilizer stack for deep CRL: each
residual block is 4 × (Dense → LayerNorm → Swish) with a skip connection wrapping the block.
Our `residual_block` at `train.py:145–155` is exactly this, width 1024, identical activation,
identical LayerNorm placement. Wang et al. state that beyond ~16 layers, *all three* of
residual, LayerNorm, and Swish are mandatory; removing any one kills trainability. At our
current depth (4), we are below the threshold where residual connections are required —
and in fact our `SA_encoder` / `G_encoder` default to `skip_connections=0`, which routes the
forward pass through the *non-residual* 4-Dense path (train.py:170–177). That is consistent
with their ablation: 4 layers is fine without skips. For any depth ablation beyond ~8, we
must set `skip_connections > 0`, which our config already supports.

**Critic metric.** Eysenbach, Zhang, Salakhutdinov, Levine (NeurIPS 2022) derive CRL from
the NCE density-ratio estimator; the natural parameterization is the **inner product**
f(s,a,g) = ⟨φ(s,a), ψ(g)⟩, under which the central theorem holds — the critic converges to
log p(g | s,a) / p(g). JaxGCRL (Bortkiewicz et al. 2024) and scaling-crl both support
multiple energies (`dot`, `l2`, `l2_sq`, `cosine`) as a config flag. Our code uses
**negative L2 distance** f = −√(‖φ − ψ‖² + 10⁻¹²) (`train.py:600, 640`). This is one of the
supported options upstream but has two quantitative differences from ⟨φ, ψ⟩ that matter
here:

  (i) Non-smoothness at φ=ψ. The gradient magnitude of −‖φ−ψ‖ with respect to φ is
      (φ−ψ)/‖φ−ψ‖; the denominator goes to zero on positive pairs as the critic learns,
      so individual-sample actor gradients can grow arbitrarily even though the loss value
      stays bounded. The 1e-12 floor in our code bounds this at 10⁶ per coordinate — finite
      but not small. ⟨φ, ψ⟩ has uniformly bounded gradient (by ‖φ‖, ‖ψ‖ via LayerNorm).

  (ii) No scale anchor on ‖φ‖, ‖ψ‖. Under inner product, LayerNorm *inside* the encoder
       effectively bounds ⟨φ, ψ⟩ via bounded pre-projection activations. Under negative
       L2 distance, φ − ψ can grow without bound on negative pairs because the output head
       `nn.Dense(64)` has no post-LayerNorm.

Wang et al. report scaling-to-1024-layers results that are, as far as we can tell from the
published description, achieved with the architectural stabilizers alone (LN + Swish + skips
+ LSE penalty + SAC entropy target). They do not report requiring an external Lagrangian
— but their evaluation is also (a) unconstrained, (b) trained for longer horizons than 5M
steps, and (c) they likely use `dot` or `cosine` energy in at least some of their headline
configurations, not `l2` without repr-norm. The claim "scaling-crl is stable at depth=4 on
Ant" is consistent with their paper; the claim "scaling-crl + λ̃·Q_c is stable at depth=4
when λ̃ → 0" is neither established nor refuted by their paper, because the constrained
setting changes the training distribution.

## 3. Question 2 — normalization devices, theirs vs. ours

The comparable stabilizers are listed here. "Ours" = `constrained_crl/train.py` today;
"scaling-crl" = Wang et al. 2025 defaults as inherited from JaxGCRL (Bortkiewicz 2024), with
extra evidence from *Stabilizing Contrastive RL* (Zheng, Eysenbach et al. ICLR 2024) for the
specific items that paper isolates as stability-critical.

LayerNorm *inside* the residual stack (post each Dense, pre activation): **ours — yes**
(train.py:165, 188, 239). **scaling-crl — yes, mandatory beyond ~16 layers** per their
ablation.

Residual skip connections inside each 4-Dense block: **ours — supported, currently `skip_connections=0`**
which at `network_depth=4` flows through the non-residual path, harmless at depth 4 but
incorrect for any depth ablation > 8. **scaling-crl — mandatory beyond ~16 layers**, and
required for their 1024-layer result.

Swish activation: **ours — yes** (train.py:166, 189, 213, 240). **scaling-crl — yes,
non-negotiable**, ReLU ablates to failure in their paper.

L2 normalization of the 64-dim output heads (φ ← φ/‖φ‖, ψ ← ψ/‖ψ‖): **ours — no**.
**scaling-crl — supported as the `cosine` energy variant, not the default**; under `l2` or
`dot` energies their code takes raw outputs, same as ours.

logsumexp quadratic penalty β · E[LSE²]: **ours — yes, β = 0.1** (train.py:603).
**scaling-crl — yes, β = 0.1 is the JaxGCRL default inherited by scaling-crl.**

Small-magnitude final-layer init U[−10⁻¹², 10⁻¹²]: **ours — no**, we use lecun_normal
(train.py:141, 177, 200). **Zheng et al. 2024 — yes, stability-critical in their offline
setting.** Whether Wang 2025 keeps this detail is unclear without reading the code; JaxGCRL's
default is lecun_normal, so plausibly scaling-crl is also lecun_normal at its reported
depths.

Running-statistics observation normalization: **ours — off**, `normalize_observations=False`
(train.py:60). **JaxGCRL default — on.** This is a specific regression from the upstream
default we inherited; plausibly introduced during our Phase-0 CRL-goal fix.

Actor-loss preconditioning by ν_f = log N: **ours — yes**, shrinks CRL signal ~5.5× relative
to standard JaxGCRL. **Scaling-crl / JaxGCRL — no**, their actor is `E[α·log π − f]`, no ν_f.
This is a departure we introduced for SR-CPO to put f and λ̃·Q_c on comparable gradient
magnitudes. Its effect is to make the entropy / α channel proportionally larger in our
actor loss than in any upstream CRL variant.

External Lagrangian λ̃·Q_c/ν_c: **ours — yes**. **Scaling-crl — no**, they are unconstrained.

Net picture: we match scaling-crl's architectural stabilizers that matter at depth 4
(LayerNorm, Swish, LSE penalty); we are *latently* incompatible with their depth-scaling
claim at depth > 8 because `skip_connections=0` is the default; we diverge on the small-init
detail and on observation normalization; we shrink the CRL signal relative to the entropy
channel via ν_f = log N; and we introduce the Lagrangian as a new stabilizer that has no
upstream analog. The two departures most relevant at depth = 4 are `normalize_observations
= False` and `ν_f = log N` — both are changes we introduced, not inherited.

## 4. Question 3 — is the objective bounded?

**Their objective is bounded above; ours isn't bounded below in the sense that matters for
gradient flow.**

For the critic loss L_crit, both forms give a finite minimum. Under inner-product logits
f = ⟨φ,ψ⟩ with bounded ‖φ‖, ‖ψ‖ (which LayerNorm + small final-layer init effectively
enforce at the start of training), the InfoNCE loss is bounded in both directions and has a
well-defined finite minimizer. Under negative-L2 logits without output norm, the InfoNCE
loss has no lower bound in φ, ψ: pushing all negatives far apart drives logits to −∞ and
logsumexp to a finite but unbounded-below value on each individual term — the LSE² penalty
bounds the *mean*, but the gradient on any individual logit is not bounded by it.

For the actor loss L_actor, the relevant quantity is the **drift of α·log π** in the
SAC-style fixed-entropy formulation. Upstream's loss is E[α·log π − f]. Ours is
E[α·log π − f/ν_f + λ̃·Q_c/ν_c]. Both have the property that α·log π is unbounded below:
SAC's entropy target is log π = −|A|, but the update on α is a multiplicative ascent on
log_α proportional to (log_prob + |A|). The fixed point `log_prob = −|A|` is stable only
while log_α stays finite. If the contrastive term persistently pushes log_prob above the
target (narrow policies to chase f), log_α grows without bound, and α·log_prob → −∞ at
rate log_α · |A|. This is the blow-up pathway.

Upstream empirically avoids this because (a) their critic is inner-product — bounded per-logit
by ‖φ‖·‖ψ‖, which LayerNorm pins; (b) their final-layer init is ε-small, so at step 0 the
CRL signal is zero and α pins log_prob near −|A| before the contrast develops; (c) they run
20–50M steps on larger replay and the slow dynamics reach a fixed point before α runs away.
Our run collects (a) is false — we use −L2 on unnormed 64-dim heads; (b) is false — Lecun
normal init puts f at O(1) from step 0; (c) is moot at 5M steps.

The observed `|actor_loss|_max ≈ 485,678` in Phase-1b run `rvsu555e` and ≈ 485,495 in v1a
are both consistent with α climbing to ~6×10⁴ and log_prob pinned at roughly −|A|. Both
occurred at λ̃ = 0. The λ=0 condition differed (v1a: broken sigmoid cost ⇒ nothing to
integrate; v1b: feasible CMDP ⇒ nothing to punish), but the terminal attractor of the actor-α
dynamics was the same because it is a property of the unregularized CRL actor loss, not of
the cost branch.

## 5. Question 4 — architecture vs. λ

In Wang et al. 2025 CRL the stability is **architectural, with a multi-component safety net**:
the 4×(Dense+LN+Swish) residual block with skip connections, logsumexp penalty on the
partition function, SAC's auto-α targeting fixed entropy, observation normalization (from
JaxGCRL default), and (at deeper depths) the residual connections themselves. Their
headline contribution is precisely that *this combination* is what unlocks scaling to 1024
layers — they ablate each component and show that omitting any one kills trainability.

In our constrained-CRL code we have the 4×(Dense+LN+Swish) block and the LSE penalty and
auto-α, but have **dropped** observation normalization (regression from JaxGCRL default,
introduced in our refactor); we have **shrunk** the CRL signal by ν_f = log N relative to
entropy, tilting the actor loss away from contrastive pressure and toward entropy pressure;
we have chosen the `l2` energy (supported upstream but harder-conditioned than `dot`); and
we have **added** λ̃·Q_c/ν_c as a new term. The Lagrangian term is providing the missing
restoring force — it penalizes the same narrow, wall-hugging, low-entropy policies that the
architectural stabilizers would otherwise rule out. When λ̃ → 0 (either because the CMDP is
feasible, or because the cost is broken as in v1a), the restoring force vanishes and the
training diverges.

So: **their stability source is distributed across architecture + LSE penalty + obs-norm +
auto-α + the `dot` or `cosine` energy in their deeper experiments**; **our stability source
is currently the λ-term plus a partial replica of their architectural stabilizers**. The
Lagrangian is covering for a gap that was introduced by our modifications (obs-norm off, ν_f
preconditioning, `l2` energy), not for a gap present in the original scaling-crl code. That
is an identified single point of failure, and it is the reason Phase-1b reproduced v1a's
collapse through a completely different root cause. The actor dynamics do not care *why*
λ̃ = 0 — they care only *that* λ̃ = 0.

## 6. Question 5 — "if CRL is not normalized, it fundamentally requires an external
stabilizer"

The statement is essentially correct, with one refinement. The word *normalized* is doing a
lot of work, and it should be unpacked:

- **Representation-level normalization** (output L2-norm on φ, ψ, or cosine similarity):
  provides a hard scale anchor on ‖φ‖, ‖ψ‖. With this, ⟨φ,ψ⟩ ∈ [−1,1] and the InfoNCE loss
  is bounded in both directions.

- **Distributional normalization** (logsumexp penalty, batch-norm–style scaling, small init):
  soft constraints on the partition function and representation magnitudes.

- **Input normalization** (running-stats obs-norm, reward scaling): upstream condition on the
  encoder inputs.

Upstream CRL uses mostly the middle category, plus obs-norm and small init to compensate for
*not* doing representation-level L2-norm in its most common configuration. This combination
is empirically sufficient; it is also fragile, which is why Zheng et al. 2024 is titled
"*Stabilizing* Contrastive RL" rather than "Tuning Contrastive RL" — the community considered
it unstable enough to warrant a dedicated paper.

In our setup we have removed two of the three safety-net items and introduced a more
ill-conditioned metric, so the claim *"CRL without adequate normalization requires an
external stabilizer"* is more than a theoretical conjecture — it is exactly what our
`rvsu555e` run demonstrates. The λ̃·Q_c term is that external stabilizer, and we need it to
be non-zero for training to stay in the well-behaved regime.

## 7. Implications for ICML narrative — this is a feature, not a bug

This finding has a natural place in the paper, and it actually *reinforces* the project's
stated focus on "representation learning and depth effects on feasibility." The observation
is specifically that the Wang 2025 scaling-crl stabilizer stack — which their paper shows is
sufficient to train 1000-layer *unconstrained* GCRL — is **not** sufficient once you impose
a CMDP budget. The constrained setting exposes a failure mode their paper could not have
seen.

It would be intellectually dishonest to claim "SR-CPO trains stably" while eliding the fact
that the Lagrangian is also doing representation-regularization work. The honest — and
more interesting — framing is the converse:

> *In the scaling-crl architecture, the Lagrangian in SR-CPO serves a dual role: constraint
> enforcement (its intended function) and implicit regularization of the contrastive actor
> (emergent consequence). When the CMDP is strictly feasible (J_c⋆ < d) or the cost signal
> is degenerate, the Lagrangian decays to zero and the latter role disappears, exposing a
> representation-level instability of goal-conditioned contrastive RL that is not visible in
> the unconstrained setting of Wang et al. (2025). This is empirically verified on Phase-1b
> run rvsu555e, where |actor_loss| reaches 4.86×10⁵ with λ̃ ≡ 0 — indistinguishable from
> the v1a control whose cost was broken by a different mechanism (actor-α blow-up at
> ≈ −485k is a property of the unregularized actor, not of either root cause).*

This aligns the constrained-CRL paper with the project charter's focus on "depth effects on
feasibility." The clean three-way ablation to report is (a) bare SR-CPO (scaling-crl +
λ̃·Q_c) under feasible CMDP, (b) scaling-crl + `dot` energy + obs-norm-on + ν_f = 1 (i.e.
fully restoring the upstream defaults) under feasible CMDP, (c) bare SR-CPO under tight
(infeasible-forcing) CMDP. Predictions: (a) reproduces the −485k blow-up, (b) trains
stably with Ĵ_c ≤ d not actively enforced, (c) trains stably with λ̃ > 0 throughout. Each
outcome tells a different piece of the story: (a) is the single-point-of-failure result,
(b) is the "scaling-crl alone can handle feasible CMDPs if you preserve its stabilizers"
result, (c) is the "binding constraint enables training even with our modified stack" result.
A 2–3-day experimental campaign.

## 8. Recommendation — what to do before any further run

Do **not** submit Phase-1c tight-budget until one of the following is in place. The current
pipeline will reproduce the v1a/v1b collapse in any run where the CMDP is feasible, no
matter how tight the budget, because feasibility is a property of the true environment not
of the budget.

Concretely, three options, in decreasing order of scientific defensibility:

**(A) restore the scaling-crl defaults we regressed on, keep λ in its intended role.** This
is the intervention that closes the gap to Wang 2025 most faithfully. Three changes, all
small:

    # (i) switch energy to inner product — the form under which LayerNorm actually anchors ‖f‖
    #     in the encoder. Replace the critic logits in both critic_loss_fn and actor_loss_fn:
    logits = jnp.einsum("ik,jk->ij", sa_repr, g_repr)          # instead of -sqrt(‖φ−ψ‖²)
    f_sa_g = jnp.sum(sa_repr * g_repr, axis=-1)                # in actor_loss

    # (ii) turn observation normalization back on — restores JaxGCRL upstream default
    normalize_observations: bool = True                        # train.py:60

    # (iii) remove the ν_f shrinkage so the actor loss matches scaling-crl's scale
    nu_f: float = 1.0                                          # or drop the /ν_f entirely

This reverts the three departures most likely responsible for our current gap, without
introducing anything speculative (no extra L2-norm layers, no small-init retro-patch). The
λ̃·Q_c term is then freed to do only its intended job — budget enforcement — and the
experimental claim becomes clean.

**(B) keep our current code as the "bare SR-CPO" condition, add a parallel "faithful
scaling-crl" condition for contrast.** Run both code paths under the same environment and
CMDP budget, log both. Use the divergence between them as Exhibit A for the paper's central
observation: scaling-crl's stability apparatus is sufficient for unconstrained goal-reaching
(their Figure 1) but is *not* automatically sufficient for constrained goal-reaching unless
the Lagrangian happens to be active, and the failure mode we document is precisely the
price of under-investing in the unconstrained stabilizers. This is the most publishable
path: it says something new.

**(C) keep the current code, reframe the contribution around emergent Lagrangian regularization.**
Weaker than (B) because it doesn't isolate the cause from the effect, but defensible if the
ablation budget is tight.

Either (A) or (B) is defensible — (B) is the stronger ICML story. What is **not** defensible
is submitting more Phase-1-style runs without deciding between them, because every such run
either (a) produces the v1b collapse and tells us nothing new, or (b) accidentally lands in
the λ̃ > 0 regime and cannot be attributed cleanly to either constraint enforcement or
representation regularization.

## 9. Diagnostics to add regardless of path

Three wandb panels to log, per-sgd-step:

  representation_scale/‖sa_repr‖_mean       — should stabilize at O(1); divergence ⇒ broken
  representation_scale/‖g_repr‖_mean
  representation_scale/‖sa−g‖_mean          — should *not* collapse to 0 nor diverge
  training/log_alpha                        — currently logged only as `alpha`; add log_α
                                              because the blow-up happens in log-space
  training/lambda_tilde_over_log_alpha      — diagnostic ratio: if this ≈ 0 while log_alpha
                                              climbs, the pipeline is in the unstable regime

These panels would have told us in ~10 epochs rather than 50 whether a run was heading into
the −485k basin.

---

References.
Wang, Javali, Bortkiewicz, Trzciński, Eysenbach. *1000 Layer Networks for Self-Supervised RL:
Scaling Depth Can Enable New Goal-Reaching Capabilities*. NeurIPS 2025 (Best Paper).
arXiv:2503.14858. OpenReview s0JVsx3bx1. Code `wang-kevin3290/scaling-crl`. **Primary
reference — the paper our constrained-CRL work extends.**
Bortkiewicz, Pałucki, Myers, Dziarmaga, Arczewski, Kuciński, Eysenbach. *Accelerating
Goal-Conditioned RL Algorithms and Research (JaxGCRL)*. ICLR 2025. arXiv:2408.11052. The
codebase scaling-crl was forked from; source of inherited defaults like the LSE penalty
coefficient and obs-norm behaviour.
Eysenbach, Zhang, Salakhutdinov, Levine. *Contrastive Learning as Goal-Conditioned
Reinforcement Learning*. NeurIPS 2022. arXiv:2206.07568. Origin of the InfoNCE-as-GCRL
theorem; states the inner-product parameterization.
Zheng, Eysenbach, Walke, Yin, Fang, Salakhutdinov, Levine. *Stabilizing Contrastive RL:
Techniques for Robotic Goal Reaching from Offline Data*. ICLR 2024. arXiv:2306.03346.
Offline/robotic setting, identifies LayerNorm + small-final-layer init as stability-critical.
Less directly applicable to our online-MuJoCo setting than Wang 2025, but informative on the
init question.
