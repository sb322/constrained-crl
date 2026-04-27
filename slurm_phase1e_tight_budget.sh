#!/bin/bash -l
#SBATCH --job-name=ccrl_phase1e_tight_d0p05
#SBATCH --output=phase1e_tight_d0p05.%j.out
#SBATCH --error=phase1e_tight_d0p05.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE-1e — TIGHT BUDGET  d = 0.05   (BINDING CMDP, λ̃ expected > 0)
# ─────────────────────────────────────────────────────────────────────────────
#  One-knob successor to Phase-1f. Submit only AFTER Phase-1f finishes and
#  passes its stability check. If Phase-1f fails (actor_loss diverges even
#  with row-L2 + τ=0.1), do NOT launch this — Phase-1e would pile a second
#  intervention onto an already-unstable system and muddle causal attribution.
#
#  The SOLE difference from Phase-1f is   --cost_budget_d  0.15 → 0.05.
#  Everything else — code, architecture, PID gains, τ, ν_f, ν_c, cost
#  function, representation normalization, norm-logging — is byte-identical
#  to Phase-1f. This is the "change lambda the CMDP way" experiment: the
#  budget tightens, J_c > d organically, the PID integrator drives λ̃ > 0,
#  and we observe whether a binding constraint stabilizes the actor or not.
#
#  Why "Phase-1e" after "Phase-1f":
#    Phase-1e was originally staged earlier in the project (see
#    slurm_phase1c_tight_budget.sh, which this supersedes), but Phase-1d
#    Option-A invalidated the code path that Phase-1c tested. Renumbering
#    as 1e keeps the narrative: 1b (baseline) → 1d (Option-A fail) →
#    1f (row-L2 + τ fix) → 1e (tight budget on the fixed code).
#
#  Background (why d = 0.05 is informative):
#    • Phase-1b (job 963416) at d = 0.15: J_c_hat ≈ 0.019 — strictly feasible,
#      λ̃ decayed to 0 within ~5 epochs. Constraint never bit.
#    • Phase-1f (preceding this run) holds d = 0.15; tests whether row-L2
#      representation scaling alone stabilizes the actor at λ̃ ≈ 0.
#    • Phase-1e drops d to 0.05 — well below Phase-1b's J_c_hat — so the
#      CMDP becomes binding, λ̃ stays positive, and the constraint term
#      λ̃·Q_c/ν_c in the actor loss is the DOMINANT regularizer.
#
#  Hypotheses this run bears on:
#    • Confirmatory: if Phase-1f passed, Phase-1e tests whether the fix
#      continues to hold under a binding CMDP (the paper's main regime).
#    • Depth × feasibility: this is the first tight-budget data point in the
#      planned depth × feasibility matrix — puts a real constraint on the
#      depth-4 policy before we sweep depth.
#    • λ-sensitivity: the PID drives λ̃ up organically. We see the actor
#      under λ̃ > 0 without pinning or ablating the controller.
#
#  Metrics to watch on wandb (group phase1e_tight_d0p05):
#
#    PRIMARY (stability under binding constraint):
#      train/actor_loss          EXPECTED |·| < 1e3 throughout
#      train/critic_loss         EXPECTED finite, monotone-ish decrease from log(N)≈5.5
#      train/accuracy            EXPECTED rising above 1/N = 1/256 within ~5 epochs
#      train/logits_pos          EXPECTED in [ 3.0 , 10.0 ] once trained (= [0.3, 1.0]/τ)
#      train/logits_neg          EXPECTED in [−2.0 ,  3.0 ]              (= [−0.2, 0.3]/τ)
#
#    LAGRANGIAN DIAGNOSTICS  (central to this run — REPLACES the λ̃→0 band):
#      train/lambda_tilde        EXPECTED > 0 after warmup (binding regime)
#                                median over last 20 epochs in roughly [0.3, 30]
#                                if near 0, the budget is not actually binding —
#                                re-examine J_c_hat
#      train/jhat_c              EXPECTED ≈ d = 0.05 once PID has converged
#                                (small positive band above d is normal, large
#                                 sustained violations indicate the actor can't
#                                 meet the constraint at this architecture/depth)
#      train/constraint_violation  max(0, jhat_c − d) — EXPECTED decaying to 0
#      train/mean_qc_pi            EXPECTED > mean_qc at start, converging
#
#    REPRESENTATION-SCALING DIAGNOSTICS  (inherited from Phase-1f):
#      train/sa_repr_norm_critic  mean ‖φ‖ (pre-normalize) in critic pass
#      train/g_repr_norm_critic   mean ‖ψ‖ (pre-normalize) in critic pass
#      train/sa_repr_norm_actor   mean ‖φ‖ (pre-normalize) in actor  pass
#      train/g_repr_norm_actor    mean ‖ψ‖ (pre-normalize) in actor  pass
#      EXPECTED: still bounded and roughly stationary. A binding λ̃ should
#      NOT cause encoder-norm drift; if it does, the constraint term is
#      entering through an unscaled representation path and the stabilization
#      story is not what we claim.
#
#    SECONDARY (constraint accounting):
#      train/hard_violation_rate EXPECTED monotone ↓ after warmup;
#                                ≤ 0.05 by mid-training
#      train/mean_step_cost      EXPECTED ↓ across training
#
#  Pass criterion (all four must hold):
#    (a) |train/actor_loss|_max  over all 50 epochs  <  1e3
#    (b) no NaN in any loss
#    (c) median train/lambda_tilde over last 20 epochs  >  0.1
#        (verifies the constraint is actually binding — the whole point of
#         this run is to see the binding regime, so λ̃ ≈ 0 means the budget
#         d = 0.05 was too loose for this architecture and the test did not
#         actually happen)
#    (d) final train/jhat_c  ≤  d + 0.02  (constraint is roughly met by end)
#
#  Failure interpretations:
#    • (a) fails but Phase-1f (a) passed: the λ̃·Q_c/ν_c term is destabilizing
#      at this scale — either ν_c is mis-tuned or Q_c is ill-conditioned.
#      Diagnose via mean_qc trajectory and nu_c computation.
#    • (c) fails (λ̃ ≈ 0 at d=0.05): the agent met d=0.05 trivially. Drop d
#      further (0.02 or 0.01) — NOT in this run, as a separate resubmission.
#    • (d) fails (jhat_c > d + 0.02 persistent): the policy class at depth 4
#      can't meet the tight budget. Ups the ante for the depth sweep: Phase-2
#      should include deeper policies or a wider maze.
# ═════════════════════════════════════════════════════════════════════════════

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

WORKDIR="/mmfs1/home/sb3222/projects/constrained-crl"
VENV="$WORKDIR/.venv"
PYTHON="$VENV/bin/python"

cd "$WORKDIR" || exit 1

export PATH="$VENV/bin:$PATH"

# ── LD_LIBRARY_PATH: pip nvidia packages FIRST, system CUDA as fallback ──
PY_SITE=$("$PYTHON" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

PIP_NVIDIA_LIBS=""
for subpkg in cudnn cusolver cublas cuda_runtime cuda_nvrtc cufft cuda_cupti cusparse nvjitlink nccl; do
    d="$PY_SITE/nvidia/$subpkg/lib"
    [ -d "$d" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$d"
done

SYS_CUDA_LIB="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0/lib64"
export LD_LIBRARY_PATH="${PIP_NVIDIA_LIBS}:${SYS_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="/mmfs1/home/sb3222/.cache/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 | head -20

echo ""
echo "===== PHASE-1f CODE VERIFICATION  (inherited fix required by Phase-1e) ====="
"$PYTHON" - <<'PYCHECK'
# Verify Phase-1f AND the surviving Option-A knobs are all in the shipped code.
# If any of these asserts fail, the wrong train.py is on PYTHONPATH.
import pathlib
src = pathlib.Path("train.py").read_text()

# --- Surviving Option-A knobs ------------------------------------------------
assert "normalize_observations: bool = True" in src, \
    "normalize_observations default is not True — Option-A regressed"
assert "nu_f: float = 1.0" in src, \
    "nu_f default is not 1.0 — Option-A regressed"

# --- Phase-1f row-L2 + temperature -------------------------------------------
# This run's τ = 0.1 (experiment spec: clean stability test in isolation).
# If this assert fires, either the wrong train.py is on PYTHONPATH or τ was
# regressed to an earlier value (1.0 or 0.125).
assert "tau: float = 0.1" in src, \
    "args.tau default is not 0.1 — Phase-1f τ-scaling missing"

# Representation-norm logging must be wired through (both loss functions +
# per-step metrics dict + epoch log_dict).
for probe, where in [
    ("sa_repr_norm = jnp.mean(jnp.linalg.norm(sa_repr, axis=-1))",
     "critic_loss_fn pre-normalize ‖φ‖ probe"),
    ("g_repr_norm  = jnp.mean(jnp.linalg.norm(g_repr,  axis=-1))",
     "critic_loss_fn pre-normalize ‖ψ‖ probe"),
    ("sa_repr_norm_actor = jnp.mean(jnp.linalg.norm(sa_repr, axis=-1))",
     "actor_loss_fn pre-normalize ‖φ‖ probe"),
    ("g_repr_norm_actor  = jnp.mean(jnp.linalg.norm(g_repr,  axis=-1))",
     "actor_loss_fn pre-normalize ‖ψ‖ probe"),
    ('"sa_repr_norm_critic": sa_rn_crit,',
     "per-step metrics wiring (sa_repr_norm_critic)"),
    ('"g_repr_norm_critic":  g_rn_crit,',
     "per-step metrics wiring (g_repr_norm_critic)"),
    ('"sa_repr_norm_actor":  sa_rn_act,',
     "per-step metrics wiring (sa_repr_norm_actor)"),
    ('"g_repr_norm_actor":   g_rn_act,',
     "per-step metrics wiring (g_repr_norm_actor)"),
]:
    assert probe in src, f"norm-logging probe missing — {where}"

# Phase-1f autograd-safe row-L2 (epsilon INSIDE sqrt, in BOTH losses).
# Job 988003 forensics confirmed the OLD form (`+ 1e-8` outside the norm)
# is forward-finite but autograd-NaN on near-zero rows. The fix moves the
# epsilon inside the sqrt so the radicand is strictly positive. Phase-1e
# inherits this fix unchanged.
assert src.count("sa_norm_safe = jnp.sqrt(") == 2, \
    "autograd-safe sa_norm_safe must appear in BOTH critic_loss_fn and actor_loss_fn"
assert src.count("g_norm_safe  = jnp.sqrt(") == 2, \
    "autograd-safe g_norm_safe must appear in BOTH critic_loss_fn and actor_loss_fn"
assert src.count(
    "jnp.sum(sa_repr * sa_repr, axis=-1, keepdims=True) + 1e-12"
) == 2, "sa_repr safe-norm radicand sum(sa²)+1e-12 must appear in both losses"
assert src.count(
    "jnp.sum(g_repr  * g_repr,  axis=-1, keepdims=True) + 1e-12"
) == 2, "g_repr safe-norm radicand sum(g²)+1e-12 must appear in both losses"
assert src.count("sa_repr = sa_repr / sa_norm_safe") == 2, \
    "sa_repr divide-by-safe-norm must appear in both losses"
assert src.count("g_repr  = g_repr  / g_norm_safe")  == 2, \
    "g_repr divide-by-safe-norm must appear in both losses"
# Stale unsafe form must NOT survive anywhere.
assert "jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-8" not in src, \
    "stale unsafe row-L2 form on sa_repr still present in train.py"
assert "jnp.linalg.norm(g_repr,  axis=-1, keepdims=True) + 1e-8" not in src, \
    "stale unsafe row-L2 form on g_repr still present in train.py"

# Both energies must be divided by args.tau
assert 'jnp.einsum("ik,jk->ij", sa_repr, g_repr) / args.tau' in src, \
    "critic_loss_fn logits must be divided by args.tau (Phase-1f)"
assert "f_sa_g  = jnp.sum(sa_repr * g_repr, axis=-1) / args.tau" in src, \
    "actor_loss_fn f_sa_g must be divided by args.tau (Phase-1f)"

# Probes alive (kept for this 50-epoch run; per Phase-1f at d=0.15 success,
# they cost essentially nothing and give one-line per-epoch confirmation
# that nothing is NaN at any point during training).
assert "f\"         nan[obs_c=" in src, \
    "per-epoch forward NaN one-liner missing — probes were stripped"
assert "f\"         grad[c=" in src, \
    "per-epoch grad/param one-liner missing — probes were stripped"
assert "[prefill probe] buffer NaN anywhere:" in src, \
    "post-prefill buffer-NaN probe missing — probes were stripped"
assert "[epoch1 forensics]" in src, \
    "epoch-1 time-series dump missing — probes were stripped"

# Phase-1g α-cap. Job 1005888 (Phase-1f at d=0.15, 50 epochs) confirmed
# that without the cap actor_loss diverges to ~−484K, with α_max=1.0 it
# plateaus at −1.42. Phase-1e at d=0.05 ADDS a binding constraint term
# (λ̃·Q_c/ν_c) on top of the same actor; running it without the α-cap
# would re-introduce the divergence regardless of how λ̃ behaves. The
# cap is non-negotiable for this run.
assert "alpha_max: float = 1.0" in src, \
    "Args.alpha_max field missing or default changed — α-cap not wired"
assert "log_alpha_cap = jnp.log(jnp.maximum(args.alpha_max, 1e-12))" in src, \
    "α-cap clip computation missing from sgd_step"
assert "alpha_state = alpha_state.replace(params=clipped_log_alpha)" in src, \
    "α-cap clip not applied to alpha_state — divergence will recur"
assert "f\"         actor[α=" in src, \
    "actor-component per-epoch one-liner missing"
assert "α_clip={log_dict['alpha_clip_active']:.2f}" in src, \
    "α_clip token missing from actor-component print line"

# The broken Phase-1d claim must be gone
assert "LayerNorm inside the encoders anchors" not in src, \
    "the false 'LayerNorm anchors ‖φ‖,‖ψ‖' comment from Option-A is still in train.py"

# The old negative-L2 energy must be gone (already gone in Option-A, guard anyway)
assert "-jnp.sqrt(_d2 + 1e-12)" not in src, \
    "old negative-L2 energy still present"

print("Phase-1e static diff (autograd-safe row-L2 + probes alive + α-cap) verified in train.py.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Phase-1f code verification failed — Phase-1e cannot run without the row-L2 + τ fix. Aborting."
    exit 1
fi

echo ""
echo "===== JAX GPU CHECK ====="
"$PYTHON" - <<'PYCHECK'
import jax
devices = jax.devices()
backend = jax.default_backend()
print(f"devices = {devices}")
print(f"default_backend = {backend}")
if backend != "gpu":
    print("FATAL: JAX did not detect GPU. Aborting.")
    import sys; sys.exit(1)
print("GPU OK — proceeding with Phase-1e run.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "JAX GPU check failed — aborting job."
    exit 1
fi

# ── Env preflight: same env check as Phase-1d / Phase-1f ────────────────────
echo ""
echo "===== PHASE-1e ENV PREFLIGHT ====="
"$PYTHON" - <<'PYCHECK'
import envs  # registers ant/humanoid mazes
from brax import envs as brax_envs
import jax

env = brax_envs.get_environment(
    "ant_big_maze",
    exclude_current_positions_from_observation=False,
)
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
env = VmapWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))

keys = jax.random.split(jax.random.PRNGKey(0), 2)
state = env.reset(keys)
print(f"obs.shape        = {state.obs.shape}")
print(f"action_size      = {env.action_size}")
print(f"torso xy @ reset = {state.pipeline_state.x.pos[:, 0, :2]}")
print(f"obs[:, 0:3]      = {state.obs[:, 0:3]}   # should be [x, y, z]")
PYCHECK

# ── Dynamic normalization check (verifies the Phase-1f row-L2 fix is live) ──
#  Static grep is necessary but not sufficient. The Option-A failure taught us
#  that a code comment can confidently claim "‖φ‖ is bounded" while the
#  instantiated encoder makes that claim false on a real batch. So: build the
#  encoders, run one batch, and ACTUALLY assert ‖φ̂‖₂ = 1 to 5 decimal places.
#  Phase-1e runs on this fix; if it regressed, abort before burning GPU hours.
echo ""
echo "===== PHASE-1e DYNAMIC NORMALIZATION ASSERTION ====="
"$PYTHON" - <<'PYCHECK'
# Build the actual SA_encoder and G_encoder classes from train.py (safe to
# import: train.py is guarded by `if __name__ == "__main__"`, so importing
# does NOT launch a training run).
#
# Phase-1e CLI config we will launch with:
#   --obs_dim 29    --goal_start_idx 0  --goal_end_idx 3
#   --critic_depth 4   --critic_network_width 256
#   (Ant action_size = 8, so sa input dim = 29 + 8 = 37; goal dim = 3)
import jax, jax.numpy as jnp
from train import SA_encoder, G_encoder, Args

args = Args()  # Phase-1f defaults (inherited by Phase-1e): tau=0.1, nu_f=1.0,
               # normalize_observations=True, row-L2 in both loss fns

# Phase-1e runs on ant_big_maze → explicit dims (NOT Args.obs_dim which
# defaults to 265 for the Humanoid config in train.py).
OBS_DIM    = 29
ACTION_DIM = 8          # Ant
GOAL_DIM   = 3          # x,y,z torso position
WIDTH      = 256
DEPTH      = 4
SKIP       = 0          # args.critic_skip_connections default

sa_enc = SA_encoder(network_width=WIDTH, network_depth=DEPTH, skip_connections=SKIP)
g_enc  = G_encoder(network_width=WIDTH, network_depth=DEPTH, skip_connections=SKIP)

key = jax.random.PRNGKey(0)
k_sa, k_g, k_batch = jax.random.split(key, 3)
sa_params = sa_enc.init(k_sa, jnp.zeros((4, OBS_DIM)), jnp.zeros((4, ACTION_DIM)))
g_params  = g_enc.init(k_g,  jnp.zeros((4, GOAL_DIM)))

# Batch of Gaussian inputs — non-trivial, so we really exercise the encoder.
N = 64
obs_batch  = jax.random.normal(k_batch, (N, OBS_DIM))
act_batch  = jax.random.normal(k_batch, (N, ACTION_DIM))
goal_batch = jax.random.normal(k_batch, (N, GOAL_DIM))

sa = sa_enc.apply(sa_params, obs_batch, act_batch)
g  = g_enc.apply(g_params,  goal_batch)

# Pre-normalization norms — expected to span orders of magnitude because the
# trailing Dense layer is unconstrained. This is the Phase-1d failure mode we
# are guarding against; document it in the log before normalizing.
sa_raw_norms = jnp.linalg.norm(sa, axis=-1)
g_raw_norms  = jnp.linalg.norm(g,  axis=-1)
print(f"‖φ‖ (raw)  min/mean/max = {float(sa_raw_norms.min()):.4f} / "
      f"{float(sa_raw_norms.mean()):.4f} / {float(sa_raw_norms.max()):.4f}")
print(f"‖ψ‖ (raw)  min/mean/max = {float(g_raw_norms.min()):.4f} / "
      f"{float(g_raw_norms.mean()):.4f} / {float(g_raw_norms.max()):.4f}")

# Apply the EXACT autograd-safe row-L2 normalization train.py uses
# (epsilon INSIDE sqrt, so the radicand is strictly positive and grad of
# sqrt is finite even when sum(x²) underflows). Mirrors the form in
# losses post-Phase-1f-fix; if you change one, change the other.
sa_norm_safe = jnp.sqrt(jnp.sum(sa * sa, axis=-1, keepdims=True) + 1e-12)
g_norm_safe  = jnp.sqrt(jnp.sum(g  * g,  axis=-1, keepdims=True) + 1e-12)
sa_hat = sa / sa_norm_safe
g_hat  = g  / g_norm_safe

sa_norms = jnp.linalg.norm(sa_hat, axis=-1)
g_norms  = jnp.linalg.norm(g_hat,  axis=-1)
print(f"‖φ̂‖       min/mean/max = {float(sa_norms.min()):.6f} / "
      f"{float(sa_norms.mean()):.6f} / {float(sa_norms.max()):.6f}")
print(f"‖ψ̂‖       min/mean/max = {float(g_norms.min()):.6f} / "
      f"{float(g_norms.mean()):.6f} / {float(g_norms.max()):.6f}")

assert jnp.allclose(sa_norms, 1.0, atol=1e-5), \
    "‖φ̂‖ not unit-norm on a real batch — row-L2 normalize broken"
assert jnp.allclose(g_norms,  1.0, atol=1e-5), \
    "‖ψ̂‖ not unit-norm on a real batch — row-L2 normalize broken"

# Sanity: cosine energy must lie in [−1/τ, 1/τ]
logits = jnp.einsum("ik,jk->ij", sa_hat, g_hat) / args.tau
print(f"logits     min/max      = {float(logits.min()):.4f} / "
      f"{float(logits.max()):.4f}   (τ = {args.tau})")
assert float(logits.max()) <=  1.0 / args.tau + 1e-5, \
    "cosine logits exceed +1/τ — normalization broken"
assert float(logits.min()) >= -1.0 / args.tau - 1e-5, \
    "cosine logits fall below −1/τ — normalization broken"
print("Phase-1f dynamic normalization OK — φ̂, ψ̂ on the unit sphere, "
      "logits ∈ [−1/τ, 1/τ].")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Phase-1f dynamic normalization check failed — aborting job."
    exit 1
fi

echo ""
echo "===== PHASE-1e LAUNCH  $(date) ====="

# Note: Phase-1e inherits the Phase-1f representation-scaling defaults from
# train.py (tau=0.1, row-L2 normalize, normalize_observations=True, nu_f=1.0).
# The ONLY CLI override vs. Phase-1f is  --cost_budget_d 0.15 → 0.05.
# That single knob is the entire point of this run: overriding stability
# defaults (tau, normalization, nu_f) would confound the causal attribution
# and is prohibited.

"$PYTHON" train.py \
    --env_id            ant_big_maze \
    --eval_env_id       ant_big_maze_eval \
    --episode_length    1000 \
    --total_env_steps   5000000 \
    --num_epochs        50 \
    --num_envs          256 \
    --eval_envs         32 \
    --eval_every        5 \
    --unroll_length     62 \
    --num_minibatches   32 \
    --num_update_epochs 4 \
    --batch_size        256 \
    --seed              0 \
    --obs_dim           29 \
    --goal_start_idx    0 \
    --goal_end_idx      3 \
    --actor_depth       4 \
    --critic_depth      4 \
    --cost_critic_depth 4 \
    --actor_network_width  256 \
    --critic_network_width 256 \
    --actor_lr          3e-4 \
    --critic_lr         3e-4 \
    --alpha_lr          3e-4 \
    --use_constraints   True \
    --cost_type         quadratic \
    --cost_epsilon      2.0 \
    --cost_epsilon_hard 0.1 \
    --cost_tau          0.05 \
    --cost_budget_d     0.05 \
    --pid_kp            0.1 \
    --pid_ki            0.003 \
    --pid_kd            0.001 \
    --lambda_max        100.0 \
    --wandb_project     constrained-crl \
    --wandb_group       phase1e_tight_d0p05 \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1e DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
