#!/bin/bash -l
#SBATCH --job-name=ccrl_phase1f_l2norm
#SBATCH --output=phase1f_l2norm.%j.out
#SBATCH --error=phase1f_l2norm.%j.err
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
#  PHASE-1f — ROW-L2 NORMALIZATION + TEMPERATURE  (feasible CMDP, λ̃ expected ≈ 0)
# ─────────────────────────────────────────────────────────────────────────────
#  Context.
#  Phase-1b (job 963416) and Phase-1d Option-A (job 976587) both ran 50/50
#  epochs in the feasible regime (d = 0.15, J_c_hat ≈ 0.019, λ̃ → 0) and both
#  blew up to |train/actor_loss| ≈ 4.7–4.9 × 10⁵. Option-A's claim was that
#  LayerNorm inside the 4-block Wang residual encoders was sufficient to
#  anchor ‖φ‖, ‖ψ‖ and therefore bound the inner-product energy ⟨φ, ψ⟩ at
#  λ̃ = 0. That claim was empirically falsified by Phase-1d:
#
#    • LayerNorm normalizes hidden activations per-sample along the feature
#      axis; it does NOT bound the final encoder output. Trailing Dense layers
#      can rescale φ, ψ by arbitrary factors.
#    • Therefore ⟨φ, ψ⟩ is unbounded above, and the actor minimizes
#        L_actor = −⟨φ, ψ⟩ / ν_f + α log π   (at λ̃ = 0)
#      by driving ‖φ‖ ‖ψ‖ → ∞. Same failure mode as the negative-L2 energy of
#      Phase-1a; Option-A only changed the functional form.
#
#  Phase-1f fix (single knob on top of Option-A):
#    • Row-L2 normalize φ, ψ inside BOTH critic_loss_fn and actor_loss_fn
#          φ̂ = φ / ‖φ‖₂ ,   ψ̂ = ψ / ‖ψ‖₂
#      so that ⟨φ̂, ψ̂⟩ = cos(φ, ψ) ∈ [−1, 1] is genuinely norm-bounded.
#    • Divide the InfoNCE logits by a fixed temperature  τ = 0.1  (per this
#      run's experiment spec — slightly sharper than the paper-standard
#      τ = 1/√D = 0.125 for D=64; still well within Wang 2025 / JaxGCRL
#      operating range). One knob per run; τ-tuning is a separate branch.
#    • With τ = 0.1 and ‖φ̂‖ = ‖ψ̂‖ = 1,  f = ⟨φ̂, ψ̂⟩ / τ  ∈  [−10, 10].
#    • NEW diagnostic logging: mean ‖φ‖, ‖ψ‖ pre-normalize (both the critic
#      pass and the actor pass). These four metrics are the smoking gun for
#      the Phase-1d failure mode — if they grow monotonically the encoder is
#      attempting to escape via rescaling and the row-L2 is the only reason
#      actor_loss looks bounded.
#
#  All other knobs UNCHANGED from Phase-1d:
#    env_id=ant_big_maze, num_envs=256, unroll_length=62, depth=4,
#    batch_size=256, PID=(0.1, 0.003, 0.001), 5M steps, 50 epochs,
#    cost_type=quadratic, ε_train=2.0, ε_hard=0.1,
#    cost_budget_d=0.15, seed=0,
#    normalize_observations=True, nu_f=1.0, inner-product energy (from Option-A).
#
#  Hypotheses distinguished by this run:
#    H1′ (implementation-driven, but ALSO needs row-L2 + τ):
#        If |actor_loss|_max < 1e3 through all 50 epochs,
#        H1′ wins and depth-feasibility program resumes.
#    H2  (intrinsic instability at λ̃ = 0):
#        If actor still diverges to O(10⁵) with representations on the unit
#        sphere, constraints are a necessary CRL stabilizer. Reframe paper
#        around emergent-Lagrangian regularization.
#
#  Metrics to watch on wandb (group phase1f_l2norm):
#
#    PRIMARY (stability):
#      train/actor_loss          EXPECTED |·| < 1e3 throughout (was 4.67e5 in 1d)
#      train/critic_loss         EXPECTED finite, monotone-ish decrease from log(N)≈5.5
#      train/accuracy            EXPECTED rising above 1/N = 1/256 within ~5 epochs
#      train/logits_pos          EXPECTED in [ 3.0 , 10.0 ] once trained (= [0.3, 1.0]/τ)
#      train/logits_neg          EXPECTED in [−2.0 ,  3.0 ]              (= [−0.2, 0.3]/τ)
#      train/lambda_tilde        EXPECTED ≈ 0 after warmup (feasible regime)
#
#    REPRESENTATION-SCALING DIAGNOSTICS  (NEW — central to this experiment):
#      train/sa_repr_norm_critic  mean ‖φ‖ (pre-normalize) in critic pass
#      train/g_repr_norm_critic   mean ‖ψ‖ (pre-normalize) in critic pass
#      train/sa_repr_norm_actor   mean ‖φ‖ (pre-normalize) in actor  pass
#      train/g_repr_norm_actor    mean ‖ψ‖ (pre-normalize) in actor  pass
#      EXPECTED: all four bounded, roughly stationary (O(1)–O(10)).
#      RED FLAG:  any of the four grows monotonically across epochs. That means
#                 the encoder is trying to escape via ‖φ‖·‖ψ‖, and the row-L2
#                 is the ONLY reason actor_loss looks bounded — the underlying
#                 instability is still there, just masked. Report this as a
#                 first-class finding, do not call Phase-1f a pass.
#
#    SECONDARY (constraint accounting — sanity only):
#      train/jhat_c              EXPECTED < 0.15 throughout
#      train/hard_violation_rate EXPECTED ≤ 0.05
#
#    DIAGNOSTIC (τ health check — if these fire, τ is misconfigured):
#      train/critic_loss plateaus at log(batch_size) ≈ 5.5 for > 10 epochs
#      train/accuracy stays at ≈ 1/N throughout
#      → τ too flat; drop to 0.05 and resubmit as phase1f_tau0p05.
#      Opposite: logits saturate near ±10, accuracy ≈ 1 with large pos/neg gap
#      → τ too sharp; bump to 0.2. Either way, one knob per resubmission.
#
#  Pass criterion (same form as Phase-1d, all three must hold):
#    (a) |train/actor_loss|_max  over all 50 epochs  <  1e3
#    (b) no NaN in any loss
#    (c) train/lambda_tilde ≤ 1.0 for ≥ 80 % of epochs (confirms feasible regime
#        was actually reached — the test is well-posed only in that regime)
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
echo "===== PHASE-1f STATIC DIFF VERIFICATION ====="
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

# Exactly two row-L2 normalizations of sa_repr and two of g_repr (one in
# critic_loss_fn, one in actor_loss_fn) — AUTOGRAD-SAFE FORM (epsilon
# INSIDE sqrt). The earlier "+ 1e-8 outside the norm" form was forward-
# finite but autograd-NaN at near-zero rows: jnp.linalg.norm internally
# computes sqrt(sum(x²)), and grad of sqrt(s) at s=0 is +inf, which the
# chain rule then multiplies by upstream zeros to produce NaN. Job 988003
# epoch-1 forensics confirmed the failure (c_grad_nan=1 at sgd-step 0
# with critic FORWARD finite); job 989229 confirmed the fix (every NaN
# flag (never), training stable, gradients huge-but-finite at step 0,
# normal by step 1).
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
# Belt-and-suspenders: the old unsafe form must NOT survive anywhere,
# else the autograd-NaN singularity would re-fire on near-zero rows.
assert "jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-8" not in src, \
    "stale unsafe row-L2 form on sa_repr still present in train.py"
assert "jnp.linalg.norm(g_repr,  axis=-1, keepdims=True) + 1e-8" not in src, \
    "stale unsafe row-L2 form on g_repr still present in train.py"

# Both energies must be divided by args.tau
assert 'jnp.einsum("ik,jk->ij", sa_repr, g_repr) / args.tau' in src, \
    "critic_loss_fn logits must be divided by args.tau (Phase-1f)"
assert "f_sa_g  = jnp.sum(sa_repr * g_repr, axis=-1) / args.tau" in src, \
    "actor_loss_fn f_sa_g must be divided by args.tau (Phase-1f)"

# Probes (kept alive for this 50-epoch run; per the smoke result, they
# cost essentially nothing and give one-line per-epoch confirmation that
# nothing's NaN at any depth/seed).
assert "f\"         nan[obs_c=" in src, \
    "per-epoch forward NaN one-liner missing — probes were stripped"
assert "f\"         grad[c=" in src, \
    "per-epoch grad/param one-liner missing — probes were stripped"
assert "[prefill probe] buffer NaN anywhere:" in src, \
    "post-prefill buffer-NaN probe missing — probes were stripped"
assert "[epoch1 forensics]" in src, \
    "epoch-1 time-series dump missing — probes were stripped"

# The broken Phase-1d claim must be gone
assert "LayerNorm inside the encoders anchors" not in src, \
    "the false 'LayerNorm anchors ‖φ‖,‖ψ‖' comment from Option-A is still in train.py"

# The old negative-L2 energy must be gone (already gone in Option-A, guard anyway)
assert "-jnp.sqrt(_d2 + 1e-12)" not in src, \
    "old negative-L2 energy still present"

print("Phase-1f static diff (autograd-safe row-L2 + probes alive) verified in train.py.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Phase-1f diff verification failed — aborting job."
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
print("GPU OK — proceeding with Phase-1f run.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "JAX GPU check failed — aborting job."
    exit 1
fi

# ── Phase-1f env preflight: same env check as Phase-1d ──────────────────────
echo ""
echo "===== PHASE-1f ENV PREFLIGHT ====="
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

# ── Phase-1f DYNAMIC normalization check ────────────────────────────────────
#  Static grep is necessary but not sufficient. The Option-A failure taught us
#  that a code comment can confidently claim "‖φ‖ is bounded" while the
#  instantiated encoder makes that claim false on a real batch. So: build the
#  encoders, run one batch, and ACTUALLY assert ‖φ̂‖₂ = 1 to 5 decimal places.
echo ""
echo "===== PHASE-1f DYNAMIC NORMALIZATION ASSERTION ====="
"$PYTHON" - <<'PYCHECK'
# Build the actual SA_encoder and G_encoder classes from train.py (safe to
# import: train.py is guarded by `if __name__ == "__main__"`, so importing
# does NOT launch a training run).
#
# Phase-1f CLI config we will launch with:
#   --obs_dim 29    --goal_start_idx 0  --goal_end_idx 3
#   --critic_depth 4   --critic_network_width 256
#   (Ant action_size = 8, so sa input dim = 29 + 8 = 37; goal dim = 3)
import jax, jax.numpy as jnp
from train import SA_encoder, G_encoder, Args

args = Args()  # Phase-1f defaults: tau=0.1, nu_f=1.0, normalize_observations=True

# Phase-1f run uses ant_big_maze → explicit dims (NOT Args.obs_dim which
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

# Apply the EXACT autograd-safe row-L2 normalization the loss functions
# use post-fix (epsilon INSIDE sqrt, so the radicand is strictly positive
# and grad of sqrt is finite even when sum(x²) underflows to 0). This
# mirrors the form in train.py — if you change one, change the other.
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
echo "===== PHASE-1f LAUNCH  $(date) ====="

# Note: Phase-1f defaults live in train.py (tau=0.1, plus all surviving
# Option-A defaults). We do NOT override them on the CLI. Overriding would
# silently defeat the purpose of this run.

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
    --cost_budget_d     0.15 \
    --pid_kp            0.1 \
    --pid_ki            0.003 \
    --pid_kd            0.001 \
    --lambda_max        100.0 \
    --wandb_project     constrained-crl \
    --wandb_group       phase1f_l2norm \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1f DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
