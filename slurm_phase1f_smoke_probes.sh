#!/bin/bash -l
#SBATCH --job-name=ccrl_1f_probes
#SBATCH --output=phase1f_smoke_probes.%j.out
#SBATCH --error=phase1f_smoke_probes.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# ── Phase-1g actor-loss component forensics (PRODUCTION-knobs smoke).
#    Knobs match production: τ=0.1, 15 epochs, 1.5M env steps. Reuses the
#    `phase1f_smoke_probes` job name and output filenames so existing
#    diagnostic-pull scripts continue to work, but the EXPERIMENTAL
#    QUESTION has changed since the NaN cascade was closed by the
#    autograd-safe row-L2 fix.
#
#    Question this run answers:
#       Phase-1f at d=0.15 (job 990871, 50 epochs) ran with no NaN but
#       a_loss = −484K by epoch 50. Since
#           a_loss = α·mean(log_p) − mean(f_sa_g)/ν_f
#       and mean(f_sa_g)/ν_f ≤ 10/ν_f under row-L2, the −484K must
#       originate in α·mean(log_p). NEW probes split this into:
#         alpha       = α = exp(log_α)             unbounded above?
#         log_p       = mean(log_prob)             distribution density
#         α·log_p     = product                    direct read on driver
#         gauss_lp    = mean(Gaussian logpdf)      Gaussian density only
#         sat_corr    = mean(−Σ log((1−a²)+1e-6))  tanh-Jacobian term
#         log_std     = mean(log σ)                policy concentration
#         f_term      = mean(f_sa_g)/ν_f           reward-critic term
#       (sum-decomposes log_p = gauss_lp + sat_corr).
#
#    Expected behavior at 15 epochs (extrapolating the production trajectory
#    a_loss[1]=+1, a_loss[5]=−10.8, a_loss[15]≈ −80 to −300):
#       • |a_loss| should be in the range [50, 500].
#       • Components should make exactly one of the four candidate sources
#         dominant by epoch 15. Read the third per-epoch print line
#         (`actor[α=… log_p=… α·log_p=… gauss_lp=… sat_corr=… …]`) and
#         flag the term whose magnitude tracks a_loss.
#
#    Wallclock budget: 15 epochs at ~13 min/epoch (production rate) + JIT
#    cache warmup ≈ 4 h. Walltime set to 04:00:00 with safety margin.
#
#    No causal knob changed in train.py losses, optimizer, or environment.
#    The only changes vs job 990871 are (a) duration shortened from 50→15
#    epochs and (b) the actor-loss component probes added.

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE-1g ACTOR-LOSS COMPONENT FORENSICS (production-knobs smoke).
#  Context. Job 990871 (Phase-1f, 50 epochs, d=0.15) showed a_loss diverged
#  to −484K despite zero NaN, with all forward NaN flags clean and row-L2 +
#  τ=0.1 holding. The Cauchy-Schwarz bound on f_sa_g rules out the
#  reward-critic term as the source, so the divergence is in α·log_p.
#  This run instruments the components to identify which sub-term carries
#  the magnitude — α, log_p, the Gaussian piece, the tanh-saturation piece,
#  or log σ. The result determines the Phase-2 fix surface (clip α, switch
#  to numerically-stable softplus tanh-Jacobian, smooth actions, or constrain
#  σ from below differently).
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
echo "===== PHASE-1f STATIC DIFF + PROBE VERIFICATION ====="
"$PYTHON" - <<'PYCHECK'
# Verify (a) the Phase-1f base fix is still present, AND (b) the new
# NaN-forensics probes are wired through into the loss aux tuples, the
# per-step metrics dict, and the per-epoch log_dict. If any assert fires,
# the wrong train.py is on PYTHONPATH or the probe patch was dropped.
import pathlib
src = pathlib.Path("train.py").read_text()

# --- Surviving Option-A + Phase-1f knobs -------------------------------------
assert "normalize_observations: bool = True" in src, \
    "normalize_observations default is not True"
assert "nu_f: float = 1.0" in src, \
    "nu_f default is not 1.0"
assert "tau: float = 0.1" in src, \
    "args.tau default is not 0.1"

# --- Phase-1f base row-L2 with autograd-safe sqrt (NEW: epsilon INSIDE sqrt) -
# Original form was sa_repr / (jnp.linalg.norm(sa_repr) + 1e-8). That form is
# forward-finite but autograd-NaN whenever sum(sa²) underflows to 0, because
# JAX's gradient of sqrt(s) at s=0 is +inf. Fix: compute the norm as
# sqrt(sum(x²) + 1e-12) so the radicand is strictly positive. Job 988003
# epoch-1 forensics ⇒ critic forward finite at sgd=0, c_grad_nan = 1 at
# sgd=0 — exact textbook signature of this autograd singularity.
assert src.count(
    "sa_norm_safe = jnp.sqrt("
) == 2, "autograd-safe row-L2 (sa_norm_safe = sqrt(sum + eps)) must appear in BOTH losses"
assert src.count(
    "g_norm_safe  = jnp.sqrt("
) == 2, "autograd-safe row-L2 (g_norm_safe = sqrt(sum + eps)) must appear in BOTH losses"
assert src.count(
    "jnp.sum(sa_repr * sa_repr, axis=-1, keepdims=True) + 1e-12"
) == 2, "sa_repr safe-norm radicand (sum(sa²)+1e-12) must appear in both losses"
assert src.count(
    "jnp.sum(g_repr  * g_repr,  axis=-1, keepdims=True) + 1e-12"
) == 2, "g_repr safe-norm radicand (sum(g²)+1e-12) must appear in both losses"
assert src.count("sa_repr = sa_repr / sa_norm_safe") == 2, \
    "sa_repr divide-by-safe-norm must appear in both losses"
assert src.count("g_repr  = g_repr  / g_norm_safe")  == 2, \
    "g_repr divide-by-safe-norm must appear in both losses"
# Belt-and-suspenders: the OLD unsafe form must NOT survive anywhere, else
# we have a partial fix and the autograd-NaN singularity would re-fire.
assert "jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-8" not in src, \
    "stale unsafe row-L2 form on sa_repr still present — fix did not land"
assert "jnp.linalg.norm(g_repr,  axis=-1, keepdims=True) + 1e-8" not in src, \
    "stale unsafe row-L2 form on g_repr still present — fix did not land"
assert 'jnp.einsum("ik,jk->ij", sa_repr, g_repr) / args.tau' in src, \
    "critic InfoNCE logits must be divided by args.tau"
assert "f_sa_g  = jnp.sum(sa_repr * g_repr, axis=-1) / args.tau" in src, \
    "actor f_sa_g must be divided by args.tau"

# --- NaN-forensics probe wiring (the new payload for this run) ---------------
# 1. Critic-side probe computation
for probe, where in [
    ("sa_has_nan_c  = jnp.any(jnp.isnan(sa_repr)).astype(jnp.float32)",
     "critic sa NaN flag"),
    ("g_has_nan_c   = jnp.any(jnp.isnan(g_repr)).astype(jnp.float32)",
     "critic g NaN flag"),
    ("obs_has_nan_c = jnp.any(jnp.isnan(obs)).astype(jnp.float32)",
     "critic-pass obs NaN flag (buffer contamination detector)"),
    ("sa_norm_min_c = jnp.min(jnp.linalg.norm(sa_repr, axis=-1))",
     "critic sa pre-norm min"),
    ("g_norm_min_c  = jnp.min(jnp.linalg.norm(g_repr,  axis=-1))",
     "critic g pre-norm min"),
    ("logits_has_nan_c = jnp.any(jnp.isnan(logits)).astype(jnp.float32)",
     "post-division logits NaN flag"),
    # 2. Actor-side probe computation
    ("action_has_nan_a = jnp.any(jnp.isnan(action)).astype(jnp.float32)",
     "actor action NaN flag"),
    ("action_max_a     = jnp.max(jnp.abs(action))",
     "actor |a| max"),
    ("sa_has_nan_a     = jnp.any(jnp.isnan(sa_repr)).astype(jnp.float32)",
     "actor sa NaN flag"),
    ("g_has_nan_a      = jnp.any(jnp.isnan(g_repr)).astype(jnp.float32)",
     "actor g NaN flag"),
    ("sa_norm_min_a    = jnp.min(jnp.linalg.norm(sa_repr, axis=-1))",
     "actor sa pre-norm min"),
    ("g_norm_min_a     = jnp.min(jnp.linalg.norm(g_repr,  axis=-1))",
     "actor g pre-norm min"),
    ("f_has_nan_a = jnp.any(jnp.isnan(f_sa_g)).astype(jnp.float32)",
     "actor f (post-τ division) NaN flag"),
]:
    assert probe in src, f"NaN-forensics probe missing — {where}"

# 3. Aux-tuple returns extended
assert ("sa_has_nan_c, g_has_nan_c, obs_has_nan_c,\n"
        "                      sa_norm_min_c, g_norm_min_c, logits_has_nan_c)") in src, \
    "critic_loss_fn aux return is missing the 6 probe scalars"
assert ("sa_has_nan_a, g_has_nan_a,\n"
        "                            sa_norm_min_a, g_norm_min_a,\n"
        "                            action_has_nan_a, action_max_a, f_has_nan_a,") in src, \
    ("actor_loss_fn aux return is missing the 7 NaN-forensics probe scalars "
     "(NB: the closing `)` moved when Phase-1g added the 7 actor-component "
     "probes after f_has_nan_a; this assert now matches `f_has_nan_a,` "
     "since more probes follow on the next lines).")

# 4. Call-site unpacking
assert ("(c_loss, (lp, ln, acc, lse, sa_rn_crit, g_rn_crit,\n"
        "                  sa_nan_c, g_nan_c, obs_nan_c,\n"
        "                  sa_nmin_c, g_nmin_c, logits_nan_c))") in src, \
    "critic_loss_fn call site does not unpack the 6 new probe scalars"
assert ("(a_loss, (log_prob, mean_qc_pi, sa_rn_act, g_rn_act,\n"
        "                  sa_nan_a, g_nan_a, sa_nmin_a, g_nmin_a,\n"
        "                  action_nan_a, action_max_a, f_nan_a,") in src, \
    ("actor_loss_fn call site does not unpack the 7 NaN-forensics probe "
     "scalars (NB: closing `))` moved when Phase-1g added the 7 actor-"
     "component probes after f_nan_a; this assert now matches `f_nan_a,` "
     "since more unpack tokens follow on the next lines).")

# 5. Metrics dict keys
for key, where in [
    ('"nan_obs_critic":      obs_nan_c,',       "metrics: nan_obs_critic"),
    ('"nan_sa_critic":       sa_nan_c,',        "metrics: nan_sa_critic"),
    ('"nan_g_critic":        g_nan_c,',         "metrics: nan_g_critic"),
    ('"nan_logits_critic":   logits_nan_c,',    "metrics: nan_logits_critic"),
    ('"sa_norm_min_critic":  sa_nmin_c,',       "metrics: sa_norm_min_critic"),
    ('"g_norm_min_critic":   g_nmin_c,',        "metrics: g_norm_min_critic"),
    ('"nan_sa_actor":        sa_nan_a,',        "metrics: nan_sa_actor"),
    ('"nan_g_actor":         g_nan_a,',         "metrics: nan_g_actor"),
    ('"nan_action_actor":    action_nan_a,',    "metrics: nan_action_actor"),
    ('"nan_f_actor":         f_nan_a,',         "metrics: nan_f_actor"),
    ('"sa_norm_min_actor":   sa_nmin_a,',       "metrics: sa_norm_min_actor"),
    ('"g_norm_min_actor":    g_nmin_a,',        "metrics: g_norm_min_actor"),
    ('"action_max_actor":    action_max_a,',    "metrics: action_max_actor"),
]:
    assert key in src, f"metrics dict missing — {where}"

# 6. Per-epoch print contains the probe one-liner
assert "f\"         nan[obs_c=" in src, \
    "per-epoch probe one-liner missing from training print"

# 7. Post-prefill probe (one-shot, before training loop). Localizes whether
#    NaN enters the buffer during prefill (env emit under random action) or
#    after prefill (first training step poisons actor params, next rollout
#    contaminates buffer).
assert "[prefill probe] buffer.data shape=" in src, \
    "post-prefill buffer-shape probe missing from train.py"
assert "[prefill probe] buffer NaN anywhere:" in src, \
    "post-prefill buffer-NaN probe missing from train.py"
assert "[prefill probe] env_state.obs NaN:" in src, \
    "post-prefill env_state.obs NaN probe missing from train.py"

# 8. Gradient/post-update param NaN forensics (the new payload for THIS run).
#    We localize *which* of the three gradient steps within sgd_step first
#    produces NaN. Reading order: forward NaN flag set + grad_nan = 0
#    → corruption rode in via buffer. grad_nan = 1 → autograd path of that
#    loss is the source. grad_nan = 0 + params_nan = 1 → optimizer apply
#    produced NaN params. Both must be wired through sgd_step → metrics →
#    log_dict → print.

# 8a. Helper functions
assert "def _grads_have_nan(grads):" in src, \
    "_grads_have_nan helper missing"
assert "def _grads_global_norm(grads):" in src, \
    "_grads_global_norm helper missing"
assert "def _params_have_nan(params):" in src, \
    "_params_have_nan helper missing"

# 8b. Probe computation (3 sites: critic, actor, cost critic)
for line, where in [
    ("c_grad_nan  = _grads_have_nan(c_grads)",       "critic grad NaN"),
    ("c_grad_norm = _grads_global_norm(c_grads)",    "critic grad norm"),
    ("c_params_nan = _params_have_nan(critic_state.params)",
     "critic post-update params NaN"),
    ("a_grad_nan  = _grads_have_nan(a_grads)",       "actor grad NaN"),
    ("a_grad_norm = _grads_global_norm(a_grads)",    "actor grad norm"),
    ("a_params_nan = _params_have_nan(actor_state.params)",
     "actor post-update params NaN"),
    ("cc_grad_nan  = _grads_have_nan(cc_grads)",     "cost-critic grad NaN"),
    ("cc_grad_norm = _grads_global_norm(cc_grads)",  "cost-critic grad norm"),
    ("cc_params_nan = _params_have_nan(cost_critic_state_new.params)",
     "cost-critic post-update params NaN"),
]:
    assert line in src, f"grad/param probe computation missing — {where}"

# 8c. Zero-placeholders in the no-constraints branch (so jax.lax.scan sees
#     a consistent metrics pytree shape across both code paths)
assert "cc_grad_nan   = jnp.zeros(())" in src, \
    "cc_grad_nan zero placeholder missing in no-constraints branch"
assert "cc_grad_norm  = jnp.zeros(())" in src, \
    "cc_grad_norm zero placeholder missing in no-constraints branch"
assert "cc_params_nan = jnp.zeros(())" in src, \
    "cc_params_nan zero placeholder missing in no-constraints branch"

# 8d. Metrics dict keys (sgd_step return)
for key, where in [
    ('"c_grad_nan":     c_grad_nan,',   "metrics: c_grad_nan"),
    ('"c_grad_norm":    c_grad_norm,',  "metrics: c_grad_norm"),
    ('"c_params_nan":   c_params_nan,', "metrics: c_params_nan"),
    ('"a_grad_nan":     a_grad_nan,',   "metrics: a_grad_nan"),
    ('"a_grad_norm":    a_grad_norm,',  "metrics: a_grad_norm"),
    ('"a_params_nan":   a_params_nan,', "metrics: a_params_nan"),
    ('"cc_grad_nan":    cc_grad_nan,',  "metrics: cc_grad_nan"),
    ('"cc_grad_norm":   cc_grad_norm,', "metrics: cc_grad_norm"),
    ('"cc_params_nan":  cc_params_nan,',"metrics: cc_params_nan"),
]:
    assert key in src, f"metrics dict missing grad/param key — {where}"

# 8e. log_dict aggregations (max for NaN flags; max for grad norms — runaway
#     spike must surface even if mean is finite)
for key, where in [
    ('"c_grad_nan":     float(jnp.max(epoch_metrics["c_grad_nan"])),',
     "log_dict: c_grad_nan"),
    ('"c_grad_norm":    float(jnp.max(epoch_metrics["c_grad_norm"])),',
     "log_dict: c_grad_norm"),
    ('"c_params_nan":   float(jnp.max(epoch_metrics["c_params_nan"])),',
     "log_dict: c_params_nan"),
    ('"a_grad_nan":     float(jnp.max(epoch_metrics["a_grad_nan"])),',
     "log_dict: a_grad_nan"),
    ('"a_grad_norm":    float(jnp.max(epoch_metrics["a_grad_norm"])),',
     "log_dict: a_grad_norm"),
    ('"a_params_nan":   float(jnp.max(epoch_metrics["a_params_nan"])),',
     "log_dict: a_params_nan"),
    ('"cc_grad_nan":    float(jnp.max(epoch_metrics["cc_grad_nan"])),',
     "log_dict: cc_grad_nan"),
    ('"cc_grad_norm":   float(jnp.max(epoch_metrics["cc_grad_norm"])),',
     "log_dict: cc_grad_norm"),
    ('"cc_params_nan":  float(jnp.max(epoch_metrics["cc_params_nan"])),',
     "log_dict: cc_params_nan"),
]:
    assert key in src, f"log_dict missing grad/param aggregation — {where}"

# 8f. Per-epoch grad/param print line
assert "f\"         grad[c=" in src, \
    "per-epoch grad/param print line missing from training loop"

# 9. Epoch-1 time-series forensics (host-side, one-shot at epoch == 0).
#    Job 985769 confirmed every NaN flag fires by end of epoch 1, but
#    per-epoch jnp.max cannot localize WHICH sgd-step within epoch 1
#    fired first. This dump emits the flat index of the first 1-flag
#    for each NaN signal plus the first 5 grad-norm values, so we can
#    distinguish (a) loss autograd is broken from clean init from
#    (b) optimizer apply poisons params later in the epoch.
assert "[epoch1 forensics]" in src, \
    "epoch-1 forensics one-shot dump missing from training loop"
assert "def _first_one_idx(arr):" in src, \
    "epoch-1 forensics _first_one_idx helper missing"
assert "def _decode(flat_idx):" in src, \
    "epoch-1 forensics _decode helper missing"
assert "sgd_per_step = args.num_update_epochs * args.num_minibatches" in src, \
    "epoch-1 forensics sgd_per_step calculation missing"
# Confirm key signals are queried in the dump (sample three to keep the
# assert tight without enumerating all 13).
for query, where in [
    ('"c_grad_nan",       "c_grad_nan"',     "c_grad_nan first-index query"),
    ('"c_params_nan",     "c_params_nan"',   "c_params_nan first-index query"),
    ('"nan_obs_critic",   "nan_obs_critic"', "nan_obs_critic first-index query"),
]:
    assert query in src, f"epoch-1 forensics missing — {where}"
# Confirm grad-norm head dump is wired (catches dropping the [:5] block)
assert '"c_grad_norm",   "c_grad_norm"' in src, \
    "epoch-1 forensics grad_norm head dump missing"

# 10. Phase-1g actor-loss component forensics. Job 990871 confirmed
#     a_loss = −484K with no NaN and row-L2 holding f_sa_g ∈ [−10, 10].
#     The −484K must originate in α·mean(log_p). These probes split that
#     into α, log_p, the Gaussian piece, the tanh-saturation piece, and
#     log σ. The smoke runs 15 epochs at production τ=0.1; even if a_loss
#     hasn't reached −484K by epoch 15, the relative magnitudes of the
#     components identify the divergent term unambiguously.

# 10a. Probe definitions inside actor_loss_fn
for probe, where in [
    ("sat_correction_per_dim = jnp.log((1 - jnp.square(action)) + 1e-6)",
     "saturation-correction per-dim probe"),
    ("gaussian_logp_full = jax.scipy.stats.norm.logpdf(",
     "Gaussian logpdf full probe"),
    ("alpha_metric        = alpha",
     "alpha probe"),
    ("log_prob_mean       = jnp.mean(log_prob)",
     "mean(log_prob) probe"),
    ("alpha_logprob_mean  = alpha * log_prob_mean",
     "α·log_prob probe"),
    ("gaussian_logp_mean  = jnp.mean(gaussian_logp_full)",
     "mean Gaussian logpdf probe"),
    ("sat_correction_mean = jnp.mean(sat_correction_full)",
     "mean saturation correction probe"),
    ("log_std_mean        = jnp.mean(log_stds)",
     "mean log_std probe"),
    ("f_term_mean         = jnp.mean(f_sa_g) / args.nu_f",
     "f_term mean probe"),
]:
    assert probe in src, f"actor-loss component probe missing — {where}"

# 10b. Aux tuple extension
assert ("alpha_metric, log_prob_mean, alpha_logprob_mean,\n"
        "                            gaussian_logp_mean, sat_correction_mean,\n"
        "                            log_std_mean, f_term_mean)") in src, \
    "actor_loss_fn aux return missing the 7 new component probes"

# 10c. Call-site unpacking
assert ("alpha_metric_a, log_prob_mean_a, alpha_logprob_mean_a,\n"
        "                  gaussian_logp_mean_a, sat_correction_mean_a,\n"
        "                  log_std_mean_a, f_term_mean_a)") in src, \
    "actor_loss_fn call site does not unpack the 7 new component probes"

# 10d. Metrics dict keys
for k in [
    '"alpha_actor":           alpha_metric_a,',
    '"log_prob_mean_actor":   log_prob_mean_a,',
    '"alpha_logprob_actor":   alpha_logprob_mean_a,',
    '"gaussian_logp_actor":   gaussian_logp_mean_a,',
    '"sat_correction_actor":  sat_correction_mean_a,',
    '"log_std_mean_actor":    log_std_mean_a,',
    '"f_term_mean_actor":     f_term_mean_a,',
]:
    assert k in src, f"metrics dict missing actor component — {k}"

# 10e. log_dict aggregation (mean)
for k in [
    '"alpha_actor":          float(jnp.mean(epoch_metrics["alpha_actor"])),',
    '"log_prob_mean_actor":  float(jnp.mean(epoch_metrics["log_prob_mean_actor"])),',
    '"alpha_logprob_actor":  float(jnp.mean(epoch_metrics["alpha_logprob_actor"])),',
    '"gaussian_logp_actor":  float(jnp.mean(epoch_metrics["gaussian_logp_actor"])),',
    '"sat_correction_actor": float(jnp.mean(epoch_metrics["sat_correction_actor"])),',
    '"log_std_mean_actor":   float(jnp.mean(epoch_metrics["log_std_mean_actor"])),',
    '"f_term_mean_actor":    float(jnp.mean(epoch_metrics["f_term_mean_actor"])),',
]:
    assert k in src, f"log_dict missing actor component aggregation — {k}"

# 10f. Per-epoch actor-component print line
assert "f\"         actor[α=" in src, \
    "per-epoch actor-component print line missing from training loop"

print("Phase-1f base fix + forward NaN probes + prefill probe + grad/param NaN probes + epoch-1 time-series dump + actor-component probes verified in train.py.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Phase-1f probe verification failed — aborting job."
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
print("GPU OK — proceeding with Phase-1f probed smoke.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "JAX GPU check failed — aborting job."
    exit 1
fi

# ── Phase-1f env preflight (inherited from smoke_tau1p0; unchanged) ─────────
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

echo ""
echo "===== PHASE-1f PROBED SMOKE LAUNCH  $(date) ====="

# Production knobs (τ=0.1 default, depth=4 everywhere, batch=256, etc.).
# Only changes vs job 990871 are duration: 15 epochs / 1.5M env steps.
# τ is NOT overridden — we want the production τ=0.1 default to drive the
# same divergence we observed at scale.

"$PYTHON" train.py \
    --env_id            ant_big_maze \
    --eval_env_id       ant_big_maze_eval \
    --episode_length    1000 \
    --total_env_steps   1500000 \
    --num_epochs        15 \
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
    --wandb_group       phase1g_actor_components \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1g ACTOR-COMPONENT SMOKE DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
