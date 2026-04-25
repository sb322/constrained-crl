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
#SBATCH --time=01:00:00
# ── Phase-1f NaN forensics (probed re-run).
#    Same config as slurm_phase1f_smoke_tau1p0.sh: τ=1.0, 3 epochs, 300K steps.
#    New payload: train.py now emits per-epoch NaN-flag probes (obs/sa/g
#    NaN in the critic pass; sa/g/action/f NaN in the actor pass; pre-norm
#    minima ‖φ‖min, ‖ψ‖min; max absolute action). Purpose is to localize
#    the NaN origin that the prior tau=1.0 smoke established is NOT
#    τ-amplified gradient blow-up (τ=0.1 and τ=1.0 produced identical
#    step-1 NaN signatures, so the driver is upstream of the InfoNCE
#    division).
#
#  Expected routing from the one-liner probe print at epoch 1:
#    obs_nan_c=1          → buffer contamination (env emitted NaN)
#    sa_nan_c=1/g_nan_c=1 → encoder forward NaN on finite input
#    *_norm_min ≈ 0       → row-L2 divides by +1e-8 on a zero-length vector
#    logits_nan_c=1 alone → cosine / τ pathology despite finite φ̂, ψ̂
#    action_nan_a=1       → actor produces NaN action at step 0/1
#    |a|max_a > 1 + ε     → tanh-saturation precursor
#
#  No causal knob is changed relative to the prior smoke — this is pure
#  observability. The run MUST NaN identically to slurm_phase1f_smoke_tau1p0
#  (job 981715). If it NaNs differently, the patch introduced a bug.

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE-1f PROBED SMOKE — NaN forensics on τ=1.0 config.
#  Context. Two prior smokes (tau=0.1 full at job 980196 over 20/50 epochs,
#  tau=1.0 smoke at job 981715 over 3/3 epochs) both produced:
#       c_loss = NaN from epoch 1
#       acc = 0.00391 = 1/256          (argmax over NaN logits → always 0)
#       a_loss = NaN, cost = NaN, hard_viol = 0 (NaN < ε returns False)
#  This run is one commit of new diagnostics on top; no causal knob changed.
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

# --- Phase-1f base row-L2 (still required; the probes don't remove it) -------
assert src.count(
    "sa_repr = sa_repr / (jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-8)"
) == 2, "row-L2 normalize of sa_repr must appear in BOTH critic_loss_fn and actor_loss_fn"
assert src.count(
    "g_repr  = g_repr  / (jnp.linalg.norm(g_repr,  axis=-1, keepdims=True) + 1e-8)"
) == 2, "row-L2 normalize of g_repr must appear in BOTH critic_loss_fn and actor_loss_fn"
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
        "                            action_has_nan_a, action_max_a, f_has_nan_a)") in src, \
    "actor_loss_fn aux return is missing the 7 probe scalars"

# 4. Call-site unpacking
assert ("(c_loss, (lp, ln, acc, lse, sa_rn_crit, g_rn_crit,\n"
        "                  sa_nan_c, g_nan_c, obs_nan_c,\n"
        "                  sa_nmin_c, g_nmin_c, logits_nan_c))") in src, \
    "critic_loss_fn call site does not unpack the 6 new probe scalars"
assert ("(a_loss, (log_prob, mean_qc_pi, sa_rn_act, g_rn_act,\n"
        "                  sa_nan_a, g_nan_a, sa_nmin_a, g_nmin_a,\n"
        "                  action_nan_a, action_max_a, f_nan_a))") in src, \
    "actor_loss_fn call site does not unpack the 7 new probe scalars"

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

print("Phase-1f base fix + forward NaN probes + prefill probe + grad/param NaN probes + epoch-1 time-series dump verified in train.py.")
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

# τ=1.0 matches slurm_phase1f_smoke_tau1p0.sh exactly. The new payload is the
# probe instrumentation inside train.py, not any CLI knob. 3 epochs, ~40 min.

"$PYTHON" train.py \
    --env_id            ant_big_maze \
    --eval_env_id       ant_big_maze_eval \
    --episode_length    1000 \
    --total_env_steps   300000 \
    --num_epochs        3 \
    --tau               1.0 \
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
    --wandb_group       phase1f_smoke_probes \
    --track             True

EXIT_CODE=$?
echo ""
echo "===== PHASE-1f PROBED SMOKE DONE  $(date)  exit=$EXIT_CODE ====="
exit $EXIT_CODE
