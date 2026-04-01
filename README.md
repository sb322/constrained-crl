# Constrained Contrastive RL (C-CRL)

CMDP extension of [scaling-crl](https://github.com/wang-kevin3290/scaling-crl) with Lagrangian wall-avoidance constraints for maze environments.

## What this adds

This fork extends the original Contrastive RL codebase with a **Constrained MDP (CMDP)** formulation that penalizes wall proximity and collisions during goal-conditioned navigation:

- **Hybrid step cost**: `c(s,a,s') = alpha * 1{collision} + (1-alpha) * exp(-d_wall / sigma)` computed from known maze geometry
- **Cost critic** `Q_c(s,a,g)`: separate network trained with Bellman backups and Polyak-averaged target
- **Lagrangian actor loss**: `L = E[alpha_sac * log pi - Q + lambda * Q_c]`
- **Dual ascent**: `lambda <- max(0, lambda + lr_lambda * (J_c - d_budget))`
- Full **wandb logging** of cost metrics (collision rate, wall distance, lambda, cost critic loss)

The original CRL logic (InfoNCE contrastive critic, deep residual networks, SAC entropy, geometric goal relabeling) is preserved unchanged.

## Installation

```bash
uv sync
```

Then apply the two Brax bug fixes described in the [original repo](https://github.com/wang-kevin3290/scaling-crl#fixing-two-bugs-in-brax-0101).

## Usage

### Constrained training (recommended)

```bash
uv run train.py \
    --env_id "ant_big_maze" \
    --eval_env_id "ant_big_maze_eval" \
    --num_epochs 100 \
    --total_env_steps 100000000 \
    --critic_depth 8 --actor_depth 8 \
    --actor_skip_connections 4 --critic_skip_connections 4 \
    --batch_size 512 \
    --use_constraints True \
    --cost_budget_d 0.1 \
    --alpha_cost 0.5 \
    --sigma_wall 1.0 \
    --lambda_lr 1e-3
```

### Unconstrained baseline

```bash
uv run train.py \
    --env_id "ant_big_maze" \
    --eval_env_id "ant_big_maze_eval" \
    --use_constraints False \
    ...
```

### SLURM (NJIT Wulver)

```bash
sbatch job_constrained.slurm   # single constrained run
sbatch job_baseline.slurm      # unconstrained baseline
sbatch job_sweep.slurm         # budget x lr x seed sweep (12 jobs)
```

## Key constraint hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--use_constraints` | `True` | Toggle CMDP machinery |
| `--alpha_cost` | `0.5` | Weight on binary collision vs. proximity |
| `--sigma_wall` | `1.0` | Length-scale for proximity soft-cost |
| `--contact_threshold` | `0.1` | Distance below which collision = 1 |
| `--cost_budget_d` | `0.1` | Per-step cost budget d |
| `--lambda_init` | `0.0` | Initial Lagrange multiplier |
| `--lambda_lr` | `1e-3` | Dual ascent step size |
| `--lambda_max` | `100.0` | Lambda clamp |
| `--cost_critic_lr` | `3e-4` | Cost critic learning rate |
| `--cost_critic_width` | `256` | Cost critic hidden dim |
| `--cost_critic_depth` | `4` | Cost critic depth |
| `--cost_critic_tau` | `0.005` | Polyak averaging coefficient |
| `--cost_discount` | `0.99` | Cost Bellman discount |

## Project structure

```
constrained_crl/
  train.py              # Modified: adds CostCritic, Lagrangian actor, dual update
  cost_utils.py         # NEW: wall distance, hybrid cost from maze geometry
  buffer.py             # Unchanged from upstream
  evaluator.py          # Unchanged from upstream
  envs/
    __init__.py          # Environment registration
    ant_maze.py          # Unchanged from upstream
    humanoid_maze.py     # Unchanged from upstream
    ant.py               # Compatibility shim
    assets/              # MuJoCo XML files (copy from upstream)
  job_constrained.slurm  # SLURM: single constrained run
  job_baseline.slurm     # SLURM: unconstrained baseline
  job_sweep.slurm        # SLURM: hyperparameter sweep
  pyproject.toml         # Dependencies (same as upstream + cost_utils)
  README.md              # This file
```

## Important: MuJoCo assets

The `envs/assets/` directory must contain the MuJoCo XML model files (`ant_maze.xml`, `humanoid_maze.xml`). Copy these from the original scaling-crl repo:

```bash
cp -r /path/to/scaling-crl/envs/assets envs/
```

## Based on

- [scaling-crl](https://github.com/wang-kevin3290/scaling-crl) — Wang et al., "1000 Layer Networks for Self-Supervised RL", NeurIPS 2025 (Best Paper)
- [JaxGCRL](https://github.com/MichalBorworski/JaxGCRL) — upstream JAX goal-conditioned RL framework
