#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  setup_repo.sh — Create the GitHub repo and push all files
# ─────────────────────────────────────────────────────────────
#
#  Prerequisites:
#    1. GitHub CLI installed:  brew install gh  (or see https://cli.github.com)
#    2. Logged in:  gh auth login
#    3. Copy envs/assets/ from scaling-crl (see step below)
#
#  Usage:
#    cd constrained_crl/
#    bash setup_repo.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

REPO_NAME="constrained-crl"
REPO_DESC="CMDP extension of scaling-crl with Lagrangian wall-avoidance constraints for maze environments"

echo "=== Step 1: Create GitHub repo ==="
gh repo create "$REPO_NAME" --public --description "$REPO_DESC" --confirm 2>/dev/null || \
gh repo create "$REPO_NAME" --public -d "$REPO_DESC" 2>/dev/null || \
echo "Repo may already exist, continuing..."

echo "=== Step 2: Initialize git ==="
git init
git branch -M main

echo "=== Step 3: Add remote ==="
GITHUB_USER=$(gh api user --jq .login)
git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git" 2>/dev/null || \
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "=== Step 4: Create .gitignore ==="
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Weights & Biases
wandb/

# Checkpoints (large files)
checkpoints/

# Logs
logs/
*.out
*.err

# uv
.python-version
uv.lock
GITIGNORE

echo "=== Step 5: Stage all files ==="
git add -A

echo "=== Step 6: Commit ==="
git commit -m "Initial commit: Constrained CRL (CMDP extension of scaling-crl)

- Modified train.py with CostCritic, Lagrangian actor loss, dual ascent
- New cost_utils.py: wall distance + hybrid cost from maze geometry
- SLURM scripts for NJIT Wulver HPC (constrained, baseline, sweep)
- Original CRL logic preserved unchanged (buffer.py, evaluator.py, envs/)
"

echo "=== Step 7: Push ==="
git push -u origin main

echo ""
echo "Done! Your repo is at: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""
echo "IMPORTANT: Copy the MuJoCo asset files from scaling-crl:"
echo "  git clone https://github.com/wang-kevin3290/scaling-crl.git /tmp/scaling-crl"
echo "  cp -r /tmp/scaling-crl/envs/assets envs/"
echo "  git add envs/assets && git commit -m 'Add MuJoCo XML assets' && git push"
