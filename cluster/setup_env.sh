#!/bin/bash
# One-time: create the venv on a login node.
set -euo pipefail
module load python/3.11 2>/dev/null || module load anaconda3 2>/dev/null || true
python3 -m venv "$HOME/venvs/bmd"
source "$HOME/venvs/bmd/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$(dirname "$0")/../requirements.txt"
python -c "import catboost, xgboost, pymatgen, pandas; print('env ok')"
