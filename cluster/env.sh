# Activates the Python env on the cluster (created once by setup_env.sh).
module load python/3.11 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$HOME/venvs/bmd/bin/activate"
