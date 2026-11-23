#!/bin/bash
# Run from the laptop (on IITK VPN):  bash cluster/submit.sh [partition]
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=sanganak; DIR='~/battery'
rsync -az --exclude 'data/raw' --exclude 'results' --exclude 'models' \
      --exclude 'paper' --exclude '__pycache__' ./ "$REMOTE:battery/"
N=$(grep -vc '^#' cluster/jobs.txt)
P=${1:+-p $1}
ssh "$REMOTE" "cd $DIR && mkdir -p cluster/logs && ( [ -d ~/venvs/bmd ] || bash cluster/setup_env.sh ) && cd cluster && sbatch $P --array=1-$N run.sbatch"
