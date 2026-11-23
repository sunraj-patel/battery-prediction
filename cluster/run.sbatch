#!/bin/bash
#SBATCH --job-name=bmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Partition/account: set via `sbatch -p <partition>` in submit.sh if the default queue is not right.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."
source cluster/env.sh
line=$(grep -v '^#' cluster/jobs.txt | sed -n "$((SLURM_ARRAY_TASK_ID))p")
read -r target protocol tier extra <<<"$line"
echo "[$(date)] task $SLURM_ARRAY_TASK_ID: $target $protocol $tier $extra on $(hostname)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/02_train.py --target $target --protocol $protocol --tier $tier $extra
echo "[$(date)] done"
