#!/bin/bash
#SBATCH --job-name=bmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=cluster/logs/node_%x_%j.out
# One whole node per shard: run the shard's tasks 4 at a time with 12 threads each.
# Submit from the project root:  sbatch -p <partition> --export=ALL,SHARD=cluster/shard1.txt cluster/node.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/sunrajp23/env311/bin/python3
export BMD_THREADS=12 OMP_NUM_THREADS=12
mkdir -p cluster/logs/tasks
echo "[$(date)] $(hostname) shard=$SHARD"
xargs -P 4 -L 1 bash -c 'tag=$(echo "$0 $*" | tr " " "_" | tr -d "-"); \
  echo "[$(date +%T)] START $tag"; \
  if '"$PY"' scripts/02_train.py --target $0 --protocol $1 --tier $2 ${@:3} > cluster/logs/tasks/$tag.log 2>&1; \
  then echo "[$(date +%T)] DONE  $tag"; else echo "[$(date +%T)] FAIL  $tag"; fi' < "$SHARD"
echo "[$(date)] shard finished"
