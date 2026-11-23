#!/bin/bash
# Run one shard file sequentially on this Mac (all cores per task).  bash cluster/run_shard_local.sh cluster/shard4.txt
cd "$(dirname "$0")/.."; mkdir -p cluster/logs/tasks
while read -r t p tier rest; do
  tag=$(echo "$t $p $tier $rest" | tr " " "_" | tr -d "-")
  echo "[$(date +%T)] START $tag"
  if python3 scripts/02_train.py --target $t --protocol $p --tier $tier $rest > cluster/logs/tasks/$tag.log 2>&1
  then echo "[$(date +%T)] DONE  $tag"; else echo "[$(date +%T)] FAIL  $tag"; fi
done < "$1"
echo "[$(date +%T)] shard finished"
