#!/bin/bash
# Run every line of cluster/jobs.txt sequentially on this Mac (resumable).
#   nohup caffeinate -is bash cluster/run_local.sh > cluster/logs/local.log 2>&1 &
# Finished tasks leave a marker in cluster/logs/done/ and are skipped on restart.
# Models are saved only for the main t2 DOI runs (used by 04_predict.py) to save disk.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs/done
i=0
grep -v '^#' cluster/jobs.txt | while read -r target protocol tier extra; do
  i=$((i + 1))
  tag="${i}_${target}_${protocol}_${tier}${extra:+_${extra//[^a-z]/}}"
  [ -e "cluster/logs/done/$tag" ] && { echo "skip $tag"; continue; }
  save=""
  if [[ "$protocol" != "doi" || "$tier" != "t2" || "$extra" == *drop-series* ]] && [[ "$extra" != *no-save-models* ]]; then
    save="--no-save-models"
  fi
  echo "[$(date '+%F %T')] START $tag"
  if python3 scripts/02_train.py --target $target --protocol $protocol --tier $tier $extra $save \
       > "cluster/logs/$tag.log" 2>&1; then
    touch "cluster/logs/done/$tag"; echo "[$(date '+%F %T')] DONE  $tag"
  else
    echo "[$(date '+%F %T')] FAIL  $tag (see cluster/logs/$tag.log)"
  fi
done
echo "[$(date '+%F %T')] all tasks finished"
