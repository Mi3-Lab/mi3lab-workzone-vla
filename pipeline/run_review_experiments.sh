#!/bin/bash
# Reviewer-requested experiment chain (P2 ablations + P6 greedy), run
# sequentially after the paired YOLO+CLIP comparison releases the GPU.
# Each run: full 208-video validation split, latency-honest protocol,
# own cache dir + log. ~2h each, ~12h total.
set -uo pipefail
cd "$(dirname "$0")"

while pgrep -f "evaluate_yolo_baseline.py" > /dev/null; do sleep 30; done

log() { echo "[$(date '+%F %T')] $1" >> ~/review_experiments_progress.log; }
run() { # $1=name  $2...=extra flags
  local name=$1; shift
  log "START $name"
  mkdir -p ~/eval_cache/vlm_$name
  python3 -u evaluate_cascade.py --split-file eval_split_full.json --split validation \
      --cache-dir ~/eval_cache/vlm_$name "$@" > ~/eval_abl_$name.log 2>&1
  log "DONE $name (exit $?)"
}

run greedy          --temperature 0
run nosign          --no-sign
run binarysign      --binary-sign
run nofastentry     --no-fast-entry
run nofilter        --no-qualifier-filter
run unrestbayes     --unrestricted-bayes

log "ALL DONE"
touch ~/review_experiments_ALLDONE.marker
