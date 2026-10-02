#!/bin/bash
# Master sequential runner: VLM + YOLO-only over ALL 520 annotated videos
# (eval_split_full.json: 312 calibration / 208 validation). YOLO+CLIP skipped
# for now (not needed). Now also reports paper-exact metrics (arXiv:2606.08860
# Sec V: event-level P/R, per-state IoU, entry timing offset, transition-
# tolerance table) via paper_metrics.py, and caches each video's per-frame
# predicted/gt arrays to ~/eval_cache/{vlm,yolo}/ so future metric additions
# never require re-running inference again.
# Must run sequentially -- both systems are GPU-bound and the eval harnesses
# use real per-frame processing latency to drive the simulated camera clock,
# so running two at once would distort both systems' timing-based sampling.
set -uo pipefail
cd "$(dirname "$0")"

mkdir -p ~/eval_cache/vlm ~/eval_cache/yolo

echo "=== starting full 520-video eval chain v2 (paper metrics + cache): $(date) ===" >> ~/eval_full_progress.log

echo "[1/4] VLM calibration (312 videos)..." >> ~/eval_full_progress.log
python3 -u evaluate_cascade.py --split-file eval_split_full.json --split calibration \
    --cache-dir ~/eval_cache/vlm > ~/eval_full_vlm_calibration.log 2>&1
echo "[1/4] done: $(date)" >> ~/eval_full_progress.log

echo "[2/4] VLM validation (208 videos)..." >> ~/eval_full_progress.log
python3 -u evaluate_cascade.py --split-file eval_split_full.json --split validation \
    --cache-dir ~/eval_cache/vlm > ~/eval_full_vlm_validation.log 2>&1
echo "[2/4] done: $(date)" >> ~/eval_full_progress.log

echo "[3/4] YOLO-only calibration (312 videos)..." >> ~/eval_full_progress.log
~/workzone/venv/bin/python -u evaluate_yolo_baseline.py --split-file eval_split_full.json --split calibration \
    --cache-dir ~/eval_cache/yolo > ~/eval_full_yolo_calibration.log 2>&1
echo "[3/4] done: $(date)" >> ~/eval_full_progress.log

echo "[4/4] YOLO-only validation (208 videos)..." >> ~/eval_full_progress.log
~/workzone/venv/bin/python -u evaluate_yolo_baseline.py --split-file eval_split_full.json --split validation \
    --cache-dir ~/eval_cache/yolo > ~/eval_full_yolo_validation.log 2>&1
echo "[4/4] done: $(date)" >> ~/eval_full_progress.log

echo "=== ALL DONE: $(date) ===" >> ~/eval_full_progress.log
touch ~/eval_full_ALLDONE_v2.marker
