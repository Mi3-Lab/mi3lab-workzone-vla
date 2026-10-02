#!/usr/bin/env python3
"""Precision/recall/timing sweep for a minimum-duration debounce filter on
predicted INSIDE events, computed directly from cached per-frame predictions
(no inference re-run). Used to find the cheapest Jetson-only precision fix:
requiring a predicted INSIDE run to persist d seconds before counting it as
an event, mirroring the paired baseline's EMA + minimum-frames-inside rule.

Usage:
  python3 debounce_sweep.py --cache-dir ~/eval_cache/vlm
  python3 debounce_sweep.py --cache-dir ~/eval_cache/vlm_greedy
"""
import argparse
import json
import os

import numpy as np

from paper_metrics import extract_events, _overlaps

FPS = 30.0


def load_cache(cache_dir, videos):
    data = {}
    for v in videos:
        p = os.path.join(cache_dir, f"{v}.npz")
        if os.path.exists(p):
            d = np.load(p)
            data[v] = (d["predicted"].tolist(), d["gt"].tolist())
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split-file", default="eval_split_full.json")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--durations", default="0,0.5,1,1.5,2,2.5,3")
    args = ap.parse_args()

    videos = json.load(open(args.split_file))[args.split]
    data = load_cache(os.path.expanduser(args.cache_dir), videos)
    print(f"loaded {len(data)}/{len(videos)} cached videos from {args.cache_dir}\n")

    durations = [float(x) for x in args.durations.split(",")]
    print(f"{'d (s)':>6s}{'precision':>11s}{'recall':>9s}{'MAE (kept)':>12s}")
    for d in durations:
        min_frames = d * FPS
        tp_p = n_p = tp_g = n_g = 0
        offsets = []
        for pred, gt in data.values():
            P = [e for e in extract_events(pred, "inside") if (e.end - e.start + 1) >= min_frames]
            G = extract_events(gt, "inside")
            n_p += len(P)
            n_g += len(G)
            for p in P:
                if any(_overlaps(p, g) for g in G):
                    tp_p += 1
            for g in G:
                cands = [p for p in P if _overlaps(p, g)]
                if cands:
                    tp_g += 1
                    if g.start != 0:
                        best = min(cands, key=lambda p: abs(p.start - g.start))
                        offsets.append(abs(best.start - g.start) / FPS)
        prec = tp_p / n_p if n_p else 0.0
        rec = tp_g / n_g if n_g else 0.0
        mae = np.mean(offsets) if offsets else float("nan")
        print(f"{d:6.1f}{prec:10.1%} {rec:8.1%} {mae:10.2f}s")


if __name__ == "__main__":
    main()
