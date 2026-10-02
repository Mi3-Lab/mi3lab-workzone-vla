#!/usr/bin/env python3
"""Per-channel latency distribution (p50/p95/p99) and end-to-end cascade-path
latency on the Jetson AGX Orin, addressing review W5/W7 (Q6). Runs each channel
many times on a real work-zone frame through the persistent engine; power is
logged separately via tegrastats.

Usage: python3 bench_latency_power.py --frame test/sample_frame.jpg --reps 60
"""
import argparse
import os
import statistics

import cv2
import numpy as np

from live_cascade_demo import (PersistentEngine, GATE_PROMPT, SIGN_PROMPT,
                               EGO_PROMPT, DESC_PROMPT, WORKERS_PROMPT, SHM_FRAME,
                               DEFAULT_ENGINE_DIR, DEFAULT_MM_ENGINE_DIR,
                               DEFAULT_PLUGIN, DEFAULT_BINARY)

CHANNELS = [
    ("gate", GATE_PROMPT, 3),
    ("ego", EGO_PROMPT, 8),
    ("sign", SIGN_PROMPT, 15),
    ("desc", DESC_PROMPT, 40),
    ("workers", WORKERS_PROMPT, 25),
]


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="test/sample_frame.jpg")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--infer-width", type=int, default=480)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f"cannot read {args.frame}")
    if args.infer_width and frame.shape[1] > args.infer_width:
        h = int(frame.shape[0] * args.infer_width / frame.shape[1])
        frame = cv2.resize(frame, (args.infer_width, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(SHM_FRAME, frame)

    engine = PersistentEngine(DEFAULT_BINARY, DEFAULT_PLUGIN, DEFAULT_ENGINE_DIR,
                              DEFAULT_MM_ENGINE_DIR, temperature=args.temperature)

    # warmup
    for _ in range(3):
        engine.infer(SHM_FRAME, GATE_PROMPT, 3)

    results = {}
    for name, prompt, budget in CHANNELS:
        lats = []
        for _ in range(args.reps):
            _, ms = engine.infer(SHM_FRAME, prompt, budget)
            if ms > 0:
                lats.append(ms)
        results[name] = lats
        print(f"{name:8s} budget={budget:2d}  n={len(lats):3d}  "
              f"p50={pct(lats,50):6.1f}  p95={pct(lats,95):6.1f}  p99={pct(lats,99):6.1f}  "
              f"mean={statistics.mean(lats):6.1f}  max={max(lats):6.1f}")

    # composite cascade paths (sum of channel medians)
    p50 = {k: pct(v, 50) for k, v in results.items()}
    p95 = {k: pct(v, 95) for k, v in results.items()}
    print("\n--- composite cascade-cycle latency (channel medians / p95) ---")
    paths = {
        "OUTSIDE patrol (gate+sign)": ["gate", "sign"],
        "OUTSIDE + corroboration (gate+sign+desc)": ["gate", "sign", "desc"],
        "active cycle (gate+ego)": ["gate", "ego"],
        "state-only gate": ["gate"],
    }
    for label, chans in paths.items():
        s50 = sum(p50[c] for c in chans)
        s95 = sum(p95[c] for c in chans)
        print(f"  {label:44s} p50={s50:6.1f}ms ({1000/s50:4.1f}Hz)  p95={s95:6.1f}ms ({1000/s95:4.1f}Hz)")

    engine.close()


if __name__ == "__main__":
    main()
