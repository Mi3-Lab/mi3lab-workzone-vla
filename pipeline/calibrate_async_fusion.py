#!/usr/bin/env python3
"""Calibrate the two constants the asynchronous stack introduces.

The stack adds a bias magnitude (how far an unverified positive gate may move
the detector's score) and a half-life (how fast a semantic answer goes stale).
Choosing them by eye and then reporting on validation would be tuning on the
test set -- the objection this paper spends its length avoiding.  They are
therefore swept on the calibration split.

The sweep is exact, not an approximation.  The world model's query schedule
depends only on its own latency and the replay clock, never on the bias or the
half-life, so every setting observes the same frames and consumes the same
semantic answers at the same times.  Recording one trace per video is enough
to replay the whole grid offline.

Usage:
  python3 calibrate_async_fusion.py --trace ~/eval_cache/async_trace_cal
"""
import argparse
import glob
import os

import numpy as np

import evaluate_yolo_baseline as B
from evaluate_async_fusion import update_state_seconds
from paper_metrics import frame_accuracy, macro_f1

STATES = ["outside", "approaching", "inside", "exiting"]


def load(trace_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(trace_dir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        t = z["trace"]
        if t.size:
            out.append((t, [str(x) for x in z["gt"]], int(z["n_frames"])))
    return out


def replay(trace, n_frames, bias, halflife, bypass_on=True):
    """Re-run the fusion and the seconds-mode state machine over one trace."""
    pred = np.full(n_frames, "outside", dtype=object)
    state, f_ema, dur_s, out_s = "OUT", None, 0.0, 0.0
    for frame_idx, clock, dt, y_s, alpha, sem_val, sem_ver, sem_t in trace:
        score = y_s
        did_bypass = False
        if sem_val >= 0.0:
            decay = 0.5 ** ((clock - sem_t) / halflife)
            if bypass_on and sem_ver > 0.5 and decay > 0.5 and state == "OUT":
                did_bypass = True
            score = B.clamp01(score + bias * decay * (2.0 * sem_val - 1.0))
        f_ema = B.ema(f_ema, B.clamp01(score), alpha)
        if did_bypass:
            state, dur_s, out_s = "APPROACHING", 0.0, 0.0
        else:
            state, dur_s, out_s = update_state_seconds(state, f_ema, dur_s, out_s, dt)
        pred[max(0, int(frame_idx)):] = B.STATE_MAP[state]
    return pred


def score(data, bias, halflife, bypass_on=True):
    conf = {}
    for trace, gt, n_frames in data:
        pred = replay(trace, n_frames, bias, halflife, bypass_on)
        n = min(len(pred), len(gt))
        for a, b in zip(gt[:n], pred[:n]):
            conf[(a, str(b))] = conf.get((a, str(b)), 0) + 1
    return frame_accuracy(conf, STATES), macro_f1(conf, STATES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--bias", default="0,0.1,0.15,0.2,0.3,0.4,0.5")
    ap.add_argument("--halflife", default="0.5,1,2,4,8")
    args = ap.parse_args()

    data = load(args.trace)
    print(f"{len(data)} calibration videos, "
          f"{sum(len(t) for t, _, _ in data)} recorded cycles\n")

    results = []
    print(f"{'bias':>6s} {'half-life':>10s} {'accuracy':>10s} {'macro-F1':>10s}")
    for b in [float(x) for x in args.bias.split(",")]:
        for h in [float(x) for x in args.halflife.split(",")]:
            acc, f1 = score(data, b, h)
            results.append((f1, acc, b, h))
            print(f"{b:6.2f} {h:9.1f}s {acc:9.1%} {f1:10.3f}")
    results.sort(key=lambda r: -r[0])
    f1, acc, b, h = results[0]
    print(f"\n>>> calibrated: bias {b:.2f}, half-life {h:.1f}s "
          f"(accuracy {acc:.1%}, macro-F1 {f1:.3f})")

    # what each ingredient contributes, on the calibration split
    a0, f0 = score(data, 0.0, h)                       # no semantic channel
    ab, fb = score(data, b, h, bypass_on=False)        # bias only, no bypass
    print(f"    detector alone (bias 0):        macro-F1 {f0:.3f}")
    print(f"    + score bias, no bypass:        macro-F1 {fb:.3f}")
    print(f"    + verified-evidence bypass:     macro-F1 {f1:.3f}")


if __name__ == "__main__":
    main()
