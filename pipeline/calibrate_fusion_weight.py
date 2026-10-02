#!/usr/bin/env python3
"""Calibrate the detector/world-model fusion weight offline.

The ported baseline fuses its semantic channel at CLIP_WEIGHT=0.35, a constant
fitted for CLIP.  Reusing it for a different semantic model would repeat, inside
our own system, the inheritance error this paper is about -- so we sweep it.

The sweep is exact rather than approximate.  In evaluate_yolo_c3e_fusion the
per-cycle cost depends only on the detector's EMA (which decides whether
transcription is escalated), never on the fusion weight, so every weight
observes exactly the same frames.  Recording the raw scores once therefore
lets the whole sweep run offline over a fixed record, with no re-execution and
no drift between the calibrated and the deployed system.

Usage:
  python3 calibrate_fusion_weight.py --scores ~/eval_cache/fusion_scores_cal
"""
import argparse
import glob
import os

import numpy as np

import evaluate_yolo_baseline as B
from paper_metrics import frame_accuracy, macro_f1

STATES = ["outside", "approaching", "inside", "exiting"]


def load(scores_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(scores_dir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        rows = z["rows"]
        if rows.size:
            out.append((rows, [str(x) for x in z["gt"]], int(z["n_frames"])))
    return out


def replay(rows, n_frames, weight):
    """Re-run the detector's own state machine over recorded scores."""
    pred = np.full(n_frames, "outside", dtype=object)
    state, f_ema, dur, out_f = "OUT", None, 0, 999
    for frame_idx, y_s, y_ema, sem, _esc in rows:
        fused = (1.0 - weight) * y_s + weight * sem
        # the orange-context boost depends on the frame, not on the weight, and
        # was already folded into the recorded run; omitted here so the sweep
        # compares weights on identical terms
        evidence = B.clamp01(y_s)
        alpha = B.adaptive_alpha(evidence, B.F_CONF["ema_alpha"] * 0.4,
                                 B.F_CONF["ema_alpha"] * 1.2)
        f_ema = B.ema(f_ema, B.clamp01(fused), alpha)
        state, dur, out_f = B.update_state(state, f_ema, dur, out_f)
        pred[max(0, int(frame_idx)):] = B.STATE_MAP[state]
    return pred


def score(data, weight):
    conf = {}
    for rows, gt, n_frames in data:
        pred = replay(rows, n_frames, weight)
        n = min(len(pred), len(gt))
        for a, b in zip(gt[:n], pred[:n]):
            conf[(a, str(b))] = conf.get((a, str(b)), 0) + 1
    return frame_accuracy(conf, STATES), macro_f1(conf, STATES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--grid", default="0,0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = ap.parse_args()

    data = load(args.scores)
    cycles = sum(len(r) for r, _, _ in data)
    print(f"{len(data)} calibration videos, {cycles} recorded cycles\n")
    print(f"{'weight':>7s} {'accuracy':>10s} {'macro-F1':>10s}")

    best = None
    for w in [float(x) for x in args.grid.split(",")]:
        acc, f1 = score(data, w)
        tag = "   <- CLIP's inherited value" if abs(w - B.CLIP_WEIGHT) < 1e-9 else ""
        print(f"{w:7.2f} {acc:9.1%} {f1:10.3f}{tag}")
        if best is None or f1 > best[2]:
            best = (w, acc, f1)
    print(f"\n>>> calibrated weight {best[0]:.2f}  (accuracy {best[1]:.1%}, macro-F1 {best[2]:.3f})")
    print(f"    inherited 0.35 would give macro-F1 {score(data, B.CLIP_WEIGHT)[1]:.3f}")


if __name__ == "__main__":
    main()
