#!/usr/bin/env python3
"""Sweep the joint filter's remaining free constants on the calibration split.

Two constants are not counted from data by calibrate_joint_filter.py and would
otherwise be my choice rather than a measurement:

  EVIDENCE_TAU   how many seconds of observation count as one application of the
                 likelihood.  Without it the filter multiplies the same evidence
                 once per 24 ms loop iteration, treating ~30 near-duplicate looks
                 at one world-model answer as 30 independent observations.  The
                 measured consequence was a false-alarm rate of 135/h.

  bucket edges   how the detector's calibrated score is discretised.

Both are swept here by replaying the recorded evidence, which is exact: the
recorded run's frame sampling and world-model schedule depend on inference
latency alone, never on these constants, so every setting sees the same stream.
Leave-one-out over videos would be better still; with 100 videos and two scalars
the risk of overfitting the sweep is small but real, and is noted in the paper.

Usage:
  python3 sweep_joint_filter.py --dump ~/eval_cache/joint_dump_cal
"""
import argparse
import glob
import os

import numpy as np

import joint_filter as J
from calibrate_joint_filter import IDX, S, STATES
from paper_metrics import frame_accuracy, macro_f1

FPS = 30.0


def load(dump_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(dump_dir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        rows = z["obs"]
        if rows.size:
            out.append((rows, [str(x) for x in z["gt"]], int(z["n_frames"])))
    return out


def encode(rows, edges):
    def bucket(v):
        for i, e in enumerate(edges):
            if v < e:
                return i
        return len(edges)
    return [(int(r[0]), r[2],
             (bucket(r[3]) * 6 + ((2 if r[5] > 0.5 else 1 if r[4] > 0.5 else 0) * 2
                                  + (1 if r[6] > 0.5 else 0)))
             * (len(J.AGE_BUCKETS) + 1) + J.age_bucket(r[7]))
            for r in rows]


def fit_from(data, edges, n_obs, alpha=1.0):
    emit = np.full((S, n_obs), alpha)
    trans = np.zeros((S, S))
    dwell = np.zeros(S)
    marg = np.zeros(S)
    for rows, gt, _ in data:
        enc = encode(rows, edges)
        prev, prev_clock = None, None
        for (fi, dt, obs), r in zip(enc, rows):
            if fi < 0 or fi >= len(gt):
                continue
            si = IDX.get(gt[fi])
            if si is None:
                continue
            emit[si, obs] += 1.0
            marg[si] += 1.0
            if prev is not None:
                dwell[prev] += max(r[1] - prev_clock, 1e-6)
                if si != prev:
                    trans[prev, si] += 1.0
            prev, prev_clock = si, r[1]
    emit /= emit.sum(axis=1, keepdims=True)
    rate = np.zeros((S, S))
    for i in range(S):
        if dwell[i] > 0:
            for j in range(S):
                if j != i:
                    rate[i, j] = trans[i, j] / dwell[i]
            rate[i, i] = -rate[i].sum()
    prior = marg / marg.sum() if marg.sum() > 0 else np.full(S, 1.0 / S)
    return emit, rate, prior


def replay(data, emit, rate, prior, edges, tau):
    log_emit = np.log(emit + 1e-9)
    conf = {}
    eye = np.eye(S)
    for rows, gt, n_frames in data:
        enc = encode(rows, edges)
        pred = np.full(n_frames, "outside", dtype=object)
        b = prior.copy()
        for fi, dt, obs in enc:
            T = np.clip(eye + rate * dt, 1e-9, None)
            T /= T.sum(axis=1, keepdims=True)
            b = b @ T
            b = b * np.exp(log_emit[:, obs] * (dt / tau))
            ssum = b.sum()
            b = b / ssum if ssum > 0 else prior.copy()
            pred[max(0, fi):] = STATES[int(np.argmax(b))]
        n = min(len(pred), len(gt))
        for a, c in zip(gt[:n], pred[:n]):
            conf[(a, str(c))] = conf.get((a, str(c)), 0) + 1
    return frame_accuracy(conf, STATES), macro_f1(conf, STATES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tau", default="0.05,0.1,0.25,0.5,1.0,2.0")
    args = ap.parse_args()

    data = load(args.dump)
    print(f"{len(data)} calibration videos, "
          f"{sum(len(r) for r, _, _ in data)} recorded steps\n")

    edge_sets = {
        "4 buckets": [0.2, 0.45, 0.7],
        "5 buckets": [0.15, 0.35, 0.55, 0.75],
        "6 buckets": [0.1, 0.25, 0.4, 0.55, 0.75],
    }
    best = None
    print(f"{'edges':>12} {'tau (s)':>9} {'accuracy':>10} {'macro-F1':>10}")
    for name, edges in edge_sets.items():
        n_obs = (len(edges) + 1) * 6 * (len(J.AGE_BUCKETS) + 1)
        emit, rate, prior = fit_from(data, edges, n_obs)
        for tau in [float(x) for x in args.tau.split(",")]:
            acc, f1 = replay(data, emit, rate, prior, edges, tau)
            print(f"{name:>12} {tau:9.2f} {acc:9.1%} {f1:10.3f}")
            if best is None or f1 > best[0]:
                best = (f1, acc, name, edges, tau)
    f1, acc, name, edges, tau = best
    print(f"\n>>> calibrated: {name}, edges {edges}, tau {tau:.2f}s "
          f"(accuracy {acc:.1%}, macro-F1 {f1:.3f})")


if __name__ == "__main__":
    main()
