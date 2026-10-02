#!/usr/bin/env python3
"""Channel ablation of the joint estimator: refit AND re-evaluate with one
evidence stream removed.

The estimator's claim is that the interaction between the detector's count and
the world model's answer is the signal -- that a factorised or single-stream
version cannot express "the world model's APPROACHING is trustworthy when the
detector corroborates and mostly a false alarm when it does not".  That claim is
testable on the recorded dumps alone: mask a channel in the calibration dump,
count the emission table again, then replay the validation dump with the same
channel masked.  Fitting and inference stay consistent, so this measures a
system built without the channel, not a system surprised by its absence.

No GPU pass is repeated: masking never changes which frames were observed,
because the recorded schedule depends on inference latency alone.

Usage:
  python3 ablate_joint.py --cal ~/eval_cache/joint_dump_cal2 \
      --val ~/eval_cache/joint_dump_val
"""
import argparse
import glob
import os

import numpy as np

import joint_filter as J
from paper_metrics import (EventLevelAccumulator, extract_events, _overlaps,
                           per_state_iou, frame_accuracy, macro_f1)

STATES = J.STATES
IDX = {s: i for i, s in enumerate(STATES)}
FPS = 30.0

MASKS = {
    "full":      dict(),
    "no det":    dict(det=True),
    "no gate":   dict(gate=True),
    "no sign":   dict(sign=True),
    "no desc":   dict(corr=True),
    "no world":  dict(gate=True, sign=True, corr=True),
    "no age":    dict(age=True),
}


def obs_of(row, m):
    _fi, _clock, _dt, y_s, gate, sign, corr, age = row
    if m.get("det"):
        y_s = 0.0
    if m.get("gate"):
        gate = 0.0
    if m.get("sign"):
        sign = 0.0
    if m.get("corr"):
        corr = 0.0
    if m.get("age"):
        age = 0.0
    return J.obs_index(J.det_bucket(y_s), gate > 0.5, sign > 0.5, corr > 0.5,
                       J.age_bucket(age))


def load(dump):
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(dump), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        if z["obs"].size == 0:
            continue
        yield z["obs"], [str(x) for x in z["gt"]], int(z["n_frames"])


def fit(cal, m, alpha=1.0):
    emit = np.full((len(STATES), J.N_OBS), alpha)
    trans, dwell, marg = np.zeros((4, 4)), np.zeros(4), np.zeros(4)
    for rows, gt, _n in load(cal):
        prev, prev_clock = None, None
        for row in rows:
            fi = int(row[0])
            if not (0 <= fi < len(gt)):
                continue
            si = IDX.get(gt[fi])
            if si is None:
                continue
            emit[si, obs_of(row, m)] += 1.0
            marg[si] += 1.0
            if prev is not None:
                dwell[prev] += max(row[1] - prev_clock, 1e-6)
                if si != prev:
                    trans[prev, si] += 1.0
            prev, prev_clock = si, row[1]
    emit /= emit.sum(axis=1, keepdims=True)
    rate = np.zeros((4, 4))
    for i in range(4):
        if dwell[i] > 0:
            for j in range(4):
                if j != i:
                    rate[i, j] = trans[i, j] / dwell[i]
            rate[i, i] = -rate[i].sum()
    return {"emission": emit.tolist(), "rate": rate.tolist(),
            "prior": (marg / marg.sum()).tolist()}


def score(val, params, m):
    filt_params = params
    conf, ev = {}, EventLevelAccumulator(STATES)
    fa, frames = 0, 0
    for rows, gt, n_frames in load(val):
        f = J.JointFilter(filt_params)
        pred = np.full(n_frames, "outside", dtype=object)
        for row in rows:
            pred[max(0, int(row[0])):] = f.step(obs_of(row, m), float(row[2]))
        n = min(len(pred), len(gt))
        ps, gs = [str(x) for x in pred[:n]], gt[:n]
        frames += n
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs)
        pa = ["active" if s != "outside" else "outside" for s in ps]
        ga = ["active" if s != "outside" else "outside" for s in gs]
        G = extract_events(ga, "active")
        fa += sum(1 for e in extract_events(pa, "active")
                  if not any(_overlaps(e, g) for g in G))
    iou, r = per_state_iou(conf, STATES), ev.result()
    return (frame_accuracy(conf, STATES), macro_f1(conf, STATES),
            iou["approaching"], iou["inside"], iou["exiting"],
            r["inside"]["precision"] or 0.0, r["inside"]["recall"] or 0.0,
            fa / (frames / FPS / 3600.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--val", required=True)
    args = ap.parse_args()
    hdr = (f"{'variant':10s} {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} "
           f"{'IoUex':>6s} {'inP':>6s} {'inR':>6s} {'FA/h':>6s}")
    print(hdr); print("-" * len(hdr))
    for name, m in MASKS.items():
        p = fit(args.cal, m)
        a, f1, ap_, ins, ex, pr, rc, fa = score(args.val, p, m)
        print(f"{name:10s} {a*100:5.1f}% {f1:6.3f} {ap_:6.3f} {ins:6.3f} "
              f"{ex:6.3f} {pr*100:5.1f}% {rc*100:5.1f}% {fa:6.1f}", flush=True)


if __name__ == "__main__":
    main()
