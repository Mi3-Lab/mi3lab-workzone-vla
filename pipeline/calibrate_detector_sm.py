#!/usr/bin/env python3
"""Reviewer W2: give the detector's hand-tuned state machine the same per-model
calibration the paper gives the VLMs.

Evidence: the detector score y_s recorded on every detector cycle by the joint
estimator's latency-honest runs (joint_dump_cal2: 312 calibration videos;
joint_dump_val: 208 validation videos).  The detector-only stream (no CLIP), so
this compares detector-only machines; the paper's counted filter "without world
model" uses exactly this stream (0.536).

Machine: the verbatim update_state of evaluate_yolo_baseline (EMA -> 4 states
with thresholds, min_out cycles and a maximum approach duration).  One
approximation: the baseline's adaptive EMA rate uses an evidence term that also
counts objects, which the dump does not keep; here evidence = y_s.

Search (selection on calibration macro-F1 only, as for C3E):
  approach_th 5 x enter_th 4 x exit_th 4 x min_out 5 x max_approach 4 x alpha 3
  = 4,800 configurations (C3E search: 4,896).
All configurations run in parallel (vectorized over the config axis).
"""
import glob
import itertools
import json
import os

import numpy as np

import compare_all_systems as C

H = os.path.expanduser
ST = ["outside", "approaching", "inside", "exiting"]
OUT, APP, INS, EXI = range(4)

HAND = dict(approach_th=0.25, enter_th=0.5, exit_th=0.3, min_out=20, max_app=150, alpha=0.1)
GRID = dict(approach_th=[0.15, 0.2, 0.25, 0.3, 0.35], enter_th=[0.35, 0.4, 0.5, 0.6],
            exit_th=[0.2, 0.25, 0.3, 0.35], min_out=[5, 10, 20, 40, 80],
            max_app=[75, 150, 300, 600], alpha=[0.05, 0.1, 0.2])


def load(dump):
    vids = []
    for p in sorted(glob.glob(H(f"{dump}/*.npz"))):
        z = np.load(p, allow_pickle=True)
        o = z["obs"]
        if o.size == 0:
            continue
        gt = np.array([ST.index(str(x)) if str(x) in ST else 0 for x in z["gt"]])
        n = len(gt)
        fi = np.clip(o[:, 0].astype(int), 0, n - 1)
        # frames each cycle's prediction covers (held until the next cycle), by GT state
        ends = np.r_[fi[1:], n]
        span = np.zeros((len(fi), 4))
        for k, (a, b) in enumerate(zip(fi, ends)):
            if b > a:
                span[k] = np.bincount(gt[a:b], minlength=4)
        span[0] += np.bincount(gt[:fi[0]], minlength=4)    # frames before the first cycle: OUT
        vids.append(dict(name=os.path.basename(p), y=o[:, 3], fi=fi, n=n, span=span, gt=gt))
    return vids


def run(vids, cfgs, record=None):
    """cfgs: dict of arrays (C,).  Returns confusion [C, gt, pred].  record: dict filled with
    per-frame predictions of config 0, keyed by video name."""
    C_ = len(cfgs["alpha"])
    conf = np.zeros((C_, 4, 4))
    lo, hi = cfgs["alpha"] * 0.4, cfgs["alpha"] * 1.2
    for v in vids:
        state = np.zeros(C_, int); dur = np.zeros(C_); outf = np.zeros(C_); ema = np.zeros(C_)
        pred = np.zeros(v['n'], int) if record is not None else None
        for k, y in enumerate(v["y"]):
            a = lo + (hi - lo) * min(max(y, 0.0), 1.0)
            ema = ema + a * (y - ema)
            s = ema
            ns, nd, no = state.copy(), dur.copy(), outf.copy()
            m = state == OUT
            go = m & (s >= cfgs["approach_th"]); ns[go] = APP; nd[go] = 0; no[go] = 0
            st = m & ~go; no[st] = outf[st] + 1; nd[st] = 0
            m = state == APP
            t1 = m & (dur > cfgs["max_app"]); ns[t1] = OUT; nd[t1] = 0; no[t1] = 0
            r = m & ~t1
            t2 = r & (s >= cfgs["enter_th"]); ns[t2] = INS; nd[t2] = 0; no[t2] = 0
            r2 = r & ~t2
            low = r2 & (s <= cfgs["approach_th"] - 0.05)
            t3 = low & (outf >= 2 * cfgs["min_out"]); ns[t3] = OUT; nd[t3] = 0; no[t3] = 0
            t4 = low & ~t3; nd[t4] = dur[t4] + 1; no[t4] = outf[t4] + 1
            t5 = r2 & ~low; nd[t5] = dur[t5] + 1; no[t5] = 0
            m = state == INS
            t6 = m & (s < cfgs["exit_th"]); ns[t6] = EXI; nd[t6] = 0; no[t6] = 0
            t7 = m & ~t6; nd[t7] = dur[t7] + 1; no[t7] = 0
            m = state == EXI
            t8 = m & (s >= cfgs["enter_th"]); ns[t8] = INS; no[t8] = 0
            r3 = m & ~t8
            t9 = r3 & (outf >= cfgs["min_out"]); ns[t9] = OUT; nd[t9] = 0; no[t9] = 0
            t10 = r3 & ~t9; no[t10] = outf[t10] + 1
            state, dur, outf = ns, nd, no
            conf[np.arange(C_), :, state] += v["span"][k]
            if pred is not None:
                pred[v['fi'][k]:] = state[0]
        if record is not None:
            record[v['name']] = ([ST[x] for x in pred], [ST[x] for x in v['gt']])
    return conf


def macro_f1(conf):
    tp = np.einsum("cii->ci", conf); fp = conf.sum(1) - tp; fn = conf.sum(2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0); r = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    return f.mean(1)


GRID2 = dict(approach_th=[0.3, 0.35, 0.4, 0.45, 0.5], enter_th=[0.5, 0.6, 0.7, 0.8],
             exit_th=[0.25, 0.3, 0.35, 0.4], min_out=[40, 80, 160, 320, 640],
             max_app=[150, 300, 600, 1200], alpha=[0.02, 0.05, 0.1])


GRID3 = dict(approach_th=[0.2, 0.25, 0.3, 0.35, 0.4], enter_th=[0.7, 0.8, 0.9, 0.95],
             exit_th=[0.3, 0.4, 0.5, 0.6], min_out=[80, 160, 320, 640],
             max_app=[75, 150, 300, 600], alpha=[0.005, 0.01, 0.02, 0.03])


def main():
    import sys
    global GRID
    if "--refine" in sys.argv:   # second stage: the first stage's optimum sat on grid edges
        GRID = GRID2
    if "--refine3" in sys.argv:  # third and last stage, same reason
        GRID = GRID3
    cal, val = load("~/eval_cache/joint_dump_cal2"), load("~/eval_cache/joint_dump_val")
    keys = list(GRID)
    combos = [c for c in itertools.product(*GRID.values()) if c[2] < c[1]]   # exit < enter
    cfgs = {k: np.array([c[i] for c in combos], float) for i, k in enumerate(keys)}
    print(f"{len(combos)} configurations (exit < enter), {len(cal)} cal / {len(val)} val videos")
    f_cal = macro_f1(run(cal, cfgs))
    best = int(np.argmax(f_cal))
    hand = {k: np.array([HAND[k]], float) for k in keys}
    pick = {k: cfgs[k][best:best + 1] for k in keys}
    print("hand-tuned :", HAND)
    print("calibrated :", {k: float(pick[k][0]) for k in keys})
    print(f"{'':12s} {'cal F1':>7s} | validation: {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} {'IoUex':>6s} {'inP':>6s} {'inR':>6s} {'FA/h':>6s}")
    for name, c in (("hand-tuned", hand), ("calibrated", pick)):
        rec = {}
        run(val, c, record=rec)
        m = C.metrics(rec)
        print(f"{name:12s} {macro_f1(run(cal, c))[0]:7.3f} | {'':11s} {m['acc']*100:5.1f}% {m['f1']:6.3f} {m['iou_appr']:6.3f} "
              f"{m['iou_in']:6.3f} {m['iou_exit']:6.3f} {m['in_p']*100:5.1f}% {m['in_r']*100:5.1f}% {m['fa_per_h']:6.1f}")
    json.dump({k: float(pick[k][0]) for k in keys}, open(H("~/eval_cache/detector_sm_calibrated" + ("_refined3" if "--refine3" in sys.argv else "_refined" if "--refine" in sys.argv else "") + ".json"), "w"))


if __name__ == "__main__":
    main()
