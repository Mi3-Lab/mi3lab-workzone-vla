#!/usr/bin/env python3
"""Two ROADWork-only analyses, on human four-state labels, from recorded
evidence (no inference):

1. Response bias by model.  The fine-tuned 2B, the same 2B before fine-tuning
   (base) and the zero-shot world model (C3E) were all recorded on the same 100
   calibration videos, every channel, every second.  For each model: how often
   GATE says yes in each ground-truth state, and how well it separates active
   states from OUTSIDE (AUC).  OUTSIDE in ROADWork includes visible work that
   does not concern the ego vehicle, so "yes" there mixes false perception with
   missing ego-relevance.

2. Leave-one-city-out for the joint estimator.  Its tables are counted on the
   calibration split; here they are counted on one city only and replayed on the
   other city's validation videos, next to the all-city fit on the same videos.

Usage:  python3 roadwork_analysis.py
"""
import glob
import os
import tempfile

import numpy as np

import ablate_joint as A

H = os.path.expanduser
STATES = ["outside", "approaching", "inside", "exiting"]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def bias():
    print("== 1. response bias, 100 ROADWork calibration videos, 1 Hz, human labels ==")
    models = [("2B fine-tuned", "evidence_2b"), ("2B base (no FT)", "evidence_base"),
              ("C3E zero-shot", "evidence_c3e")]
    print(f"{'model':18s} " + " ".join(f"{s[:5]+' yes':>10s}" for s in STATES) +
          f" {'AUC act/out':>12s} {'AUC in/out':>11s} {'DESC on out':>12s}")
    for name, d in models:
        g_by = {s: [] for s in STATES}; c_out = []
        for p in sorted(glob.glob(H(f"~/eval_cache/{d}/*.npz"))):
            z = np.load(p, allow_pickle=True)
            gt = [str(x) for x in z["gt"]]
            for fi, g, c in zip(z["frame_idx"], z["gate"], z["corr"]):
                st = gt[min(int(fi), len(gt) - 1)]
                g_by[st].append(g == 1)
                if st == "outside":
                    c_out.append(c == 1)
        act = g_by["approaching"] + g_by["inside"] + g_by["exiting"]
        print(f"{name:18s} " + " ".join(f"{np.mean(g_by[s]):10.1%}" for s in STATES) +
              f" {auc(act, g_by['outside']):12.2f} {auc(g_by['inside'], g_by['outside']):11.2f}"
              f" {np.mean(c_out):12.1%}")
    print(f"seconds per state: " + ", ".join(f"{s} {len(v)}" for s, v in g_by.items()))


def city_dir(src, city):
    d = tempfile.mkdtemp(prefix=f"loco_{city}_")
    for p in glob.glob(os.path.join(H(src), f"{city}_*.npz")):
        os.symlink(p, os.path.join(d, os.path.basename(p)))
    return d


def loco():
    print("\n== 2. leave-one-city-out, joint estimator (validation videos of the held-out city) ==")
    cal, val = "~/eval_cache/joint_dump_cal2", "~/eval_cache/joint_dump_val"
    fit_all = A.fit(H(cal), {})
    print(f"{'test city':10s} {'fitted on':22s} {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} {'IoUex':>6s} {'inP':>6s} {'FA/h':>6s}")
    for test, train in (("seattle", "boston"), ("boston", "seattle")):
        vdir = city_dir(val, test)
        n_cal = len(glob.glob(os.path.join(city_dir(cal, train), "*.npz")))
        for label, params in ((f"all 312 cal videos", fit_all),
                              (f"{train} only ({n_cal})", A.fit(city_dir(cal, train), {}))):
            a, f1, ap, ins, ex, p, r, fa = A.score(vdir, params, {})
            print(f"{test:10s} {label:22s} {a*100:5.1f}% {f1:6.3f} {ap:6.3f} {ins:6.3f} {ex:6.3f} {p*100:5.1f}% {fa:6.1f}")


if __name__ == "__main__":
    bias()
    loco()
