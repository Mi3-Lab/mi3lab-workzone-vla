#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for the head-to-head comparisons.

Reviewer point: the main tables carry point estimates only, so differences of
one or two points are described as "tied", "improved" or "best" without any
uncertainty.  False-alarm rate is the worst case -- it is estimated from 1.78 h
of footage, so its sampling error is operationally large.

Resampling is at the VIDEO level (the unit of independence; frames within a
video are strongly correlated), 10,000 replications, percentile intervals, and
the SAME resampled video indices are used for both systems in a pair so the
comparison is genuinely paired.

Usage:
  python3 paired_ci.py                       # all default pairs
  python3 paired_ci.py --reps 2000           # faster
"""
import argparse
import json
import os

import numpy as np

from paper_metrics import (EventLevelAccumulator, extract_events, _overlaps,
                           per_state_iou, frame_accuracy, macro_f1)

STATES = ["outside", "approaching", "inside", "exiting"]
FPS = 30.0
CACHES = {
    "2B rec.": "~/eval_cache/vlm_greedy",
    "2B recal.": "~/eval_cache/vlm_recal",
    "det pair": "~/eval_cache/yoloclip",
    "det +FE": "~/eval_cache/yoloclip_fastentry",
    "C3E calib.": "~/eval_cache/cosmos3edge_cal",
    "joint 100": "~/eval_cache/joint_v2_val",
    "joint 312": "~/eval_cache/joint_full_val",
}
PAIRS = [
    ("2B rec.", "det pair"),
    ("C3E calib.", "det pair"),
    ("C3E calib.", "2B rec."),
    ("2B recal.", "2B rec."),
    ("2B recal.", "det pair"),
    # the joint estimator: its three claims are that it beats the detector on
    # aggregate quality, beats the world model on nuisance, and finally moves
    # EXITING, which no single system did
    ("joint 312", "det pair"),
    ("joint 312", "C3E calib."),
    ("joint 312", "2B rec."),
    ("joint 312", "joint 100"),
]


def load(cache, videos):
    out = {}
    for v in videos:
        p = os.path.join(os.path.expanduser(cache), f"{v}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            out[v] = ([str(x) for x in z["predicted"]], [str(x) for x in z["gt"]])
    return out


def metrics(data, videos):
    """Aggregate metrics over a (possibly resampled) list of video keys."""
    conf, ev = {}, EventLevelAccumulator(STATES)
    fa, frames = 0, 0
    for v in videos:
        pred, gt = data[v]
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]
        frames += n
        for a, b in zip(gt, pred):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(pred, gt)
        pa = ["active" if s != "outside" else "outside" for s in pred]
        ga = ["active" if s != "outside" else "outside" for s in gt]
        G = extract_events(ga, "active")
        for e in extract_events(pa, "active"):
            if not any(_overlaps(e, g) for g in G):
                fa += 1
    r, iou = ev.result(), per_state_iou(conf, STATES)
    hours = frames / FPS / 3600.0
    return {
        "acc": frame_accuracy(conf, STATES),
        "f1": macro_f1(conf, STATES),
        "iou_ap": iou["approaching"],
        "iou_in": iou["inside"],
        "iou_ex": iou["exiting"],
        "in_p": r["inside"]["precision"],
        "fa_h": fa / hours if hours else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    val = json.load(open(os.path.join(here, "eval_split_full.json")))["validation"]
    data = {k: load(c, val) for k, c in CACHES.items()}
    data = {k: v for k, v in data.items() if len(v) >= 100}
    print("caches carregados: " + ", ".join(f"{k} ({len(v)})" for k, v in data.items()))

    keys = ["acc", "f1", "iou_ap", "iou_in", "iou_ex", "in_p", "fa_h"]
    names = {"acc": "frame acc", "f1": "macro F1", "iou_ap": "IoU appr",
             "iou_in": "IoU inside", "iou_ex": "IoU exiting",
             "in_p": "inside prec", "fa_h": "FA/h"}
    rng = np.random.default_rng(args.seed)

    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        common = sorted(set(data[a]) & set(data[b]))
        if len(common) < 100:
            continue
        pa, pb = metrics(data[a], common), metrics(data[b], common)
        # the SAME resampled indices feed both systems -> paired
        idx = rng.integers(0, len(common), size=(args.reps, len(common)))
        diffs = {k: np.empty(args.reps) for k in keys}
        for r in range(args.reps):
            vs = [common[i] for i in idx[r]]
            ma, mb = metrics(data[a], vs), metrics(data[b], vs)
            for k in keys:
                diffs[k][r] = ma[k] - mb[k]
        print(f"\n=== {a} - {b}  ({len(common)} videos, {args.reps} reps) ===")
        for k in keys:
            lo, hi = np.percentile(diffs[k], [2.5, 97.5])
            point = pa[k] - pb[k]
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  {names[k]:12s} {point:+8.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]{sig}")
        print("  (* = intervalo nao contem zero)")


if __name__ == "__main__":
    main()
