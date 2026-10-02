#!/usr/bin/env python3
"""Learned decision-level combiner for the detector and the world model.

Five hand-written fusion rules recovered almost nothing, and the oracle analysis
says why: complementarity here is state-specific and the two systems' errors are
mirrored (the detector skips APPROACHING by declaring INSIDE; the world model
lingers in APPROACHING when already INSIDE or still OUTSIDE).  A scalar weight
or a single threshold cannot express "trust A about this transition and B about
that one", so averaging cancels the very signal we are trying to exploit.

This learns the mapping instead.  The feature space is small and discrete -- the
pair of predicted states, optionally with how long that pair has persisted -- so
the combiner is a contingency table estimated by counting on the calibration
split: P(true state | detector state, world-model state).  No library, and the
whole model is 16 rows that can be printed and audited, which matters for a
paper whose argument is about not hiding fitted constants.

IMPORTANT -- what this script does and does not establish.  It combines cached
per-frame predictions from two independent executions, exactly the shortcut that
produced a 0.107 macro-F1 phantom earlier in this work.  It is therefore a
FEASIBILITY BOUND, not a system: it answers "is there enough signal in the pair
of decisions for a learned combiner to beat either alone?" before we spend GPU
hours building the real-time version.  Any number it produces must be re-earned
by an end-to-end run before it goes near the paper.

Usage:
  python3 stack_combiner.py                 # fit on calibration, report on validation
  python3 stack_combiner.py --dwell         # add persistence context to the feature
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

import paired_ci as P
from paper_metrics import (EventLevelAccumulator, extract_events, _overlaps,
                           per_state_iou, frame_accuracy, macro_f1)

STATES = ["outside", "approaching", "inside", "exiting"]
FPS = 30.0
DWELL_BUCKETS = [0.5, 2.0, 6.0]     # seconds


def dwell_bucket(seconds):
    for i, b in enumerate(DWELL_BUCKETS):
        if seconds < b:
            return i
    return len(DWELL_BUCKETS)


def features(det, c3e, use_dwell):
    """Per-frame feature stream: the pair of decisions, plus how long it held."""
    out = []
    prev, held = None, 0
    for i in range(min(len(det), len(c3e))):
        pair = (det[i], c3e[i])
        held = held + 1 if pair == prev else 0
        prev = pair
        if use_dwell:
            out.append(pair + (dwell_bucket(held / FPS),))
        else:
            out.append(pair)
    return out


def fit(det_cache, c3e_cache, videos, use_dwell):
    counts = defaultdict(lambda: defaultdict(int))
    d = P.load(det_cache, videos)
    c = P.load(c3e_cache, videos)
    common = sorted(set(d) & set(c))
    for v in common:
        (pd, gt) = d[v]
        (pc, _) = c[v]
        feats = features(pd, pc, use_dwell)
        for i, f in enumerate(feats):
            if i < len(gt):
                counts[f][gt[i]] += 1
    table = {}
    for f, dist in counts.items():
        total = sum(dist.values())
        table[f] = (max(dist, key=dist.get), total, dist)
    return table, len(common)


def apply_table(table, det, c3e, use_dwell, fallback):
    feats = features(det, c3e, use_dwell)
    out = []
    for f in feats:
        if f in table:
            out.append(table[f][0])
        else:
            out.append(fallback[len(out)] if len(out) < len(fallback) else "outside")
    return out


def score(pairs):
    conf, ev = {}, EventLevelAccumulator(STATES)
    fa, frames = 0, 0
    for pred, gt in pairs:
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
    iou = per_state_iou(conf, STATES)
    r = ev.result()
    hours = frames / FPS / 3600.0
    return dict(acc=frame_accuracy(conf, STATES), f1=macro_f1(conf, STATES),
                iou_ap=iou["approaching"], iou_in=iou["inside"],
                in_p=r["inside"]["precision"], fa_h=fa / hours if hours else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwell", action="store_true", help="add persistence to the feature")
    ap.add_argument("--export", default=None,
                    help="write the fitted table as JSON for the runtime to load")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cal = json.load(open(os.path.join(here, "eval_split_dev.json")))["calibration"]
    val = json.load(open(os.path.join(here, "eval_split_full.json")))["validation"]

    table, n_cal = fit("~/eval_cache/yoloclip_dev", "~/eval_cache/c3e_caldev",
                       cal, args.dwell)
    print(f"combiner fitted on {n_cal} calibration videos, "
          f"{len(table)} feature cells, dwell={args.dwell}\n")

    if args.export:
        out = {f"{k[0]}|{k[1]}": v[0] for k, v in table.items() if len(k) == 2}
        with open(os.path.expanduser(args.export), "w") as fh:
            json.dump({"table": out, "n_calibration_videos": n_cal}, fh, indent=1)
        print(f"table exported to {args.export}\n")

    if not args.dwell:
        print("learned table  P(true | detector, world model):")
        print(f"  {'detector':<12} {'world model':<12} {'-> predicts':<12} {'support':>8}  distribution")
        for (dd, cc), (pred, tot, dist) in sorted(table.items(), key=lambda x: -x[1][1]):
            share = ", ".join(f"{k[:4]} {100*v/tot:.0f}%" for k, v in
                              sorted(dist.items(), key=lambda x: -x[1]) if v / tot > 0.05)
            print(f"  {dd:<12} {cc:<12} {pred:<12} {tot:>8}  {share}")
        print()

    d = P.load("~/eval_cache/yoloclip", val)
    c = P.load("~/eval_cache/cosmos3edge_cal", val)
    common = sorted(set(d) & set(c))
    combined, det_only, c3e_only = [], [], []
    for v in common:
        (pd, gt) = d[v]
        (pc, _) = c[v]
        combined.append((apply_table(table, pd, pc, args.dwell, pc), gt))
        det_only.append((pd, gt))
        c3e_only.append((pc, gt))

    print(f"=== validation ({len(common)} videos), FEASIBILITY BOUND ===")
    hdr = f"{'system':<26} {'acc':>7} {'macroF1':>8} {'IoUappr':>8} {'IoUin':>8} {'precIN':>8} {'FA/h':>7}"
    print(hdr)
    for tag, pairs in [("detector", det_only), ("world model alone", c3e_only),
                       ("learned combiner", combined)]:
        m = score(pairs)
        print(f"{tag:<26} {m['acc']:6.1%} {m['f1']:8.3f} {m['iou_ap']:8.3f} "
              f"{m['iou_in']:8.3f} {m['in_p']:7.1%} {m['fa_h']:7.1f}")
    print("\nThis combines two independent executions and is therefore an upper")
    print("bound on what a real-time combiner could reach, not a system result.")


if __name__ == "__main__":
    main()
