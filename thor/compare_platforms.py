#!/usr/bin/env python3
"""Orin vs another device, from caches only (no inference).

  in-domain:  the five systems of 06_run_experiments.sh indomain, same metrics
              as the paper tables, Orin column next to this device's column.
  evidence:   per-second agreement of the world model's (and detector's)
              answers on the California and San Francisco records.

Usage:  python3 thor/compare_platforms.py eval_cache/platforms/<device>
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
import compare_all_systems as C  # noqa: E402

ORIN = os.path.join(REPO, "eval_cache")
SYSTEMS = [("2B sampled", "vlm"), ("2B greedy", "vlm_greedy"),
           ("C3E calibrated", "cosmos3edge_cal"), ("detector+CLIP", "yoloclip"),
           ("joint (312)", "joint_full_val")]


def main():
    dev = sys.argv[1]
    name = os.path.basename(dev.rstrip("/"))
    val = json.load(open(os.path.join(REPO, "pipeline", "eval_split_full.json")))["validation"]
    print(f"== in domain: 208 validation videos, Orin vs {name} ==")
    print(f"{'system':16s} {'':5s} {'n':>4s} {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} {'IoUex':>6s} {'inP':>6s} {'FA/h':>6s}")
    for label, sub in SYSTEMS:
        for plat, root in (("orin", ORIN), (name[:5], dev)):
            d = C.load(os.path.join(root, sub), val)
            if len(d) < 10:
                print(f"{label:16s} {plat:5s}   -- no cache"); continue
            m = C.metrics(d)
            print(f"{label:16s} {plat:5s} {m['n']:4d} {m['acc']*100:5.1f}% {m['f1']:6.3f} {m['iou_appr']:6.3f} "
                  f"{m['iou_in']:6.3f} {m['iou_exit']:6.3f} {m['in_p']*100:5.1f}% {m['fa_per_h']:6.1f}")
    for rec, cols in (("ood_stream", {"det_bucket": 1, "gate": 2, "sign": 3, "desc_corr": 4}),
                      ("sf_stream", {"gate": 1, "sign": 2, "desc_corr": 3})):
        files = sorted(glob.glob(os.path.join(dev, rec, "*.npz")))
        if not files:
            continue
        print(f"\n== evidence agreement, {rec} ({len(files)} videos) ==")
        agree = {k: [] for k in cols}
        for f in files:
            a = np.load(f, allow_pickle=True)["obs"]
            b_path = os.path.join(ORIN, rec, os.path.basename(f))
            if not os.path.exists(b_path):
                continue
            b = np.load(b_path, allow_pickle=True)["obs"]
            n = min(len(a), len(b))
            for k, c in cols.items():
                if k == "det_bucket":
                    import joint_filter as J
                    agree[k] += [J.det_bucket(x) == J.det_bucket(y) for x, y in zip(a[:n, c], b[:n, c])]
                else:
                    agree[k] += list((a[:n, c] > .5) == (b[:n, c] > .5))
        for k, v in agree.items():
            if v:
                print(f"  {k:10s} {np.mean(v):6.1%}  over {len(v)} s")


if __name__ == "__main__":
    main()
