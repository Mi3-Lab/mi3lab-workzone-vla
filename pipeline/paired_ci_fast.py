#!/usr/bin/env python3
"""Paired video-level bootstrap, computed from per-video sufficient statistics.

Identical estimator to paired_ci.py, which re-scores every video on every
replication and costs ~10 h once the joint estimator's pairs are added.  Every
metric in that table is a ratio of sums over videos -- the confusion counts, the
event TP/|P| and TP/|G| pools, the unmatched-alarm count and the frame count are
all additive -- so each video can be reduced ONCE to a fixed-length vector and a
replication becomes a weighted sum of those vectors.  Drawing the multinomial
count vector is equivalent to drawing indices with replacement, and the same
draw feeds both systems, so the comparison stays paired.

Point estimates are therefore exactly those of paired_ci.py; only the Monte
Carlo draws differ.

Usage:
  python3 paired_ci_fast.py [--reps 10000]
"""
import argparse
import json
import os

import numpy as np

from paper_metrics import extract_events, _overlaps

STATES = ["outside", "approaching", "inside", "exiting"]
FPS = 30.0
CACHES = {
    "2B rec.": "~/eval_cache/vlm_greedy",
    "det pair": "~/eval_cache/yoloclip",
    "det +FE": "~/eval_cache/yoloclip_fastentry",
    "C3E calib.": "~/eval_cache/cosmos3edge_cal",
    "joint 100": "~/eval_cache/joint_v2_val",
    "joint 312": "~/eval_cache/joint_full_val",
}
PAIRS = [
    ("joint 312", "det pair"),
    ("joint 312", "C3E calib."),
    ("joint 312", "2B rec."),
    ("joint 312", "joint 100"),
    ("joint 100", "det pair"),
    ("C3E calib.", "det pair"),
]
NS = len(STATES)


def video_stats(pred, gt):
    """Reduce one video to the counts every reported metric is a ratio of."""
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    conf = np.zeros((NS, NS))
    idx = {s: i for i, s in enumerate(STATES)}
    for a, b in zip(gt, pred):
        conf[idx[a], idx[b]] += 1
    tp_p = n_p = 0.0
    P, G = extract_events(pred, "inside"), extract_events(gt, "inside")
    n_p = len(P)
    tp_p = sum(1 for p in P if any(_overlaps(p, g) for g in G))
    pa = ["active" if s != "outside" else "outside" for s in pred]
    ga = ["active" if s != "outside" else "outside" for s in gt]
    GA = extract_events(ga, "active")
    fa = sum(1 for e in extract_events(pa, "active")
             if not any(_overlaps(e, g) for g in GA))
    return np.concatenate([conf.ravel(), [tp_p, n_p, fa, n]])


def metrics_from(v):
    """v: summed statistics vector (or a stack of them, one row per replication)."""
    v = np.atleast_2d(v)
    conf = v[:, :NS * NS].reshape(-1, NS, NS)
    tp = np.einsum("kii->ki", conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        in_p = np.where(v[:, -3] > 0, v[:, -4] / v[:, -3], np.nan)
    return {
        "acc": tp.sum(axis=1) / conf.sum(axis=(1, 2)),
        "f1": f1.mean(axis=1),
        "iou_ap": iou[:, 1], "iou_in": iou[:, 2], "iou_ex": iou[:, 3],
        "in_p": in_p,
        "fa_h": v[:, -2] / (v[:, -1] / FPS / 3600.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    val = json.load(open(os.path.join(here, "eval_split_full.json")))["validation"]
    S = {}
    for name, cache in CACHES.items():
        rows, keys = [], []
        for v in val:
            p = os.path.join(os.path.expanduser(cache), f"{v}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            rows.append(video_stats([str(x) for x in z["predicted"]],
                                    [str(x) for x in z["gt"]]))
            keys.append(v)
        if len(rows) >= 100:
            S[name] = (keys, np.array(rows))
            print(f"  {name:12s} {len(rows)} videos")

    keys_m = ["acc", "f1", "iou_ap", "iou_in", "iou_ex", "in_p", "fa_h"]
    names = {"acc": "frame acc", "f1": "macro F1", "iou_ap": "IoU appr",
             "iou_in": "IoU inside", "iou_ex": "IoU exiting",
             "in_p": "inside prec", "fa_h": "FA/h"}
    rng = np.random.default_rng(args.seed)

    for a, b in PAIRS:
        if a not in S or b not in S:
            continue
        common = sorted(set(S[a][0]) & set(S[b][0]))
        ia = [S[a][0].index(v) for v in common]
        ib = [S[b][0].index(v) for v in common]
        A, B = S[a][1][ia], S[b][1][ib]
        n = len(common)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=args.reps).astype(float)
        ma, mb = metrics_from(A.sum(axis=0)), metrics_from(B.sum(axis=0))
        ra, rb = metrics_from(W @ A), metrics_from(W @ B)
        print(f"\n=== {a} - {b}   ({n} videos, {args.reps} reps) ===")
        for k in keys_m:
            d = ra[k] - rb[k]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            point = float(ma[k][0] - mb[k][0])
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  {names[k]:12s} {point:+8.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]{sig}")


if __name__ == "__main__":
    main()
