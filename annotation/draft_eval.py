#!/usr/bin/env python3
"""Per-label response rates and discrimination on the California draft labels.

Uses the 1 Hz evidence record (eval_cache/ood_stream).  Every number printed
here is PROVISIONAL until california_labels.json (human-verified) replaces the
draft: pass --labels to switch.

Usage:  python3 draft_eval.py [--labels california_labels.json]
"""
import argparse
import json
import os

import numpy as np

STREAM = os.path.expanduser("~/eval_cache/ood_stream")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="california_draft.json")
    args = ap.parse_args()
    L = json.load(open(args.labels))
    rows = []
    for key, rec in L.items():
        if key.startswith("_"):
            continue
        z = np.load(os.path.join(STREAM, f"{key}.npz"), allow_pickle=True)
        o, fps = z["obs"], float(z["fps"])
        fr = (o[:, 0] * fps).astype(int)
        inside = lambda iv: np.array([any(a <= f <= b for a, b in iv) for f in fr])  # noqa: E731
        vis = inside(rec["visible"])
        act = ~inside(rec["outside"])
        cond = str(z["condition"])
        for k in range(len(o)):
            rows.append((key, cond, vis[k], act[k], o[k, 1], o[k, 2] > .5, o[k, 3] > .5))
    key, cond, vis, act, det, gate, sign = map(np.array, zip(*rows))
    print(f"labels: {args.labels}   samples: {len(rows)} s over {len(set(key))} videos\n")
    groups = [("no work visible", ~vis & ~act), ("visible, not ego-relevant", vis & ~act),
              ("ego-relevant", act)]
    print(f"{'label':28s} {'n':>5s} {'C3E gate yes':>13s} {'detector cue':>13s}")
    for name, m in groups:
        print(f"{name:28s} {m.sum():5d} {gate[m].mean():13.1%} {(det[m] > 0).mean():13.1%}")
    print("\nAUC, per-second response (0.5 = chance):")
    for tname, pos, neg in [("perception: visible vs none", vis, ~vis & ~act),
                            ("ego-relevance: ego vs visible-only", act, vis & ~act)]:
        print(f"  {tname:36s} C3E gate {auc(gate[pos], gate[neg]):.2f}   "
              f"detector score {auc(det[pos], det[neg]):.2f}")
    print("\nby condition, perception AUC (C3E gate / detector):")
    for c in ("day", "evening", "sunset", "night", "night-rain", "night-fog"):
        m = cond == c
        p, n = m & vis, m & ~vis & ~act
        if p.sum() and n.sum():
            print(f"  {c:11s} pos {p.sum():4d}s neg {n.sum():4d}s   "
                  f"{auc(gate[p], gate[n]):.2f} / {auc(det[p], det[n]):.2f}")


if __name__ == "__main__":
    main()
