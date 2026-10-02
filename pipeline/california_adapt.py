#!/usr/bin/env python3
"""California four-state results with and without unsupervised adaptation of
the joint estimator's emission table (adapt_hmm.py).  Same labels and scoring
as california_4state.py; the header says whether labels are draft."""
import argparse
import json
import os

import numpy as np

import ablate_joint as A
import adapt_hmm as H
import california_4state as C

LOAD = lambda p: json.load(open(os.path.expanduser(p)))  # noqa: E731


def seq(s, mask):
    o = np.array([A.obs_of((0, 0, 1.0, y, float(g), float(si), float(c), 0.0), mask)
                  for y, g, si, c in zip(s["det"], s["gate"], s["sign"], s["corr"])])
    return o, np.ones(len(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    ap.add_argument("--lams", default="0.5")
    args = ap.parse_args()
    path, labels = C.load_labels(args.labels)
    print(f"labels: {path}" + ("   ** DRAFT: provisional **" if "draft" in path else ""))
    vids = list(C.videos(labels))
    bases = {"joint (312)": (LOAD("~/eval_cache/joint_params_full.json"), {}),
             "joint, no detector": (LOAD("~/eval_cache/joint_params_nodet.json"), dict(det=True))}
    for name, (P, mask) in bases.items():
        seqs = [seq(s, mask) for _, _, s, _ in vids]
        rows = [("source table", [P] * len(vids))]
        for lam in map(float, args.lams.split(",")):
            Pt = H.adapt(P, seqs, lam=lam)
            rows.append((f"adapted, transductive, lam={lam}", [Pt] * len(vids)))
            rows.append((f"adapted, leave-one-out, lam={lam}",
                         [H.adapt(P, seqs[:i] + seqs[i + 1:], lam=lam) for i in range(len(vids))]))
        print(f"\n{name}")
        for label, plist in rows:
            pairs = [(C.expand(C.make_joint(p, mask)(s), s, len(g)), g)
                     for p, (_, _, s, g) in zip(plist, vids)]
            m = C.score(pairs)
            print(f"  {label:34s} acc {m['acc']*100:5.1f}%  F1 {m['f1']:.3f}  "
                  f"IoU ap/in/ex {m['iou'][1]:.3f}/{m['iou'][2]:.3f}/{m['iou'][3]:.3f}  FA/h {m['fa']:5.1f}")


if __name__ == "__main__":
    main()
