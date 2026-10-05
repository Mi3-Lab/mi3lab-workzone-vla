#!/usr/bin/env python3
"""Four-state results on California for the world model (C3E) and Qwen-Drive,
through the SAME evidence-first cascade with the SAME constants, so the only
difference is the model.  Neither constant set was calibrated for Qwen-Drive:
  c3e-calibrated  N=3, K_enter=3, K_exit=2, K_fade=1   (fitted for C3E)
  2b-calibrated   N=5, K_enter=3, K_exit=4, K_fade=2   (fitted for our 2B)
EGO is off in both (Qwen-Drive's EGO was not recorded).  Calibrating for
Qwen-Drive would mean recording it on the ROADWork calibration videos, which
are in its training data.  Labels: verified if present, else the draft.
"""
import os

import numpy as np

import california_4state as C
from cascade_state_machine import CascadeStateMachine

H = os.path.expanduser
CONSTS = {"c3e-calibrated": dict(N=3, k_enter=3, k_exit=2, k_fading=1, use_ego=False),
          "2b-calibrated": dict(N=5, k_enter=3, k_exit=4, k_fading=2, use_ego=False)}


def qd_evidence(key):
    o = np.load(H(f"~/eval_cache/qwendrive_ood_stream/{key}.npz"))["obs"]   # t, gate, sign, corr
    return dict(t=o[:, 0].astype(int), det=np.zeros(len(o)), gate=o[:, 1] > .5,
                sign=o[:, 2] > .5, corr=o[:, 3] > .5, fps=30.0)


def cascade(s, cfg):
    csm = CascadeStateMachine(**cfg)
    return [csm.update_evidence(g or si, c or si, None, fast_entry=si or (g and c)).value
            for g, si, c in zip(s["gate"], s["sign"], s["corr"])]


def main():
    path, labels = C.load_labels()
    print(f"labels: {path}" + ("   ** DRAFT: provisional **" if "draft" in path else ""))
    vids = list(C.videos(labels))
    hdr = (f"{'model':12s} {'constants':16s} {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} "
           f"{'IoUex':>6s} {'inP':>6s} {'inR':>6s} {'FA/h':>6s}")
    print(hdr); print("-" * len(hdr))
    fmt = lambda v: "  n/a" if v is None else f"{v*100:5.1f}"  # noqa: E731
    for cname, cfg in CONSTS.items():
        for mname in ("C3E", "Qwen-Drive"):
            pairs = []
            for key, _cond, s, g in vids:
                ev = s if mname == "C3E" else dict(qd_evidence(key), fps=s["fps"])
                pairs.append((C.expand(cascade(ev, cfg), ev, len(g)), g))
            m = C.score(pairs)
            print(f"{mname:12s} {cname:16s} {m['acc']*100:5.1f}% {m['f1']:6.3f} {m['iou'][1]:6.3f} "
                  f"{m['iou'][2]:6.3f} {m['iou'][3]:6.3f} {fmt(m['p'])} {fmt(m['r'])} {m['fa']:6.1f}")


if __name__ == "__main__":
    main()
