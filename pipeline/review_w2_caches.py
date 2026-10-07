#!/usr/bin/env python3
"""Write per-frame prediction caches for the reviewer-response comparisons, all
on the same detector-only evidence (joint_dump_val):
  det_sm_hand       detector state machine, hand-tuned constants (replay)
  det_sm_cal        detector state machine, constants recalibrated on calibration
  det_counted       counted filter on detector evidence only (joint w/o world model)
then paired video-level bootstrap CIs, 10,000 replications."""
import json
import os

import numpy as np

import ablate_joint as A
import calibrate_detector_sm as D
import paired_ci_fast as P

H = os.path.expanduser


def save(name, rec):
    d = H(f"~/eval_cache/{name}"); os.makedirs(d, exist_ok=True)
    for k, (p, g) in rec.items():
        np.savez_compressed(os.path.join(d, k), predicted=np.array(p), gt=np.array(g))


val = D.load("~/eval_cache/joint_dump_val")
cal_cfg = json.load(open(H("~/eval_cache/detector_sm_calibrated_refined3.json")))
for name, cfg in (("det_sm_hand", D.HAND), ("det_sm_cal", cal_cfg)):
    rec = {}; D.run(val, {k: np.array([v], float) for k, v in cfg.items()}, record=rec); save(name, rec)

world = dict(gate=True, sign=True, corr=True)
params = A.fit(H("~/eval_cache/joint_dump_cal2"), world)
import glob, joint_filter as J
rec = {}
for p in sorted(glob.glob(H("~/eval_cache/joint_dump_val/*.npz"))):
    z = np.load(p, allow_pickle=True); rows = z["obs"]
    if rows.size == 0: continue
    f = J.JointFilter(params); n = int(z["n_frames"]); pred = np.array(["outside"] * n, dtype=object)
    for r in rows:
        pred[max(0, int(r[0])):] = f.step(A.obs_of(r, world), float(r[2]))
    gt = [str(x) for x in z["gt"]]; m = min(n, len(gt))
    rec[os.path.basename(p)] = (list(pred[:m]), gt[:m])
save("det_counted", rec)

P.CACHES.update({"det SM hand": "~/eval_cache/det_sm_hand", "det SM cal": "~/eval_cache/det_sm_cal",
                 "det counted": "~/eval_cache/det_counted"})
P.PAIRS[:] = [("det SM cal", "det SM hand"), ("det counted", "det SM cal"),
              ("joint 312", "det SM cal"), ("joint 312", "det pair"),
              ("joint 312", "C3E calib."), ("C3E calib.", "det SM cal")]
import sys; sys.argv = ["x", "--reps", "10000"]
P.main()
