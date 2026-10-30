#!/usr/bin/env python3
"""Reviewers' request: the recalibrated detector baseline and the detector-only
counted filter on the DRIVE AGX Thor, with the Orin's constants (no
recalibration), against the same systems replayed on the Orin's recording.

Both devices ran dump_detclip.py (every detector+CLIP cycle of the validation
videos), so each system is replayed on each device's own cycle timing:

  det+CLIP hand      published constants
  det+CLIP recal.    constants recalibrated on the Orin's calibration recording
  det counted        detector score only, counted filter, Orin tables (age masked)
  det+CLIP counted   fused det+CLIP score in the counted filter, Orin tables

    python3 thor_detclip_transfer.py [--thor-cal ~/eval_cache/thor/detclip_dump_cal]

With --thor-cal it also recalibrates det+CLIP on the Thor's own calibration
recording.  Writes per-frame caches thor_* and orin_* for cluster_ci.
"""
import argparse
import os

import numpy as np

import ablate_joint as A
import compare_all_systems as C
import detclip_recal as R
import joint_filter as J
import review3_analysis as RA

H = os.path.expanduser
ORIN_CFG = {"approach_th": 0.3, "enter_th": 0.7, "exit_th": 0.3, "min_out": 80.0, "max_app": 150.0, "alpha": 0.1}
DET = dict(gate=True, sign=True, corr=True, age=True)


def det_counted(vids, params, rec):
    """Detector-only counted filter on the y_raw stream of a detclip recording."""
    for v in vids:
        f = J.JointFilter(params); pred = np.array(["outside"] * v["n"], dtype=object)
        for k in range(len(v["fi"])):
            row = (v["fi"][k], 0.0, v["dt"][k], v["y_raw"][k], 0.0, 0.0, 0.0, 1e9)
            pred[v["fi"][k]:] = f.step(A.obs_of(row, DET), float(v["dt"][k]))
        rec[v["name"]] = (list(pred), [R.ST[x] for x in v["gt"]])


def load(dump):
    vids = R.load(dump)
    for v in vids:   # R.load keeps y_ver; the detector-only filter needs the raw score
        z = np.load(H(f"{dump}/{v['name']}"), allow_pickle=True)
        v["y_raw"] = z["obs"][:, 2]
    return vids


def systems(vids, cal, det_params, prefix, extra_cfg=None):
    fused = {}
    R.run(cal + vids, R.one(R.HAND), fused_out=fused)
    counted_params = R.counted_fit(cal, fused)
    out = {}
    for name, cfg in (("hand", R.HAND), ("recal", ORIN_CFG)) + ((("own", extra_cfg),) if extra_cfg else ()):
        rec = {}; R.run(vids, R.one(cfg), record=rec); out[name] = rec
    rec = {}; det_counted(vids, det_params, rec); out["detcnt"] = rec
    rec = {}; R.counted_run(vids, fused, counted_params, rec); out["fcnt"] = rec
    for k, rec in out.items():
        R.save(f"{prefix}_{k}", rec); R.report(f"{prefix} {k}", rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thor-val", default="~/eval_cache/thor/detclip_dump_val")
    ap.add_argument("--thor-cal", default=None)
    args = ap.parse_args()
    orin_cal, orin_val = load("~/eval_cache/detclip_dump_cal"), load("~/eval_cache/detclip_dump_val")
    thor_val = load(args.thor_val)
    det_params = A.fit(H("~/eval_cache/joint_dump_cal2"), DET)
    print(f"Orin {len(orin_cal)} cal / {len(orin_val)} val, Thor {len(thor_val)} val", flush=True)
    systems(orin_val, orin_cal, det_params, "orin")
    own = None
    if args.thor_cal:
        thor_cal = load(args.thor_cal)
        own, f = R.search(thor_cal)
        print(f"  Thor-recalibrated config {own} (Thor cal F1 {f:.3f}, {len(thor_cal)} videos)", flush=True)
    systems(thor_val, orin_cal, det_params, "thor", own)

    vids = sorted(v["name"][:-4] for v in thor_val)
    S = {f"{d}_{k}": f"~/eval_cache/{d}_{k}" for d in ("orin", "thor") for k in ("hand", "recal", "detcnt", "fcnt")}
    pairs = [(f"thor_{k}", f"orin_{k}") for k in ("hand", "recal", "detcnt", "fcnt")]
    pairs += [("thor_recal", "thor_hand"), ("thor_detcnt", "thor_recal")]
    if own:
        S["thor_own"] = "~/eval_cache/thor_own"
        pairs += [("thor_own", "thor_recal"), ("thor_own", "orin_recal")]
    RA.cluster_ci(S, pairs, vids, 10000)


if __name__ == "__main__":
    main()
