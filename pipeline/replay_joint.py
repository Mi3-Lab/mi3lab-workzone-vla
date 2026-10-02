#!/usr/bin/env python3
"""Score the joint filter on a recorded evidence dump, without re-running the
GPU pass.

This is exact, for the reason sweep_joint_filter.py already relies on: the
recorded run's frame sampling and world-model schedule depend on inference
latency alone and never on the filter's parameters, so every parameter set
replayed here sees the identical observation stream.  What it cannot do is
change which frames were observed -- so it is valid for comparing parameter
sets and invalid for comparing systems whose latency differs.

Usage:
  python3 replay_joint.py --dump ~/eval_cache/joint_dump_val \
      --params ~/eval_cache/joint_params_v2.json [--cache-dir ~/eval_cache/...]
"""
import argparse
import glob
import json
import os

import numpy as np

import joint_filter as J
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                           extract_events, _overlaps, per_state_iou,
                           frame_accuracy, macro_f1)

STATES = J.STATES
FPS = 30.0


def replay(rows, n_frames, params):
    filt = J.JointFilter(params)
    pred = np.full(n_frames, "outside", dtype=object)
    for r in rows:
        frame_idx, _clock, dt, y_s, gate, sign, corr, age = r
        obs = J.obs_index(J.det_bucket(y_s), bool(gate), bool(sign), bool(corr),
                          J.age_bucket(age))
        pred[max(0, int(frame_idx)):] = filt.step(obs, float(dt))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--tau", type=float, default=None,
                    help="override EVIDENCE_TAU (default: the module's 0.5)")
    args = ap.parse_args()

    if args.tau is not None:
        J.EVIDENCE_TAU = args.tau
    with open(os.path.expanduser(args.params)) as fh:
        params = json.load(fh)
    if args.cache_dir:
        os.makedirs(os.path.expanduser(args.cache_dir), exist_ok=True)

    conf, ev, tm = {}, EventLevelAccumulator(STATES), TimingOffsetAccumulator(STATES)
    fa, total_frames = 0, 0
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(args.dump), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        rows, gt = z["obs"], [str(x) for x in z["gt"]]
        if rows.size == 0:
            continue
        pred = replay(rows, int(z["n_frames"]), params)
        n = min(len(pred), len(gt))
        ps, gs = [str(pred[i]) for i in range(n)], gt[:n]
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs); tm.add_video(ps, gs)
        total_frames += n
        ga = ["active" if s != "outside" else "outside" for s in gs]
        pa = ["active" if s != "outside" else "outside" for s in ps]
        G = extract_events(ga, "active")
        fa += sum(1 for e in extract_events(pa, "active")
                  if not any(_overlaps(e, g) for g in G))
        if args.cache_dir:
            np.savez_compressed(
                os.path.join(os.path.expanduser(args.cache_dir),
                             os.path.basename(p)),
                predicted=np.array(ps), gt=np.array(gs))

    r, iou, t = ev.result(), per_state_iou(conf, STATES), tm.result()
    hours = total_frames / FPS / 3600.0
    print(f"params: {args.params}  (fitted on "
          f"{params.get('n_calibration_videos','?')} videos)  tau={J.EVIDENCE_TAU}")
    print(f"  acc {frame_accuracy(conf, STATES)*100:.1f}%  macro-F1 {macro_f1(conf, STATES):.3f}")
    print(f"  IoU  out {iou['outside']:.3f}  appr {iou['approaching']:.3f}  "
          f"in {iou['inside']:.3f}  exit {iou['exiting']:.3f}")
    print(f"  INSIDE event  P {r['inside']['precision']*100:.1f}%  "
          f"R {r['inside']['recall']*100:.1f}%")
    print(f"  entry MAE {t['overall']['mae_s']:.2f}s  "
          f"INSIDE entry median {t['inside']['median_s']:.2f}s")
    print(f"  false alarms/h {fa/hours:.1f}   ({len(glob.glob(os.path.join(os.path.expanduser(args.dump),'*.npz')))} videos)")


if __name__ == "__main__":
    main()
