#!/usr/bin/env python3
"""Fit the joint filter's parameters by counting on the calibration split.

Two tables and a prior, and nothing else:

  emission   P(joint observation | state), counted directly.  The observation
             already encodes the detector's bucket together with the world
             model's evidence and its age, so the interaction between them is
             represented rather than assumed away.

  rate       a per-SECOND transition rate matrix.  For each state pair we count
             transitions in the ground truth and divide by the total dwell time
             in the source state, which gives a rate in 1/s that the filter then
             scales by whatever dt the loop actually ran at.  This is the point
             of the whole exercise: the ported detector's equivalent constants
             were counts of cycles, so they silently changed meaning whenever the
             loop rate changed, and every earlier fusion inherited that bug.

  prior      the marginal state distribution.

Laplace smoothing keeps unseen observation cells from vetoing a state outright,
which matters because 72 cells times 4 states is more than a 100-video split can
populate densely.

Usage:
  python3 calibrate_joint_filter.py --dump ~/eval_cache/joint_dump_cal \
      --out ~/eval_cache/joint_params.json
"""
import argparse
import glob
import os

import numpy as np

STATES = ["outside", "approaching", "inside", "exiting"]
S = len(STATES)
IDX = {s: i for i, s in enumerate(STATES)}


def fit(dump_dir, n_obs, alpha=1.0):
    import joint_filter as J
    emit = np.full((S, n_obs), alpha)
    trans = np.zeros((S, S))
    dwell = np.zeros(S)
    marg = np.zeros(S)
    n_videos = 0

    for p in sorted(glob.glob(os.path.join(os.path.expanduser(dump_dir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        rows = z["obs"]
        gt = [str(x) for x in z["gt"]]
        if rows.size == 0:
            continue
        n_videos += 1
        prev_state, prev_clock = None, None
        for frame_idx, clock, dt, y_s, gate, sign, corr, age in rows:
            obs = J.obs_index(J.det_bucket(y_s), gate > 0.5, sign > 0.5,
                              corr > 0.5, J.age_bucket(age))
            fi = int(frame_idx)
            if fi < 0 or fi >= len(gt):
                continue
            si = IDX.get(gt[fi])
            if si is None:
                continue
            emit[si, int(obs)] += 1.0
            marg[si] += 1.0
            if prev_state is not None:
                elapsed = max(clock - prev_clock, 1e-6)
                dwell[prev_state] += elapsed
                if si != prev_state:
                    trans[prev_state, si] += 1.0
            prev_state, prev_clock = si, clock

    emit /= emit.sum(axis=1, keepdims=True)

    # per-second off-diagonal rates; diagonal closes each row to zero net rate
    rate = np.zeros((S, S))
    for i in range(S):
        if dwell[i] > 0:
            for j in range(S):
                if j != i:
                    rate[i, j] = trans[i, j] / dwell[i]
            rate[i, i] = -rate[i].sum()
    prior = marg / marg.sum() if marg.sum() > 0 else np.full(S, 1.0 / S)
    return emit, rate, prior, n_videos, dwell, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    args = ap.parse_args()

    import joint_filter as J
    emit, rate, prior, n_videos, dwell, trans = fit(args.dump, J.N_OBS, args.alpha)

    print(f"fitted on {n_videos} calibration videos\n")
    print("marginal state distribution:")
    for s, p in zip(STATES, prior):
        print(f"  {s:<12} {p:6.1%}   dwell {dwell[IDX[s]]:8.1f} s")

    print("\nlearned transition rates (per second, off-diagonal):")
    print(f"  {'from':<12} " + " ".join(f"{s[:6]:>8}" for s in STATES))
    for i, s in enumerate(STATES):
        cells = " ".join(f"{rate[i, j]:8.3f}" if i != j else f"{'--':>8}"
                         for j in range(S))
        print(f"  {s:<12} {cells}")
    print("\n  mean dwell implied by the self-transition rate:")
    for i, s in enumerate(STATES):
        off = rate[i].sum() - rate[i, i]
        off = -rate[i, i]
        if off > 0:
            print(f"    {s:<12} {1.0/off:6.1f} s")

    payload = {
        "emission": emit.tolist(),
        "rate": rate.tolist(),
        "prior": prior.tolist(),
        "n_calibration_videos": n_videos,
        "n_obs": int(J.N_OBS),
        "alpha": args.alpha,
    }
    with open(os.path.expanduser(args.out), "w") as fh:
        import json
        json.dump(payload, fh)
    print(f"\nparameters written to {args.out}")


if __name__ == "__main__":
    main()
