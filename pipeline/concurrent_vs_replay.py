#!/usr/bin/env python3
"""Live concurrent execution vs. replay on the same videos (reviewers' concern).

Applies the counted filters fitted on joint_dump_cal2 to (a) the replayed
evidence (joint_dump_val) and (b) the evidence recorded live by
concurrent_joint.py, and reports a drive-level paired bootstrap of live - replay.

    python3 concurrent_vs_replay.py --live ~/eval_cache/concurrent_full
"""
import argparse
import glob
import os

import numpy as np

import ablate_joint as A
import compare_all_systems as C
import joint_filter as J
import paired_ci_fast as P

H = os.path.expanduser
MASKS = {"joint": {}, "joint, age masked": dict(age=True),
         "detector-only": dict(gate=True, sign=True, corr=True, age=True)}


def run(dump, names, params, mask, live):
    out = {}
    for v in names:
        z = np.load(H(f"{dump}/{v}.npz"), allow_pickle=True)
        rows, n = z["obs"], int(z["n_frames"])
        f = J.JointFilter(params)
        pred = np.array(["outside"] * n, dtype=object)
        # live rows store inference latency in column 2; the elapsed time between
        # cycles (which includes waiting for the next camera frame) is the clock step
        dts = np.r_[rows[0, 2], np.diff(rows[:, 1])] if live else rows[:, 2]
        for r, dt in zip(rows, dts):
            pred[max(0, int(r[0])):] = f.step(A.obs_of(r, mask), float(dt))
        gt = [str(x) for x in z["gt"]]
        k = min(n, len(gt))
        out[v] = ([str(x) for x in pred[:k]], gt[:k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--replay", default="~/eval_cache/joint_dump_val")
    ap.add_argument("--cal", default="~/eval_cache/joint_dump_cal2")
    args = ap.parse_args()
    names = sorted(os.path.basename(q)[:-4] for q in glob.glob(H(f"{args.live}/*.npz")))
    lat = np.concatenate([np.load(H(f"{args.live}/{v}.npz"))["obs"][:, 2] for v in names]) * 1000
    wl = np.concatenate([np.load(H(f"{args.live}/{v}.npz"))["world_lat"] for v in names])
    print(f"{len(names)} videos; live detector p50/p95 {np.percentile(lat,50):.0f}/{np.percentile(lat,95):.0f} ms"
          + (f", world model p50 {np.median(wl)*1000:.0f} ms" if len(wl) else ""))
    drive = lambda v: "_".join(v.split("_")[:2])
    ds = sorted({drive(v) for v in names})
    g = np.array([ds.index(drive(v)) for v in names])
    W = np.random.default_rng(0).multinomial(len(ds), np.ones(len(ds)) / len(ds), size=10000)
    for label, mask in MASKS.items():
        p = A.fit(H(args.cal), mask)
        rp, lv = run(args.replay, names, p, mask, False), run(args.live, names, p, mask, True)
        for nm, d in (("replay", rp), ("live", lv)):
            m = C.metrics(d)
            print(f"{label:18s} {nm:6s} F1 {m['f1']:.3f} IoUin {m['iou_in']:.3f} inP {(m['in_p'] or 0)*100:.1f} FA {m['fa_per_h']:.1f}")
        Sa = np.array([P.video_stats(*lv[v]) for v in names])
        Sb = np.array([P.video_stats(*rp[v]) for v in names])
        Ga, Gb = np.zeros((len(ds), Sa.shape[1])), np.zeros((len(ds), Sb.shape[1]))
        np.add.at(Ga, g, Sa); np.add.at(Gb, g, Sb)
        for k in ("f1", "fa_h"):
            diff = P.metrics_from(W @ Ga)[k] - P.metrics_from(W @ Gb)[k]
            d0 = P.metrics_from(Sa.sum(0))[k][0] - P.metrics_from(Sb.sum(0))[k][0]
            print(f"   live - replay {k}: {d0:+.3f} [{np.percentile(diff,2.5):+.3f}, {np.percentile(diff,97.5):+.3f}]")


if __name__ == "__main__":
    main()
