#!/usr/bin/env python3
"""Operational profile of the joint estimator, recovered from the recorded run.

The reviewers' objection to the earlier gated hybrid was that a combined system
must report what it costs to run, not only what it scores.  Everything needed is
already in the evidence dump: the per-cycle detector latency actually measured
on the Orin, and the timestamps at which world-model answers arrived.

Usage:
  python3 profile_joint.py --dump ~/eval_cache/joint_dump_val
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    dts, gaps, spans, polls_tot, cycles_tot = [], [], 0.0, 0, 0
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(args.dump), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        rows = z["obs"]
        if rows.size == 0:
            continue
        clock, dt, age = rows[:, 1], rows[:, 2], rows[:, 7]
        dts.append(dt)
        cycles_tot += len(dt)
        spans += float(clock[-1])
        # a world-model answer landed wherever the age of the current answer
        # dropped; the gap between those landings is its realised service time
        arrivals = clock[1:][age[1:] < age[:-1]]
        polls_tot += len(arrivals)
        if len(arrivals) > 1:
            gaps.append(np.diff(arrivals))

    d = np.concatenate(dts) * 1000.0
    g = np.concatenate(gaps) if gaps else np.array([np.nan])
    print(f"videos {len(dts)}   simulated driving {spans/60:.1f} min\n")
    print("detector cycle (drives the clock; never blocked by the world model)")
    print(f"  p50 {np.percentile(d,50):.0f} ms   p95 {np.percentile(d,95):.0f} ms   "
          f"mean {d.mean():.0f} ms -> {1000/d.mean():.1f} Hz")
    print(f"  cycles {cycles_tot}")
    print("\nworld model (asynchronous: polled again as soon as it answers)")
    print(f"  answers {polls_tot}  ->  {polls_tot/spans:.2f} Hz")
    print(f"  interval between answers: p50 {np.percentile(g,50)*1000:.0f} ms   "
          f"p95 {np.percentile(g,95)*1000:.0f} ms")
    print(f"\n  state updates per world-model answer: {cycles_tot/max(polls_tot,1):.1f}")
    print("  (the estimator updates on every detector cycle; the world model's\n"
          "   answer is held, with its age as part of the observation)")


if __name__ == "__main__":
    main()
