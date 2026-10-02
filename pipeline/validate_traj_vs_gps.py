#!/usr/bin/env python3
"""Validate Cosmos3-Edge generated ego-trajectory against real GPS.

The Cosmos3-Edge deploy package explicitly left this open:
    "Qual esta correta exige verdade de campo (GPS/odometria) ... Resolver isto
     antes de fixar a configuracao"
because it had no ground truth.  The self-recorded California footage ships
1 Hz GPS sidecars (.map), so we can settle it.

Usage:
  python3 validate_traj_vs_gps.py <traj_result.json> <map_file> <gps_offset_s>

`gps_offset_s` maps clip time 0 to absolute time in the .map file.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gps_map_parser import parse_map, distance_by_speed  # noqa: E402

RUNTIME = os.path.expanduser("~/jetson-deploy/cosmos3edge-orin-deploy/runtime")
sys.path.insert(0, RUNTIME)


def traj_displacement(action_60x9):
    """Integrate the 9-D relative actions into SE(3) poses and return the
    straight-line displacement and path length in metres."""
    from cosmos3edge_av_pose import integrate_av_relative_poses
    P = np.asarray(integrate_av_relative_poses(np.asarray(action_60x9, dtype=np.float64)))
    t = P[:, :3, 3]
    straight = float(np.linalg.norm(t[-1] - t[0]))
    path = float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))
    return straight, path, t


def main():
    traj_path, map_path = sys.argv[1], sys.argv[2]
    offset = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    horizon = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0

    d = json.load(open(traj_path))
    updates = d.get("updates", [])
    pts = parse_map(map_path)
    print(f"mode: {d.get('mode')} | ticks: {len(updates)} | gps fixes: {len(pts)}")
    print(f"horizon {horizon}s | clip t=0 -> gps t={offset}s\n")

    rows = []
    for u in updates:
        act = u.get("action_60x9")
        if not act:
            continue
        s_traj, p_traj, _ = traj_displacement(act)
        t_clip = float(u.get("timestamp_s", 0.0))
        causal = str(d.get("mode","")).startswith("inverse")
        w_start = offset + t_clip - horizon if causal else offset + t_clip
        gt = distance_by_speed(pts, w_start, horizon)
        if not gt:
            continue
        err = s_traj - gt["straight_m"]
        rel = err / gt["straight_m"] * 100 if gt["straight_m"] > 1e-6 else float("nan")
        rows.append((t_clip, s_traj, gt["straight_m"], err, rel, gt["mean_speed_kmh"]))
        print(f"  t={t_clip:5.1f}s  traj {s_traj:6.1f} m | gps {gt['straight_m']:6.1f} m "
              f"| err {err:+6.1f} m ({rel:+5.1f}%) | {gt['mean_speed_kmh']:4.1f} km/h")

    if rows:
        a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
        print(f"\n  n={len(rows)}  MAE {np.mean(np.abs(a[:,2])):.1f} m  "
              f"| bias {np.mean(a[:,2]):+.1f} m  | MAPE {np.mean(np.abs(a[:,3])):.1f}%")
        if len(rows) > 2:
            c = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
            print(f"  correlacao traj-vs-gps: r={c:+.3f}")


if __name__ == "__main__":
    main()
