#!/usr/bin/env python3
"""Parse the dashcam .map GPS sidecar files that accompany the self-recorded
California footage, and turn them into ground-truth ego displacement.

This is what the Cosmos3-Edge deploy package explicitly lacked: it left open
"qual resolucao e correta exige verdade de campo (GPS/odometria)".  We have the
GPS, so we can settle it.

Line format (1 Hz):
    A,DDMMYY,HHMMSS,DDMM.mmmm,N,DDDMM.mmmm,E,SPEED,ax,ay,az;
Note the recorder always writes 'E' for longitude; California is west, so the
sign is applied from the recording location, not the hemisphere letter.
"""
import math
import re

R_EARTH = 6371000.0


def _dm_to_deg(token: str) -> float:
    """DDMM.mmmm / DDDMM.mmmm -> decimal degrees."""
    val = float(token)
    deg = int(val // 100)
    minutes = val - deg * 100
    return deg + minutes / 60.0


def parse_map(path, assume_west=True):
    """Returns list of dicts: t_s (seconds from start), lat, lon, speed_kmh."""
    pts = []
    t0 = None
    for line in open(path):
        m = re.match(r"\s*A,(\d{6}),(\d{6}),([\d.]+),([NS]),([\d.]+),([EW]),([\d.]+)", line)
        if not m:
            continue
        _, hhmmss, lat_s, lat_h, lon_s, lon_h, spd = m.groups()
        hh, mm, ss = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
        t = hh * 3600 + mm * 60 + ss
        if t0 is None:
            t0 = t
        lat = _dm_to_deg(lat_s) * (1 if lat_h == "N" else -1)
        lon = _dm_to_deg(lon_s)
        lon = -lon if (assume_west or lon_h == "W") else lon
        pts.append({"t_s": float(t - t0), "lat": lat, "lon": lon,
                    "speed_kmh": float(spd)})
    return pts


def local_xy(pts):
    """Equirectangular projection to local metres, origin at first point.
    x = east, y = north."""
    if not pts:
        return []
    lat0 = math.radians(pts[0]["lat"])
    lon0 = math.radians(pts[0]["lon"])
    out = []
    for p in pts:
        dlat = math.radians(p["lat"]) - lat0
        dlon = math.radians(p["lon"]) - lon0
        x = dlon * math.cos(lat0) * R_EARTH
        y = dlat * R_EARTH
        out.append((p["t_s"], x, y, p["speed_kmh"]))
    return out


def displacement_over(pts, t_start, window_s):
    """Ground-truth straight-line displacement (m) over a time window, plus the
    path length actually travelled (integrating consecutive GPS fixes)."""
    xy = local_xy(pts)
    seg = [(t, x, y, v) for (t, x, y, v) in xy if t_start <= t <= t_start + window_s]
    if len(seg) < 2:
        return None
    straight = math.dist((seg[0][1], seg[0][2]), (seg[-1][1], seg[-1][2]))
    path = sum(math.dist((seg[i][1], seg[i][2]), (seg[i + 1][1], seg[i + 1][2]))
               for i in range(len(seg) - 1))
    mean_speed = sum(s[3] for s in seg) / len(seg)
    return {"straight_m": straight, "path_m": path,
            "mean_speed_kmh": mean_speed, "n_fixes": len(seg)}


def distance_by_speed(pts, t_start, window_s):
    """Path length over a window by integrating the GPS speed channel with
    linear interpolation.

    More robust than differencing positions when the window edges fall between
    the 1 Hz fixes: a 0.1 s shift can otherwise add/drop a whole fix and swing
    the answer by tens of percent.
    """
    if not pts:
        return None
    ts = [p["t_s"] for p in pts]
    vs = [p["speed_kmh"] / 3.6 for p in pts]  # m/s
    t_end = t_start + window_s
    if t_start < ts[0] or t_end > ts[-1]:
        return None

    def v_at(t):
        import bisect
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return vs[0]
        if i >= len(ts):
            return vs[-1]
        t0, t1 = ts[i - 1], ts[i]
        if t1 == t0:
            return vs[i]
        w = (t - t0) / (t1 - t0)
        return vs[i - 1] * (1 - w) + vs[i] * w

    # trapezoidal integration on a fine grid
    n = max(2, int(window_s * 20))
    dt = window_s / n
    dist = 0.0
    for k in range(n):
        dist += 0.5 * (v_at(t_start + k * dt) + v_at(t_start + (k + 1) * dt)) * dt
    speeds = [v_at(t_start + k * dt) * 3.6 for k in range(n + 1)]
    return {"path_m": dist, "straight_m": dist,
            "mean_speed_kmh": sum(speeds) / len(speeds), "n_fixes": n}


if __name__ == "__main__":
    import os
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/workzone/data/All_Construction_Data/With_SpeedLimit/Merced_Day_01.map")
    pts = parse_map(p)
    print(f"{os.path.basename(p)}: {len(pts)} fixes, {pts[-1]['t_s']:.0f}s")
    for t in (0, 30, 60):
        d = displacement_over(pts, t, 6.0)
        if d:
            print(f"  t={t:3d}s +6s: straight {d['straight_m']:6.1f} m | "
                  f"path {d['path_m']:6.1f} m | {d['mean_speed_kmh']:.0f} km/h")
