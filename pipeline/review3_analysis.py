#!/usr/bin/env python3
"""Third-review analyses, all from cached per-frame predictions (no inference).

  --cluster   paired bootstrap that resamples DRIVES (all videos of a drive
              together), so correlated videos of one drive no longer count as
              independent; reported next to the video-level interval.
  --lead      anticipation: for every annotated INSIDE onset, how long before it
              the system was already in an active state (signed; negative =
              late), the fraction warned >= 1/2/3 s ahead, premature warnings
              (active > 5 s before the annotated zone starts), and the signed
              INSIDE entry offset.
  --equalfa   every system through the same causal persistence filter (an
              active run is shown only after it has lasted d seconds), swept
              over d; macro-F1 and INSIDE recall interpolated at matched false
              alarms per hour.

Caches: --set table (Table I, 208 validation videos) or --set dd (drive-disjoint
predictions from drive_disjoint.py, all 520 videos).
"""
import argparse
import json
import os

import numpy as np

import compare_all_systems as C
import paired_ci_fast as P
from paper_metrics import extract_events

H = os.path.expanduser
FPS = 30.0
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = {
    "det+CLIP hand": "~/eval_cache/yoloclip",
    "det SM cal": "~/eval_cache/det_sm_cal",
    "det counted": "~/eval_cache/det_counted",
    "2B greedy": "~/eval_cache/vlm_greedy",
    "C3E cal": "~/eval_cache/cosmos3edge_cal",
    "joint": "~/eval_cache/joint_full_val",
}
DD = {
    "det SM hand": "~/eval_cache/dd_det_sm_hand",
    "det SM cal": "~/eval_cache/dd_det_sm_cal",
    "det counted": "~/eval_cache/dd_det_counted",
    "joint": "~/eval_cache/dd_joint",
    "det+CLIP hand": "~/eval_cache/dd_detclip_hand",
    "det+CLIP cal": "~/eval_cache/dd_detclip_cal",
}
PAIRS_TABLE = [("det SM cal", "det+CLIP hand"), ("joint", "det+CLIP hand"), ("joint", "det SM cal"),
               ("det counted", "det SM cal"), ("joint", "C3E cal"), ("C3E cal", "det SM cal"),
               ("joint", "det counted")]
PAIRS_DD = [("det SM cal", "det SM hand"), ("joint", "det SM cal"), ("det counted", "det SM cal"),
            ("joint", "det counted"), ("det+CLIP cal", "det+CLIP hand"), ("joint", "det+CLIP cal"),
            ("det counted", "det+CLIP cal"), ("det+CLIP cal", "det SM cal")]


def drive_of(name):
    city, h = os.path.basename(name).split("_")[:2]
    return f"{city}_{h}"


def load(cache, videos):
    out = {}
    for v in videos:
        p = os.path.join(H(cache), f"{v}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            out[v] = ([str(x) for x in z["predicted"]], [str(x) for x in z["gt"]])
    return out


def cluster_ci(systems, pairs, videos, reps, seed=0):
    data = {k: load(c, videos) for k, c in systems.items()}
    data = {k: d for k, d in data.items() if len(d) >= 0.9 * len(videos)}
    rng = np.random.default_rng(seed)
    keys = ["f1", "acc", "iou_ap", "iou_in", "iou_ex", "in_p", "fa_h"]
    for a, b in pairs:
        if a not in data or b not in data:
            continue
        common = sorted(set(data[a]) & set(data[b]))
        drives = sorted({drive_of(v) for v in common})
        di = {d: i for i, d in enumerate(drives)}
        Sa = np.array([P.video_stats(*data[a][v]) for v in common])
        Sb = np.array([P.video_stats(*data[b][v]) for v in common])
        g = np.array([di[drive_of(v)] for v in common])
        Ga = np.zeros((len(drives), Sa.shape[1])); np.add.at(Ga, g, Sa)
        Gb = np.zeros((len(drives), Sb.shape[1])); np.add.at(Gb, g, Sb)
        pa, pb = P.metrics_from(Sa.sum(0)), P.metrics_from(Sb.sum(0))
        Wd = rng.multinomial(len(drives), np.ones(len(drives)) / len(drives), size=reps)
        Wv = rng.multinomial(len(common), np.ones(len(common)) / len(common), size=reps)
        md_a, md_b = P.metrics_from(Wd @ Ga), P.metrics_from(Wd @ Gb)
        mv_a, mv_b = P.metrics_from(Wv @ Sa), P.metrics_from(Wv @ Sb)
        print(f"\n{a}  vs  {b}   ({len(common)} videos, {len(drives)} drives)")
        for k in keys:
            d0 = float(pa[k][0] - pb[k][0])
            dd = md_a[k] - md_b[k]; dv = mv_a[k] - mv_b[k]
            lo, hi = np.nanpercentile(dd, [2.5, 97.5]); vlo, vhi = np.nanpercentile(dv, [2.5, 97.5])
            print(f"   {k:7s} {d0:+8.3f}   drive-level [{lo:+.3f}, {hi:+.3f}]   video-level [{vlo:+.3f}, {vhi:+.3f}]")


def leads(pred, gt):
    """Per annotated INSIDE onset: (lead_s or None if missed, premature, signed inside-entry offset)."""
    n = len(gt)
    act = np.array([s != "outside" for s in pred])
    gact = np.array([s != "outside" for s in gt])
    res = []
    for t0 in range(1, n):
        if gt[t0] != "inside" or gt[t0 - 1] == "inside":
            continue
        g_end = t0
        while g_end + 1 < n and gt[g_end + 1] == "inside":
            g_end += 1
        g_start = t0
        while g_start > 0 and gact[g_start - 1]:
            g_start -= 1
        if act[t0]:
            s = t0
            while s > 0 and act[s - 1]:
                s -= 1
            lead = (t0 - s) / FPS
            premature = (g_start - s) / FPS > 5.0
        else:
            nxt = np.nonzero(act[t0:g_end + 1])[0]
            lead = -nxt[0] / FPS if len(nxt) else None
            premature = False
        lo = max(0, t0 - int(10 * FPS))
        ent = None
        for f in range(lo, g_end + 1):
            if pred[f] == "inside" and (f == 0 or pred[f - 1] != "inside"):
                ent = (f - t0) / FPS
                break
            if f == lo and pred[f] == "inside":
                ent = (f - t0) / FPS
                break
        res.append((lead, premature, ent, (t0 - g_start) / FPS))
    return res


def lead_table(systems, videos):
    print(f"\n{'system':16s} {'onsets':>6s} {'missed':>7s} {'>=1s':>6s} {'>=2s':>6s} {'>=3s':>6s} "
          f"{'med lead':>9s} {'prem.':>6s} {'IN entry med':>13s} {'IQR':>15s}")
    gt_ceiling = None
    for name, cache in systems.items():
        d = load(cache, videos)
        if len(d) < 0.9 * len(videos):
            continue
        R = [r for p, g in d.values() for r in leads(p[:len(g)], g[:len(p)])]
        n = len(R)
        L = np.array([r[0] if r[0] is not None else -np.inf for r in R])
        miss = np.mean([r[0] is None for r in R])
        E = np.array([r[2] for r in R if r[2] is not None])
        fin = L[np.isfinite(L)]
        print(f"{name:16s} {n:6d} {miss*100:6.1f}% {np.mean(L >= 1)*100:5.1f}% {np.mean(L >= 2)*100:5.1f}% "
              f"{np.mean(L >= 3)*100:5.1f}% {np.median(fin):8.2f}s {np.mean([r[1] for r in R])*100:5.1f}% "
              f"{np.median(E):12.2f}s [{np.percentile(E,25):+.2f},{np.percentile(E,75):+.2f}]")
        gt_ceiling = np.array([r[3] for r in R])
    if gt_ceiling is not None:
        print(f"{'annotation':16s} approach annotated before onset: >=1s {np.mean(gt_ceiling>=1)*100:.1f}%  "
              f">=2s {np.mean(gt_ceiling>=2)*100:.1f}%  >=3s {np.mean(gt_ceiling>=3)*100:.1f}%  "
              f"median {np.median(gt_ceiling):.2f}s")


def persist(pred, d):
    """Causal persistence filter: an active run is shown only after lasting d seconds."""
    if d <= 0:
        return pred
    p = list(pred)
    k = int(round(d * FPS))
    i, n = 0, len(p)
    while i < n:
        if p[i] != "outside":
            j = i
            while j < n and p[j] != "outside":
                j += 1
            for t in range(i, min(j, i + k)):
                p[t] = "outside"
            i = j
        else:
            i += 1
    return p


def equal_fa(systems, videos, targets=(20.0, 30.0)):
    ds = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]
    print(f"\n{'system':16s} " + " ".join(f"{'F1@'+str(int(t)):>7s} {'inR@'+str(int(t)):>7s} {'inP@'+str(int(t)):>7s}" for t in targets)
          + "   (d=0: F1, FA/h)")
    curves = {}
    for name, cache in systems.items():
        d = load(cache, videos)
        if len(d) < 0.9 * len(videos):
            continue
        pts = []
        for x in ds:
            m = C.metrics({k: (persist(p, x), g) for k, (p, g) in d.items()})
            pts.append((m["fa_per_h"], m["f1"], m["in_r"] or 0, m["in_p"] or 0, x))
        curves[name] = pts
        fa = np.array([q[0] for q in pts])
        cells = []
        for t in targets:
            if t > fa.max() + 1e-9:
                cells.append(f"{'(below)':>7s} {'':>7s} {'':>7s}")   # never reaches that many alarms
                continue
            if t < fa.min() - 1e-9:
                cells.append(f"{'n/a':>7s} {'':>7s} {'':>7s}")
                continue
            o = np.argsort(fa)
            vals = [np.interp(t, fa[o], np.array([q[i] for q in pts])[o]) for i in (1, 2, 3)]
            cells.append(f"{vals[0]:7.3f} {vals[1]*100:6.1f}% {vals[2]*100:6.1f}%")
        print(f"{name:16s} " + " ".join(cells) + f"   ({pts[0][1]:.3f}, {pts[0][0]:.1f})")
    json.dump(curves, open(H("~/eval_cache/equal_fa_curves.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["table", "dd"], default="table")
    ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--lead", action="store_true")
    ap.add_argument("--equalfa", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--subset", choices=["all", "validation", "calibration"], default=None)
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    split = json.load(open(os.path.join(HERE, "eval_split_full.json")))
    systems, pairs = (TABLE, PAIRS_TABLE) if args.set == "table" else (DD, PAIRS_DD)
    sub = args.subset or ("validation" if args.set == "table" else "all")
    videos = split["calibration"] + split["validation"] if sub == "all" else split[sub]
    if args.metrics:
        print(f"{'system':16s} {'n':>4s} {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} {'IoUex':>6s} {'inP':>6s} {'inR':>6s} {'FA/h':>6s}")
        for name, cache in systems.items():
            d = load(cache, videos)
            if not d:
                continue
            m = C.metrics(d)
            print(f"{name:16s} {m['n']:4d} {m['acc']*100:5.1f}% {m['f1']:6.3f} {m['iou_appr']:6.3f} {m['iou_in']:6.3f} "
                  f"{m['iou_exit']:6.3f} {(m['in_p'] or 0)*100:5.1f}% {(m['in_r'] or 0)*100:5.1f}% {m['fa_per_h']:6.1f}")
    if args.cluster:
        cluster_ci(systems, pairs, videos, args.reps)
    if args.lead:
        lead_table(systems, videos)
    if args.equalfa:
        equal_fa(systems, videos)


if __name__ == "__main__":
    main()
