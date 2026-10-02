#!/usr/bin/env python3
"""Second-review response experiments, all computed from cached per-frame
predictions (no inference re-run). Addresses:

  W1/Q12  per-city (Boston/Seattle) results, generalization signal
  W2/Q2   false-alarms/hour for the recommended greedy+1s-debounce config
  W4/Q3   SIGN channel's contribution to entries (full vs no-sign cache)
  W8/Q4   paired video-level bootstrap CIs for the APPROACHING-IoU and
          entry-timing *differences* (do they exclude zero?)

Usage: python3 review2_analysis.py
"""
import json
import os

import numpy as np

from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                            extract_events, _overlaps, per_state_iou, frame_accuracy,
                            macro_f1)

STATES = ["outside", "approaching", "inside", "exiting"]
FPS = 30.0
RNG = np.random.default_rng(0)


def load(cache, videos):
    d = {}
    for v in videos:
        p = os.path.join(os.path.expanduser(cache), f"{v}.npz")
        if os.path.exists(p):
            z = np.load(p)
            d[v] = (z["predicted"].tolist(), z["gt"].tolist())
    return d


def iou_state(pred, gt, state):
    p = np.array(pred) == state
    g = np.array(gt) == state
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / union if union else np.nan


def city_report(data, label):
    conf = {}
    ev = EventLevelAccumulator(STATES)
    for pred, gt in data.values():
        n = min(len(pred), len(gt)); pred, gt = pred[:n], gt[:n]
        for p, g in zip(pred, gt):
            conf[(g, p)] = conf.get((g, p), 0) + 1
        ev.add_video(pred, gt)
    r = ev.result()
    iou = per_state_iou(conf, STATES)
    print(f"  {label:10s} n={len(data):3d}  frameacc={frame_accuracy(conf,STATES):.1%}  "
          f"F1={macro_f1(conf,STATES):.3f}  "
          f"appr-IoU={iou['approaching']:.3f}  ins-IoU={iou['inside']:.3f}  "
          f"ins-P={r['inside']['precision']:.1%}  ins-R={r['inside']['recall']:.1%}")


def false_alarms_per_hour(data, debounce_s=0.0):
    """Active-family predicted events (approaching/inside/exiting merged) with
    no GT active overlap, per hour; plus mean/median duration and % time wrongly
    active on pure-negative videos. Debounce filters short predicted runs."""
    total_hours = sum(len(gt) for _, gt in data.values()) / FPS / 3600.0
    min_f = debounce_s * FPS
    false_events, durs = 0, []
    for pred, gt in data.values():
        pa = ["active" if s != "outside" else "outside" for s in pred]
        ga = ["active" if s != "outside" else "outside" for s in gt]
        P = [e for e in extract_events(pa, "active") if (e.end - e.start + 1) >= min_f]
        G = extract_events(ga, "active")
        for p in P:
            if not any(_overlaps(p, g) for g in G):
                false_events += 1
                durs.append((p.end - p.start + 1) / FPS)
    return false_events / total_hours, (np.mean(durs) if durs else 0), (np.median(durs) if durs else 0)


def main():
    sf = json.load(open("eval_split_full.json"))
    videos = sf["validation"]
    vlm = load("~/eval_cache/vlm", videos)
    greedy = load("~/eval_cache/vlm_greedy", videos)
    nosign = load("~/eval_cache/vlm_nosign", videos)
    yc = load("~/eval_cache/yoloclip", videos)

    # ---------- W1/Q12: per-city ----------
    print("=== W1/Q12: per-city results (generalization across cities) ===")
    for name, d in [("VLM raw", vlm), ("Detector+CLIP", yc)]:
        print(f"{name}:")
        for city in ["boston", "seattle"]:
            sub = {k: v for k, v in d.items() if k.startswith(city)}
            city_report(sub, city)

    # ---------- W2/Q2: false alarms for recommended config ----------
    print("\n=== W2/Q2: false alarms / hour ===")
    for name, d, deb in [("VLM raw (sampled)", vlm, 0.0),
                         ("VLM greedy, no debounce", greedy, 0.0),
                         ("VLM greedy + 1s debounce (REC)", greedy, 1.0),
                         ("VLM greedy + 2s debounce", greedy, 2.0),
                         ("Detector+CLIP (paired)", yc, 0.0)]:
        fa, md, mdn = false_alarms_per_hour(d, deb)
        print(f"  {name:32s} {fa:5.1f}/h  mean-dur {md:.1f}s  median {mdn:.1f}s")

    # ---------- W4/Q3: SIGN contribution ----------
    print("\n=== W4/Q3: SIGN channel contribution to entries ===")
    # count APPROACHING/INSIDE entry events (first transition into active) per config
    def first_active_entries(data):
        # per video: frame index of first transition outside->active, or None
        entries = {}
        for v, (pred, gt) in data.items():
            idx = None
            for i in range(1, len(pred)):
                if pred[i] != "outside" and pred[i - 1] == "outside":
                    idx = i; break
            entries[v] = idx
        return entries
    ef = first_active_entries(vlm)
    en = first_active_entries(nosign)
    both = [v for v in ef if ef[v] is not None and en[v] is not None]
    only_full = [v for v in ef if ef[v] is not None and en[v] is None]
    gained = []
    for v in both:
        # positive = full entered earlier than no-sign (sign accelerated it)
        gained.append((en[v] - ef[v]) / FPS)
    gained = np.array(gained)
    print(f"  videos with an entry in full system: {sum(1 for v in ef if ef[v] is not None)}")
    print(f"  entries present in full but ABSENT without SIGN: {len(only_full)} "
          f"({len(only_full)/max(sum(1 for v in ef if ef[v] is not None),1):.1%})")
    print(f"  entries in both; SIGN accelerated by (median): {np.median(gained):+.2f}s  "
          f"mean {np.mean(gained):+.2f}s   (>0 = SIGN earlier)")
    print(f"  entries SIGN made >=0.5s earlier: {int(np.sum(gained>=0.5))}/{len(both)}")

    # ---------- W8/Q4: paired bootstrap CIs on DIFFERENCES ----------
    print("\n=== W8/Q4: paired video-level bootstrap 95% CIs on VLM-minus-baseline differences ===")
    common = [v for v in videos if v in vlm and v in yc]
    # per-video approaching IoU and inside-entry median offset
    appr_v = np.array([iou_state(*vlm[v], "approaching") for v in common])
    appr_b = np.array([iou_state(*yc[v], "approaching") for v in common])

    def entry_offsets(pred, gt, state="inside"):
        P = extract_events(pred, state); G = extract_events(gt, state)
        offs = []
        for g in G:
            if g.start == 0: continue
            c = [p for p in P if _overlaps(p, g)]
            if c:
                b = min(c, key=lambda p: abs(p.start - g.start))
                offs.append((b.start - g.start) / FPS)
        return offs

    # pool entry offsets per video (mean within video for pairing)
    off_v = np.array([np.mean(entry_offsets(*vlm[v])) if entry_offsets(*vlm[v]) else np.nan for v in common])
    off_b = np.array([np.mean(entry_offsets(*yc[v])) if entry_offsets(*yc[v]) else np.nan for v in common])

    def boot_diff(a, b, reps=10000):
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        idx = np.arange(len(a))
        diffs = []
        for _ in range(reps):
            s = RNG.integers(0, len(a), len(a))
            diffs.append(np.nanmean(a[s]) - np.nanmean(b[s]))
        pt = np.mean(a) - np.mean(b)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # two-sided p ~ fraction of resamples crossing zero
        p = 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))
        return pt, lo, hi, p, mask.sum()

    pt, lo, hi, p, n = boot_diff(appr_v, appr_b)
    print(f"  APPROACHING IoU diff (VLM-base): {pt:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p={p:.3f}  (n={n} videos)")
    pt, lo, hi, p, n = boot_diff(off_v, off_b)
    print(f"  INSIDE-entry offset diff (s): {pt:+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]  p={p:.3f}  (n={n} videos)")
    print("  (offset more negative = VLM alerts earlier)")


if __name__ == "__main__":
    main()
