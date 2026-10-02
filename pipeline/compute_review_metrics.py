#!/usr/bin/env python3
"""Reviewer-requested metrics (W4/W5/W6) computed from cached per-frame state
arrays (~/eval_cache/<system>/*.npz) -- no inference re-run needed.

W4: operational false-alarm characterization
  - false alerts per hour (predicted non-outside event onsets with no GT
    overlap of the same state family)
  - specificity + false transitions + false-event duration on the pure
    negative videos (GT = outside everywhere)
W5: stricter event metrics
  - per-event temporal IoU; event P/R at temporal-IoU thresholds
  - signed entry offsets: early vs late split
W6: uncertainty
  - video-level bootstrap 95% CIs for frame accuracy and INSIDE event P/R

Usage:
  python3 compute_review_metrics.py --cache-dir ~/eval_cache/vlm \
      --split-file eval_split_full.json --split validation
"""
import argparse
import json
import os

import numpy as np

from paper_metrics import extract_events, _overlaps

STATES = ["outside", "approaching", "inside", "exiting"]
FPS = 30.0


def load_split_arrays(cache_dir, videos):
    data = {}
    for v in videos:
        p = os.path.join(cache_dir, f"{v}.npz")
        if os.path.exists(p):
            d = np.load(p)
            data[v] = (d["predicted"].tolist(), d["gt"].tolist())
    return data


def temporal_iou(a, b):
    inter = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    union = (a.end - a.start + 1) + (b.end - b.start + 1) - inter
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split-file", default="eval_split_full.json")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--annotations",
                    default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    args = ap.parse_args()

    videos = json.load(open(args.split_file))[args.split]
    ann = json.load(open(args.annotations))
    data = load_split_arrays(os.path.expanduser(args.cache_dir), videos)
    print(f"loaded {len(data)}/{len(videos)} cached videos from {args.cache_dir}")

    # ---------------- W4: false alarms ----------------
    total_hours = sum(len(gt) for _, gt in data.values()) / FPS / 3600.0
    false_events = 0        # predicted active-family event with no GT active overlap
    false_event_durs = []
    for pred, gt in data.values():
        # an "alert" = maximal run of non-outside prediction
        pred_active = ["active" if s != "outside" else "outside" for s in pred]
        gt_active = ["active" if s != "outside" else "outside" for s in gt]
        P = extract_events(pred_active, "active")
        G = extract_events(gt_active, "active")
        for p in P:
            if not any(_overlaps(p, g) for g in G):
                false_events += 1
                false_event_durs.append((p.end - p.start + 1) / FPS)
    print("\n=== W4: false-alarm characterization (whole split) ===")
    print(f"total footage: {total_hours:.2f} h")
    print(f"false alerts (no GT overlap): {false_events}  ->  {false_events / total_hours:.1f} / hour")
    if false_event_durs:
        print(f"false-event duration: mean {np.mean(false_event_durs):.1f}s  "
              f"median {np.median(false_event_durs):.1f}s  max {np.max(false_event_durs):.1f}s")

    # pure negatives
    negs = [v for v in data if not any(ann[v].get(s) for s in ["approaching", "inside", "exiting"])]
    if negs:
        spec_frames, false_trans, time_in_active = [], 0, 0.0
        neg_hours = 0.0
        for v in negs:
            pred, gt = data[v]
            spec_frames.append(np.mean([p == "outside" for p in pred]))
            false_trans += sum(1 for i in range(1, len(pred)) if pred[i] != pred[i - 1])
            time_in_active += sum(1 for p in pred if p != "outside") / FPS
            neg_hours += len(pred) / FPS / 3600.0
        print(f"\n--- pure-negative videos (n={len(negs)}) ---")
        print(f"frame specificity: mean {np.mean(spec_frames):.1%}  min {np.min(spec_frames):.1%}")
        print(f"videos with zero false alarms: "
              f"{sum(1 for v in negs if all(p == 'outside' for p in data[v][0]))}/{len(negs)}")
        print(f"false transitions: {false_trans} total  "
              f"({false_trans / max(neg_hours, 1e-9):.1f} / hour)")
        print(f"time wrongly in active states: {time_in_active:.0f}s "
              f"({time_in_active / (neg_hours * 3600) if neg_hours else 0:.1%} of negative footage)")

    # ---------------- W5: stricter event metrics ----------------
    print("\n=== W5: temporal-IoU event metrics (INSIDE) ===")
    per_event_ious = []
    n_pred_total = n_gt_total = 0
    tp_at = {0.1: 0, 0.3: 0, 0.5: 0}
    tp_at_gt = {0.1: 0, 0.3: 0, 0.5: 0}
    for pred, gt in data.values():
        P = extract_events(pred, "inside")
        G = extract_events(gt, "inside")
        n_pred_total += len(P)
        n_gt_total += len(G)
        for p in P:
            best = max((temporal_iou(p, g) for g in G), default=0.0)
            per_event_ious.append(best)
            for t in tp_at:
                if best >= t:
                    tp_at[t] += 1
        for g in G:
            best = max((temporal_iou(p, g) for p in P), default=0.0)
            for t in tp_at_gt:
                if best >= t:
                    tp_at_gt[t] += 1
    print(f"mean per-predicted-event temporal IoU: {np.mean(per_event_ious):.3f}  "
          f"median {np.median(per_event_ious):.3f}")
    for t in sorted(tp_at):
        prec = tp_at[t] / n_pred_total if n_pred_total else 0
        rec = tp_at_gt[t] / n_gt_total if n_gt_total else 0
        print(f"tIoU>={t}: precision {prec:.1%}  recall {rec:.1%}")

    print("\n--- signed entry offsets (matched INSIDE events) ---")
    offs = []
    for pred, gt in data.values():
        P = extract_events(pred, "inside")
        G = extract_events(gt, "inside")
        for g in G:
            if g.start == 0:
                continue
            cands = [p for p in P if _overlaps(p, g)]
            if cands:
                best = min(cands, key=lambda p: abs(p.start - g.start))
                offs.append((best.start - g.start) / FPS)
    offs = np.array(offs)
    if len(offs):
        print(f"n={len(offs)}  early (pred before GT): {np.mean(offs < 0):.1%}  "
              f"late: {np.mean(offs > 0):.1%}")
        print(f"signed mean {np.mean(offs):+.2f}s  median {np.median(offs):+.2f}s  "
              f"p10 {np.percentile(offs, 10):+.2f}s  p90 {np.percentile(offs, 90):+.2f}s")

    # ---------------- W6: bootstrap CIs ----------------
    print("\n=== W6: video-level bootstrap 95% CIs (10k resamples) ===")
    rng = np.random.default_rng(0)
    vids = list(data.keys())
    fa = np.array([np.mean([p == g for p, g in zip(*data[v])]) for v in vids])
    n_frames = np.array([len(data[v][1]) for v in vids])

    def boot(stat_fn):
        vals = []
        for _ in range(10000):
            idx = rng.integers(0, len(vids), len(vids))
            vals.append(stat_fn(idx))
        return np.percentile(vals, [2.5, 97.5])

    lo, hi = boot(lambda idx: np.sum(fa[idx] * n_frames[idx]) / np.sum(n_frames[idx]))
    point = np.sum(fa * n_frames) / np.sum(n_frames)
    print(f"frame accuracy: {point:.1%}  [{lo:.1%}, {hi:.1%}]")

    ev = []
    for v in vids:
        pred, gt = data[v]
        P = extract_events(pred, "inside")
        G = extract_events(gt, "inside")
        tp_p = sum(1 for p in P if any(_overlaps(p, g) for g in G))
        tp_g = sum(1 for g in G if any(_overlaps(p, g) for p in P))
        ev.append((tp_p, len(P), tp_g, len(G)))
    ev = np.array(ev)

    lo, hi = boot(lambda idx: ev[idx, 0].sum() / max(ev[idx, 1].sum(), 1))
    print(f"INSIDE event precision: {ev[:,0].sum()/max(ev[:,1].sum(),1):.1%}  [{lo:.1%}, {hi:.1%}]")
    lo, hi = boot(lambda idx: ev[idx, 2].sum() / max(ev[idx, 3].sum(), 1))
    print(f"INSIDE event recall:    {ev[:,2].sum()/max(ev[:,3].sum(),1):.1%}  [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main()
