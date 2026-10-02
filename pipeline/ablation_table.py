#!/usr/bin/env python3
"""Builds the paper's ablation table (Sec. 5.2) from already-cached per-frame
predictions -- no inference re-run. Reuses the same accumulators as the main
report (paper_metrics.py) for exact consistency with Table 2's numbers.

Usage:
  python3 ablation_table.py --split-file eval_split_full.json --split validation
"""
import argparse
import json
import os

import numpy as np

from paper_metrics import EventLevelAccumulator, TimingOffsetAccumulator, frame_accuracy, macro_f1

STATES = ["outside", "approaching", "inside", "exiting"]

VARIANTS = [
    ("full",         "Full system (ours)"),
    ("greedy",       "- greedy decoding (T=0)"),
    ("nosign",       "- SIGN channel disabled"),
    ("binarysign",   "- SIGN as binary gate"),
    ("nofastentry",  "- fast-entry bypass disabled"),
    ("nofilter",     "- qualifier-lexicon filter disabled"),
    ("unrestbayes",  "- Bayes filter unrestricted (all 4 states)"),
]

CACHE_MAP = {
    "full": os.path.expanduser("~/eval_cache/vlm"),
    "greedy": os.path.expanduser("~/eval_cache/vlm_greedy"),
    "nosign": os.path.expanduser("~/eval_cache/vlm_nosign"),
    "binarysign": os.path.expanduser("~/eval_cache/vlm_binarysign"),
    "nofastentry": os.path.expanduser("~/eval_cache/vlm_nofastentry"),
    "nofilter": os.path.expanduser("~/eval_cache/vlm_nofilter"),
    "unrestbayes": os.path.expanduser("~/eval_cache/vlm_unrestbayes"),
}


def load_cache(cache_dir, videos):
    data = {}
    for v in videos:
        p = os.path.join(cache_dir, f"{v}.npz")
        if os.path.exists(p):
            d = np.load(p)
            data[v] = (d["predicted"].tolist(), d["gt"].tolist())
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="eval_split_full.json")
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    videos = json.load(open(args.split_file))[args.split]

    rows = []
    for key, label in VARIANTS:
        cache_dir = CACHE_MAP[key]
        data = load_cache(cache_dir, videos)
        n_loaded = len(data)
        if n_loaded == 0:
            rows.append((label, n_loaded, None, None, None, None))
            continue

        confusion = {}
        event_acc = EventLevelAccumulator(STATES)
        timing_acc = TimingOffsetAccumulator(STATES)
        for pred, gt in data.values():
            n = min(len(pred), len(gt))
            pred, gt = pred[:n], gt[:n]
            for p, g in zip(pred, gt):
                confusion[(g, p)] = confusion.get((g, p), 0) + 1
            event_acc.add_video(pred, gt)
            timing_acc.add_video(pred, gt)

        acc = frame_accuracy(confusion, STATES)
        f1 = macro_f1(confusion, STATES)
        ev = event_acc.result()["inside"]
        timing = timing_acc.result()["overall"]
        rows.append((label, n_loaded, acc, f1, ev["precision"], ev["recall"], timing["mae_s"]))

    print(f"{'variant':44s}{'n':>5s}{'frame-acc':>11s}{'macroF1':>9s}"
          f"{'insideP':>9s}{'insideR':>9s}{'|Δt| MAE':>10s}")
    for r in rows:
        label, n = r[0], r[1]
        if r[2] is None:
            print(f"{label:44s}{n:>5d}{'--- cache missing ---':>30s}")
            continue
        _, _, acc, f1, p, rec, mae = r
        mae_s = f"{mae:.2f}s" if mae is not None else "n/a"
        print(f"{label:44s}{n:>5d}{acc:>10.1%} {f1:>8.3f} {p:>8.1%} {rec:>8.1%} {mae_s:>9s}")


if __name__ == "__main__":
    main()
