#!/usr/bin/env python3
"""Build the head-to-head table across every system we have cached predictions
for, on the same 208-video validation split and the same paper-exact metrics.

Systems (whichever caches exist):
  vlm                  our fine-tuned 2B (sampled)          ~/eval_cache/vlm
  vlm_greedy           our 2B, greedy                       ~/eval_cache/vlm_greedy
  yoloclip             ported detector+CLIP                 ~/eval_cache/yoloclip
  yoloclip_fastentry   detector + VLM text fast entry       ~/eval_cache/yoloclip_fastentry
  cosmos3edge          Cosmos3-Edge INT4 reasoner cascade   ~/eval_cache/cosmos3edge
  vlm_zeroshot         base Cosmos-Reason2-2B (control)     ~/eval_cache/vlm_zeroshot
"""
import json
import os

import numpy as np

from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                            extract_events, _overlaps, per_state_iou,
                            frame_accuracy, macro_f1)

STATES = ["outside", "approaching", "inside", "exiting"]
FPS = 30.0
SYSTEMS = [
    ("our 2B (sampled)", "~/eval_cache/vlm"),
    ("our 2B (greedy)", "~/eval_cache/vlm_greedy"),
    ("detector+CLIP", "~/eval_cache/yoloclip"),
    ("detector +textFE", "~/eval_cache/yoloclip_fastentry"),
    ("Cosmos3-Edge INT4", "~/eval_cache/cosmos3edge"),
    ("Cosmos3-Edge calib.", "~/eval_cache/cosmos3edge_cal"),
    ("base ckpt zero-shot", "~/eval_cache/vlm_zeroshot"),
]


def load(cache, videos):
    d = {}
    for v in videos:
        p = os.path.join(os.path.expanduser(cache), f"{v}.npz")
        if os.path.exists(p):
            z = np.load(p)
            d[v] = (z["predicted"].tolist(), z["gt"].tolist())
    return d


def metrics(data):
    conf, ev, tm = {}, EventLevelAccumulator(STATES), TimingOffsetAccumulator(STATES)
    for pred, gt in data.values():
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]
        for p, g in zip(gt, pred):
            conf[(p, g)] = conf.get((p, g), 0) + 1
        ev.add_video(pred, gt)
        tm.add_video(pred, gt)
    r, iou, t = ev.result(), per_state_iou(conf, STATES), tm.result()
    # nuisance: active-state events with no GT overlap, per hour
    total_h = sum(len(g) for _, g in data.values()) / FPS / 3600.0
    fa = 0
    for pred, gt in data.values():
        pa = ["active" if s != "outside" else "outside" for s in pred]
        ga = ["active" if s != "outside" else "outside" for s in gt]
        G = extract_events(ga, "active")
        for e in extract_events(pa, "active"):
            if not any(_overlaps(e, g) for g in G):
                fa += 1
    return {
        "n": len(data),
        "acc": frame_accuracy(conf, STATES),
        "f1": macro_f1(conf, STATES),
        "iou_out": iou["outside"], "iou_appr": iou["approaching"],
        "iou_in": iou["inside"], "iou_exit": iou["exiting"],
        "in_p": r["inside"]["precision"], "in_r": r["inside"]["recall"],
        "entry_mae": t["overall"]["mae_s"] if t.get("overall") else float("nan"),
        "in_entry_med": t["inside"]["median_s"] if t.get("inside") else float("nan"),
        "fa_per_h": fa / total_h if total_h else float("nan"),
    }


def main():
    sf = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "eval_split_full.json")))
    videos = sf["validation"]
    rows = []
    for name, cache in SYSTEMS:
        data = load(cache, videos)
        if len(data) < 10:
            continue
        rows.append((name, metrics(data)))

    hdr = (f"{'system':22s} {'n':>4s} {'acc':>7s} {'F1':>6s} {'IoUout':>7s} "
           f"{'IoUap':>6s} {'IoUin':>6s} {'IoUex':>6s} {'inP':>6s} {'inR':>6s} "
           f"{'entMAE':>7s} {'inEntMd':>8s} {'FA/h':>6s}")
    print(hdr); print("-" * len(hdr))
    def fmt(v, spec, pct=False):
        if v is None or (isinstance(v, float) and v != v):
            return f"{'n/a':>{int(spec.split('.')[0] or 6)}s}"
        return f"{v:{spec}%}" if pct else f"{v:{spec}f}"

    for name, m in rows:
        print(f"{name:22s} {m['n']:4d} {fmt(m['acc'],'6.1',True)} {fmt(m['f1'],'6.3')} "
              f"{fmt(m['iou_out'],'7.3')} {fmt(m['iou_appr'],'6.3')} {fmt(m['iou_in'],'6.3')} "
              f"{fmt(m['iou_exit'],'6.3')} {fmt(m['in_p'],'5.1',True)} {fmt(m['in_r'],'5.1',True)} "
              f"{fmt(m['entry_mae'],'6.2')}s {fmt(m['in_entry_med'],'7.2')}s "
              f"{fmt(m['fa_per_h'],'6.1')}")
    print("\n(n = videos with cached predictions; all on the validation split,")
    print(" latency-honest protocol, paper-exact metrics)")


if __name__ == "__main__":
    main()
