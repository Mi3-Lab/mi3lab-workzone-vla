#!/usr/bin/env python3
"""Search for the best fusion of the cached systems.

Motivation: each system is best at a different thing.
  Cosmos3-Edge (calibrated) -> best acc/F1/IoU on APPROACHING (0.278) AND
                   INSIDE (0.574), but the worst nuisance rate (64.6 FA/h)
  detector+CLIP     -> lowest nuisance (30.9 FA/h)
  detector+textFE   -> best INSIDE precision (81.9%)
The complementarity is now precision-vs-nuisance, not entry-vs-inside.

So combine them.  Everything runs off cached per-frame predictions, so no
inference is re-run.

Honesty protocol: the validation split is cut in half — strategies are selected
on `dev`, and only the winner is reported on `test`.  Selecting and reporting on
the same frames would inflate the result.
"""
import json
import os
import sys

import numpy as np

from paper_metrics import (EventLevelAccumulator, extract_events, _overlaps,
                            per_state_iou, frame_accuracy, macro_f1)

STATES = ["outside", "approaching", "inside", "exiting"]
ACTIVE = {"approaching", "inside", "exiting"}
FPS = 30.0
CACHES = {
    "2B": "~/eval_cache/vlm_greedy",
    "det": "~/eval_cache/yoloclip",
    "detFE": "~/eval_cache/yoloclip_fastentry",
    "c3e": "~/eval_cache/cosmos3edge_cal",
}


def load_all(videos):
    out = {}
    for name, path in CACHES.items():
        d = {}
        for v in videos:
            p = os.path.join(os.path.expanduser(path), f"{v}.npz")
            if os.path.exists(p):
                z = np.load(p)
                d[v] = (list(z["predicted"]), list(z["gt"]))
        out[name] = d
    common = set.intersection(*[set(d) for d in out.values()])
    return out, sorted(common)


# ---- fusion strategies: take a dict {sys: pred_list} -> fused pred list ----

def s_single(which):
    return lambda p: list(p[which])


def s_state_split(entry_sys, inside_sys):
    """Use one system to decide APPROACHING/entry, another to decide INSIDE."""
    def f(p):
        a, b = p[entry_sys], p[inside_sys]
        out = []
        for x, y in zip(a, b):
            if y == "inside":
                out.append("inside")
            elif x in ACTIVE or y in ACTIVE:
                # keep the entry system's flavour of "active but not inside"
                out.append(x if x in ACTIVE else y)
            else:
                out.append("outside")
        return out
    return f


def s_vote(systems):
    """Majority vote on the coarse active/outside decision; when active, take
    the most specific state any voter proposes (inside > exiting > approaching)."""
    order = {"inside": 3, "exiting": 2, "approaching": 1, "outside": 0}
    def f(p):
        cols = [p[s] for s in systems]
        out = []
        for vals in zip(*cols):
            act = sum(1 for v in vals if v in ACTIVE)
            if act * 2 > len(vals):
                best = max(vals, key=lambda v: order[v])
                out.append(best)
            else:
                out.append("outside")
        return out
    return f


def s_gated(gate_sys, state_sys):
    """gate_sys says whether we are active at all; state_sys refines which."""
    def f(p):
        g, s = p[gate_sys], p[state_sys]
        return [(s[i] if s[i] in ACTIVE else "approaching") if g[i] in ACTIVE else "outside"
                for i in range(len(g))]
    return f


STRATEGIES = {
    "2B only": s_single("2B"),
    "det only": s_single("det"),
    "detFE only": s_single("detFE"),
    "c3e only": s_single("c3e"),
    "c3e entry + 2B inside": s_state_split("c3e", "2B"),
    "c3e entry + det inside": s_state_split("c3e", "det"),
    "c3e entry + detFE inside": s_state_split("c3e", "detFE"),
    "vote(2B,det,c3e)": s_vote(["2B", "det", "c3e"]),
    "vote(2B,detFE,c3e)": s_vote(["2B", "detFE", "c3e"]),
    "c3e gate + det state": s_gated("c3e", "det"),
    "det gate + c3e state": s_gated("det", "c3e"),
}


def score(fused_by_video):
    conf, ev = {}, EventLevelAccumulator(STATES)
    fa, total_frames = 0, 0
    for pred, gt in fused_by_video:
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]
        total_frames += n
        for g, p in zip(gt, pred):
            conf[(g, p)] = conf.get((g, p), 0) + 1
        ev.add_video(pred, gt)
        pa = ["active" if s != "outside" else "outside" for s in pred]
        ga = ["active" if s != "outside" else "outside" for s in gt]
        G = extract_events(ga, "active")
        for e in extract_events(pa, "active"):
            if not any(_overlaps(e, g) for g in G):
                fa += 1
    r, iou = ev.result(), per_state_iou(conf, STATES)
    hours = total_frames / FPS / 3600.0
    return {
        "acc": frame_accuracy(conf, STATES), "f1": macro_f1(conf, STATES),
        "iou_ap": iou["approaching"], "iou_in": iou["inside"],
        "in_p": r["inside"]["precision"], "in_r": r["inside"]["recall"],
        "fa_h": fa / hours if hours else float("nan"),
    }


def apply(strategy, systems, videos):
    out = []
    for v in videos:
        preds = {s: systems[s][v][0] for s in systems}
        gt = systems["2B"][v][1]
        n = min(min(len(x) for x in preds.values()), len(gt))
        preds = {s: list(x[:n]) for s, x in preds.items()}
        out.append((strategy(preds), list(gt[:n])))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sf = json.load(open(os.path.join(here, "eval_split_full.json")))
    systems, videos = load_all(sf["validation"])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(videos))
    dev = [videos[i] for i in idx[: len(videos) // 2]]
    test = [videos[i] for i in idx[len(videos) // 2:]]
    print(f"{len(videos)} videos comuns | dev={len(dev)} test={len(test)}\n")

    hdr = f"{'strategy':26s} {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} {'inP':>6s} {'inR':>6s} {'FA/h':>6s}"
    print("=== DEV (selecao) ==="); print(hdr)
    dev_scores = {}
    for name, fn in STRATEGIES.items():
        m = score(apply(fn, systems, dev))
        dev_scores[name] = m
        print(f"{name:26s} {m['acc']:5.1%} {m['f1']:6.3f} {m['iou_ap']:6.3f} "
              f"{m['iou_in']:6.3f} {m['in_p']:5.1%} {m['in_r']:5.1%} {m['fa_h']:6.1f}")

    best = max(dev_scores, key=lambda k: dev_scores[k]["f1"])
    print(f"\n>>> melhor no DEV por macro-F1: {best}")
    print("\n=== TEST (reporte honesto) ==="); print(hdr)
    for name in ["2B only", "det only", "detFE only", "c3e only", best]:
        m = score(apply(STRATEGIES[name], systems, test))
        tag = "  <== escolhido" if name == best else ""
        print(f"{name:26s} {m['acc']:5.1%} {m['f1']:6.3f} {m['iou_ap']:6.3f} "
              f"{m['iou_in']:6.3f} {m['in_p']:5.1%} {m['in_r']:5.1%} {m['fa_h']:6.1f}{tag}")


if __name__ == "__main__":
    main()
