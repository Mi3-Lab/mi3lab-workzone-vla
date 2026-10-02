#!/usr/bin/env python3
"""Four-state evaluation on the California footage, by replay of the 1 Hz
evidence record (eval_cache/ood_stream) against the California labels.

Labels come from annotation/california_labels.json (human-verified) when it
exists, else from the draft; the header of every run says which, and draft
numbers must not be reported as results.

Every system consumes exactly the same samples.  The stream is 1 Hz, so
systems that run faster live (the joint estimator at ~22 Hz, the TCN on a
10 Hz grid) are approximated: the joint estimator steps once per second with a
fresh world-model answer, which reproduced its GPU window results exactly
(replay_ood.py); the TCN sees each second's features held over ten grid steps.

Usage:
  python3 california_4state.py [--labels PATH] [--by-condition]
"""
import argparse
import json
import os

import numpy as np

import ablate_joint as A
import joint_filter as J
from cascade_state_machine import CascadeStateMachine
from evaluate_c3e_spine import C3E_CSM
from paper_metrics import (EventLevelAccumulator, extract_events, _overlaps,
                           per_state_iou, frame_accuracy, macro_f1)

STATES = ["outside", "approaching", "inside", "exiting"]
STREAM = os.path.expanduser("~/eval_cache/ood_stream")
ANN = os.path.expanduser("~/jetson-deploy/annotation")
CAL = os.path.expanduser("~/eval_cache/joint_dump_cal2")


def load_labels(path=None):
    if path is None:
        path = os.path.join(ANN, "california_labels.json")
        if not os.path.exists(path):
            path = os.path.join(ANN, "california_draft.json")
    L = json.load(open(path))
    return path, {k: v for k, v in L.items() if not k.startswith("_")}


def gt_frames(rec):
    n = max(b for s in STATES for a, b in rec.get(s, [])) + 1
    g = np.array(["outside"] * n, dtype=object)
    for s in STATES:
        for a, b in rec.get(s, []):
            g[a:b + 1] = s
    return g


def videos(labels):
    for key, rec in labels.items():
        z = np.load(os.path.join(STREAM, f"{key}.npz"), allow_pickle=True)
        o, fps = z["obs"], float(z["fps"])
        yield key, str(z["condition"]), dict(
            t=o[:, 0].astype(int), det=o[:, 1], gate=o[:, 2] > .5,
            sign=o[:, 3] > .5, corr=o[:, 4] > .5, fps=fps), gt_frames(rec)


# ------------------------------------------------------------------ systems
def run_cascade(s):
    csm = CascadeStateMachine(**C3E_CSM)
    return [csm.update_evidence(g or si, c or si, None, fast_entry=si or (g and c)).value
            for g, si, c in zip(s["gate"], s["sign"], s["corr"])]


def make_joint(params, mask=None):
    mask = mask or {}

    def run(s):
        f = J.JointFilter(params)
        out = []
        for y, g, si, c in zip(s["det"], s["gate"], s["sign"], s["corr"]):
            row = (0, 0, 1.0, y, float(g), float(si), float(c), 0.0)
            out.append(f.step(A.obs_of(row, mask), 1.0))
        return out
    return run


def make_tcn(path, hidden=32, stages=3):
    import torch
    import tcn_segmenter as T
    m = T.MSTCN(n_hidden=hidden, n_stages=stages)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()

    def run(s):
        n = len(s["t"])
        k = int(T.GRID_HZ)
        feats = np.zeros((n * k, T.N_FEAT), dtype=np.float32)
        for i in range(n):
            for j in range(k):
                feats[i * k + j] = [s["det"][i], s["gate"][i], s["sign"][i], s["corr"][i],
                                    min(j / T.GRID_HZ, 10.0) / 10.0, 1.0]
        with torch.no_grad():
            p = m(torch.from_numpy(feats.T[None]))[-1].argmax(dim=1)[0].numpy()
        return [T.STATES[int(p[i * k + k - 1])] for i in range(n)]
    return run


def systems():
    load = lambda p: json.load(open(os.path.expanduser(p)))  # noqa: E731
    world = dict(gate=True, sign=True, corr=True)
    return {
        "C3E cascade": run_cascade,
        "detector, counted filter": make_joint(A.fit(CAL, world), world),
        "joint (312)": make_joint(load("~/eval_cache/joint_params_full.json")),
        "joint (100)": make_joint(load("~/eval_cache/joint_params_v2.json")),
        "joint, no detector": make_joint(load("~/eval_cache/joint_params_nodet.json"),
                                         dict(det=True)),
        "causal TCN": make_tcn(os.path.expanduser("~/eval_cache/tcn_journal.pt")),
    }


# ------------------------------------------------------------------ scoring
def expand(states, s, n):
    pred = np.array(["outside"] * n, dtype=object)
    for t, st in zip(s["t"], states):
        a = int(t * s["fps"])
        if a < n:
            pred[a:] = st
    return pred


def score(pairs, fps=30.0):
    conf, ev, fa, frames = {}, EventLevelAccumulator(STATES), 0, 0
    for pred, gt in pairs:
        ps, gs = [str(x) for x in pred], [str(x) for x in gt]
        frames += len(ps)
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs)
        pa = ["active" if x != "outside" else "outside" for x in ps]
        ga = ["active" if x != "outside" else "outside" for x in gs]
        G = extract_events(ga, "active")
        fa += sum(1 for e in extract_events(pa, "active") if not any(_overlaps(e, g) for g in G))
    iou, r = per_state_iou(conf, STATES), ev.result()
    return dict(acc=frame_accuracy(conf, STATES), f1=macro_f1(conf, STATES),
                iou=[iou[x] for x in STATES], p=r["inside"]["precision"],
                r=r["inside"]["recall"], fa=fa / (frames / fps / 3600.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    ap.add_argument("--by-condition", action="store_true")
    args = ap.parse_args()
    path, labels = load_labels(args.labels)
    print(f"labels: {path}" + ("   ** DRAFT: provisional **" if "draft" in path else ""))
    vids = list(videos(labels))
    print(f"{len(vids)} videos, {sum(len(g) for *_, g in vids) / 30 / 60:.1f} min\n")
    hdr = (f"{'system':26s} {'acc':>6s} {'F1':>6s} {'IoUout':>7s} {'IoUap':>6s} "
           f"{'IoUin':>6s} {'IoUex':>6s} {'inP':>6s} {'inR':>6s} {'FA/h':>6s}")
    print(hdr); print("-" * len(hdr))
    fmt = lambda v: "  n/a" if v is None else f"{v*100:5.1f}"  # noqa: E731
    for name, fn in systems().items():
        pairs = [(expand(fn(s), s, len(g)), g) for _, _, s, g in vids]
        m = score(pairs)
        print(f"{name:26s} {m['acc']*100:5.1f}% {m['f1']:6.3f} {m['iou'][0]:7.3f} "
              f"{m['iou'][1]:6.3f} {m['iou'][2]:6.3f} {m['iou'][3]:6.3f} "
              f"{fmt(m['p'])} {fmt(m['r'])} {m['fa']:6.1f}")
        if args.by_condition:
            for c in ("day", "evening", "sunset", "night", "night-rain", "night-fog"):
                sub = [p for p, (_, cc, _, _) in zip(pairs, vids) if cc == c]
                if sub:
                    mm = score(sub)
                    print(f"    {c:22s} {mm['acc']*100:5.1f}% {mm['f1']:6.3f}{'':38s}{mm['fa']:6.1f}")


if __name__ == "__main__":
    main()
