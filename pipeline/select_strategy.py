#!/usr/bin/env python3
"""Honest re-selection of the fusion strategy on a development split that is
disjoint from the validation split.

Why this file exists: fuse_systems.py selected on one half of the *validation*
split.  Under plain macro-F1 the winner was `c3e only`; a nuisance-constrained
criterion instead favoured `det gate + c3e state` -- but that constraint was
written down AFTER seeing the results, which makes it selection on the test
set.  Here the criterion is fixed up front and the development set is the
calibration split (100 videos, never used for reporting).

CRITERION (fixed before looking at any dev number, see COSMOS3_REFORMULATION.md
"Ordem de execucao"):

    maximise macro-F1  subject to  FA/h <= FA/h of the detector on the same
    development split.

Rationale for the constraint rather than raw macro-F1: the paper already treats
nuisance rate as a first-class deployment metric, and a work-zone warning that
fires twice as often on nothing is not deployable regardless of its F1.  The
detector's own rate is the reference because it is the system currently in
production, so "no worse nuisance than what is deployed today" is the bar a
replacement has to clear.

Usage:
  python3 select_strategy.py            # selects on dev, reports on validation
"""
import json
import os

import numpy as np

import fuse_systems as F

DEV_CACHES = {
    "2B": "~/eval_cache/vlm",
    "det": "~/eval_cache/yoloclip_dev",
    "detFE": "~/eval_cache/yoloclip_fastentry_dev",
    "c3e": "~/eval_cache/c3e_caldev",
}


def load(caches, videos):
    out = {}
    for name, path in caches.items():
        d = {}
        for v in videos:
            p = os.path.join(os.path.expanduser(path), f"{v}.npz")
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                d[v] = ([str(x) for x in z["predicted"]], [str(x) for x in z["gt"]])
        out[name] = d
    common = set.intersection(*[set(d) for d in out.values()]) if out else set()
    return out, sorted(common)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dev_videos = json.load(open(os.path.join(here, "eval_split_dev.json")))["calibration"]
    val_videos = json.load(open(os.path.join(here, "eval_split_full.json")))["validation"]

    dev_sys, dev = load(DEV_CACHES, dev_videos)
    if len(dev) < 20:
        missing = {k: len(v) for k, v in dev_sys.items()}
        raise SystemExit(f"dev incompleto: {len(dev)} videos comuns; por sistema {missing}")
    print(f"DEV (split de calibracao, disjunto da validacao): {len(dev)} videos")

    hdr = (f"{'strategy':26s} {'acc':>6s} {'F1':>6s} {'IoUap':>6s} {'IoUin':>6s} "
           f"{'inP':>6s} {'inR':>6s} {'FA/h':>6s}")
    print(hdr)
    dev_scores = {}
    for name, fn in F.STRATEGIES.items():
        m = F.score(F.apply(fn, dev_sys, dev))
        dev_scores[name] = m
        print(f"{name:26s} {m['acc']:5.1%} {m['f1']:6.3f} {m['iou_ap']:6.3f} "
              f"{m['iou_in']:6.3f} {m['in_p']:5.1%} {m['in_r']:5.1%} {m['fa_h']:6.1f}")

    budget = dev_scores["det only"]["fa_h"]
    eligible = {k: v for k, v in dev_scores.items() if v["fa_h"] <= budget}
    print(f"\ncriterio: max macro-F1 s.a. FA/h <= {budget:.1f} (detector no DEV)")
    print(f"elegiveis: {sorted(eligible)}")
    if not eligible:
        raise SystemExit("nenhuma estrategia satisfaz a restricao")
    best = max(eligible, key=lambda k: eligible[k]["f1"])
    print(f">>> SELECIONADO no DEV: {best}  (F1 {eligible[best]['f1']:.3f}, "
          f"FA/h {eligible[best]['fa_h']:.1f})")

    val_sys, val = load(F.CACHES, val_videos)
    print(f"\n=== VALIDACAO ({len(val)} videos, nunca usados na selecao) ===")
    print(hdr)
    for name in ["2B only", "det only", "detFE only", "c3e only", best]:
        m = F.score(F.apply(F.STRATEGIES[name], val_sys, val))
        tag = "  <== selecionado no DEV" if name == best else ""
        print(f"{name:26s} {m['acc']:5.1%} {m['f1']:6.3f} {m['iou_ap']:6.3f} "
              f"{m['iou_in']:6.3f} {m['in_p']:5.1%} {m['in_r']:5.1%} {m['fa_h']:6.1f}{tag}")


if __name__ == "__main__":
    main()
