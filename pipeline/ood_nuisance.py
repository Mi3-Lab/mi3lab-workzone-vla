#!/usr/bin/env python3
"""Recall and an UPPER BOUND on nuisance for the C3E cascade on the continuous
California record, under two parsers.

The footage has sign timestamps only, no work-zone extents.  Samples more than
MARGIN seconds from every annotated sign are treated as presumed negative, as
eval_external_negatives.py proposed.  A work zone without an annotated sign can
survive in that remainder, so activations there are an upper bound on false
alarms, not a count of them.

Parsers:
  current      sign keywords are matched in the SIGN reply only (as deployed)
  desc-all     sign keywords also matched anywhere in the DESC reply (broken:
               the reply echoes the prompt, which contains a keyword)
  desc-quoted  sign keywords also matched in quoted text of the DESC reply

The cascade runs continuously over each video at 1 Hz, as it would on the road.
Recall is scored as in the window benchmark: the state is active at some sample
within 6 s after the annotated sign.

Usage:
  python3 ood_nuisance.py --stream ~/eval_cache/ood_stream
"""
import argparse
import glob
import json
import os

import re

import numpy as np

from cascade_state_machine import CascadeStateMachine, WZState, has_corroboration
from evaluate_c3e_spine import C3E_CSM

MARGIN, WIN = 20, 6


QUOTED = re.compile(r'"([^"]{2,80})"')


def quoted_sign_hit(desc):
    """Sign keywords inside quoted text only.  Every DESC reply opens by echoing
    the prompt ("The work zone elements visible...") and "WORK ZONE" is itself a
    keyword, so matching the whole reply fires on every frame; the model reports
    what a sign actually says in quotes."""
    return any(has_corroboration("", q) for q in QUOTED.findall(desc))


def run(gate, sign, corr):
    csm = CascadeStateMachine(**C3E_CSM)
    out = []
    for g, s, c in zip(gate, sign, corr):
        out.append(csm.update_evidence(g or s, c or s, None, fast_entry=s or (g and c))
                   != WZState.OUTSIDE)
    return np.array(out)


def onsets(active):
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    args = ap.parse_args()
    d = os.path.expanduser(args.stream)
    res = {k: dict(hit=0, n=0, neg_s=0, neg_active=0, neg_onsets=0) for k in ("current", "desc-all", "desc-quoted")}
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        obs = z["obs"]; t = obs[:, 0].astype(int)
        texts = {r["t"]: r for r in json.load(open(p[:-4] + ".json"))}
        gate, sign, corr = obs[:, 2] > 0.5, obs[:, 3] > 0.5, obs[:, 4] > 0.5
        sign_all = np.array([s or has_corroboration("", texts[ti]["desc"]) for ti, s in zip(t, sign)])
        sign_q = np.array([s or quoted_sign_hit(texts[ti]["desc"]) for ti, s in zip(t, sign)])
        signs_at = np.array(z["approaches"], dtype=int)
        neg = np.array([np.all(np.abs(signs_at - ti) > MARGIN) for ti in t])
        for name, sg in (("current", sign), ("desc-all", sign_all), ("desc-quoted", sign_q)):
            a = run(gate, sg, corr); r = res[name]
            for t0 in signs_at:
                r["n"] += 1
                r["hit"] += bool(np.any(a[(t >= t0) & (t <= t0 + WIN)]))
            r["neg_s"] += int(neg.sum())
            r["neg_active"] += int((a & neg).sum())
            r["neg_onsets"] += int(np.isin(onsets(a), np.flatnonzero(neg)).sum())
    print(f"presumed-negative footage: {res['current']['neg_s']/60:.1f} min "
          f"(>{MARGIN} s from every annotated sign)\n")
    print(f"{'parser':12s} {'recall':>10s} {'active in presumed-neg':>24s} {'onsets/h there (upper bound)':>30s}")
    for name, r in res.items():
        h = r["neg_s"] / 3600
        print(f"{name:12s} {r['hit']:3d}/{r['n']} {r['hit']/r['n']:4.0%} "
              f"{r['neg_active']/r['neg_s']:>18.1%}      {r['neg_onsets']/h:>18.1f}")


if __name__ == "__main__":
    main()
