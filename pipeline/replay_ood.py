#!/usr/bin/env python3
"""Out-of-domain evaluation of any temporal layer by replay, plus the evidence
statistics that explain it.

Reads the continuous 1 Hz record of dump_ood_stream.py.  Each annotated sign
approach is the 7 samples t0..t0+6, which are the samples the window
evaluations saw, so replaying the C3E cascade and the joint estimator here must
reproduce their GPU results (62% and 69%).  A policy that is not reproduced is
not scored.

The same statistics are computed in domain on the 1 Hz C3E evidence dump
(evidence_c3e), over the 6 s following each annotated OUTSIDE->APPROACHING
transition, so the two regimes can be compared on equal terms.

Usage:
  python3 replay_ood.py --stream ~/eval_cache/ood_stream
  python3 replay_ood.py --indomain ~/eval_cache/evidence_c3e
"""
import argparse
import glob
import json
import os

import numpy as np

import joint_filter as J
from cascade_state_machine import CascadeStateMachine, WZState
from evaluate_c3e_spine import C3E_CSM

WIN = 6  # seconds; samples t0..t0+WIN inclusive


# ---------------------------------------------------------------- policies
def act_cascade(gate, sign, corr, **_):
    csm = CascadeStateMachine(**C3E_CSM)
    for g, s, c in zip(gate, sign, corr):
        st = csm.update_evidence(g or s, c or s, None, fast_entry=s or (g and c))
        if st != WZState.OUTSIDE:
            return True
    return False


def make_act_joint(params, mask_det=False):
    def act(gate, sign, corr, det, **_):
        f = J.JointFilter(params)
        age = 0.0          # every sample carries a fresh world-model answer
        for g, s, c, y in zip(gate, sign, corr, det):
            obs = J.obs_index(J.det_bucket(0.0 if mask_det else y), g, s, c,
                              J.age_bucket(age))
            if f.step(obs, 1.0) != "outside":
                return True
        return False
    return act


# ---------------------------------------------------------------- statistics
def longest_run(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def window_stats(gate, sign, corr):
    ev = [g or s for g, s in zip(gate, sign)]
    return {"n_evidence": int(sum(ev)), "n_gate": int(sum(gate)),
            "n_sign": int(sum(sign)), "n_corr": int(sum(corr)),
            "longest_gate_run": longest_run(gate), "longest_evidence_run": longest_run(ev),
            "any_evidence": any(ev)}


def summarize(label, stats):
    n = len(stats)
    arr = lambda k: np.array([s[k] for s in stats], dtype=float)
    print(f"\n{label}: {n} windows of {WIN + 1} samples")
    print(f"  windows with any evidence      {arr('any_evidence').mean():6.1%}")
    for k in ("n_evidence", "n_gate", "n_sign", "n_corr",
              "longest_gate_run", "longest_evidence_run"):
        a = arr(k)
        print(f"  {k:22s} mean {a.mean():4.2f}  median {np.median(a):3.0f}  "
              f">=3: {np.mean(a >= 3):5.1%}")


# ---------------------------------------------------------------- loaders
def ood_windows(stream_dir):
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(stream_dir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        obs = z["obs"]
        by_t = {int(r[0]): r for r in obs}
        for t0 in z["approaches"]:
            rows = [by_t[t] for t in range(int(t0), int(t0) + WIN + 1) if t in by_t]
            if not rows:
                continue
            r = np.array(rows)
            yield (os.path.basename(p)[:-4], int(t0), str(z["condition"]),
                   dict(det=r[:, 1], gate=r[:, 2] > 0.5, sign=r[:, 3] > 0.5,
                        corr=r[:, 4] > 0.5))


def indomain_windows(ev_dir):
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(ev_dir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        gt, fi = [str(x) for x in z["gt"]], z["frame_idx"]
        for k in range(1, len(fi)):
            a, b = gt[min(fi[k - 1], len(gt) - 1)], gt[min(fi[k], len(gt) - 1)]
            if a == "outside" and b == "approaching":
                sl = slice(k, k + WIN + 1)
                yield (os.path.basename(p), int(fi[k]), "in-domain",
                       dict(det=np.zeros(len(fi[sl])), gate=z["gate"][sl] == 1,
                            sign=z["sign"][sl] == 1, corr=z["corr"][sl] == 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=None)
    ap.add_argument("--indomain", default=None)
    args = ap.parse_args()

    if args.indomain:
        ws = list(indomain_windows(args.indomain))
        summarize("IN DOMAIN (ROADWork calibration, C3E, 6 s after APPROACHING onset)",
                  [window_stats(w[3]["gate"], w[3]["sign"], w[3]["corr"]) for w in ws])
        acted = np.mean([act_cascade(**w[3]) for w in ws])
        print(f"  C3E cascade acts within the window   {acted:6.1%}")

    if args.stream:
        ws = list(ood_windows(args.stream))
        summarize("OUT OF DOMAIN (California sign approaches)",
                  [window_stats(w[3]["gate"], w[3]["sign"], w[3]["corr"]) for w in ws])
        p312 = json.load(open(os.path.expanduser("~/eval_cache/joint_params_full.json")))
        p100 = json.load(open(os.path.expanduser("~/eval_cache/joint_params_v2.json")))
        pnd = json.load(open(os.path.expanduser("~/eval_cache/joint_params_nodet.json")))
        policies = {
            "evidence (any gate/sign)": lambda gate, sign, **_: bool(np.any(gate | sign)),
            "C3E cascade": act_cascade,
            "joint 312": make_act_joint(p312),
            "joint 100": make_act_joint(p100),
            "joint, no detector": make_act_joint(pnd, mask_det=True),
        }
        expected = {"evidence (any gate/sign)": 37, "C3E cascade": 24, "joint 312": 27,
                    "joint 100": 29, "joint, no detector": 31}
        print(f"\n{'policy':26s} {'acted':>7s}   {'GPU run':>8s}")
        for name, fn in policies.items():
            hits = sum(fn(**w[3]) for w in ws)
            ok = "match" if hits == expected[name] else "MISMATCH"
            print(f"{name:26s} {hits:3d}/{len(ws)}   {expected[name]:3d}/39  {ok}")


if __name__ == "__main__":
    main()
