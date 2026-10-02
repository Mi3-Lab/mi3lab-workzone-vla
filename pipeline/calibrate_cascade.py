#!/usr/bin/env python3
"""Grid-search cascade hyper-parameters from dumped per-cycle evidence.

The published thresholds (N=5, K_ENTER=3, K_EXIT=4, K_FADE=2) and the Bayesian
matrices were fixed on the calibration split *for our fine-tuned 2B*.  Running a
different model with them measures the hyper-parameters, not the model.  This
replays the recorded evidence under arbitrary settings, so a fair per-model
calibration costs CPU seconds instead of GPU hours.

Usage:
  python3 calibrate_cascade.py --evidence ~/eval_cache/evidence_c3e
"""
import argparse
import glob
import os

import numpy as np

from cascade_state_machine import CascadeStateMachine

STATES = ["outside", "approaching", "inside", "exiting"]


def load(evdir):
    out = []
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(evdir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        out.append({
            "idx": z["frame_idx"], "gate": z["gate"], "sign": z["sign"],
            "corr": z["corr"], "ego": z["ego"], "gt": z["gt"],
            "n": int(z["n_frames"]),
        })
    return out


def replay(ev, N, K_ENTER, K_EXIT, K_FADE, use_sign, use_corr, use_ego):
    """Re-run the REAL cascade state machine over recorded evidence.

    Uses CascadeStateMachine.update_evidence, so a configuration that wins here
    is the same configuration that runs live -- a hand-written re-implementation
    would calibrate one system and deploy another.
    """
    csm = CascadeStateMachine(N=N, k_enter=K_ENTER, k_exit=K_EXIT,
                              k_fading=K_FADE, use_ego=use_ego)
    pred = np.full(ev["n"], "outside", dtype=object)
    for k, f in enumerate(ev["idx"]):
        g = ev["gate"][k] == 1
        s = (ev["sign"][k] == 1) if use_sign else False
        c = (ev["corr"][k] == 1) if use_corr else False
        e = int(ev["ego"][k])
        # mirrors evaluate_cascade: sign alone can open the door (fast entry),
        # and gate+DESC in the same frame is the other high-confidence path
        corroborated = c or s
        fast_entry = s or (g and c)
        st = csm.update_evidence(g or s, corroborated,
                                 (e if e >= 0 else None) if use_ego else None,
                                 fast_entry=fast_entry)
        pred[f:] = st.value
    return pred


def score(evs, cfg):
    conf = {}
    for ev in evs:
        pred = replay(ev, **cfg)
        gt = ev["gt"]
        n = min(len(pred), len(gt))
        for a, b in zip(gt[:n], pred[:n]):
            conf[(str(a), str(b))] = conf.get((str(a), str(b)), 0) + 1
    tot = sum(conf.values())
    acc = sum(v for (a, b), v in conf.items() if a == b) / max(tot, 1)
    f1s = []
    for s in STATES:
        tp = conf.get((s, s), 0)
        fp = sum(v for (a, b), v in conf.items() if b == s and a != s)
        fn = sum(v for (a, b), v in conf.items() if a == s and b != s)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return acc, float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args()
    evs = load(args.evidence)
    print(f"{len(evs)} videos de calibracao, "
          f"{sum(len(e['idx']) for e in evs)} ciclos\n")

    base = dict(N=5, K_ENTER=3, K_EXIT=4, K_FADE=2, use_sign=True,
                use_corr=True, use_ego=True)
    a, f = score(evs, base)
    print(f"config do 2B (herdada):  acc {a:.1%}  macroF1 {f:.3f}\n")

    results = []
    for N in (3, 5, 7, 9):
        for KE in range(1, N + 1):
            for KX in range(1, N + 1):
                for KF in range(1, N + 1):
                    for ego in (True, False):
                        for sign in (True, False):
                            cfg = dict(N=N, K_ENTER=KE, K_EXIT=KX, K_FADE=KF,
                                       use_sign=sign, use_corr=True, use_ego=ego)
                            acc, f1 = score(evs, cfg)
                            results.append((f1, acc, cfg))
    results.sort(key=lambda r: -r[0])
    print("top 8 configuracoes por macro-F1 na CALIBRACAO:")
    for f1, acc, cfg in results[:8]:
        print(f"  F1 {f1:.3f} acc {acc:.1%}  N={cfg['N']} KE={cfg['K_ENTER']} "
              f"KX={cfg['K_EXIT']} KF={cfg['K_FADE']} ego={cfg['use_ego']} "
              f"sign={cfg['use_sign']}")
    best = results[0][2]
    print(f"\n>>> MELHOR: {best}")


if __name__ == "__main__":
    main()
