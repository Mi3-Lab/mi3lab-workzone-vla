#!/usr/bin/env python3
"""Reviewers' remaining request: drive-disjoint evaluation of the VLM cascades.

Per-channel evidence (one sample per second, every channel; dump_evidence.py)
for all 520 videos lets the cascade constants be re-searched per fold by
replaying the real state machine (calibrate_cascade.replay).  For one model:

  standard   search on the 312 calibration videos, replay on the 208 validation
             videos (the paper's protocol, in replay)
  inherited  the 2B's constants (N=5, K=3/4/2, EGO on), no search
  disjoint   the same 5 drive folds as drive_disjoint.py: search on four
             folds, replay on the fifth, pooled over all 520 videos
  --cal-only the same folds restricted to the 312 calibration videos (used when
             no validation evidence was dumped): search on the calibration
             videos of four folds' drives, replay on the fifth's

The grid is the paper's (N in {3,5,7,9}, K in 1..N, EGO/SIGN switches) plus the
later extension N in {1,2,4}.  Writes per-frame caches for cluster_ci.

    python3 vlm_drive_disjoint.py --model c3e
    python3 vlm_drive_disjoint.py --model 2b
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

import calibrate_cascade as CC
import compare_all_systems as C

H = os.path.expanduser
HERE = os.path.dirname(os.path.abspath(__file__))
INHERITED = dict(N=5, K_ENTER=3, K_EXIT=4, K_FADE=2, use_sign=True, use_corr=True, use_ego=True)


def grid():
    out = []
    for N in (1, 2, 3, 4, 5, 7, 9):
        for ke, kx, kf in itertools.product(range(1, N + 1), repeat=3):
            for ego in (True, False):
                for sign in (True, False):
                    out.append(dict(N=N, K_ENTER=ke, K_EXIT=kx, K_FADE=kf,
                                    use_sign=sign, use_corr=True, use_ego=ego))
    return out


_EVS = None


def _init(evs):
    global _EVS
    _EVS = evs


def _score(cfg):
    return CC.score(_EVS, cfg)[1]


def search(evs, cfgs, procs=12):
    with Pool(procs, initializer=_init, initargs=(evs,)) as pool:
        f1 = pool.map(_score, cfgs, chunksize=64)
    i = int(np.argmax(f1))
    return cfgs[i], f1[i]


def predict(evs, names, cfg):
    rec = {}
    for ev, nm in zip(evs, names):
        pred = CC.replay(ev, **cfg)
        gt = [str(x) for x in ev["gt"]]
        n = min(len(pred), len(gt))
        rec[nm] = ([str(x) for x in pred[:n]], gt[:n])
    return rec


def save(name, rec):
    d = H(f"~/eval_cache/{name}")
    os.makedirs(d, exist_ok=True)
    for k, (p, g) in rec.items():
        np.savez_compressed(os.path.join(d, k), predicted=np.array(p), gt=np.array(g))


def load_named(evdir, videos):
    evs, names = [], []
    for v in videos:
        p = H(f"{evdir}/{v}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            evs.append({"idx": z["frame_idx"], "gate": z["gate"], "sign": z["sign"],
                        "corr": z["corr"], "ego": z["ego"], "gt": z["gt"], "n": int(z["n_frames"])})
            names.append(f"{v}.npz")
    return evs, names


def report(label, rec):
    m = C.metrics(rec)
    print(f"  {label:34s} n={m['n']:3d} F1 {m['f1']:.3f} acc {m['acc']*100:.1f} IoUin {m['iou_in']:.3f} "
          f"inP {(m['in_p'] or 0)*100:.1f} FA {m['fa_per_h']:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["c3e", "2b", "base"], required=True)
    ap.add_argument("--cal-only", action="store_true", help="drive folds within the calibration videos only")
    args = ap.parse_args()
    sp = json.load(open(os.path.join(HERE, "eval_split_full.json")))
    cal_dir, val_dir = f"~/eval_cache/evidence_{args.model}", f"~/eval_cache/evidence_{args.model}_val"
    cal, cal_n = load_named(cal_dir, sp["calibration"])
    val, val_n = load_named(val_dir, sp["validation"])
    print(f"{args.model}: {len(cal)} calibration / {len(val)} validation videos with evidence", flush=True)
    cfgs = grid()
    print(f"{len(cfgs)} configurations", flush=True)

    if not args.cal_only:
        report("inherited 2B constants, validation", predict(val, val_n, INHERITED))
        best, f = search(cal, cfgs)
        print(f"  standard split: best {best} (cal F1 {f:.3f})", flush=True)
        rec = predict(val, val_n, best)
        report("searched on calibration, validation", rec)
        save(f"vlmrep_{args.model}_std_val", rec)

    folds = json.load(open(H("~/eval_cache/dd_folds.json")))["folds"]
    allv, alln = (cal, cal_n) if args.cal_only else (cal + val, cal_n + val_n)
    tag = "_cal" if args.cal_only else ""
    drive = lambda nm: "_".join(nm.split("_")[:2])
    dd, inh = {}, {}
    for fd in folds:
        test = [i for i, nm in enumerate(alln) if drive(nm) in fd["drives"]]
        train = [i for i in range(len(alln)) if i not in set(test)]
        b, fb = search([allv[i] for i in train], cfgs)
        dd.update(predict([allv[i] for i in test], [alln[i] for i in test], b))
        inh.update(predict([allv[i] for i in test], [alln[i] for i in test], INHERITED))
        print(f"  fold {fd['fold']}: {len(test)} test, best {b} (train F1 {fb:.3f})", flush=True)
    report(f"inherited 2B constants, all {len(alln)}", inh)
    report(f"drive-disjoint search, all {len(alln)}", dd)
    save(f"dd_vlm_{args.model}{tag}", dd)
    save(f"dd_vlm_{args.model}{tag}_inherited", inh)


if __name__ == "__main__":
    main()
