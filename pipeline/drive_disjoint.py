#!/usr/bin/env python3
"""Reviewers' W2/C2: drive-disjoint evaluation of every system whose parameters
are fitted by replay.

All 520 benchmark videos (312 calibration + 208 validation) are pooled and the
25 drives split into 5 folds balanced by video count.  For each fold, every
fitted system is refitted on the videos of the other 20 drives only and
replayed on the fold's videos, so no test video shares a drive with any video
used to fit it.  Pooling the five folds gives one out-of-drive prediction per
video.

Systems (all on the recorded latency-honest detector stream, joint_dump_*):
  det_sm_hand   detector state machine, hand-tuned constants (no fitting)
  det_sm_cal    the same machine, constants searched on the training drives
                with the union of the paper's three grids (>= its budget)
  det_counted   counted filter on the detector stream only
  joint         counted filter on detector + world-model evidence
  (detclip_*    added by detclip_recal.py once the CLIP dump exists)

Outputs per-frame caches ~/eval_cache/dd_<system>/ for cluster_ci.py.

    python3 drive_disjoint.py
"""
import glob
import itertools
import json
import os
import tempfile

import numpy as np

import ablate_joint as A
import calibrate_detector_sm as D
import joint_filter as J

H = os.path.expanduser
DUMPS = ["~/eval_cache/joint_dump_cal2", "~/eval_cache/joint_dump_val"]
WORLD_OFF = dict(gate=True, sign=True, corr=True)


def drive_of(name):
    city, h = os.path.basename(name).split("_")[:2]
    return f"{city}_{h}"


def drive_folds(names, k=5):
    count = {}
    for n in names:
        count[drive_of(n)] = count.get(drive_of(n), 0) + 1
    folds, load = [[] for _ in range(k)], np.zeros(k)
    for d, c in sorted(count.items(), key=lambda t: (-t[1], t[0])):
        i = int(np.argmin(load))
        folds[i].append(d)
        load[i] += c
    return folds


def linkdir(paths):
    d = tempfile.mkdtemp(prefix="dd_")
    for p in paths:
        os.symlink(p, os.path.join(d, os.path.basename(p)))
    return d


def save(name, rec):
    d = H(f"~/eval_cache/dd_{name}")
    os.makedirs(d, exist_ok=True)
    for k, (p, g) in rec.items():
        np.savez_compressed(os.path.join(d, k), predicted=np.array(p), gt=np.array(g))


def search_sm(train):
    """Best calibration macro-F1 over the union of the paper's three grids."""
    best, best_f = None, -1.0
    for grid in (D.GRID, D.GRID2, D.GRID3):
        keys = list(grid)
        combos = [c for c in itertools.product(*grid.values()) if c[2] < c[1]]
        cfgs = {k: np.array([c[i] for c in combos], float) for i, k in enumerate(keys)}
        f = D.macro_f1(D.run(train, cfgs))
        i = int(np.argmax(f))
        if f[i] > best_f:
            best_f, best = float(f[i]), {k: float(cfgs[k][i]) for k in keys}
    return best, best_f


def replay_filter(paths, params, mask):
    rec = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        rows = z["obs"]
        if rows.size == 0:
            continue
        f = J.JointFilter(params)
        n = int(z["n_frames"])
        pred = np.array(["outside"] * n, dtype=object)
        for r in rows:
            pred[max(0, int(r[0])):] = f.step(A.obs_of(r, mask), float(r[2]))
        gt = [str(x) for x in z["gt"]]
        m = min(n, len(gt))
        rec[os.path.basename(p)] = (list(pred[:m]), gt[:m])
    return rec


def main():
    paths = sorted(p for d in DUMPS for p in glob.glob(H(f"{d}/*.npz")))
    paths = [p for p in paths if np.load(p, allow_pickle=True)["obs"].size > 0]
    folds = drive_folds(paths)
    vids = {}
    for d in DUMPS:
        for v in D.load(d):
            vids[v["name"]] = v
    print(f"{len(paths)} videos, {len({drive_of(p) for p in paths})} drives")
    out = {s: {} for s in ("det_sm_hand", "det_sm_cal", "det_counted", "joint")}
    log = []
    hand = {k: np.array([v], float) for k, v in D.HAND.items()}
    for k, fd in enumerate(folds):
        test = [p for p in paths if drive_of(p) in fd]
        train = [p for p in paths if drive_of(p) not in fd]
        print(f"fold {k}: {len(fd)} drives, {len(test)} test / {len(train)} train videos", flush=True)
        tv = [vids[os.path.basename(p)] for p in test]
        D.run(tv, hand, record=out["det_sm_hand"])
        cfg, fcal = search_sm([vids[os.path.basename(p)] for p in train])
        D.run(tv, {kk: np.array([vv], float) for kk, vv in cfg.items()}, record=out["det_sm_cal"])
        tdir = linkdir(train)
        out["det_counted"].update(replay_filter(test, A.fit(tdir, WORLD_OFF), WORLD_OFF))
        out["joint"].update(replay_filter(test, A.fit(tdir, {}), {}))
        log.append(dict(fold=k, drives=fd, n_test=len(test), sm_cfg=cfg, sm_train_f1=fcal))
        print(f"   det SM config {cfg} (train F1 {fcal:.3f})", flush=True)
    for s, rec in out.items():
        save(s, rec)
    json.dump(dict(folds=log), open(H("~/eval_cache/dd_folds.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
