#!/usr/bin/env python3
"""Reviewers' W1/C4: recalibrate the COMPLETE detector+CLIP pipeline, not only
its detector score, using the evidence recorded by dump_detclip.py.

Replay (vectorized over configurations) of evaluate_yolo_baseline's loop:
CLIP-verified counts -> y_s, adaptive EMA, global-CLIP fusion gated on the EMA
(every third cycle), orange-context boost, fused EMA, 4-state machine.  The
CLIP and orange weights stay at the app's values; the search covers the same
temporal constants, with the same three grids, as the detector-only search
(calibrate_detector_sm.py), so both baselines get the same budget.

Also: the fused per-cycle score through the counted filter (detector+CLIP in
the HMM of joint_filter, no world model).

Outputs (per-frame caches):
  detclip_hand_val / detclip_cal_val / detclip_counted_val   standard split
  dd_detclip_hand / dd_detclip_cal / dd_detclip_counted      drive-disjoint folds
"""
import glob
import itertools
import json
import os

import numpy as np

import calibrate_detector_sm as D
import compare_all_systems as C
import joint_filter as J

H = os.path.expanduser
ST = D.ST
OUT, APP, INS, EXI = range(4)
CLIP_W, CLIP_TRIG, ORANGE_W, CTX_BELOW = 0.35, 0.2, 0.25, 0.5


def load(dump):
    vids = []
    for p in sorted(glob.glob(H(f"{dump}/*.npz"))):
        z = np.load(p, allow_pickle=True)
        o = z["obs"]
        if o.size == 0:
            continue
        gt = np.array([ST.index(str(x)) if str(x) in ST else 0 for x in z["gt"]])
        n = len(gt)
        fi = np.clip(o[:, 0].astype(int), 0, n - 1)
        ends = np.r_[fi[1:], n]
        span = np.zeros((len(fi), 4))
        for k, (a, b) in enumerate(zip(fi, ends)):
            if b > a:
                span[k] = np.bincount(gt[a:b], minlength=4)
        span[0] += np.bincount(gt[:fi[0]], minlength=4)
        vids.append(dict(name=os.path.basename(p), fi=fi, n=n, span=span, gt=gt, dt=o[:, 1],
                         y=o[:, 4], tot=o[:, 5], clip=o[:, 6], orange=o[:, 7],
                         cyc=o[:, 8].astype(int)))
    return vids


def run(vids, cfgs, record=None, fused_out=None):
    """Same state logic as calibrate_detector_sm.run, on the fused CLIP score."""
    C_ = len(cfgs["alpha"])
    conf = np.zeros((C_, 4, 4))
    lo, hi = cfgs["alpha"] * 0.4, cfgs["alpha"] * 1.2
    for v in vids:
        state = np.zeros(C_, int); dur = np.zeros(C_); outf = np.full(C_, 999.0)
        yema = None; fema = None; last_clip = np.full(C_, 0.5)
        pred = np.zeros(v["n"], int) if record is not None else None
        fseq = []
        for k in range(len(v["y"])):
            y = v["y"][k]
            ev = min(max(0.5 * min(max(v["tot"][k] / 8.0, 0), 1) + 0.5 * min(max(y, 0), 1), 0), 1)
            a = lo + (hi - lo) * ev
            yema = np.full(C_, y) if yema is None else a * y + (1 - a) * yema
            fused = np.full(C_, y)
            g = yema >= CLIP_TRIG
            if v["cyc"][k] % 3 == 0:
                last_clip = np.where(g, v["clip"][k], last_clip)
            fused = np.where(g, (1 - CLIP_W) * fused + CLIP_W * last_clip, fused)
            c = yema < CTX_BELOW
            fused = np.where(c, (1 - ORANGE_W) * fused + ORANGE_W * v["orange"][k], fused)
            fused = np.clip(fused, 0, 1)
            fseq.append(fused[0])
            fema = fused.copy() if fema is None else a * fused + (1 - a) * fema
            s = fema
            ns, nd, no = state.copy(), dur.copy(), outf.copy()
            m = state == OUT
            go = m & (s >= cfgs["approach_th"]); ns[go] = APP; nd[go] = 0; no[go] = 0
            st = m & ~go; no[st] = outf[st] + 1; nd[st] = 0
            m = state == APP
            t1 = m & (dur > cfgs["max_app"]); ns[t1] = OUT; nd[t1] = 0; no[t1] = 0
            r = m & ~t1
            t2 = r & (s >= cfgs["enter_th"]); ns[t2] = INS; nd[t2] = 0; no[t2] = 0
            r2 = r & ~t2
            low = r2 & (s <= cfgs["approach_th"] - 0.05)
            t3 = low & (outf >= 2 * cfgs["min_out"]); ns[t3] = OUT; nd[t3] = 0; no[t3] = 0
            t4 = low & ~t3; nd[t4] = dur[t4] + 1; no[t4] = outf[t4] + 1
            t5 = r2 & ~low; nd[t5] = dur[t5] + 1; no[t5] = 0
            m = state == INS
            t6 = m & (s < cfgs["exit_th"]); ns[t6] = EXI; nd[t6] = 0; no[t6] = 0
            t7 = m & ~t6; nd[t7] = dur[t7] + 1; no[t7] = 0
            m = state == EXI
            t8 = m & (s >= cfgs["enter_th"]); ns[t8] = INS; no[t8] = 0
            r3 = m & ~t8
            t9 = r3 & (outf >= cfgs["min_out"]); ns[t9] = OUT; nd[t9] = 0; no[t9] = 0
            t10 = r3 & ~t9; no[t10] = outf[t10] + 1
            state, dur, outf = ns, nd, no
            conf[np.arange(C_), :, state] += v["span"][k]
            if pred is not None:
                pred[v["fi"][k]:] = state[0]
        if record is not None:
            record[v["name"]] = ([ST[x] for x in pred], [ST[x] for x in v["gt"]])
        if fused_out is not None:
            fused_out[v["name"]] = np.array(fseq)
    return conf


def search(train):
    best, best_f = None, -1.0
    for grid in (D.GRID, D.GRID2, D.GRID3):
        keys = list(grid)
        combos = [c for c in itertools.product(*grid.values()) if c[2] < c[1]]
        cfgs = {k: np.array([c[i] for c in combos], float) for i, k in enumerate(keys)}
        f = D.macro_f1(run(train, cfgs))
        i = int(np.argmax(f))
        if f[i] > best_f:
            best_f, best = float(f[i]), {k: float(cfgs[k][i]) for k in keys}
    return best, best_f


def one(cfg):
    return {k: np.array([v], float) for k, v in cfg.items()}


HAND = dict(D.HAND)


def counted_fit(vids, fused):
    emit = np.ones((4, J.N_OBS)); trans = np.zeros((4, 4)); dwell = np.zeros(4); marg = np.zeros(4)
    for v in vids:
        f = fused[v["name"]]; clock = np.cumsum(np.r_[0, v["dt"][:-1]]); prev = None
        for k in range(len(f)):
            si = v["gt"][v["fi"][k]]
            emit[si, J.obs_index(J.det_bucket(f[k]), False, False, False, J.age_bucket(1e9))] += 1
            marg[si] += 1
            if prev is not None:
                dwell[prev] += max(clock[k] - clock[k - 1], 1e-6)
                if si != prev:
                    trans[prev, si] += 1
            prev = si
    emit /= emit.sum(1, keepdims=True)
    rate = np.zeros((4, 4))
    for i in range(4):
        if dwell[i] > 0:
            for j in range(4):
                if j != i:
                    rate[i, j] = trans[i, j] / dwell[i]
            rate[i, i] = -rate[i].sum()
    return {"emission": emit.tolist(), "rate": rate.tolist(), "prior": (marg / marg.sum()).tolist()}


def counted_run(vids, fused, params, rec):
    for v in vids:
        filt = J.JointFilter(params); pred = np.array(["outside"] * v["n"], dtype=object)
        for k, x in enumerate(fused[v["name"]]):
            o = J.obs_index(J.det_bucket(x), False, False, False, J.age_bucket(1e9))
            pred[v["fi"][k]:] = filt.step(o, float(v["dt"][k]))
        rec[v["name"]] = (list(pred), [ST[x] for x in v["gt"]])


def save(name, rec):
    d = H(f"~/eval_cache/{name}"); os.makedirs(d, exist_ok=True)
    for k, (p, g) in rec.items():
        np.savez_compressed(os.path.join(d, k), predicted=np.array(p), gt=np.array(g))


def report(name, rec):
    m = C.metrics(rec)
    print(f"  {name:22s} n={m['n']} acc {m['acc']*100:.1f} F1 {m['f1']:.3f} ap {m['iou_appr']:.3f} in {m['iou_in']:.3f} "
          f"ex {m['iou_exit']:.3f} inP {(m['in_p'] or 0)*100:.1f} inR {(m['in_r'] or 0)*100:.1f} FA {m['fa_per_h']:.1f}", flush=True)


def main():
    cal, val = load("~/eval_cache/detclip_dump_cal"), load("~/eval_cache/detclip_dump_val")
    print(f"{len(cal)} cal / {len(val)} val videos")
    fused = {}
    run(cal + val, one(HAND), fused_out=fused)
    # 1. standard split
    rec = {}; run(val, one(HAND), record=rec); save("detclip_hand_val", rec); report("hand (val)", rec)
    cfg, fcal = search(cal)
    print("  calibrated config", cfg, f"cal F1 {fcal:.3f}  (hand cal F1 {D.macro_f1(run(cal, one(HAND)))[0]:.3f})")
    rec = {}; run(val, one(cfg), record=rec); save("detclip_cal_val", rec); report("recalibrated (val)", rec)
    rec = {}; counted_run(val, fused, counted_fit(cal, fused), rec); save("detclip_counted_val", rec)
    report("counted (val)", rec)
    # 2. drive-disjoint folds (same folds as drive_disjoint.py)
    folds = json.load(open(H("~/eval_cache/dd_folds.json")))["folds"]
    allv = cal + val
    out = {"hand": {}, "cal": {}, "counted": {}}
    for f in folds:
        test = [v for v in allv if v["name"].split("_")[0] + "_" + v["name"].split("_")[1] in f["drives"]]
        train = [v for v in allv if v not in test]
        run(test, one(HAND), record=out["hand"])
        c, fc = search(train)
        run(test, one(c), record=out["cal"])
        counted_run(test, fused, counted_fit(train, fused), out["counted"])
        print(f"  fold {f['fold']}: {len(test)} test, config {c} (train F1 {fc:.3f})", flush=True)
    for k, r in out.items():
        save(f"dd_detclip_{k}", r); report(f"drive-disjoint {k}", r)


if __name__ == "__main__":
    main()
