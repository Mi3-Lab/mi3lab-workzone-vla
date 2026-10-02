#!/usr/bin/env python3
"""Detector and world model as an ASYNCHRONOUS multi-rate stack.

Three structural defects killed the two earlier combinations, and this design
fixes each one rather than retuning around it.

1. THE SLOW MODEL MUST NOT BLOCK THE FAST ONE.  Both earlier attempts ran the
   VLM inside the detector's cycle, so the combined cycle was 3x slower than
   the detector alone.  Decision-level fusion over multi-rate asynchronous
   pipelines is the standard answer in deployed perception stacks: every
   component runs at its own rate and the fusion consumes the most recent
   estimate available, compensating for its age.  Here the detector keeps its
   native 43 ms cycle and the world model runs beside it, its answer applied
   from the moment it would actually have arrived.

2. TEMPORAL CONSTANTS MUST BE IN SECONDS, NOT CYCLES.  The ported detector
   counts cycles (MAX_APPROACH_DUR=150, min_out_frames), which at 43 ms means
   6.5 s.  Slowing the cycle silently stretches every timeout: measured event
   duration inflated 2.76x when the cycle inflated 2.95x.  We convert the
   counters to seconds so that the same wall-clock behaviour survives a change
   of rate -- the pipeline-level analogue of this paper's threshold-portability
   result.

3. VERIFIABLE EVIDENCE MUST NOT BE AVERAGED AWAY.  The detector enters at
   f_ema >= 0.5, so a semantic channel fused at weight w contributes at most
   0.95w and cannot open a detection alone for any w that keeps performance.
   Weighted averaging structurally denies the world model the one thing it was
   added for.  A transcribed work-zone sign is checkable evidence, so it takes
   the same bypass our own cascade gives it: it opens the state directly,
   while an unverified positive gate only biases the score.

Run with the workzone venv (numpy<2 for ultralytics).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

import evaluate_yolo_baseline as B
from cascade_state_machine import has_corroboration, parse_gate
from live_cascade_demo import GATE_PROMPT, SIGN_PROMPT, PersistentEngine
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                           TransitionToleranceAccumulator, print_paper_report)

SHM = "/dev/shm/async_fusion.jpg"
STATES = ["outside", "approaching", "inside", "exiting"]
C3E_SIZE = (736, 416)

# the ported detector's cycle counters, restated in seconds at its native rate
DET_CYCLE_S = 0.043
MAX_APPROACH_S = 150 * DET_CYCLE_S      # 6.45 s
MIN_OUT_S = B.F_CONF["min_out_frames"] * DET_CYCLE_S

SEM_HALFLIFE_S = 2.0     # a semantic answer loses half its weight every 2 s
SEM_BIAS = 0.20          # how far an unverified positive gate may move the score


def update_state_seconds(prev, score, dur_s, out_s, dt):
    """B.update_state with its cycle counters expressed in seconds."""
    f = B.F_CONF
    if prev == "OUT":
        if score >= f["approach_th"]:
            return "APPROACHING", 0.0, 0.0
        return "OUT", 0.0, out_s + dt
    if prev == "APPROACHING":
        if dur_s > MAX_APPROACH_S:
            return "OUT", 0.0, 0.0
        if score >= f["enter_th"]:
            return "INSIDE", 0.0, 0.0
        if score <= (f["approach_th"] - 0.05):
            if out_s >= 2 * MIN_OUT_S:
                return "OUT", 0.0, 0.0
            return "APPROACHING", dur_s + dt, out_s + dt
        return "APPROACHING", dur_s + dt, 0.0
    if prev == "INSIDE":
        if score < f["exit_th"]:
            return "EXITING", 0.0, 0.0
        return "INSIDE", dur_s + dt, 0.0
    if prev == "EXITING":
        if score >= f["enter_th"]:
            return "INSIDE", dur_s, 0.0
        if out_s >= MIN_OUT_S:
            return "OUT", 0.0, 0.0
        return "EXITING", dur_s, out_s + dt
    return prev, dur_s, out_s



def update_state_authority(prev, det_score, sem_fresh, sem_verified, dur_s, out_s, dt):
    """State machine in which each transition is decided by whichever component
    the calibration data shows is right about it.

    Measured on the validation split, per ground-truth state, the fraction of
    frames where exactly one component is correct:

        approaching   C3E 41.9%  vs detector  7.8%   -> C3E owns entry
        inside        C3E  9.0%  vs detector 15.8%   -> detector owns arrival
        outside       C3E  9.4%  vs detector 11.5%   -> detector owns exit

    and the errors are mirrored: the detector skips approaching by declaring
    inside (12.4k frames), while C3E lingers in approaching when already inside
    (10.2k) or still outside (8.0k).  One component knows something is coming;
    the other knows you have arrived.  A single fused score cannot express that,
    which is why global fusion recovered almost nothing.  So we give each
    transition to its owner instead of averaging them.
    """
    f = B.F_CONF
    if prev == "OUT":
        # entry: either source may open, and C3E is the better opener
        if sem_fresh or det_score >= f["approach_th"]:
            return "APPROACHING", 0.0, 0.0
        return "OUT", 0.0, out_s + dt
    if prev == "APPROACHING":
        if dur_s > MAX_APPROACH_S:
            return "OUT", 0.0, 0.0
        # arrival: the detector decides; C3E is not allowed to declare inside
        if det_score >= f["enter_th"]:
            return "INSIDE", 0.0, 0.0
        # leaving without arriving needs BOTH quiet, so C3E can hold an
        # anticipatory state the detector cannot yet see
        if det_score <= (f["approach_th"] - 0.05) and not sem_fresh:
            if out_s >= 2 * MIN_OUT_S:
                return "OUT", 0.0, 0.0
            return "APPROACHING", dur_s + dt, out_s + dt
        return "APPROACHING", dur_s + dt, 0.0
    if prev == "INSIDE":
        if det_score < f["exit_th"]:
            return "EXITING", 0.0, 0.0
        return "INSIDE", dur_s + dt, 0.0
    if prev == "EXITING":
        if det_score >= f["enter_th"]:
            return "INSIDE", dur_s, 0.0
        if out_s >= MIN_OUT_S:
            return "OUT", 0.0, 0.0
        return "EXITING", dur_s, out_s + dt
    return prev, dur_s, out_s


def run(model, engine, video_path, seconds_mode=True, use_sem=True, trace=None,
        authority=False):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predicted = np.full(n_frames, "outside", dtype=object)

    state, f_ema, dur_s, out_s = "OUT", None, 0.0, 0.0
    dur_c, out_c = 0, 0                      # cycle counters, for seconds_mode=False
    frame_idx, sim_clock = -1, 0.0
    # asynchronous world-model channel
    vlm_free_at, vlm_pending = 0.0, None     # (result, ready_at)
    sem_value, sem_time, sem_verified = 0.5, -1e9, False
    det_cycles, vlm_calls, bypasses = [], 0, 0

    while True:
        target = int(sim_clock * fps)
        if target >= n_frames:
            break
        while frame_idx < target:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.retrieve()
        if not ok:
            break

        # ---- the world model, running beside the detector on its own clock ----
        if use_sem and vlm_pending is not None and sim_clock >= vlm_pending[1]:
            sem_value, sem_verified = vlm_pending[0]
            sem_time = vlm_pending[1]
            vlm_pending = None
        if use_sem and vlm_pending is None and sim_clock >= vlm_free_at:
            t_v = time.monotonic()
            cv2.imwrite(SHM, cv2.resize(frame, C3E_SIZE, interpolation=cv2.INTER_AREA))
            g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
            g = parse_gate(g_txt or "")
            verified = False
            if g:
                s_txt, _ = engine.infer(SHM, SIGN_PROMPT, 32)
                verified = has_corroboration("", (s_txt or "").strip())
            vlm_calls += 1
            lat = time.monotonic() - t_v
            # the answer describes THIS frame but only arrives lat seconds later
            vlm_pending = ((1.0 if g else 0.0, verified), sim_clock + lat)
            vlm_free_at = sim_clock + lat

        # ---- the detector, never blocked, at its native rate ----
        t0 = time.monotonic()
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        counts = {}
        for cid in res.boxes.cls.cpu().tolist():
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        y_s, total = B.yolo_frame_score(counts)
        evidence = B.clamp01(0.5 * B.clamp01(total / 8.0) + 0.5 * B.clamp01(y_s))
        alpha = B.adaptive_alpha(evidence, B.F_CONF["ema_alpha"] * 0.4,
                                 B.F_CONF["ema_alpha"] * 1.2)

        score = y_s
        bypass = False
        if use_sem and sem_time > -1e8:
            age = sim_clock - sem_time
            decay = 0.5 ** (age / SEM_HALFLIFE_S)
            if sem_verified and decay > 0.5 and state == "OUT":
                bypass = True            # checkable evidence opens the state
            score = B.clamp01(score + SEM_BIAS * decay * (2.0 * sem_value - 1.0))
        f_ema = B.ema(f_ema, B.clamp01(score), alpha)

        dt_det = time.monotonic() - t0
        det_cycles.append(dt_det * 1000.0)
        if authority:
            fresh = (sem_time > -1e8 and sem_value > 0.5
                     and 0.5 ** ((sim_clock - sem_time) / SEM_HALFLIFE_S) > 0.5)
            if fresh and state == "OUT":
                bypasses += 1
            state, dur_s, out_s = update_state_authority(
                state, f_ema, fresh, sem_verified, dur_s, out_s, dt_det)
        elif bypass and state == "OUT":
            state, dur_s, out_s, dur_c, out_c = "APPROACHING", 0.0, 0.0, 0, 0
            bypasses += 1
        elif seconds_mode:
            state, dur_s, out_s = update_state_seconds(state, f_ema, dur_s, out_s, dt_det)
        else:
            state, dur_c, out_c = B.update_state(state, f_ema, dur_c, out_c)

        if trace is not None:
            trace.append((frame_idx, sim_clock, dt_det, y_s, alpha,
                          sem_value if sem_time > -1e8 else -1.0,
                          float(sem_verified), sem_time))
        predicted[max(0, frame_idx):] = B.STATE_MAP[state]
        sim_clock += dt_det          # only the detector advances the clock

    cap.release()
    return predicted, det_cycles, vlm_calls, bypasses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    ap.add_argument("--annotations",
                    default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    ap.add_argument("--weights",
                    default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    ap.add_argument("--split-file", default="eval_split_full.json")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--engine-dir", default=os.path.expanduser("~/cosmos3edge-engines"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--authority", action="store_true",
                    help="each transition decided by the component the data shows owns it")
    ap.add_argument("--trace", default=None,
                    help="record per-cycle state so SEM_BIAS/half-life can be swept offline")
    ap.add_argument("--cycles-mode", action="store_true",
                    help="keep the detector's cycle counters (ablation of fix 2)")
    ap.add_argument("--no-semantic", action="store_true",
                    help="detector only, seconds mode (ablation of the world model)")
    args = ap.parse_args()

    videos = json.load(open(args.split_file))[args.split][:args.limit]
    ann = json.load(open(args.annotations))
    model = YOLO(args.weights)
    engine = None
    if not args.no_semantic:
        engine = PersistentEngine(
            os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
            os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
            args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)
    if args.cache_dir:
        os.makedirs(os.path.expanduser(args.cache_dir), exist_ok=True)

    ev, tm = EventLevelAccumulator(STATES), TimingOffsetAccumulator(STATES)
    tr = TransitionToleranceAccumulator()
    conf, cyc, calls_tot, byp_tot = {}, [], 0, 0
    for i, v in enumerate(videos, 1):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p) or v not in ann:
            continue
        cap = cv2.VideoCapture(p); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        gt = [s.value if hasattr(s, "value") else str(s)
              for s in B.load_ground_truth(ann, v, nf)]
        t0 = time.monotonic()
        trace = [] if args.trace else None
        pred, cycles, calls, byp = run(model, engine, p,
                                       seconds_mode=not args.cycles_mode,
                                       use_sem=not args.no_semantic, trace=trace,
                                       authority=args.authority)
        n = min(len(pred), len(gt))
        ps, gs = [str(pred[j]) for j in range(n)], [str(gt[j]) for j in range(n)]
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs); tm.add_video(ps, gs); tr.add_video(ps, gs)
        cyc.extend(cycles); calls_tot += calls; byp_tot += byp
        if args.cache_dir:
            np.savez_compressed(os.path.join(os.path.expanduser(args.cache_dir), f"{v}.npz"),
                                predicted=np.array(ps), gt=np.array(gs))
        if args.trace:
            os.makedirs(os.path.expanduser(args.trace), exist_ok=True)
            np.savez_compressed(os.path.join(os.path.expanduser(args.trace), f"{v}.npz"),
                                trace=np.array(trace, dtype=np.float64),
                                gt=np.array(gs), n_frames=nf)
        print(f"[{i}/{len(videos)}] {v}  det_cycles={len(cycles)} vlm={calls} "
              f"bypass={byp} p50={np.percentile(cycles,50):.0f}ms "
              f"({time.monotonic()-t0:.1f}s)", flush=True)

    if engine:
        engine.close()
    c = np.array(cyc)
    print(f"\n=== async stack (seconds_mode={not args.cycles_mode}, "
          f"semantic={not args.no_semantic}) ===")
    print(f"  detector cycle p50 {np.percentile(c,50):.0f} ms | p95 {np.percentile(c,95):.0f} ms "
          f"| {1000/np.mean(c):.1f} Hz")
    print(f"  world-model queries {calls_tot}, verified-evidence bypasses {byp_tot}")
    print_paper_report(conf, ev, tm, tr, STATES)


if __name__ == "__main__":
    main()
