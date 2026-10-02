#!/usr/bin/env python3
"""Detector and world model, each keeping its own calibrated state machine,
combined by a table learned on the calibration split.  Run end-to-end.

This is the real-time counterpart of the feasibility bound in stack_combiner.py,
which reached macro-F1 0.536 by combining two independent executions.  That
number is not a system result, for the reason this project learned the hard way:
under latency-honest replay two separate runs observe different frames, and a
single chained execution observes a third set.  This script earns the number, or
fails to.

Design, in the order the failures taught it:

  The detector runs at its native rate and is never blocked.  Five earlier
  attempts put the world model inside the detector's cycle and tripled it, which
  starved the estimator of frames exactly inside work zones.

  Each component keeps its OWN state machine, at the rate that machine was
  calibrated for -- the detector's EMA and hysteresis at ~24 ms, the world
  model's voting window once per poll.  Stepping either at the wrong rate
  silently rescales every temporal constant, the pipeline-level version of the
  threshold-portability result this paper reports.

  The two state estimates are combined by a 16-cell lookup table fitted by
  counting on the calibration split.  Hand-written rules failed because the
  complementarity is asymmetric in a way that is easy to measure and hard to
  guess: the world model wins a disagreement when the detector is already active
  (detector INSIDE, model APPROACHING -> APPROACHING, 49%), and loses it when
  the detector is quiet (detector OUTSIDE, model APPROACHING -> OUTSIDE, 73%).
  A rule giving the world model entry authority -- which the per-state analysis
  seemed to justify -- scored worst of everything we tried.

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
from cascade_state_machine import CascadeStateMachine, WZState, has_corroboration, parse_gate
from live_cascade_demo import (DESC_PROMPT, EGO_PROMPT, GATE_PROMPT, SIGN_PROMPT,
                                PersistentEngine)
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                           TransitionToleranceAccumulator, print_paper_report)

SHM = "/dev/shm/stacked.jpg"
STATES = ["outside", "approaching", "inside", "exiting"]
C3E_SIZE = (736, 416)
C3E_CSM = dict(N=3, k_enter=3, k_exit=2, k_fading=1, use_ego=False)
POLL_S = 1.0          # world-model poll period: the stride its window was calibrated at


def load_table(path):
    with open(os.path.expanduser(path)) as fh:
        d = json.load(fh)
    return {tuple(k.split("|")): v for k, v in d["table"].items()}, d["n_calibration_videos"]


def run(model, engine, video_path, table, poll_s=POLL_S, combine=True):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predicted = np.full(n_frames, "outside", dtype=object)

    # the detector's own machine, stepped every cycle at its native rate
    det_state, y_ema, f_ema, dur, out_f = "OUT", None, None, 0, 999
    # the world model's own cascade, stepped once per poll
    csm = CascadeStateMachine(**C3E_CSM)
    c3e_state = WZState.OUTSIDE

    frame_idx, sim_clock = -1, 0.0
    vlm_pending, vlm_free_at, next_poll = None, 0.0, 0.0
    det_cycles, polls, misses = [], 0, 0

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

        # ---- world model, on its own clock, never blocking the detector ----
        if vlm_pending is not None and sim_clock >= vlm_pending[1]:
            gate, corr, fast = vlm_pending[0]
            c3e_state = csm.update_evidence(gate, corr, None, fast_entry=fast)
            vlm_pending = None
        if vlm_pending is None and sim_clock >= max(next_poll, vlm_free_at):
            t_v = time.monotonic()
            cv2.imwrite(SHM, cv2.resize(frame, C3E_SIZE, interpolation=cv2.INTER_AREA))
            # The world-model arm must be the SAME cascade that scores 0.515
            # standalone, not a reduced one.  Three earlier designs quietly
            # dropped the SIGN channel and with it the anticipatory entry that
            # transcription provides, which is most of what the world model
            # contributes; each time the arm scored far below its own baseline.
            g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
            gate = bool(parse_gate(g_txt or ""))
            sign_txt, desc_txt = "", ""
            fast = False
            if c3e_state == WZState.OUTSIDE:
                sign_txt, _ = engine.infer(SHM, SIGN_PROMPT, 15)
                sign_txt = sign_txt or ""
                sign_detected = has_corroboration("", sign_txt)
                candidate = gate or sign_detected
                if candidate:
                    desc_txt, _ = engine.infer(SHM, DESC_PROMPT, 40)
                    desc_txt = desc_txt or ""
                    fast = sign_detected or (gate and has_corroboration(desc_txt, ""))
                gate = candidate
            else:
                desc_txt, _ = engine.infer(SHM, DESC_PROMPT, 40)
                desc_txt = desc_txt or ""
            corr = has_corroboration(desc_txt, sign_txt)
            lat = time.monotonic() - t_v
            vlm_pending = ((gate, corr, fast), sim_clock + lat)
            vlm_free_at = sim_clock + lat
            next_poll = sim_clock + poll_s
            polls += 1

        # ---- detector, every cycle ----
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
        y_ema = B.ema(y_ema, y_s, alpha)
        f_ema = B.ema(f_ema, B.clamp01(y_s), alpha)
        det_state, dur, out_f = B.update_state(det_state, f_ema, dur, out_f)
        dt = time.monotonic() - t0
        det_cycles.append(dt * 1000.0)

        d_lbl, c_lbl = B.STATE_MAP[det_state], c3e_state.value
        if combine:
            out = table.get((d_lbl, c_lbl))
            if out is None:
                out, misses = c_lbl, misses + 1
        else:
            out = c_lbl
        predicted[max(0, frame_idx):] = out
        sim_clock += dt

    cap.release()
    return predicted, det_cycles, polls, misses


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
    ap.add_argument("--table", default=os.path.expanduser("~/eval_cache/stack_table.json"))
    ap.add_argument("--poll-s", type=float, default=POLL_S)
    ap.add_argument("--no-combine", action="store_true",
                    help="ablation: report the world model's own state, ignore the table")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    table, n_cal = load_table(args.table)
    print(f"combiner: {len(table)} cells fitted on {n_cal} calibration videos")

    videos = json.load(open(args.split_file))[args.split][:args.limit]
    ann = json.load(open(args.annotations))
    model = YOLO(args.weights)
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)
    if args.cache_dir:
        os.makedirs(os.path.expanduser(args.cache_dir), exist_ok=True)

    ev, tm = EventLevelAccumulator(STATES), TimingOffsetAccumulator(STATES)
    tr = TransitionToleranceAccumulator()
    conf, cyc, polls_tot, miss_tot = {}, [], 0, 0
    for i, v in enumerate(videos, 1):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p) or v not in ann:
            continue
        cap = cv2.VideoCapture(p); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        gt = [s.value if hasattr(s, "value") else str(s)
              for s in B.load_ground_truth(ann, v, nf)]
        t0 = time.monotonic()
        pred, cycles, polls, misses = run(model, engine, p, table, args.poll_s,
                                          combine=not args.no_combine)
        n = min(len(pred), len(gt))
        ps, gs = [str(pred[j]) for j in range(n)], [str(gt[j]) for j in range(n)]
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs); tm.add_video(ps, gs); tr.add_video(ps, gs)
        cyc.extend(cycles); polls_tot += polls; miss_tot += misses
        if args.cache_dir:
            np.savez_compressed(os.path.join(os.path.expanduser(args.cache_dir), f"{v}.npz"),
                                predicted=np.array(ps), gt=np.array(gs))
        print(f"[{i}/{len(videos)}] {v}  det={len(cycles)} polls={polls} "
              f"({time.monotonic()-t0:.1f}s)", flush=True)

    engine.close()
    c = np.array(cyc)
    print(f"\n=== stacked, end-to-end (combine={not args.no_combine}) ===")
    print(f"  detector cycle p50 {np.percentile(c,50):.0f} ms | p95 {np.percentile(c,95):.0f} ms "
          f"| {1000/np.mean(c):.1f} Hz")
    print(f"  world-model polls {polls_tot}; table misses {miss_tot}")
    print_paper_report(conf, ev, tm, tr, STATES)


if __name__ == "__main__":
    main()
