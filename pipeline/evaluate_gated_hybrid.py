#!/usr/bin/env python3
"""The detector-gated world model, run as an actual system on the device.

Why this file exists: the gated configuration reported in the paper was, until
now, a post-hoc combination of two independently cached prediction arrays
(fuse_systems.s_gated).  That is fine as a feasibility estimate but it is not a
system: under latency-honest replay the detector (43 ms/cycle) and the C3E
cascade (~160-650 ms/cycle) observe *different* frames, and a real gated
pipeline observes a third set, because C3E is invoked only while the detector
gate is open and the combined cycle time drives the frame dropping.

This runs the real thing:
  every cycle  -> YOLO detector decides whether a zone is active
  gate open    -> C3E channels decide WHICH active state, through the same
                  CascadeStateMachine, with the thresholds calibrated for C3E
  gate closed  -> OUTSIDE, and the C3E state machine is reset

so the reported accuracy and the reported cost come from one execution.

Run with the workzone venv (numpy<2 for ultralytics):
  ~/workzone/venv/bin/python evaluate_gated_hybrid.py --split validation
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

import evaluate_yolo_baseline as B
from cascade_state_machine import (CascadeStateMachine, WZState, has_corroboration,
                                   parse_gate)
from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, PersistentEngine
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                           TransitionToleranceAccumulator, print_paper_report)

SHM = "/dev/shm/gated_hybrid.jpg"
STATES = ["outside", "approaching", "inside", "exiting"]
# thresholds calibrated for C3E on the calibration split (see the paper)
C3E_CSM = dict(N=3, k_enter=3, k_exit=2, k_fading=1, use_ego=False)
C3E_SIZE = (736, 416)


def run(model, engine, video_path, desc_len=40, cheap=False):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    predicted = np.full(n_frames, "outside", dtype=object)
    csm = CascadeStateMachine(**C3E_CSM)
    det_state, y_ema, f_ema, dur, out_f = "OUT", None, None, 0, 999
    frame_idx, sim_clock, cycle_idx = -1, 0.0, 0
    cycles, active_cycles, c3e_calls = [], 0, 0

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

        t0 = time.monotonic()

        # ---- stage 1: the detector decides WHETHER a zone is active ----
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
        gate_open = B.STATE_MAP[det_state] != "outside"

        # ---- stage 2: only when the gate is open, C3E decides WHICH state ----
        if gate_open:
            active_cycles += 1
            cv2.imwrite(SHM, cv2.resize(frame, C3E_SIZE, interpolation=cv2.INTER_AREA))
            g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
            c3e_calls += 1
            g = parse_gate(g_txt or "")
            if cheap:
                # The detector already supplies object evidence, so paying the
                # VLM again for DESC (488 ms, ~70% of the active cycle) buys a
                # second opinion at the cost of observing 3x fewer frames.
                # Corroboration therefore comes from the gate that opened.
                corr = True
            else:
                d_txt, _ = engine.infer(SHM, DESC_PROMPT, desc_len)
                c3e_calls += 1
                corr = has_corroboration(d_txt or "", "")
            state = csm.update_evidence(bool(g), corr, None,
                                        fast_entry=bool(g) and corr)
            mapped = state.value if state != WZState.OUTSIDE else "approaching"
        else:
            if not cheap:
                csm = CascadeStateMachine(**C3E_CSM)  # v1 reset on gate close
            mapped = "outside"

        dt = time.monotonic() - t0
        cycles.append(dt * 1000.0)
        cycle_idx += 1
        predicted[max(0, frame_idx):] = mapped
        sim_clock += dt

    cap.release()
    duty = active_cycles / cycle_idx if cycle_idx else 0.0
    return predicted, cycles, duty, c3e_calls, cycle_idx


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
    ap.add_argument("--cheap", action="store_true",
                    help="gated cycle without DESC and with persistent state")
    args = ap.parse_args()

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
    conf, all_cycles, duties = {}, [], []
    for i, v in enumerate(videos, 1):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p) or v not in ann:
            continue
        cap = cv2.VideoCapture(p)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        gt = [s.value if hasattr(s, "value") else str(s)
              for s in B.load_ground_truth(ann, v, nf)]
        t0 = time.monotonic()
        pred, cycles, duty, calls, ncyc = run(model, engine, p, cheap=args.cheap)
        n = min(len(pred), len(gt))
        ps, gs = [str(pred[j]) for j in range(n)], [str(gt[j]) for j in range(n)]
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs); tm.add_video(ps, gs); tr.add_video(ps, gs)
        all_cycles.extend(cycles); duties.append(duty)
        if args.cache_dir:
            np.savez_compressed(
                os.path.join(os.path.expanduser(args.cache_dir), f"{v}.npz"),
                predicted=np.array(ps), gt=np.array(gs))
        print(f"[{i}/{len(videos)}] {v}  cycles={ncyc} duty={duty:.0%} "
              f"p50={np.percentile(cycles,50):.0f}ms  ({time.monotonic()-t0:.1f}s)",
              flush=True)

    engine.close()
    c = np.array(all_cycles)
    print("\n=== perfil operacional do sistema encadeado ===")
    print(f"  ciclo p50 {np.percentile(c,50):.0f} ms | p95 {np.percentile(c,95):.0f} ms "
          f"| p99 {np.percentile(c,99):.0f} ms")
    print(f"  taxa efetiva {1000/np.mean(c):.1f} Hz (media {np.mean(c):.0f} ms)")
    print(f"  duty cycle do C3E: {100*np.mean(duties):.1f}% dos ciclos")
    print_paper_report(conf, ev, tm, tr, STATES)


if __name__ == "__main__":
    main()
