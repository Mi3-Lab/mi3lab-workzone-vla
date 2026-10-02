#!/usr/bin/env python3
"""The world model as the backbone, the detector as its evidence channel.

Four earlier combinations all failed, and they shared an assumption we never
questioned: the detector's state machine was the spine and the world model was
an accessory.  Measured on the same videos, that spine is the weaker component
(macro-F1 0.426 for the ported detector against 0.593 for the recalibrated C3E
cascade), so every one of those designs was building on the worse of the two
and trying to repair it with the better.  This inverts the roles.

Spine: the C3E cascade of the main paper, with the thresholds calibrated for
that model (N=3, K_enter=3, K_exit=2, K_fade=1, ego disabled).

Detector: the corroboration channel.  Entry in that cascade requires a gate
vote AND corroboration from a second channel, which until now came from the
world model's own DESC query -- one 488 ms opinion per cycle, from the same
model that produced the gate, and therefore correlated with it.  The detector
supplies the same signal from an independent modality, at 24 ms, aggregated
over every frame since the previous world-model answer instead of sampled once.
That attacks C3E's measured weakness (64.6 false alarms/h against the
detector's 30.9) at the point where it is generated: the entry decision.

Rates: the detector runs continuously and never waits for the world model; the
world model is queried on its own schedule, and additionally on a rising edge
of detector evidence, so a zone appearing between polls is not missed.  The
cascade steps once per world-model answer, which is the rate its window
constants were calibrated at -- stepping it at the detector's rate would repeat
the cycles-versus-seconds error diagnosed earlier in this work.

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
from cascade_state_machine import CascadeStateMachine, WZState, parse_gate
from live_cascade_demo import GATE_PROMPT, PersistentEngine
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                           TransitionToleranceAccumulator, print_paper_report)

SHM = "/dev/shm/c3e_spine.jpg"
STATES = ["outside", "approaching", "inside", "exiting"]
C3E_SIZE = (736, 416)
C3E_CSM = dict(N=3, k_enter=3, k_exit=2, k_fading=1, use_ego=False)

POLL_S = 1.0        # world-model polling period, the stride its window was calibrated at
DET_MIN_OBJ = 1     # detector corroborates when it sees at least this many cues


def run(model, engine, video_path, poll_s=POLL_S, det_corr=True, edge_trigger=True):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predicted = np.full(n_frames, "outside", dtype=object)

    csm = CascadeStateMachine(**C3E_CSM)
    state = WZState.OUTSIDE
    frame_idx, sim_clock = -1, 0.0
    next_poll, prev_det_active = 0.0, False
    det_hits, det_frames = 0, 0          # detector evidence since the last poll
    det_cycles, polls, edge_polls = [], 0, 0

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

        # ---- detector: every cycle, never waiting on the world model ----
        t0 = time.monotonic()
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        counts = {}
        for cid in res.boxes.cls.cpu().tolist():
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        n_obj = sum(counts.values())
        det_active = n_obj >= DET_MIN_OBJ
        det_frames += 1
        det_hits += int(det_active)
        dt = time.monotonic() - t0
        det_cycles.append(dt * 1000.0)

        # ---- world model: on schedule, or when detector evidence rises ----
        rising = edge_trigger and det_active and not prev_det_active
        prev_det_active = det_active
        if sim_clock >= next_poll or rising:
            t1 = time.monotonic()
            cv2.imwrite(SHM, cv2.resize(frame, C3E_SIZE, interpolation=cv2.INTER_AREA))
            g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
            gate = bool(parse_gate(g_txt or ""))
            dt_v = time.monotonic() - t1
            polls += 1
            edge_polls += int(rising and sim_clock < next_poll)

            # corroboration from the detector, aggregated over the interval
            if det_corr:
                corr = det_frames > 0 and (det_hits / det_frames) >= 0.5
            else:
                corr = det_active
            det_hits, det_frames = 0, 0

            state = csm.update_evidence(gate, corr, None,
                                        fast_entry=gate and corr)
            next_poll = sim_clock + poll_s
            dt += dt_v          # the poll costs this cycle its own latency

        predicted[max(0, frame_idx):] = state.value
        sim_clock += dt

    cap.release()
    return predicted, det_cycles, polls, edge_polls


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
    ap.add_argument("--poll-s", type=float, default=POLL_S)
    ap.add_argument("--no-det-corr", action="store_true",
                    help="ablation: use the instantaneous detector answer, not the aggregate")
    ap.add_argument("--no-edge", action="store_true",
                    help="ablation: poll on schedule only, no rising-edge trigger")
    ap.add_argument("--cache-dir", default=None)
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
    conf, cyc, polls_tot, edge_tot = {}, [], 0, 0
    for i, v in enumerate(videos, 1):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p) or v not in ann:
            continue
        cap = cv2.VideoCapture(p); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        gt = [s.value if hasattr(s, "value") else str(s)
              for s in B.load_ground_truth(ann, v, nf)]
        t0 = time.monotonic()
        pred, cycles, polls, edges = run(model, engine, p, args.poll_s,
                                         det_corr=not args.no_det_corr,
                                         edge_trigger=not args.no_edge)
        n = min(len(pred), len(gt))
        ps, gs = [str(pred[j]) for j in range(n)], [str(gt[j]) for j in range(n)]
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs); tm.add_video(ps, gs); tr.add_video(ps, gs)
        cyc.extend(cycles); polls_tot += polls; edge_tot += edges
        if args.cache_dir:
            np.savez_compressed(os.path.join(os.path.expanduser(args.cache_dir), f"{v}.npz"),
                                predicted=np.array(ps), gt=np.array(gs))
        print(f"[{i}/{len(videos)}] {v}  det={len(cycles)} polls={polls} "
              f"edge={edges} ({time.monotonic()-t0:.1f}s)", flush=True)

    engine.close()
    c = np.array(cyc)
    print(f"\n=== world-model spine, detector corroboration (poll {args.poll_s}s) ===")
    print(f"  detector cycle p50 {np.percentile(c,50):.0f} ms | {1000/np.mean(c):.1f} Hz")
    print(f"  world-model polls {polls_tot} ({edge_tot} triggered by detector edge)")
    print_paper_report(conf, ev, tm, tr, STATES)


if __name__ == "__main__":
    main()
