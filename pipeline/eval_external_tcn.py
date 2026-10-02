#!/usr/bin/env python3
"""The TCN segmenter on the out-of-domain California footage.

This is the test the system can fail.  It fuses a detector that collapses off
distribution (72% overall, 44% at night, 40% in fog) with a world model that
does not (95%, 100%, 80%).  If the fusion inherits the detector's fragility, it
trades an in-domain gain for a loss exactly where the paper argues the world
model earns its place.  And its parameters were counted on ROADWork, so this
also asks whether a fitted temporal model transfers across camera, region and
weather at all.

Scoring matches the other three systems exactly so the numbers are comparable
to the 39 annotated approaches already reported: a detection counts if the
system leaves OUTSIDE at any point inside the 6 s window following an annotated
sign approach.  The estimator is reset between windows so that one detection
cannot leak into the next.

Run with the workzone venv (numpy<2 for ultralytics).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

import eval_external_cosmos3 as E
import evaluate_yolo_baseline as B
import joint_filter as J
from cascade_state_machine import has_corroboration, parse_gate
from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, SIGN_PROMPT, PersistentEngine

SHM = "/dev/shm/joint_ood.jpg"
WIN_S = 6.0
GRID_HZ = 10.0   # the grid the stack was trained on; a dilated stack's receptive
                 # field is measured in STEPS, so sampling it at 1 Hz would hand
                 # it ten times the context it ever saw.  The Bayesian filter has
                 # no such constraint -- its rates are per second by construction.


def detect_in_window(model, engine, tcn, cap, fps, t0, device):
    """Run the segmenter across one approach window at its training rate.

    The detector is sampled on the 10 Hz grid; the world model is polled at its
    own latency, and its answer is held (with its age) between polls, exactly as
    in the deployed loop.
    """
    import torch
    import tcn_segmenter as T
    n = int(WIN_S * GRID_HZ)
    feats = np.zeros((n, T.N_FEAT), dtype=np.float32)
    gate = sign = corr = 0.0
    sem_time = -1e9
    next_poll = 0.0
    for k in range(n):
        t = k / GRID_HZ
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + t) * fps))
        ok, frame = cap.read()
        if not ok:
            break
        if t >= next_poll:
            cv2.imwrite(SHM, cv2.resize(frame, J.C3E_SIZE, interpolation=cv2.INTER_AREA))
            g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
            gate = 1.0 if parse_gate(g_txt or "") else 0.0
            s_txt, _ = engine.infer(SHM, SIGN_PROMPT, 15)
            sign = 1.0 if has_corroboration("", (s_txt or "").strip()) else 0.0
            d_txt, _ = engine.infer(SHM, DESC_PROMPT, 40)
            corr = 1.0 if has_corroboration(d_txt or "", "") else 0.0
            sem_time = t
            next_poll = t + 0.7        # the world model's own cycle
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        counts = {}
        for cid in res.boxes.cls.cpu().tolist():
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        y_s, _ = B.yolo_frame_score(counts)
        age = min(t - sem_time, 10.0) if sem_time > -1e8 else 10.0
        feats[k] = [y_s, gate, sign, corr, age / 10.0, 1.0]
    with torch.no_grad():
        x = torch.from_numpy(feats.T[None]).to(device)
        pred = tcn(x)[-1].argmax(dim=1)[0].cpu().numpy()
    for k, p in enumerate(pred):
        if T.STATES[int(p)] != "outside":
            return True, k / GRID_HZ
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    ap.add_argument("--engine-dir", default=os.path.expanduser("~/cosmos3edge-engines"))
    ap.add_argument("--model", required=True, help="trained TCN checkpoint")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--stages", type=int, default=2)
    args = ap.parse_args()

    import torch, tcn_segmenter as T
    device = "cpu"
    tcn = T.MSTCN(n_hidden=args.hidden, n_stages=args.stages).to(device)
    tcn.load_state_dict(torch.load(os.path.expanduser(args.model), map_location=device))
    tcn.eval()
    print(f"TCN segmenter from {args.model}, evaluated on its {GRID_HZ:.0f} Hz grid\n")

    model = YOLO(args.weights)
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)

    by_cond, det_tot, tot = {}, 0, 0
    for key, tss in E.parse_txt():
        path = E.find_video(key)
        if not path:
            continue
        cond = E.condition(key)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t0 in tss:
            t_start = time.monotonic()
            fired, first = detect_in_window(model, engine, tcn, cap, fps, t0, device)
            c = by_cond.setdefault(cond, [0, 0])
            c[1] += 1
            tot += 1
            if fired:
                c[0] += 1
                det_tot += 1
            print(f"{key} @{t0//60}:{t0%60:02d} [{cond}] -> "
                  f"{'DET +' + str(int(first)) + 's' if fired else 'miss'} "
                  f"({time.monotonic()-t_start:.1f}s)", flush=True)
        cap.release()

    engine.close()
    print("\n=== TCN segmenter, out-of-domain detection rate ===")
    for cond, (d, n) in sorted(by_cond.items()):
        print(f"  {cond:12s} {d}/{n} = {d/n:.0%}")
    if tot:
        print(f"  {'OVERALL':12s} {det_tot}/{tot} = {det_tot/tot:.0%}")


if __name__ == "__main__":
    main()
