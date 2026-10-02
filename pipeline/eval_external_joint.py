#!/usr/bin/env python3
"""The joint estimator on the out-of-domain California footage.

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
WIN_S, STEP_S = 6.0, 1.0


def detect_in_window(model, engine, cap, fps, t0, params, mask_det=False):
    """Run the estimator across one approach window; True if it ever activates."""
    filt = J.JointFilter(params)
    gate = sign = corr = False
    sem_time = -1e9
    clock = 0.0
    fired, first = False, None
    # The three systems of the OOD table are scored on RAW EVIDENCE -- a gate
    # activation or a work-zone keyword anywhere in the window.  A state machine
    # leaving OUTSIDE is a strictly harder bar, so scoring this system that way
    # and comparing it to those numbers would not be the paired comparison the
    # table claims.  Both are recorded here on the same samples: the gap between
    # them is exactly how much recovered evidence the estimator discards.
    evid = False
    t = 0.0
    while t <= WIN_S:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + t) * fps))
        ok, frame = cap.read()
        if not ok:
            break
        # world model on this sample
        cv2.imwrite(SHM, cv2.resize(frame, J.C3E_SIZE, interpolation=cv2.INTER_AREA))
        g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
        gate = bool(parse_gate(g_txt or ""))
        s_txt, _ = engine.infer(SHM, SIGN_PROMPT, 15)
        sign = has_corroboration("", (s_txt or "").strip())
        d_txt, _ = engine.infer(SHM, DESC_PROMPT, 40)
        corr = has_corroboration(d_txt or "", "")
        sem_time = clock
        # detector on the same frame
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        counts = {}
        for cid in res.boxes.cls.cpu().tolist():
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        y_s, _ = B.yolo_frame_score(counts)
        if mask_det:
            # the ablation of ablate_joint.py, carried out of domain: the same
            # filter with the detector stream removed at BOTH fit and inference.
            # In domain that variant loses little; the question here is whether
            # it recovers the world model's out-of-domain recall, which the
            # joint system does not.
            y_s = 0.0
        obs = J.obs_index(J.det_bucket(y_s), gate, sign, corr,
                          J.age_bucket(clock - sem_time))
        evid = evid or gate or sign
        state = filt.step(obs, STEP_S)
        if state != "outside" and not fired:
            fired, first = True, t
        clock += STEP_S
        t += STEP_S
    return fired, first, evid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    ap.add_argument("--engine-dir", default=os.path.expanduser("~/cosmos3edge-engines"))
    ap.add_argument("--params", required=True)
    ap.add_argument("--mask-det", action="store_true",
                    help="zero the detector stream (pair with params fitted the same way)")
    args = ap.parse_args()

    with open(os.path.expanduser(args.params)) as fh:
        params = json.load(fh)
    print(f"joint estimator, parameters counted on "
          f"{params.get('n_calibration_videos','?')} ROADWork calibration videos\n")

    model = YOLO(args.weights)
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)

    by_cond, det_tot, ev_tot, tot = {}, 0, 0, 0
    for key, tss in E.parse_txt():
        path = E.find_video(key)
        if not path:
            continue
        cond = E.condition(key)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t0 in tss:
            t_start = time.monotonic()
            fired, first, evid = detect_in_window(model, engine, cap, fps, t0,
                                                  params, args.mask_det)
            c = by_cond.setdefault(cond, [0, 0, 0])
            c[1] += 1
            tot += 1
            if fired:
                c[0] += 1
                det_tot += 1
            if evid:
                c[2] += 1
                ev_tot += 1
            print(f"{key} @{t0//60}:{t0%60:02d} [{cond}] -> "
                  f"{'DET +' + str(int(first)) + 's' if fired else 'miss'} "
                  f"({time.monotonic()-t_start:.1f}s)", flush=True)
        cap.release()

    engine.close()
    print("\n=== out-of-domain: evidence recovered vs. acted upon ===")
    print(f"  {'condition':12s} {'state leaves OUTSIDE':>22s} {'raw evidence':>16s}")
    for cond, (d, n, e) in sorted(by_cond.items()):
        print(f"  {cond:12s} {d}/{n} = {d/n:6.0%}{'':>9s} {e}/{n} = {e/n:4.0%}")
    if tot:
        print(f"  {'OVERALL':12s} {det_tot}/{tot} = {det_tot/tot:6.0%}{'':>9s} "
              f"{ev_tot}/{tot} = {ev_tot/tot:4.0%}")


if __name__ == "__main__":
    main()
