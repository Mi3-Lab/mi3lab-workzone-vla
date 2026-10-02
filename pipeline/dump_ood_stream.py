#!/usr/bin/env python3
"""Record every evidence channel over the WHOLE California footage, once, at 1 Hz.

Every out-of-domain number so far came from a GPU pass per method over 6 s
windows.  That makes each new temporal layer cost a run, gives no view of the
evidence around a window, and gives a test-time method nothing to adapt on.
This records the continuous stream instead: detector score and the world
model's three channels at every integer second of every video, plus the raw
replies.  The sign-approach windows are then slices of this record, so any
temporal layer can be scored out of domain by replay, in seconds.

Frames are taken at int(t * fps) for integer t, which is exactly how the window
evaluations seek, so a replay over a window must reproduce their results.

Run with the workzone venv.  Resumable: finished videos are skipped.

Usage:
  python3 dump_ood_stream.py --out ~/eval_cache/ood_stream
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
from cascade_state_machine import has_corroboration, parse_gate
from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, SIGN_PROMPT, PersistentEngine

SHM = "/dev/shm/ood_stream.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights",
                    default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    model = YOLO(args.weights)
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        E.ENG, E.VIS, temperature=0.0)

    annotations = {k: tss for k, tss in E.parse_txt()}
    for key, tss in annotations.items():
        dst = os.path.join(out, f"{key}.npz")
        if os.path.exists(dst):
            print(f"skip {key}", flush=True)
            continue
        path = E.find_video(key)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        rows, texts = [], []
        t_start = time.monotonic()
        for t in range(n_s + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(SHM, cv2.resize(frame, (E.W, E.H), interpolation=cv2.INTER_AREA))
            g_txt = engine.infer(SHM, GATE_PROMPT, 8)[0] or ""
            s_txt = (engine.infer(SHM, SIGN_PROMPT, 15)[0] or "").strip()
            d_txt = engine.infer(SHM, DESC_PROMPT, 40)[0] or ""
            res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
            counts = {}
            for cid in res.boxes.cls.cpu().tolist():
                cat = B.get_cue_category(model.names[int(cid)])
                if cat:
                    counts[cat] = counts.get(cat, 0) + 1
            y_s, _ = B.yolo_frame_score(counts)
            rows.append((t, y_s, float(bool(parse_gate(g_txt))),
                         float(has_corroboration("", s_txt)),
                         float(has_corroboration(d_txt, ""))))
            texts.append({"t": t, "gate": g_txt, "sign": s_txt, "desc": d_txt,
                          "det": counts})
        cap.release()
        np.savez_compressed(dst, obs=np.array(rows, dtype=np.float64),
                            approaches=np.array(tss, dtype=np.int64),
                            condition=E.condition(key), fps=fps)
        with open(os.path.join(out, f"{key}.json"), "w") as fh:
            json.dump(texts, fh)
        print(f"{key}: {len(rows)} s recorded, {len(tss)} approaches "
              f"({time.monotonic()-t_start:.0f}s)", flush=True)
    engine.close()
    print("done")


if __name__ == "__main__":
    main()
