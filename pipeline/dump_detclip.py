#!/usr/bin/env python3
"""Record the full detector+CLIP pipeline's evidence on every cycle, so the
complete baseline (not only its detector score) can be recalibrated by replay.

Reviewers' W1/C4: the recalibrated detector used only the YOLO score because
CLIP scores had not been recorded.  This runs the same loop as
evaluate_yolo_baseline.run_yolo_on_video, latency-honest, but computes every
scheduled component unconditionally and stores it:

  obs[:, 0] frame index        obs[:, 1] cycle time (s)
  obs[:, 2] y_s from raw counts            obs[:, 3] total raw objects
  obs[:, 4] y_s from CLIP-verified counts  obs[:, 5] total verified objects
  obs[:, 6] CLIP global score (logistic(3*sim), fresh on cycle % 3 == 0, else held)
  obs[:, 7] orange context score            obs[:, 8] cycle index

Per-cue verification runs on cycle % 3 == 0, as in the app, so columns 4-5
equal 2-3 on other cycles.  The global CLIP score is computed on every third
cycle regardless of the EMA gate; the replay applies the gate itself, so the
hand-tuned configuration is reproduced exactly except that cycle times include
the always-on CLIP call (slightly sparser sampling, i.e. conservative).

    ~/workzone/venv/bin/python pipeline/dump_detclip.py --split calibration --out ~/eval_cache/detclip_dump_cal
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

import evaluate_yolo_baseline as B


def dump_video(model, clip, path, n_frames, fps):
    cap = cv2.VideoCapture(path)
    rows, frame_idx, clock, cycle = [], -1, 0.0, 0
    last_clip = 0.5
    while True:
        target = int(clock * fps)
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
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        h, w = frame.shape[:2]
        cands = []
        for box, cid, conf in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.cls.cpu().tolist(),
                                  res.boxes.conf.cpu().tolist()):
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = max(0, x1 - 10), max(0, y1 - 10), min(w, x2 + 10), min(h, y2 + 10)
                cands.append({"cat": cat, "conf": conf, "crop": frame[y1:y2, x1:x2]})
        raw = {}
        for c in cands:
            raw[c["cat"]] = raw.get(c["cat"], 0) + 1
        ver = dict(raw)
        if cycle % B.PER_CUE_INTERVAL == 0 and cands:
            cands.sort(key=lambda c: c["conf"], reverse=True)
            top, rest = cands[:4], cands[4:]
            scores = clip.verify_batch([c["crop"] for c in top], [c["cat"] for c in top])
            ver = {}
            for c, s in zip(top, scores):
                if s > B.PER_CUE_TH:
                    ver[c["cat"]] = ver.get(c["cat"], 0) + 1
            for c in rest:
                ver[c["cat"]] = ver.get(c["cat"], 0) + 1
        y_raw, n_raw = B.yolo_frame_score(raw)
        y_ver, n_ver = B.yolo_frame_score(ver)
        if cycle % B.CLIP_INTERVAL == 0:
            last_clip = B.logistic(clip.global_score(frame) * 3.0)
        orange = B.orange_context_score(frame)
        dt = time.monotonic() - t0
        rows.append([frame_idx, dt, y_raw, n_raw, y_ver, n_ver, last_clip, orange, cycle])
        cycle += 1
        clock += dt
    cap.release()
    return np.array(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["calibration", "validation"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "eval_split_full.json"))
    ap.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    ap.add_argument("--annotations", default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    ap.add_argument("--weights", default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    ann = json.load(open(args.annotations))
    vids = json.load(open(args.split_file))[args.split]
    model = YOLO(args.weights, task="detect")
    model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=960, verbose=False)
    clip = B.ClipFusion()
    for i, v in enumerate(vids):
        dst = os.path.join(out, f"{v}.npz")
        if os.path.exists(dst):
            continue
        path = os.path.join(args.videos_dir, v)
        if not os.path.exists(path):
            print(f"[{i+1}/{len(vids)}] SKIP {v}", flush=True)
            continue
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        gt = B.load_ground_truth(ann, v, n)
        t0 = time.monotonic()
        obs = dump_video(model, clip, path, n, fps)
        np.savez_compressed(dst, obs=obs, gt=np.array(gt.tolist()), n_frames=n, fps=fps)
        print(f"[{i+1}/{len(vids)}] {v} cycles={len(obs)} "
              f"cycle={1000*obs[:,1].mean():.0f}ms ({time.monotonic()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
