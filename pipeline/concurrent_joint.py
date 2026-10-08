#!/usr/bin/env python3
"""Reviewers' concurrency concern: run the detector and the world model truly
concurrently on the Orin, in wall-clock time, instead of charging each its
separately measured latency on a simulated clock.

Three threads:
  camera    decodes the video at its native frame rate in wall-clock time and
            publishes the latest frame (frames the consumers miss are dropped,
            as with a live camera)
  detector  YOLO on the latest frame, back to back (main thread)
  world     C3E gate/sign/desc on the latest frame, re-polled as soon as it
            answers; runs in its own TensorRT process, so both models share the
            GPU for real

Records rows in the joint-dump format (frame, clock, dt, y_s, gate, sign, corr,
age) so the counted filter can be applied to exactly what the live system saw,
plus per-cycle latencies.  --detector-only skips the world thread (reference
latency distribution).

    ~/workzone/venv/bin/python pipeline/concurrent_joint.py --n 20 --out ~/eval_cache/concurrent
    ~/workzone/venv/bin/python pipeline/concurrent_joint.py --n 20 --detector-only --out ~/eval_cache/concurrent_detonly
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

import evaluate_yolo_baseline as B
from cascade_state_machine import has_corroboration, parse_gate
from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, SIGN_PROMPT, PersistentEngine

C3E_SIZE = (736, 416)
SHM = "/dev/shm/concurrent_world.jpg"


class Camera(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.lock = threading.Lock()
        self.frame, self.idx, self.done = None, -1, False
        self.t0 = None

    def run(self):
        self.t0 = time.monotonic()
        i = 0
        while i < self.n:
            ok, f = self.cap.read()
            if not ok:
                break
            due = self.t0 + i / self.fps
            wait = due - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            with self.lock:
                self.frame, self.idx = f, i
            i += 1
        self.done = True

    def latest(self):
        with self.lock:
            return self.frame, self.idx


class World(threading.Thread):
    def __init__(self, engine, cam):
        super().__init__(daemon=True)
        self.engine, self.cam = engine, cam
        self.lock = threading.Lock()
        self.answer, self.t_answer = (False, False, False), None
        self.lat = []
        self.stop = False

    def run(self):
        while not self.stop and not self.cam.done:
            f, _ = self.cam.latest()
            if f is None:
                time.sleep(0.005)
                continue
            t = time.monotonic()
            cv2.imwrite(SHM, cv2.resize(f, C3E_SIZE, interpolation=cv2.INTER_AREA))
            g, _ = self.engine.infer(SHM, GATE_PROMPT, 8)
            s, _ = self.engine.infer(SHM, SIGN_PROMPT, 15)
            d, _ = self.engine.infer(SHM, DESC_PROMPT, 40)
            ans = (bool(parse_gate(g or "")), has_corroboration("", (s or "").strip()),
                   has_corroboration(d or "", ""))
            now = time.monotonic()
            self.lat.append(now - t)
            with self.lock:
                self.answer, self.t_answer = ans, now

    def current(self):
        with self.lock:
            return self.answer, self.t_answer


def run_video(model, engine, path, detector_only):
    cam = Camera(path)
    cam.start()
    while cam.t0 is None or cam.latest()[0] is None:
        time.sleep(0.002)
    world = None
    if not detector_only:
        world = World(engine, cam)
        world.start()
    rows, last_idx = [], -1
    while not cam.done:
        f, idx = cam.latest()
        if idx == last_idx:
            time.sleep(0.001)
            continue
        last_idx = idx
        t = time.monotonic()
        res = model.predict(f, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        counts = {}
        for cid in res.boxes.cls.cpu().tolist():
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        y_s, _ = B.yolo_frame_score(counts)
        dt = time.monotonic() - t
        if world is not None:
            (g, s, c), ta = world.current()
            age = (t - ta) if ta is not None else 1e9
        else:
            g = s = c = False
            age = 1e9
        rows.append((idx, t - cam.t0, dt, y_s, float(g), float(s), float(c), age))
    wl = []
    if world is not None:
        world.stop = True
        world.join(timeout=10)
        wl = world.lat
    return np.array(rows, dtype=np.float64), cam.n, np.array(wl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector-only", action="store_true")
    ap.add_argument("--split-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_split_full.json"))
    ap.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    ap.add_argument("--annotations", default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    ap.add_argument("--weights", default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    ap.add_argument("--engine-dir", default=os.path.expanduser("~/cosmos3edge-engines"))
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    vids = json.load(open(args.split_file))["validation"]
    rng = np.random.default_rng(0)
    vids = [vids[i] for i in sorted(rng.choice(len(vids), args.n, replace=False))]
    ann = json.load(open(args.annotations))
    model = YOLO(args.weights, task="detect")
    model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=960, verbose=False)
    engine = None
    if not args.detector_only:
        engine = PersistentEngine(
            os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
            os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
            args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)
    for i, v in enumerate(vids, 1):
        p = os.path.join(args.videos_dir, v)
        rows, n, wl = run_video(model, engine, p, args.detector_only)
        gt = [str(x) for x in B.load_ground_truth(ann, v, n)]
        np.savez_compressed(os.path.join(out, f"{v}.npz"), obs=rows, gt=np.array(gt),
                            n_frames=n, world_lat=wl)
        d = rows[:, 2] * 1000
        print(f"[{i}/{len(vids)}] {v} det cycles={len(rows)} p50={np.percentile(d,50):.0f}ms "
              f"p95={np.percentile(d,95):.0f}ms world answers={len(wl)}"
              + (f" p50={np.percentile(wl,50)*1000:.0f}ms" if len(wl) else ""), flush=True)
    if engine is not None:
        engine.close()


if __name__ == "__main__":
    main()
