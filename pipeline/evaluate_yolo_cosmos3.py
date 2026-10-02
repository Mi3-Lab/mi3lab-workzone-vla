#!/usr/bin/env python3
"""Detector + Cosmos3-Edge as the semantic verifier (replacing CLIP).

Idea (user's): keep YOLO as the fast detector (43 ms/cycle) and call the
Cosmos3-Edge reasoner at low frequency as the verifier, instead of CLIP.
CLIP is already the low-rate component in the ported baseline (every
CLIP_INTERVAL cycles), and it is the weakest part under domain shift —
Cosmos3-Edge detects work zones at 92% out-of-domain vs the fine-tuned VLM's 4%.

Cost: 43 ms + 162 ms / interval.  At interval 3 that is ~97 ms/cycle (~10 Hz),
versus 1.4 Hz for the pure-VLM cascade.

Run with the workzone venv (numpy<2 for ultralytics):
    ~/workzone/venv/bin/python pipeline/evaluate_yolo_cosmos3.py --split validation
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

import evaluate_yolo_baseline as B
from cascade_state_machine import parse_gate
from live_cascade_demo import GATE_PROMPT, PersistentEngine
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                            TransitionToleranceAccumulator, print_paper_report)

SHM = "/dev/shm/yolo_c3e_frame.jpg"
STATE_MAP = B.STATE_MAP
F_CONF = B.F_CONF


class Cosmos3Verifier:
    """Drop-in replacement for ClipFusion.global_score using the Cosmos3-Edge
    reasoner.  Returns a signed score so the caller's logistic(3x) maps it to a
    confident probability, matching how the CLIP cosine score was consumed."""

    def __init__(self, engine, size=(736, 416)):
        self.engine = engine
        self.size = size

    def global_score(self, frame_bgr):
        cv2.imwrite(SHM, cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA))
        txt, _ = self.engine.infer(SHM, GATE_PROMPT, 8)
        g = parse_gate(txt or "")
        if g is True:
            return 1.0
        if g is False:
            return -1.0
        return 0.0


def run(model, verifier, video_path, verify_interval=3, always=False):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predicted = np.full(n_frames, "outside", dtype=object)
    transitions = []
    state, y_ema_v, f_ema_v, state_dur, out_f = "OUT", None, None, 0, 999
    last_score, cycle_idx, frame_idx, sim_clock = 0.5, 0, -1, 0.0
    lat = []

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
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        counts = {}
        for cid in res.boxes.cls.cpu().tolist():
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                counts[cat] = counts.get(cat, 0) + 1

        y_s, total = B.yolo_frame_score(counts)
        evidence = B.clamp01(0.5 * B.clamp01(total / 8.0) + 0.5 * B.clamp01(y_s))
        alpha = B.adaptive_alpha(evidence, F_CONF["ema_alpha"] * 0.4, F_CONF["ema_alpha"] * 1.2)
        y_ema_v = B.ema(y_ema_v, y_s, alpha)

        fused = y_s
        # always=True removes CLIP's two gates on the semantic channel: the
        # y_ema trigger and the interval.  With them, the verifier can only
        # confirm what the detector already suspects -- which is why the ported
        # baseline collapses out of domain, where the detector never clears the
        # trigger in the first place.  Always-on lets the world model *raise*
        # evidence the detector missed, at a uniform 43+162 ms cycle.
        if always or y_ema_v >= B.CLIP_TRIGGER_TH:
            if always or cycle_idx % verify_interval == 0:
                last_score = B.logistic(verifier.global_score(frame) * 3.0)
            fused = (1.0 - B.CLIP_WEIGHT) * fused + B.CLIP_WEIGHT * last_score
        if y_ema_v < B.CONTEXT_TRIGGER_BELOW:
            fused = (1.0 - B.ORANGE_WEIGHT) * fused + B.ORANGE_WEIGHT * B.orange_context_score(frame)
        f_ema_v = B.ema(f_ema_v, B.clamp01(fused), alpha)

        prev = state
        state, state_dur, out_f = B.update_state(state, f_ema_v, state_dur, out_f)
        cycle_idx += 1
        dt = time.monotonic() - t0
        lat.append(dt * 1000)
        mapped = STATE_MAP[state]
        if state != prev:
            transitions.append((frame_idx, STATE_MAP[prev], mapped))
        predicted[max(0, frame_idx):] = mapped
        sim_clock += dt

    cap.release()
    return predicted, transitions, (float(np.mean(lat)) if lat else 0.0)


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
    ap.add_argument("--verify-interval", type=int, default=3)
    ap.add_argument("--always", action="store_true",
                    help="run the world-model channel every cycle, ungated")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--engine-dir", default=os.path.expanduser("~/cosmos3edge-engines"))
    args = ap.parse_args()

    videos = json.load(open(args.split_file))[args.split][:args.limit]
    ann = json.load(open(args.annotations))
    model = YOLO(args.weights)
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)
    verifier = Cosmos3Verifier(engine)
    if args.cache_dir:
        os.makedirs(os.path.expanduser(args.cache_dir), exist_ok=True)

    STATES = ["outside", "approaching", "inside", "exiting"]
    ev, tm = EventLevelAccumulator(STATES), TimingOffsetAccumulator(STATES)
    tr = TransitionToleranceAccumulator()
    conf, lats = {}, []
    for i, v in enumerate(videos, 1):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        gt = B.load_ground_truth(ann, v, nf)
        t0 = time.monotonic()
        pred, trans, ml = run(model, verifier, p, args.verify_interval, args.always)
        lats.append(ml)
        n = min(len(pred), len(gt))
        ps, gs = [pred[j] for j in range(n)], [gt[j] for j in range(n)]
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs); tm.add_video(ps, gs); tr.add_video(ps, gs)
        if args.cache_dir:
            np.savez_compressed(os.path.join(os.path.expanduser(args.cache_dir), f"{v}.npz"),
                                predicted=np.array(ps), gt=np.array(gs))
        print(f"[{i}/{len(videos)}] {v}  cycle={ml:.0f}ms  ({time.monotonic()-t0:.1f}s)", flush=True)

    engine.close()
    print(f"\nlatencia media de ciclo: {np.mean(lats):.0f} ms ({1000/np.mean(lats):.1f} Hz)")
    print_paper_report(conf, ev, tm, tr, STATES)


if __name__ == "__main__":
    main()
