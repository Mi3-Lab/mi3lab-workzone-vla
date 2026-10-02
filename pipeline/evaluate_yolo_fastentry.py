#!/usr/bin/env python3
"""Reviewer control (W2): detector+CLIP baseline WITH the VLM's text-based
fast entry.  Tests whether the pure-VLM system's boundary-localization edge
is simply the sign channel---addable to a detector---or something more.

The detector+CLIP pipeline runs exactly as in evaluate_yolo_baseline.py, but
while the state machine is in OUT (patrol) we additionally run the VLM
\textsc{sign} transcription channel on the frame; a lexicon-matching
transcription forces an immediate OUT->APPROACHING transition (the same fast
entry the VLM cascade uses).  Latency is honest: the SIGN inference time is
added to the patrol-cycle wall clock, so the hybrid samples frames more
sparsely in OUT, exactly as it would in deployment.

Run with the workzone venv (numpy<2):
    ~/workzone/venv/bin/python pipeline/evaluate_yolo_fastentry.py \
        --split validation --clip --cache-dir ~/eval_cache/yoloclip_fastentry
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

import evaluate_yolo_baseline as B
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                            TransitionToleranceAccumulator, print_paper_report)
from cascade_state_machine import has_corroboration
from live_cascade_demo import (PersistentEngine, SIGN_PROMPT, SHM_FRAME,
                               DEFAULT_ENGINE_DIR, DEFAULT_MM_ENGINE_DIR,
                               DEFAULT_PLUGIN, DEFAULT_BINARY)

STATE_MAP = B.STATE_MAP
F_CONF = B.F_CONF


def run_hybrid(model, engine, video_path, clip=None, infer_w=480):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predicted = np.full(n_frames, "outside", dtype=object)
    transitions = []
    state, y_ema_v, f_ema_v, state_dur, out_f = "OUT", None, None, 0, 999
    last_clip_score = 0.5
    cycle_idx = 0
    frame_idx = -1
    sim_clock = 0.0

    while True:
        target_idx = int(sim_clock * fps)
        if target_idx >= n_frames:
            break
        while frame_idx < target_idx:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.retrieve()
        if not ok:
            break

        t0 = time.monotonic()
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        h_img, w_img = frame.shape[:2]
        candidates = []
        for box, cid, conf in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.cls.cpu().tolist(),
                                   res.boxes.conf.cpu().tolist()):
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                x1, y1, x2, y2 = map(int, box)
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
                candidates.append({"cat": cat, "conf": conf, "crop": frame[y1:y2, x1:x2]})

        counts = {}
        should_verify = clip is not None and (cycle_idx % B.PER_CUE_INTERVAL == 0)
        if should_verify and candidates:
            candidates.sort(key=lambda c: c["conf"], reverse=True)
            to_verify, remaining = candidates[:4], candidates[4:]
            scores = clip.verify_batch([c["crop"] for c in to_verify], [c["cat"] for c in to_verify])
            for c, s in zip(to_verify, scores):
                if s > B.PER_CUE_TH:
                    counts[c["cat"]] = counts.get(c["cat"], 0) + 1
            for c in remaining:
                counts[c["cat"]] = counts.get(c["cat"], 0) + 1
        else:
            for c in candidates:
                counts[c["cat"]] = counts.get(c["cat"], 0) + 1

        y_s, total_objs = B.yolo_frame_score(counts)
        evidence = B.clamp01(0.5 * B.clamp01(total_objs / 8.0) + 0.5 * B.clamp01(y_s))
        alpha = B.adaptive_alpha(evidence, F_CONF["ema_alpha"] * 0.4, F_CONF["ema_alpha"] * 1.2)
        y_ema_v = B.ema(y_ema_v, y_s, alpha)
        fused = y_s
        if clip is not None and y_ema_v >= B.CLIP_TRIGGER_TH:
            if cycle_idx % B.CLIP_INTERVAL == 0:
                last_clip_score = B.logistic(clip.global_score(frame) * 3.0)
            fused = (1.0 - B.CLIP_WEIGHT) * fused + B.CLIP_WEIGHT * last_clip_score
        if clip is not None and y_ema_v < B.CONTEXT_TRIGGER_BELOW:
            ctx = B.orange_context_score(frame)
            fused = (1.0 - B.ORANGE_WEIGHT) * fused + B.ORANGE_WEIGHT * ctx
        f_ema_v = B.ema(f_ema_v, B.clamp01(fused), alpha)

        # ── VLM sign fast-entry, only while patrolling (OUT) ─────────────
        sign_fast = False
        if state == "OUT":
            small = frame
            if frame.shape[1] > infer_w:
                hh = int(frame.shape[0] * infer_w / frame.shape[1])
                small = cv2.resize(frame, (infer_w, hh), interpolation=cv2.INTER_AREA)
            cv2.imwrite(SHM_FRAME, small)
            sign_txt, _ = engine.infer(SHM_FRAME, SIGN_PROMPT, 15)
            sign_fast = has_corroboration("", sign_txt or "")

        prev_state = state
        if state == "OUT" and sign_fast:
            state, state_dur, out_f = "APPROACHING", 0, 0
        else:
            state, state_dur, out_f = B.update_state(state, f_ema_v, state_dur, out_f)
        cycle_idx += 1

        elapsed_s = time.monotonic() - t0
        mapped = STATE_MAP[state]
        if state != prev_state:
            transitions.append((frame_idx, STATE_MAP[prev_state], mapped))
        predicted[max(0, frame_idx):] = mapped
        sim_clock += elapsed_s

    cap.release()
    return predicted, transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    ap.add_argument("--annotations",
                    default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    ap.add_argument("--weights", default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    ap.add_argument("--split-file", default="eval_split_full.json")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--limit", type=int, default=10000)
    args = ap.parse_args()

    videos = json.load(open(args.split_file))[args.split][:args.limit]
    ann = json.load(open(args.annotations))
    model = YOLO(args.weights)
    clip = B.ClipFusion() if args.clip else None
    engine = PersistentEngine(DEFAULT_BINARY, DEFAULT_PLUGIN, DEFAULT_ENGINE_DIR,
                              DEFAULT_MM_ENGINE_DIR, temperature=0.0)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    STATES = ["outside", "approaching", "inside", "exiting"]
    event_acc = EventLevelAccumulator(STATES)
    timing_acc = TimingOffsetAccumulator(STATES)
    transition_acc = TransitionToleranceAccumulator()
    confusion = {}
    for i, v in enumerate(videos):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        gt = B.load_ground_truth(ann, v, nf)
        t0 = time.monotonic()
        pred, trans = run_hybrid(model, engine, p, clip=clip)
        n = min(len(pred), len(gt))
        ps = [pred[j] for j in range(n)]; gs = [gt[j] for j in range(n)]
        for a, b in zip(gs, ps):
            confusion[(a, b)] = confusion.get((a, b), 0) + 1
        event_acc.add_video(ps, gs); timing_acc.add_video(ps, gs)
        transition_acc.add_video(ps, gs)
        if args.cache_dir:
            np.savez_compressed(os.path.join(args.cache_dir, f"{v}.npz"),
                                predicted=np.array(ps), gt=np.array(gs))
        print(f"[{i+1}/{len(videos)}] {v}  trans={len(trans)}  ({time.monotonic()-t0:.1f}s)",
              flush=True)

    engine.close()
    print_paper_report(confusion, event_acc, timing_acc, transition_acc, STATES)


if __name__ == "__main__":
    main()
