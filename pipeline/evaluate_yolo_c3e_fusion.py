#!/usr/bin/env python3
"""Detector + world model as a fused perception stack: C3E replaces CLIP, but
without the two restrictions that make CLIP useless out of domain.

The ported baseline gates its semantic channel twice: CLIP runs only when the
detector's EMA already clears a trigger, and then only every third cycle.  The
verifier can therefore confirm the detector but never contradict it, which is
exactly why the pipeline collapses at night -- the detector never clears the
trigger, so the semantic channel is never consulted.

This stack changes three things, in order of how much they matter:

  1. The world-model GATE runs EVERY cycle, ungated, so it can *raise* evidence
     the detector missed.  Uniform 43+162 ms cycle: unlike a state-gating
     design, there is no regime where the expensive model is needed most and
     costs most.

  2. Disagreement escalates to transcription.  When the detector is quiet but
     the world model says yes, the SIGN channel (556 ms) is spent to arbitrate,
     because a transcribed 'ROAD WORK AHEAD' is verifiable evidence while two
     binary answers in conflict are not.  This is the paper's evidence-first
     principle applied inside the fusion rather than around it.  Agreement
     costs nothing extra, so the average cycle stays near 205 ms.

  3. The fusion weight is CALIBRATED for this model rather than inherited from
     CLIP.  Inheriting it would repeat, inside our own system, the error this
     paper is about.  --dump-scores writes per-cycle raw scores so the weight
     can be swept offline on the calibration split in CPU seconds.

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
from cascade_state_machine import has_corroboration, parse_gate
from live_cascade_demo import GATE_PROMPT, SIGN_PROMPT, PersistentEngine
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                           TransitionToleranceAccumulator, print_paper_report)

SHM = "/dev/shm/yolo_c3e_fusion.jpg"
STATES = ["outside", "approaching", "inside", "exiting"]
C3E_SIZE = (736, 416)
QUIET_TH = B.CLIP_TRIGGER_TH   # "the detector is quiet" uses its own threshold


def c3e_scores(engine, frame, quiet):
    """Semantic evidence for one frame.

    Returns (score in [0,1], escalated).  GATE is always paid; SIGN is paid
    only to arbitrate a disagreement, where its transcription is checkable.
    """
    cv2.imwrite(SHM, cv2.resize(frame, C3E_SIZE, interpolation=cv2.INTER_AREA))
    g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
    g = parse_gate(g_txt or "")
    if g is None:
        return 0.5, False
    if g and quiet:
        # detector quiet, world model positive -> spend transcription
        s_txt, _ = engine.infer(SHM, SIGN_PROMPT, 32)
        if has_corroboration("", (s_txt or "").strip()):
            return 1.0, True      # verifiable work-zone text
        return 0.75, True         # positive gate, unverified
    return (0.95 if g else 0.05), False


def run(model, engine, video_path, weight, dump=None):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predicted = np.full(n_frames, "outside", dtype=object)
    state, y_ema, f_ema, dur, out_f = "OUT", None, None, 0, 999
    frame_idx, sim_clock = -1, 0.0
    cycles, escalations, ncyc = [], 0, 0
    rows = []

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
        alpha = B.adaptive_alpha(evidence, B.F_CONF["ema_alpha"] * 0.4,
                                 B.F_CONF["ema_alpha"] * 1.2)
        y_ema = B.ema(y_ema, y_s, alpha)

        sem, esc = c3e_scores(engine, frame, quiet=(y_ema < QUIET_TH))
        escalations += int(esc)
        fused = (1.0 - weight) * y_s + weight * sem
        if y_ema < B.CONTEXT_TRIGGER_BELOW:
            fused = (1.0 - B.ORANGE_WEIGHT) * fused + B.ORANGE_WEIGHT * B.orange_context_score(frame)
        f_ema = B.ema(f_ema, B.clamp01(fused), alpha)
        state, dur, out_f = B.update_state(state, f_ema, dur, out_f)

        dt = time.monotonic() - t0
        cycles.append(dt * 1000.0)
        ncyc += 1
        predicted[max(0, frame_idx):] = B.STATE_MAP[state]
        if dump is not None:
            rows.append((frame_idx, y_s, y_ema, sem, int(esc)))
        sim_clock += dt

    cap.release()
    if dump is not None:
        dump.append((os.path.basename(video_path), rows, n_frames, fps))
    return predicted, cycles, escalations, ncyc


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
    ap.add_argument("--weight", type=float, default=B.CLIP_WEIGHT,
                    help="fusion weight; calibrate it, do not inherit CLIP's")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--dump-scores", default=None,
                    help="write per-cycle raw scores so the weight can be swept offline")
    args = ap.parse_args()

    videos = json.load(open(args.split_file))[args.split][:args.limit]
    ann = json.load(open(args.annotations))
    model = YOLO(args.weights)
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)
    for d in (args.cache_dir, args.dump_scores):
        if d:
            os.makedirs(os.path.expanduser(d), exist_ok=True)

    ev, tm = EventLevelAccumulator(STATES), TimingOffsetAccumulator(STATES)
    tr = TransitionToleranceAccumulator()
    conf, all_cycles, esc_tot, cyc_tot = {}, [], 0, 0
    for i, v in enumerate(videos, 1):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p) or v not in ann:
            continue
        cap = cv2.VideoCapture(p)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        gt = [s.value if hasattr(s, "value") else str(s)
              for s in B.load_ground_truth(ann, v, nf)]
        dump = [] if args.dump_scores else None
        t0 = time.monotonic()
        pred, cycles, esc, ncyc = run(model, engine, p, args.weight, dump)
        n = min(len(pred), len(gt))
        ps, gs = [str(pred[j]) for j in range(n)], [str(gt[j]) for j in range(n)]
        for a, b in zip(gs, ps):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        ev.add_video(ps, gs); tm.add_video(ps, gs); tr.add_video(ps, gs)
        all_cycles.extend(cycles); esc_tot += esc; cyc_tot += ncyc
        if args.cache_dir:
            np.savez_compressed(os.path.join(os.path.expanduser(args.cache_dir), f"{v}.npz"),
                                predicted=np.array(ps), gt=np.array(gs))
        if args.dump_scores:
            name, rows, nfr, fps = dump[0]
            arr = np.array(rows, dtype=np.float32)
            np.savez_compressed(os.path.join(os.path.expanduser(args.dump_scores), f"{v}.npz"),
                                rows=arr, gt=np.array(gs), n_frames=nfr, fps=fps)
        print(f"[{i}/{len(videos)}] {v}  cycles={ncyc} esc={esc} "
              f"p50={np.percentile(cycles,50):.0f}ms ({time.monotonic()-t0:.1f}s)", flush=True)

    engine.close()
    c = np.array(all_cycles)
    print(f"\n=== fused stack (weight {args.weight:.2f}) ===")
    print(f"  cycle p50 {np.percentile(c,50):.0f} ms | p95 {np.percentile(c,95):.0f} ms "
          f"| mean {np.mean(c):.0f} ms ({1000/np.mean(c):.1f} Hz)")
    print(f"  transcription escalations: {esc_tot}/{cyc_tot} cycles "
          f"({100*esc_tot/max(cyc_tot,1):.1f}%)")
    print_paper_report(conf, ev, tm, tr, STATES)


if __name__ == "__main__":
    main()
