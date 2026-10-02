#!/usr/bin/env python3
"""Dump per-cycle channel evidence so cascade hyper-parameters can be calibrated
offline for ANY model, instead of inheriting the ones tuned for our 2B.

Why this exists: the cascade thresholds (N, K_ENTER, K_EXIT, K_FADE), the
Bayesian filter matrices and the qualifier lexicon were all fixed on the
calibration split *for our fine-tuned 2B*.  Running a different model (e.g.
Cosmos3-Edge) with those numbers measures the hyper-parameters, not the model.

To make an offline grid search exact, every channel is run on EVERY sampled
frame at a fixed stride (instead of the state-dependent schedule the live
cascade uses).  The replay can then apply any policy to the same evidence.

Usage:
  python3 dump_evidence.py --split calibration --limit 100 --stride-s 1.0 \
      --engine-dir ~/cosmos3edge-engines --out ~/eval_cache/evidence_c3e
"""
import argparse
import json
import os

import cv2
import numpy as np

from cascade_state_machine import classify_ego, has_corroboration, parse_gate
from live_cascade_demo import (DESC_PROMPT, EGO_PROMPT, GATE_PROMPT, SIGN_PROMPT,
                                PersistentEngine)

SHM = "/dev/shm/dump_frame.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    ap.add_argument("--annotations",
                    default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    ap.add_argument("--split-file", default="eval_split_full.json")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--stride-s", type=float, default=1.0)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--visual-dir", default=None)
    ap.add_argument("--infer-size", default="736x416")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plugin-path",
                    default=os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"))
    ap.add_argument("--binary",
                    default=os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"))
    args = ap.parse_args()

    eng_dir = os.path.expanduser(args.engine_dir)
    vis_dir = os.path.expanduser(args.visual_dir) if args.visual_dir else os.path.join(eng_dir, "visual")
    w, h = (int(x) for x in args.infer_size.lower().split("x"))
    os.makedirs(os.path.expanduser(args.out), exist_ok=True)

    videos = json.load(open(args.split_file))[args.split][:args.limit]
    ann = json.load(open(args.annotations))
    engine = PersistentEngine(args.binary, args.plugin_path, eng_dir, vis_dir, temperature=0.0)

    for i, name in enumerate(videos, 1):
        path = os.path.join(args.videos_dir, name)
        outp = os.path.join(os.path.expanduser(args.out), f"{name}.npz")
        if not os.path.exists(path) or os.path.exists(outp):
            continue
        cap = cv2.VideoCapture(path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(args.stride_s * fps))

        idxs, gates, signs, corrs, egos = [], [], [], [], []
        f = 0
        while f < n_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(SHM, cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
            g, _ = engine.infer(SHM, GATE_PROMPT, 8)
            s, _ = engine.infer(SHM, SIGN_PROMPT, 32)
            d, _ = engine.infer(SHM, DESC_PROMPT, 64)
            e, _ = engine.infer(SHM, EGO_PROMPT, 16)
            gv = parse_gate(g or "")
            idxs.append(f)
            gates.append(1 if gv is True else (0 if gv is False else -1))
            signs.append(1 if has_corroboration("", s or "") else 0)
            corrs.append(1 if has_corroboration(d or "", "") else 0)
            ec = classify_ego(e or "")
            egos.append(-1 if ec is None else ec)
            f += step
        cap.release()

        gt = np.full(n_frames, "outside", dtype=object)
        for st, spans in ann.get(name, {}).items():
            for a, b in spans:
                gt[a:min(b + 1, n_frames)] = st
        np.savez_compressed(outp, frame_idx=np.array(idxs), gate=np.array(gates),
                            sign=np.array(signs), corr=np.array(corrs),
                            ego=np.array(egos), gt=np.array(gt), fps=fps,
                            n_frames=n_frames)
        print(f"[{i}/{len(videos)}] {name}  cycles={len(idxs)}", flush=True)

    engine.close()


if __name__ == "__main__":
    main()
