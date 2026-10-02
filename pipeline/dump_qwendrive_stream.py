#!/usr/bin/env python3
"""Qwen-Drive-1.0-4B as a third evidence source, on exactly the samples the
world model's records use, with the same three prompts and token budgets.

Why it matters for the paper: Qwen-Drive is a driving-specialized foundation
model whose training mix INCLUDES ROADWork.  It sits between the world model
(never saw a work zone) and our fine-tuned 2B (trained on ROADWork work zones).
Its Boston/Seattle numbers would be contaminated; the California footage is not
in ROADWork, so California is its clean test.

Only the VLM is loaded (VQA mode); the planning and BEV heads are not used.
Input is 736x416, which the Qwen3.5 patch grid turns into 299 image tokens,
the same budget as the world model.  Runs with PyTorch (BF16, SDPA attention),
not TensorRT: its latency is NOT comparable with the INT4 engines.

Environment (no venv on this Orin):
  PYTHONPATH=~/qwendrive-libs:~/models/qwen-drive/src python3 dump_qwendrive_stream.py ...

Usage:
  ... dump_qwendrive_stream.py --source california --out ~/eval_cache/qwendrive_ood_stream
  ... dump_qwendrive_stream.py --source sf         --out ~/eval_cache/qwendrive_sf_stream
  ... dump_qwendrive_stream.py --pilot             # raw replies on a few frames, no files
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from cascade_state_machine import has_corroboration, parse_gate
from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, SIGN_PROMPT

H = os.path.expanduser
MODEL = H("~/models/Qwen-Drive-1.0-4B")
SIZE = (736, 416)
CHANNELS = (("gate", GATE_PROMPT, 8), ("sign", SIGN_PROMPT, 15), ("desc", DESC_PROMPT, 40))
NO_THINK = False   # set by --no-think; see main()


def load_model():
    from qwen_drive import QwenDriveForPlanning
    m = QwenDriveForPlanning.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    return m.to("cuda").eval()


def ask(model, img, prompt, n):
    from qwen_drive import CameraFrame
    q = prompt
    t0 = time.monotonic()
    out = model.generate_text([CameraFrame(img, target_size=SIZE)], q, max_new_tokens=n)
    torch.cuda.synchronize()
    return out.text.strip(), (time.monotonic() - t0) * 1000


def frame_at(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) if ok else None


def sources(which):
    if which == "california":
        import eval_external_cosmos3 as E
        for p in sorted(glob.glob(H("~/eval_cache/ood_stream/*.npz"))):
            key = os.path.basename(p)[:-4]
            ts = np.load(p, allow_pickle=True)["obs"][:, 0].astype(int)
            yield key, E.find_video(key), ts
    else:
        for p in sorted(glob.glob(H("~/eval_cache/sf_stream/*.npz"))):
            key = os.path.basename(p)[:-4]
            ts = np.load(p, allow_pickle=True)["obs"][:, 0].astype(int)
            yield key, H(f"~/workzone/data/videos/{key}_snippet.mp4"), ts


def pilot(model):
    import eval_external_cosmos3 as E
    tests = [("sample_frame (Boston, obra)", Image.open(H("~/jetson-deploy/test/sample_frame.jpg")).convert("RGB"))]
    cap = cv2.VideoCapture(E.find_video("CA99_Day_02")); fps = cap.get(cv2.CAP_PROP_FPS)
    tests += [("CA99_Day_02 t=20 (rascunho: sem obra)", frame_at(cap, fps, 20)),
              ("CA99_Day_02 t=150 (rascunho: obra relevante)", frame_at(cap, fps, 150))]
    for name, img in tests:
        print(f"\n== {name}")
        for ch, prompt, n in CHANNELS:
            txt, ms = ask(model, img, prompt, 64)   # generous budget: see if it thinks first
            print(f"  {ch:4s} ({ms:6.0f} ms, budget 64): {txt[:160]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["california", "sf"])
    ap.add_argument("--out")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--channels", default="gate,sign,desc",
                    help="subset to run; missing channels are stored as NaN")
    ap.add_argument("--merge-from", default=None,
                    help="earlier output dir whose channels are copied instead of re-run")
    args = ap.parse_args()
    run_ch = [c for c in CHANNELS if c[0] in args.channels.split(",")]
    model = load_model()
    if args.pilot:
        pilot(model); return
    out = H(args.out); os.makedirs(out, exist_ok=True)
    for key, vpath, ts in sources(args.source):
        dst = os.path.join(out, f"{key}.npz")
        if os.path.exists(dst):
            print(f"skip {key}", flush=True); continue
        cap = cv2.VideoCapture(vpath); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rows, texts, t0 = [], [], time.monotonic()
        prev = {}
        if args.merge_from:
            jp = os.path.join(H(args.merge_from), f"{key}.json")
            if os.path.exists(jp):
                prev = {r["t"]: r for r in json.load(open(jp))}
        for t in ts:
            img = frame_at(cap, fps, t)
            if img is None:
                break
            r = dict(prev.get(int(t), {"t": int(t)}))
            for ch, prompt, n in run_ch:
                r[ch], r[ch + "_ms"] = ask(model, img, prompt, n)
            nan = float("nan")
            rows.append((t, float(bool(parse_gate(r["gate"]))) if "gate" in r else nan,
                         float(has_corroboration("", r["sign"])) if "sign" in r else nan,
                         float(has_corroboration(r["desc"], "")) if "desc" in r else nan))
            texts.append(r)
        cap.release()
        np.savez_compressed(dst, obs=np.array(rows, dtype=np.float64), fps=fps)
        json.dump(texts, open(os.path.join(out, f"{key}.json"), "w"))
        empty = sum(1 for r in texts if not r.get("gate", "x"))
        print(f"{key}: {len(rows)} s, {empty} empty gate replies ({time.monotonic()-t0:.0f}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
