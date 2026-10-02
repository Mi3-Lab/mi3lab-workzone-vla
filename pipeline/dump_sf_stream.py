#!/usr/bin/env python3
"""Same 1 Hz evidence dump as dump_ood_stream.py, but on ROADWork's own San
Francisco footage instead of our California dashcam recordings.  This is the
control the OOD benchmark was missing: same collection format and camera as
Boston/Seattle (1920x1080 ROADWork snippets), same state-estimation prompts,
a DIFFERENT city never used for the 4-state benchmark or for calibration.
It separates "camera/night/rain" from "this geography" as explanations for
the response-bias gap, because here neither camera nor conditions differ from
Boston -- only the city does.

No ground truth exists for these clips (no 4-state annotation, no
visible/ego-relevant label): the point is to let a person watch the answers
against the frames, not to score them automatically.

Usage:
  python3 dump_sf_stream.py --out ~/eval_cache/sf_stream
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, SIGN_PROMPT, PersistentEngine
from cascade_state_machine import has_corroboration, parse_gate

VDIR = os.path.expanduser("~/workzone/data/videos")
SHM = "/dev/shm/sf_stream.jpg"
W, H = 736, 416  # C3E visual tower is FROZEN at a 26x46 patch grid; a proportional resize (e.g. the 2B's 480px convention) breaks inference silently -- confirmed: all-empty replies

FILES = [
    "san_francisco_032bee7c11054832906b1eae93ebccb1_000000_15780_snippet.mp4",
    "san_francisco_0873632ec728436da0ddbdb2a65dcc0c_000002_13650_snippet.mp4",
    "san_francisco_59b9e9f8a40a4642ba6280f6bc5964cf_000002_01500_snippet.mp4",
    "san_francisco_7dd719a023a342359327c0d5cf6a3615_000002_01140_snippet.mp4",
    "san_francisco_0873632ec728436da0ddbdb2a65dcc0c_000004_06750_snippet.mp4",
    "san_francisco_7dd719a023a342359327c0d5cf6a3615_000003_02640_snippet.mp4",
    "san_francisco_0efc638c166b4d05b3e6b50f9115ac38_000002_02820_snippet.mp4",
    "san_francisco_27cb92abad5c4c4caee70d5779ac7ae7_000001_00540_snippet.mp4",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        os.path.expanduser("~/cosmos3edge-engines"),
        os.path.expanduser("~/cosmos3edge-engines/visual"), temperature=0.0)

    for fname in FILES:
        key = fname[:-len("_snippet.mp4")]
        dst = os.path.join(out, f"{key}.npz")
        if os.path.exists(dst):
            print(f"skip {key}", flush=True)
            continue
        path = os.path.join(VDIR, fname)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        rows, texts = [], []
        t0 = time.monotonic()
        for t in range(n_s + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(SHM, cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA))
            g_txt = engine.infer(SHM, GATE_PROMPT, 8)[0] or ""
            s_txt = (engine.infer(SHM, SIGN_PROMPT, 15)[0] or "").strip()
            d_txt = engine.infer(SHM, DESC_PROMPT, 40)[0] or ""
            rows.append((t, float(bool(parse_gate(g_txt))),
                        float(has_corroboration("", s_txt)),
                        float(has_corroboration(d_txt, ""))))
            texts.append({"t": t, "gate": g_txt, "sign": s_txt, "desc": d_txt})
        cap.release()
        np.savez_compressed(dst, obs=np.array(rows, dtype=np.float64), fps=fps)
        json.dump(texts, open(os.path.join(out, f"{key}.json"), "w"))
        print(f"{key}: {len(rows)} s ({time.monotonic()-t0:.0f}s)", flush=True)
    engine.close()
    print("done")


if __name__ == "__main__":
    main()
