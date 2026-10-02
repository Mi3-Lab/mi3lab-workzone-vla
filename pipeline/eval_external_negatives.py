#!/usr/bin/env python3
"""Out-of-domain NUISANCE, on the same California footage as the sign benchmark.

The sign benchmark scores only the 6 s window around each annotated approach,
which measures detection but says nothing about false firing.  A reviewer's
objection follows directly: a system that detects 95% of signs may simply fire
more often.  This scores the complement.

Negative definition (stated because it is the weak point): the annotations mark
sign *approaches*, not work-zone extents, so footage shortly after a sign may
still be inside a zone.  We therefore exclude a +/-20 s margin around every
annotated approach and treat the remaining footage as presumed negative.  This
is conservative in the sense that it removes the frames most likely to be
genuinely positive, but it cannot guarantee that no work zone survives in the
remainder, so the resulting rate is an upper bound on nuisance, not an exact
false-positive count.

Detection uses the identical rule as the positive benchmark: the GATE channel
fires, or the SIGN channel transcribes a work-zone keyword.

Usage:
  python3 eval_external_negatives.py --system c3e
  ~/workzone/venv/bin/python eval_external_negatives.py --system yolo
"""
import argparse
import os

import cv2

import eval_external_cosmos3 as E
from cascade_state_machine import has_corroboration, parse_gate

MARGIN_S = 20.0
STRIDE_S = 2.0
WIN_S = 6.0


def negative_times(duration, signs):
    """Sample times at STRIDE_S that are far from every annotated approach."""
    out, t = [], 0.0
    while t < duration:
        if all(not (s - MARGIN_S <= t <= s + WIN_S + MARGIN_S) for s in signs):
            out.append(t)
        t += STRIDE_S
    return out


def run_vlm(engine, size, prompts):
    """Returns a callable frame->bool using the positive benchmark's rule."""
    gate_p, sign_p = prompts

    def fire(frame):
        cv2.imwrite(E.SHM_FRAME, cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        g, _ = engine.infer(E.SHM_FRAME, gate_p, 8)
        if parse_gate(g or "") is True:
            return True
        s, _ = engine.infer(E.SHM_FRAME, sign_p, 32)
        return has_corroboration("", (s or "").strip())
    return fire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["c3e", "2b", "yolo"], required=True)
    args = ap.parse_args()

    from live_cascade_demo import GATE_PROMPT, SIGN_PROMPT, PersistentEngine
    BIN = os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video")
    PLG = os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so")

    engine = None
    if args.system == "c3e":
        eng = os.path.expanduser("~/cosmos3edge-engines")
        engine = PersistentEngine(BIN, PLG, eng, os.path.join(eng, "visual"), temperature=0.0)
        fire = run_vlm(engine, (736, 416), (GATE_PROMPT, SIGN_PROMPT))
    elif args.system == "2b":
        eng = os.path.expanduser("~/jetson-deploy/engines")
        engine = PersistentEngine(BIN, PLG, os.path.join(eng, "llm"),
                                  os.path.join(eng, "visual"), temperature=0.0)
        fire = run_vlm(engine, (480, 270), (GATE_PROMPT, SIGN_PROMPT))
    else:
        from ultralytics import YOLO
        import evaluate_yolo_baseline as B
        model = YOLO(os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))

        def fire(frame):
            res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
            counts = {}
            for cid in res.boxes.cls.cpu().tolist():
                cat = B.get_cue_category(model.names[int(cid)])
                if cat:
                    counts[cat] = counts.get(cat, 0) + 1
            score, _ = B.yolo_frame_score(counts)
            return score >= B.F_CONF["enter_th"]

    fired = total = 0
    secs = 0.0
    for key, ts in E.parse_txt():
        path = E.find_video(key)
        if not path:
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        times = negative_times(dur, ts)
        hits = 0
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            total += 1
            secs += STRIDE_S
            if fire(fr):
                fired += 1
                hits += 1
        cap.release()
        print(f"{key:24s} {len(times):4d} negative samples, {hits:3d} fired "
              f"({E.condition(key)})", flush=True)

    if engine:
        engine.close()
    hours = secs / 3600.0
    print(f"\n=== {args.system} on presumed-negative out-of-domain footage ===")
    print(f"  {total} samples over {secs/60:.1f} min ({hours:.2f} h)")
    print(f"  fired on {fired} ({100*fired/max(total,1):.1f}% of samples)")
    print(f"  upper-bound nuisance rate: {fired/max(hours,1e-9):.1f} firings/h")


if __name__ == "__main__":
    main()
