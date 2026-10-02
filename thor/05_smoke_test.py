#!/usr/bin/env python3
"""Smoke test + regression against the Orin + latency, on whatever device runs it.

  1. The 2B answers a real work-zone frame with sensible text (catches the
     tie_word_embeddings regression: replies starting with ".DATA").
  2. The world model (C3E) answers at its fixed 736x416 input and NOT with an
     empty string (empty replies are how a wrong input size fails: silently).
  3. Regression: C3E (and the detector, if available) on every second of
     CA99_Day_01, compared with what the Orin recorded in eval_cache/ood_stream.
     INT4 kernels differ across GPUs, so a few flips are expected; many are not.
  4. Per-channel latency (p50/p95) for both models and the detector.

Writes eval_cache/thor/smoke_<platform>_<date>.json and prints PASS/WARN/FAIL.

Usage:  ~/workzone/venv/bin/python thor/05_smoke_test.py [--quick]
"""
import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
from cascade_state_machine import has_corroboration, parse_gate  # noqa: E402
from live_cascade_demo import (DESC_PROMPT, EGO_PROMPT, GATE_PROMPT,  # noqa: E402
                               SIGN_PROMPT, PersistentEngine)

H = os.path.expanduser
BIN = H("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video")
PLUGIN = H("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so")
C3E = H("~/cosmos3edge-engines")
SAMPLE = os.path.join(REPO, "test", "sample_frame.jpg")
CA_KEY = "CA99_Day_01"
CA_VIDEO = H("~/workzone/data/All_Construction_Data/With_SpeedLimit/CA99_Day_01.MP4")
SHM = "/dev/shm/thor_smoke.jpg"
CHANNELS = [("GATE", GATE_PROMPT, 8), ("SIGN", SIGN_PROMPT, 15),
            ("EGO", EGO_PROMPT, 8), ("DESC", DESC_PROMPT, 40)]
WZ_WORDS = ("cone", "barric", "barrier", "marker", "drum", "sign", "work", "worker")

results = {"checks": {}}
status = []


def verdict(name, level, msg):
    status.append((name, level, msg))
    color = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}[level]
    print(f"{color}[{level}]\033[0m {name}: {msg}", flush=True)


def platform():
    try:
        model = open("/proc/device-tree/model").read().strip("\x00").strip()
    except OSError:
        model = "unknown"
    return model


def write_frame(frame, size):
    if size == "2b":   # 2B: proportional 480 px wide, as in the paper
        h = int(frame.shape[0] * 480 / frame.shape[1])
        img = cv2.resize(frame, (480, h), interpolation=cv2.INTER_AREA)
    else:              # C3E: frozen 26x46 patch grid -> exactly 736x416
        img = cv2.resize(frame, (736, 416), interpolation=cv2.INTER_AREA)
    cv2.imwrite(SHM, img)


def latency(engine, reps):
    out = {}
    for name, prompt, n in CHANNELS:
        engine.infer(SHM, prompt, n)   # warm-up
        ms = [engine.infer(SHM, prompt, n)[1] for _ in range(reps)]
        out[name] = {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95))}
        print(f"    {name:5s} p50 {out[name]['p50_ms']:6.1f} ms   p95 {out[name]['p95_ms']:6.1f} ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 s of regression instead of the whole video")
    args = ap.parse_args()
    reps = 5 if args.quick else 15
    frame = cv2.imread(SAMPLE)
    if frame is None:
        sys.exit(f"missing {SAMPLE}")
    results.update(host=socket.gethostname(), device=platform(),
                   date=datetime.datetime.now().isoformat(timespec="seconds"))
    print(f"device: {results['device']}\n")

    # ---------------------------------------------------------------- 1. 2B
    print("1) 2B fine-tuned (480 px)")
    eng = PersistentEngine(BIN, PLUGIN, os.path.join(REPO, "engines", "llm"),
                           os.path.join(REPO, "engines", "visual"), temperature=0.0)
    write_frame(frame, "2b")
    desc, _ = eng.infer(SHM, DESC_PROMPT, 40)
    desc = (desc or "").strip()
    print(f"    DESC: {desc[:120]!r}")
    results["checks"]["2b_desc"] = desc
    if not desc:
        verdict("2B resposta", "FAIL", "resposta vazia (engine não respondeu)")
    elif desc.startswith("."):
        verdict("2B resposta", "FAIL", "começa com lixo ('.DATA'?): lm_head trocado pela embedding")
    elif any(w in desc.lower() for w in WZ_WORDS):
        verdict("2B resposta", "PASS", "descreve elementos de obra")
    else:
        verdict("2B resposta", "WARN", "resposta sem elementos de obra; confira o texto acima")
    results["latency_2b"] = latency(eng, reps)
    eng.close()

    # ---------------------------------------------------------------- 2. C3E
    print("\n2) Cosmos3-Edge (736x416)")
    eng = PersistentEngine(BIN, PLUGIN, C3E, os.path.join(C3E, "visual"), temperature=0.0)
    write_frame(frame, "c3e")
    gate, _ = eng.infer(SHM, GATE_PROMPT, 8)
    gate = (gate or "").replace("<|im_end|>", "").strip()
    print(f"    GATE: {gate!r}")
    results["checks"]["c3e_gate"] = gate
    if not gate:
        verdict("C3E resposta", "FAIL", "resposta vazia: tamanho de entrada errado ou engine quebrado")
    elif gate.lower().startswith("yes"):
        verdict("C3E resposta", "PASS", "responde 'Yes' num frame com obra")
    else:
        verdict("C3E resposta", "WARN", f"respondeu {gate!r} num frame com obra")
    results["latency_c3e"] = latency(eng, reps)

    # ---------------------------------------------------------------- 3. regression
    print(f"\n3) regressão contra o Orin: {CA_KEY}")
    rec_path = os.path.join(REPO, "eval_cache", "ood_stream", f"{CA_KEY}.npz")
    if not (os.path.exists(CA_VIDEO) and os.path.exists(rec_path)):
        verdict("regressão", "WARN", "vídeo ou registro do Orin ausente (01_sync_from_orin.sh smoke)")
        eng.close()
    else:
        rec = np.load(rec_path, allow_pickle=True)["obs"]   # t, det, gate, sign, corr
        if args.quick:
            rec = rec[:20]
        cap = cv2.VideoCapture(CA_VIDEO)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        mine, frames = [], {}
        t0 = time.monotonic()
        for row in rec:
            t = int(row[0])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                break
            frames[t] = fr
            write_frame(fr, "c3e")
            g = eng.infer(SHM, GATE_PROMPT, 8)[0] or ""
            s = (eng.infer(SHM, SIGN_PROMPT, 15)[0] or "").strip()
            d = eng.infer(SHM, DESC_PROMPT, 40)[0] or ""
            mine.append((bool(parse_gate(g)), has_corroboration("", s), has_corroboration(d, ""), bool(g.strip())))
        cap.release(); eng.close()
        n = len(mine)
        m = np.array(mine)
        empty = int((~m[:, 3]).sum())
        agree = {k: float((m[:, i] == (rec[:n, j] > .5)).mean())
                 for k, i, j in (("gate", 0, 2), ("sign", 1, 3), ("desc_corr", 2, 4))}
        results["regression_c3e"] = dict(seconds=n, empty_replies=empty, agreement=agree,
                                         wall_s=time.monotonic() - t0)
        print(f"    {n} s; concordância com o Orin: " +
              ", ".join(f"{k} {v:.0%}" for k, v in agree.items()) + f"; vazias: {empty}")
        if empty:
            verdict("regressão C3E", "FAIL", f"{empty} respostas vazias")
        elif agree["gate"] >= 0.90:
            verdict("regressão C3E", "PASS", f"GATE concorda {agree['gate']:.0%}")
        elif agree["gate"] >= 0.75:
            verdict("regressão C3E", "WARN", f"GATE concorda só {agree['gate']:.0%} (numérica INT4 diferente?)")
        else:
            verdict("regressão C3E", "FAIL", f"GATE concorda {agree['gate']:.0%}")

        # detector, if this platform has torch + ultralytics
        try:
            from ultralytics import YOLO
            import evaluate_yolo_baseline as B
            import joint_filter as J
            model = YOLO(H("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
            ys, lat = [], []
            for t in sorted(frames):
                t1 = time.monotonic()
                res = model.predict(frames[t], conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
                lat.append((time.monotonic() - t1) * 1000)
                counts = {}
                for cid in res.boxes.cls.cpu().tolist():
                    cat = B.get_cue_category(model.names[int(cid)])
                    if cat:
                        counts[cat] = counts.get(cat, 0) + 1
                ys.append(B.yolo_frame_score(counts)[0])
            ys = np.array(ys); ref = rec[:len(ys), 1]
            same_bucket = float(np.mean([J.det_bucket(a) == J.det_bucket(b) for a, b in zip(ys, ref)]))
            results["regression_yolo"] = dict(mean_abs_diff=float(np.abs(ys - ref).mean()),
                                              same_bucket=same_bucket)
            results["latency_yolo"] = {"p50_ms": float(np.percentile(lat[1:], 50)),
                                       "p95_ms": float(np.percentile(lat[1:], 95))}
            print(f"    detector: mesmo bucket em {same_bucket:.0%}, |Δscore| médio "
                  f"{results['regression_yolo']['mean_abs_diff']:.3f}, "
                  f"p50 {results['latency_yolo']['p50_ms']:.1f} ms")
            verdict("regressão detector", "PASS" if same_bucket >= 0.9 else "WARN",
                    f"mesmo bucket em {same_bucket:.0%}")
        except ImportError as e:
            verdict("regressão detector", "WARN", f"pulado: sem torch/ultralytics ({e.name})")

    # ---------------------------------------------------------------- report
    tag = results["device"].replace(" ", "_").replace("/", "_")[:40]
    out_dir = os.path.join(REPO, "eval_cache", "thor"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"smoke_{tag}_{datetime.date.today()}.json")
    results["status"] = status
    json.dump(results, open(out, "w"), indent=1)
    fails = sum(1 for _, lv, _ in status if lv == "FAIL")
    print(f"\nrelatório: {out}")
    print("RESULTADO:", "\033[31mFALHOU\033[0m" if fails else "\033[32mOK\033[0m")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
