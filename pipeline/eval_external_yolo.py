#!/usr/bin/env python3
"""Detector counterpart to eval_external_ca.py: on the same annotated-sign
approach windows of the self-recorded California footage, does the
ROADWork-trained YOLO detector fire any work-zone class?  Enables a paired
external OOD-generalization comparison VLM vs detector.  Run with workzone venv.
"""
import os
import re
import sys

import cv2
from ultralytics import YOLO

D = os.path.expanduser("~/workzone/data/All_Construction_Data")
INFER_IMGSZ = 960
WIN_S, STEP_S = 6, 1.0
WZ_CLASSES = {"Cone", "Drum", "Barricade", "Barrier", "Vertical Panel",
              "Tubular Marker", "Fence", "Worker", "Work Vehicle",
              "Temporary Traffic Control Sign", "Arrow Board",
              "Temporary Traffic Control Message Board"}


def find_video(key):
    for sub in ["With_SpeedLimit", "Only_Signs", "Lane_Change"]:
        for ext in [".MP4", ".mp4"]:
            p = os.path.join(D, sub, key + ext)
            if os.path.exists(p):
                return p
    base = key.split("_")[0]
    for sub in ["With_SpeedLimit", "Only_Signs", "Lane_Change"]:
        for f in os.listdir(os.path.join(D, sub)):
            if f.lower().startswith(base.lower()) and f.lower().endswith(".mp4"):
                return os.path.join(D, sub, f)
    return None


def condition(key):
    k = key.lower()
    if "fog" in k: return "night-fog"
    if "rain" in k: return "night-rain"
    if "night" in k: return "night"
    if "evening" in k: return "evening"
    if "sunset" in k: return "sunset"
    return "day"


def parse_txt():
    p = [f for f in os.listdir(os.path.join(D, "With_SpeedLimit")) if f.endswith(".txt")][0]
    items = []
    for line in open(os.path.join(D, "With_SpeedLimit", p)):
        m = re.match(r"\s*([A-Za-z0-9_]+):\s*(?:Sign at\s*)?(.+)", line)
        if not m:
            continue
        rest = m.group(2)
        if "impossible" in rest.lower():
            continue
        ts = [int(a) * 60 + int(b) for a, b in re.findall(r"(\d{1,2}):(\d{2})", rest)]
        if ts:
            items.append((m.group(1), ts))
    return items


def main():
    model = YOLO(os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    only = sys.argv[1:] or None
    by_cond, td, tt = {}, 0, 0
    for key, tss in parse_txt():
        if only and key not in only:
            continue
        path = find_video(key)
        if not path:
            continue
        cond = condition(key)
        cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t0 in tss:
            det = False
            for dt in [i * STEP_S for i in range(int(WIN_S / STEP_S) + 1)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + dt) * fps))
                ok, fr = cap.read()
                if not ok:
                    continue
                r = model.predict(fr, conf=0.25, iou=0.7, imgsz=INFER_IMGSZ, verbose=False)[0]
                if any(model.names[int(c)] in WZ_CLASSES for c in r.boxes.cls.cpu().tolist()):
                    det = True; break
            c = by_cond.setdefault(cond, [0, 0]); c[1] += 1; tt += 1
            if det:
                c[0] += 1; td += 1
            print(f"{key} @{t0//60}:{t0%60:02d} [{cond}] -> {'DET' if det else 'miss'}", flush=True)
        cap.release()
    print("\n=== detector external detection rate (any work-zone class in window) ===")
    for cond, (d, n) in sorted(by_cond.items()):
        print(f"  {cond:12s} {d}/{n} = {d/n:.0%}")
    print(f"  {'OVERALL':12s} {td}/{tt} = {td/tt:.0%}")


if __name__ == "__main__":
    main()
