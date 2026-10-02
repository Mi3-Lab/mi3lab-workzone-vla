#!/usr/bin/env python3
"""Cosmos3-Edge INT4 reasoner on the self-recorded California footage — the same
annotated-sign approach windows used by eval_external_ca.py (our 2B: 4%) and
eval_external_yolo.py (detector: 72%).  Gives a three-way OOD comparison.

The Cosmos3-Edge visual tower is frozen at grid 26x46 patches => 736x416 input.
"""
import os
import re
import sys

import cv2

from live_cascade_demo import (PersistentEngine, GATE_PROMPT, SIGN_PROMPT, SHM_FRAME,
                               DEFAULT_PLUGIN, DEFAULT_BINARY)
from cascade_state_machine import parse_gate, has_corroboration

D = os.path.expanduser("~/workzone/data/All_Construction_Data")
ENG = os.path.expanduser("~/cosmos3edge-engines")
VIS = os.path.join(ENG, "visual")
W, H = 736, 416          # fixed by the visual tower's 26x46 patch grid
WIN_S, STEP_S = 6, 1.0


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
    engine = PersistentEngine(DEFAULT_BINARY, DEFAULT_PLUGIN, ENG, VIS, temperature=0.0)
    only = sys.argv[1:] or None
    by_cond, td, tt = {}, 0, 0
    for key, tss in parse_txt():
        if only and key not in only:
            continue
        path = find_video(key)
        if not path:
            continue
        cond = condition(key)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t0 in tss:
            det, hit = False, ""
            for dt in [i * STEP_S for i in range(int(WIN_S / STEP_S) + 1)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + dt) * fps))
                ok, fr = cap.read()
                if not ok:
                    continue
                cv2.imwrite(SHM_FRAME, cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA))
                g, _ = engine.infer(SHM_FRAME, GATE_PROMPT, 8)
                s, _ = engine.infer(SHM_FRAME, SIGN_PROMPT, 32)
                s = (s or "").strip()
                if parse_gate(g or "") is True or has_corroboration("", s):
                    det, hit = True, f"+{dt:.0f}s"
                    break
            c = by_cond.setdefault(cond, [0, 0]); c[1] += 1; tt += 1
            if det:
                c[0] += 1; td += 1
            print(f"{key} @{t0//60}:{t0%60:02d} [{cond}] -> {'DET '+hit if det else 'miss'}", flush=True)
        cap.release()
    engine.close()
    print("\n=== Cosmos3-Edge external detection rate ===")
    for cond, (d, n) in sorted(by_cond.items()):
        print(f"  {cond:12s} {d}/{n} = {d/n:.0%}")
    if tt:
        print(f"  {'OVERALL':12s} {td}/{tt} = {td/tt:.0%}")


if __name__ == "__main__":
    main()
