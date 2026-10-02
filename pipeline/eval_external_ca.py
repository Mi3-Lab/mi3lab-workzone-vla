#!/usr/bin/env python3
"""External generalization test on self-recorded California work-zone footage
(All_Construction_Data), disjoint from ROADWork training/eval (zero leakage;
new geography; day/evening/night/rain/fog).  Ground truth = annotated
construction-sign timestamps.  For each annotated sign we sample an approach
window and count it detected if GATE fires or SIGN transcribes a sign at any
frame in the window.  Reports detection rate overall and by condition.
"""
import os
import re
import sys

import cv2

from live_cascade_demo import (PersistentEngine, GATE_PROMPT, SIGN_PROMPT, SHM_FRAME,
                               DEFAULT_ENGINE_DIR, DEFAULT_MM_ENGINE_DIR,
                               DEFAULT_PLUGIN, DEFAULT_BINARY)
from cascade_state_machine import parse_gate, has_corroboration

D = os.path.expanduser("~/workzone/data/All_Construction_Data")
INFER_W = 720
WIN_S = 6          # approach window length (s) from the annotated timestamp
STEP_S = 1.0

NO_SIGN_RE = re.compile(r"no (temporary )?(traffic control|sign)", re.I)


def find_video(key):
    for sub in ["With_SpeedLimit", "Only_Signs", "Lane_Change"]:
        for ext in [".MP4", ".mp4"]:
            p = os.path.join(D, sub, key + ext)
            if os.path.exists(p):
                return p
    # fuzzy: Fresno_Sunset_01 -> Fresno_01
    base = key.split("_")[0]
    for sub in ["With_SpeedLimit", "Only_Signs", "Lane_Change"]:
        for f in os.listdir(os.path.join(D, sub)):
            if f.lower().startswith(base.lower()) and f.lower().endswith((".mp4",)):
                return os.path.join(D, sub, f)
    return None


def condition(key):
    k = key.lower()
    if "nightfog" in k or "fog" in k: return "night-fog"
    if "nightrain" in k or "rain" in k: return "night-rain"
    if "night" in k: return "night"
    if "evening" in k: return "evening"
    if "sunset" in k: return "sunset"
    return "day"


def parse_txt():
    txt = open(os.path.join(D, "With_SpeedLimit",
                            [f for f in os.listdir(os.path.join(D, "With_SpeedLimit"))
                             if f.endswith(".txt")][0])).read()
    items = []
    for line in txt.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_]+):\s*(?:Sign at\s*)?(.+)", line)
        if not m:
            continue
        key, rest = m.group(1), m.group(2)
        hard = "hard" in rest.lower() or "impossible" in rest.lower()
        impossible = "impossible" in rest.lower()
        ts = []
        for t in re.findall(r"(\d{1,2}):(\d{2})", rest):
            ts.append(int(t[0]) * 60 + int(t[1]))
        if ts:
            items.append((key, ts, hard, impossible))
    return items


def main():
    engine = PersistentEngine(DEFAULT_BINARY, DEFAULT_PLUGIN, DEFAULT_ENGINE_DIR,
                              DEFAULT_MM_ENGINE_DIR, temperature=0.0)
    items = parse_txt()
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    by_cond = {}
    total_det = total = 0
    for key, tss, hard, impossible in items:
        if only and key not in only:
            continue
        if impossible:
            continue
        path = find_video(key)
        if not path:
            print(f"SKIP {key}: video not found"); continue
        cond = condition(key)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t0 in tss:
            detected = False
            hit = ""
            for dt in [i * STEP_S for i in range(int(WIN_S / STEP_S) + 1)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + dt) * fps))
                ok, fr = cap.read()
                if not ok:
                    continue
                h = int(fr.shape[0] * INFER_W / fr.shape[1])
                cv2.imwrite(SHM_FRAME, cv2.resize(fr, (INFER_W, h)))
                g, _ = engine.infer(SHM_FRAME, GATE_PROMPT, 3)
                s, _ = engine.infer(SHM_FRAME, SIGN_PROMPT, 15)
                s = (s or "").strip()
                gate_yes = parse_gate(g or "") is True
                sign_yes = has_corroboration("", s)  # real work-zone sign keyword
                if gate_yes or sign_yes:
                    detected = True
                    hit = f"+{dt:.0f}s gate={g.strip() if g else '?'} sign={s[:40]}"
                    break
            cap_c = by_cond.setdefault(cond, [0, 0])
            cap_c[1] += 1; total += 1
            if detected:
                cap_c[0] += 1; total_det += 1
            tag = " [hard]" if hard else ""
            print(f"{key} @{t0//60}:{t0%60:02d}{tag} [{cond}] -> "
                  f"{'DET '+hit if detected else 'miss'}", flush=True)
        cap.release()
    engine.close()
    print("\n=== external detection rate (GATE or SIGN within approach window) ===")
    for cond, (d, n) in sorted(by_cond.items()):
        print(f"  {cond:12s} {d}/{n}  = {d/n:.0%}" if n else f"  {cond}: 0")
    print(f"  {'OVERALL':12s} {total_det}/{total} = {total_det/total:.0%}" if total else "none")


if __name__ == "__main__":
    main()
