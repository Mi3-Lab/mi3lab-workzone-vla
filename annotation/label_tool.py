#!/usr/bin/env python3
"""Verify (or correct) draft work-zone labels by stepping through a video.

The window shows the frame, the current state label and the visibility label.
Labels are edited as change points: pressing a state key sets that state from
the current second until the next change point, which is how the benchmark's
interval annotations are made.

Keys
  space        play / pause
  a / d        back / forward 1 s          A / D   back / forward 10 s
  1 2 3 4      state from here on: outside / approaching / inside / exiting
  v            toggle "work zone visible" from here on
  n            jump to the next draft note (low-confidence stretch)
  s            save                        q       save and quit

The verified file records who verified it; only verified labels are used for
reported results (GUIDELINE.md).

Usage
  python3 label_tool.py CA99_Night_01 --annotator "Name"
  python3 label_tool.py --list
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/jetson-deploy/pipeline"))
import eval_external_cosmos3 as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFT = os.path.join(HERE, "california_draft.json")
OUT = os.path.join(HERE, "california_labels.json")
STATES = ["outside", "approaching", "inside", "exiting"]
COLORS = {"outside": (0, 160, 0), "approaching": (0, 140, 255),
          "inside": (0, 0, 220), "exiting": (200, 0, 200)}


def to_seconds(rec, n_s, fps):
    """Per-second state and visibility arrays from a frame-interval record."""
    state = np.array(["outside"] * n_s, dtype=object)
    vis = np.zeros(n_s, bool)
    for st in STATES:
        for a, b in rec.get(st, []):
            state[int(a // fps):int(b // fps) + 1] = st
    for a, b in rec.get("visible", []):
        vis[int(a // fps):int(b // fps) + 1] = True
    return state, vis


def to_intervals(state, vis, fps, n_frames):
    rec = {st: [] for st in STATES}
    start = 0
    for t in range(1, len(state) + 1):
        if t == len(state) or state[t] != state[start]:
            a = int(start * fps)
            b = min(int(t * fps) - 1, n_frames - 1)
            rec[state[start]].append([a, b])
            start = t
    rec["visible"] = []
    t = 0
    while t < len(vis):
        if vis[t]:
            u = t
            while u < len(vis) and vis[u]:
                u += 1
            rec["visible"].append([int(t * fps), min(int(u * fps) - 1, n_frames - 1)])
            t = u
        else:
            t += 1
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?")
    ap.add_argument("--annotator", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    draft = json.load(open(DRAFT)) if os.path.exists(DRAFT) else {}
    done = json.load(open(OUT)) if os.path.exists(OUT) else {"_meta": {}}
    if args.list or not args.key:
        for k, _ in E.parse_txt():
            tag = "verified" if k in done else ("draft" if k in draft else "-")
            print(f"  {k:24s} {tag}")
        return
    if not args.annotator:
        sys.exit("--annotator is required: verified labels record who verified them")

    cap = cv2.VideoCapture(E.find_video(args.key))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_s = int(n_frames / fps) + 1
    rec = done.get(args.key) or draft.get(args.key) or {}
    state, vis = to_seconds(rec, n_s, fps)
    notes = [(int(a // fps), int(b // fps), txt) for a, b, txt in rec.get("notes", [])]

    t, playing = 0, False
    win = f"label {args.key}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)

    def save():
        out = to_intervals(state, vis, fps, n_frames)
        out["notes"] = rec.get("notes", [])
        out["verified_by"] = args.annotator
        done[args.key] = out
        done["_meta"] = {"guideline": "annotation/GUIDELINE.md",
                         "note": "human-verified; see verified_by per video"}
        json.dump(done, open(OUT, "w"), indent=1)
        print(f"saved {args.key} to {OUT}")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((720, 1280, 3), np.uint8)
        frame = cv2.resize(frame, (1280, 720))
        bar = np.zeros((80, 1280, 3), np.uint8)
        st = state[t]
        cv2.rectangle(bar, (0, 0), (330, 80), COLORS[st], -1)
        cv2.putText(bar, st.upper(), (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.putText(bar, f"{t // 60}:{t % 60:02d}   visible: {'YES' if vis[t] else 'no'}",
                    (350, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        note = next((txt for a, b, txt in notes if a <= t <= b), "")
        if note:
            cv2.putText(frame, "NOTE: " + note[:110], (10, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # timeline strip
        strip = np.zeros((20, 1280, 3), np.uint8)
        for s in range(n_s):
            x0, x1 = int(s * 1280 / n_s), int((s + 1) * 1280 / n_s)
            strip[:, x0:x1] = COLORS[state[s]]
        strip[:, int(t * 1280 / n_s):int(t * 1280 / n_s) + 3] = (255, 255, 255)
        cv2.imshow(win, np.vstack([bar, strip, frame]))

        k = cv2.waitKey(1000 if playing else 0) & 0xFF
        if playing and k == 255:
            t = min(t + 1, n_s - 1)
            continue
        if k == ord(" "):
            playing = not playing
        elif k == ord("d"):
            t = min(t + 1, n_s - 1)
        elif k == ord("a"):
            t = max(t - 1, 0)
        elif k == ord("D"):
            t = min(t + 10, n_s - 1)
        elif k == ord("A"):
            t = max(t - 10, 0)
        elif k in (ord("1"), ord("2"), ord("3"), ord("4")):
            new = STATES[k - ord("1")]
            old = state[t]
            u = t
            while u < n_s and state[u] == old:
                state[u] = new
                u += 1
        elif k == ord("v"):
            old = vis[t]
            u = t
            while u < n_s and vis[u] == old:
                vis[u] = not old
                u += 1
        elif k == ord("n"):
            nxt = [a for a, _, _ in notes if a > t]
            if nxt:
                t = nxt[0]
        elif k == ord("s"):
            save()
        elif k == ord("q"):
            save()
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
