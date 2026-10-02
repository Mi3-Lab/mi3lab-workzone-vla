#!/usr/bin/env python3
"""Contact sheets for annotating a video: one frame every STEP seconds, 3x3 per
sheet, timestamp burned in large.  Used for the draft pass of the California
four-state labels; the draft is then verified by a person with label_tool.py.

Usage:
  python3 contact_sheets.py VIDEO_KEY --step 2 --out sheets/
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/jetson-deploy/pipeline"))
import eval_external_cosmos3 as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--out", default="sheets")
    ap.add_argument("--tile", default="640x360")
    args = ap.parse_args()
    w, h = map(int, args.tile.split("x"))
    path = E.find_video(args.key)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    out = os.path.join(args.out, args.key)
    os.makedirs(out, exist_ok=True)
    times = np.arange(0, dur, args.step)
    for s in range(0, len(times), 9):
        tiles = []
        for t in times[s:s + 9]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, f = cap.read()
            f = cv2.resize(f, (w, h)) if ok else np.zeros((h, w, 3), np.uint8)
            label = f"{int(t)//60}:{int(t)%60:02d} ({t:.0f}s)"
            cv2.rectangle(f, (0, 0), (230, 40), (0, 0, 0), -1)
            cv2.putText(f, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            tiles.append(f)
        while len(tiles) < 9:
            tiles.append(np.zeros((h, w, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])
        cv2.imwrite(os.path.join(out, f"sheet_{s // 9:03d}.jpg"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()
    print(f"{args.key}: {len(times)} frames, {(len(times) + 8) // 9} sheets, {dur:.0f}s at {fps:.2f} fps")


if __name__ == "__main__":
    main()
