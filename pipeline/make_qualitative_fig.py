#!/usr/bin/env python3
"""Regenerate the paper's qualitative 4-panel figure (fig_qualitative.png) from
a single Seattle work-zone traversal, with the REAL per-channel VLM outputs
below each frame. Picks frames whose ground-truth state is OUTSIDE /
APPROACHING / INSIDE / EXITING and where the work zone is clearly visible.

Usage: python3 make_qualitative_fig.py
"""
import os
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from live_cascade_demo import (PersistentEngine, GATE_PROMPT, SIGN_PROMPT,
                               EGO_PROMPT, DESC_PROMPT, SHM_FRAME,
                               DEFAULT_ENGINE_DIR, DEFAULT_MM_ENGINE_DIR,
                               DEFAULT_PLUGIN, DEFAULT_BINARY)

VDIR = os.path.expanduser("~/workzone/data/videos")
VIDEO = "seattle_83d7f710e54b48f5a7730576be90ddfc_000001_11100_snippet.mp4"
# ground truth: outside 0-120, approaching 121-240, inside 241-810, exiting 811-850
PANELS = [
    ("OUTSIDE",     60,  (48, 168, 96)),    # RGB green
    ("APPROACHING", 180, (240, 200, 0)),    # yellow
    ("INSIDE",      520, (210, 40, 40)),    # red
    ("EXITING",     830, (240, 140, 0)),    # orange
]

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INFER_W = 480
PANEL_W = 640          # rendered panel width
IMG_H = 360
BANNER_H = 40
TEXT_H = 210
GAP = 6

LABEL_COLORS = {  # RGB, matching the channel accent colors
    "GATE": (40, 150, 60),
    "SIGN": (30, 110, 210),
    "EGO":  (150, 60, 170),
    "DESC": (60, 60, 60),
}


def run_channels(engine, frame):
    small = frame
    if frame.shape[1] > INFER_W:
        h = int(frame.shape[0] * INFER_W / frame.shape[1])
        small = cv2.resize(frame, (INFER_W, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(SHM_FRAME, small)
    out = {}
    out["GATE"], _ = engine.infer(SHM_FRAME, GATE_PROMPT, 3)
    out["SIGN"], _ = engine.infer(SHM_FRAME, SIGN_PROMPT, 15)
    out["EGO"], _ = engine.infer(SHM_FRAME, EGO_PROMPT, 8)
    out["DESC"], _ = engine.infer(SHM_FRAME, DESC_PROMPT, 40)
    return {k: (v or "").strip() for k, v in out.items()}


def desc_prefix(txt):
    return txt.split("Scene:", 1)[0].strip()


def build_panel(frame, state, fnum, color, outputs):
    fb = ImageFont.truetype(FONT_B, 22)
    fbb = ImageFont.truetype(FONT_B, 16)
    ft = ImageFont.truetype(FONT, 15)
    H = BANNER_H + IMG_H + TEXT_H
    panel = Image.new("RGB", (PANEL_W, H), (255, 255, 255))
    d = ImageDraw.Draw(panel)

    # banner
    d.rectangle([0, 0, PANEL_W, BANNER_H], fill=(0, 0, 0))
    d.rectangle([0, 0, 250, BANNER_H], fill=color)
    d.text((12, 8), state, font=fb, fill=(255, 255, 255))
    d.text((270, 10), f"frame {fnum}", font=fbb, fill=color)

    # image
    ar = frame.shape[1] / frame.shape[0]
    iw, ih = PANEL_W, int(PANEL_W / ar)
    if ih > IMG_H:
        ih = IMG_H; iw = int(IMG_H * ar)
    img = cv2.cvtColor(cv2.resize(frame, (iw, ih)), cv2.COLOR_BGR2RGB)
    xoff = (PANEL_W - iw) // 2
    panel.paste(Image.fromarray(img), (xoff, BANNER_H))

    # outputs
    y = BANNER_H + IMG_H + 8
    disp = {
        "GATE": outputs["GATE"],
        "SIGN": outputs["SIGN"],
        "EGO":  outputs["EGO"],
        "DESC": desc_prefix(outputs["DESC"]),
    }
    for lab in ["GATE", "SIGN", "EGO", "DESC"]:
        d.text((10, y), lab + ":", font=fbb, fill=LABEL_COLORS[lab])
        wrapped = textwrap.wrap(disp[lab], width=58) or [""]
        for i, line in enumerate(wrapped[:3]):
            d.text((70, y + i * 18), line, font=ft, fill=(20, 20, 20))
        y += max(len(wrapped[:3]), 1) * 18 + 8
    return panel


def main():
    engine = PersistentEngine(DEFAULT_BINARY, DEFAULT_PLUGIN, DEFAULT_ENGINE_DIR,
                              DEFAULT_MM_ENGINE_DIR, temperature=0.0)
    cap = cv2.VideoCapture(os.path.join(VDIR, VIDEO))
    panels = []
    for state, fnum, color in PANELS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"cannot read frame {fnum}")
        outs = run_channels(engine, frame)
        print(f"\n=== {state} (frame {fnum}) ===")
        for k, v in outs.items():
            print(f"  {k}: {v}")
        panels.append(build_panel(frame, state, fnum, color, outs))
    cap.release()
    engine.close()

    H = max(p.height for p in panels)
    strip = Image.new("RGB", (sum(p.width for p in panels) + GAP * (len(panels) - 1), H),
                      (255, 255, 255))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width + GAP
    out = os.path.expanduser("~/jetson-deploy/paper/wacv2027/fig_qualitative.png")
    strip.save(out, dpi=(200, 200))
    print(f"\nwrote {out}  ({strip.width}x{strip.height})")


if __name__ == "__main__":
    main()
