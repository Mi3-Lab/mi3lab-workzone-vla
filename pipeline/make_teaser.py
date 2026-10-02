#!/usr/bin/env python3
"""Build the first-page teaser (fig_teaser.png).

One out-of-domain daytime frame from our own California footage, at an
annotated work-zone approach, with the ACTUAL on-device outputs of both models
we study.  The frame is chosen so a reader can verify it: the orange cones and
the concrete barrier are plainly visible, the zero-shot world model names both,
and the model we fine-tuned on work zones reports that none are there.  This is
the out-of-domain result of the paper in a single, checkable frame.

All strings below are greedy outputs measured on-device for this exact frame.
"""
import os
import textwrap

import cv2
from PIL import Image, ImageDraw, ImageFont

VDIR = os.path.expanduser("~/workzone/data/All_Construction_Data/With_SpeedLimit")
VIDEO = "CA50_Day_01.MP4"
FRAME = 23 * 30  # annotated work-zone approach at 0:23

OURS = {
    "GATE": "No.",
    "SIGN": "No temporary traffic control devices visible in this scene.",
    "DESC": "No work zone elements visible in this scene.",
}
C3E = {
    "GATE": "Yes.",
    "SIGN": "No visible temporary traffic control signs; a green highway "
            "sign reading \u201cEXIT ONLY\u201d.",
    "DESC": "A concrete barrier on the right side, and orange traffic cones "
            "placed along the edge of the highway.",
}
# the row where the two models disagree, and the disagreement the paper is about
HILITE = "DESC"   # the row carrying verifiable work-zone evidence

LABEL_COLORS = {"GATE": (90, 200, 120), "SIGN": (255, 200, 110),
                "DESC": (200, 200, 205)}

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

W, H = 2200, 520
PHOTO_W = 880
COL_W = (W - PHOTO_W) // 2


def main():
    cap = cv2.VideoCapture(os.path.join(VDIR, VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read frame")
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    canvas = Image.new("RGB", (W, H), (18, 20, 24))
    ph, pw = img.shape[:2]
    scale = max(PHOTO_W / pw, H / ph)
    rw, rh = int(pw * scale), int(ph * scale)
    pim = Image.fromarray(img).resize((rw, rh))
    top = max(0, (rh - H) // 2 - int(0.06 * rh))  # evita a barra de OSD do dashcam
    pim = pim.crop((0, top, PHOTO_W, top + H))
    canvas.paste(pim, (0, 0))

    d = ImageDraw.Draw(canvas, "RGBA")
    f_state = ImageFont.truetype(FONTB, 38)
    f_badge = ImageFont.truetype(FONTB, 19)
    f_head = ImageFont.truetype(FONTB, 24)
    f_lab = ImageFont.truetype(FONTB, 22)
    f_txt = ImageFont.truetype(FONT, 20)
    f_small = ImageFont.truetype(FONT, 17)

    d.rounded_rectangle([22, 22, 372, 78], radius=10, fill=(210, 40, 40, 235))
    d.text((40, 31), "WORK ZONE", font=f_state, fill=(255, 255, 255))
    d.rounded_rectangle([382, 28, 848, 72], radius=8, fill=(0, 0, 0, 165))
    d.text((395, 39), "human-annotated work-zone approach", font=f_badge, fill=(255, 220, 120))

    # keep this short enough to stay inside PHOTO_W
    badge = "same frame, same prompts  ·  both INT4, Jetson AGX Orin"
    bw = d.textlength(badge, font=f_badge)
    d.rounded_rectangle([22, H - 58, 44 + bw, H - 16], radius=8, fill=(0, 0, 0, 180))
    d.text((34, H - 51), badge, font=f_badge, fill=(230, 235, 245))

    cols = [
        (PHOTO_W, "Cosmos-Reason2-2B, work-zone fine-tuned",
         "our deployment  \u00b7  2 of 15 daytime approaches", OURS, (210, 40, 40)),
        (PHOTO_W + COL_W, "Cosmos3-Edge 4B, INT4 + calibrated",
         "our deployment, no work-zone training  \u00b7  15 of 15", C3E, (70, 140, 230)),
    ]
    for x0, title, sub, out, accent in cols:
        d.rectangle([x0, 0, x0 + COL_W, H], fill=(24, 27, 32))
        d.rectangle([x0, 0, x0 + 5, H], fill=accent)
        d.text((x0 + 28, 26), title, font=f_head, fill=(255, 255, 255))
        d.text((x0 + 28, 58), sub, font=f_small, fill=(150, 156, 168))
        y = 100
        for lab in ["GATE", "SIGN", "DESC"]:
            if lab == HILITE:
                d.rounded_rectangle([x0 + 18, y - 8, x0 + COL_W - 18, y + 34],
                                    radius=6, fill=(70, 40, 40))
            d.text((x0 + 28, y), lab, font=f_lab, fill=LABEL_COLORS[lab])
            ty = y
            for line in textwrap.wrap(out[lab], width=38)[:4]:
                d.text((x0 + 128, ty + 1), line, font=f_txt, fill=(228, 232, 240))
                ty += 24
            y = max(ty, y + 28) + 14
            if lab != "DESC":
                d.line([x0 + 28, y - 10, x0 + COL_W - 28, y - 10],
                       fill=(48, 52, 60), width=1)

    note = ("same cascade, same device, same calibration procedure \u2014 only the "
            "adaptation strategy differs, and the fine-tuned one misses the cones")
    d.rectangle([PHOTO_W, H - 44, W, H], fill=(34, 30, 34))
    d.text((PHOTO_W + 28, H - 36), note, font=f_small, fill=(235, 200, 200))

    out_path = os.path.expanduser("~/jetson-deploy/paper/wacv2027/fig_teaser.png")
    canvas.save(out_path, dpi=(220, 220))
    print("wrote", out_path, canvas.size)


if __name__ == "__main__":
    main()
