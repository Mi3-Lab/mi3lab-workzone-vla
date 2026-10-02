#!/usr/bin/env python3
"""Watchable review video: the STATE each temporal layer decides, next to the
ground truth, with the raw per-second evidence underneath.

Top to bottom:
  1. predicted state from the world model's cascade (always)
  2. predicted state from the joint estimator (only when the record has the
     detector column -- California yes, San Francisco no)
  3. ground truth state (only for videos with labels)
  4. raw evidence: GATE answer (red = yes, green = no), SIGN and DESC text
  5. the frame
  6. two timelines over the WHOLE video (predicted vs ground truth) with a
     cursor, so the full pattern is visible at a glance

States are computed by the same replay functions that produce the paper's
tables (california_4state.run_cascade / make_joint), stepped once per recorded
second, causally, and held between steps as the live system holds its state.

Colors: outside green, approaching orange, inside red, exiting magenta.

Usage:
  python3 make_overlay_video.py --stream ~/eval_cache/ood_stream \
      --video-dir ~/workzone/data/All_Construction_Data --key CA99_Day_02 \
      --labels ~/jetson-deploy/annotation/california_draft.json --out DIR
  python3 make_overlay_video.py --stream ~/eval_cache/sf_stream \
      --video-dir ~/workzone/data/videos --key san_francisco_..._NNNNN --out DIR
"""
import argparse
import json
import os
import textwrap

import cv2
import numpy as np

import california_4state as C

STATES = ["outside", "approaching", "inside", "exiting"]
COLOR = {"outside": (0, 160, 0), "approaching": (0, 140, 255),
         "inside": (0, 0, 220), "exiting": (200, 0, 200)}   # BGR


def find_video(video_dir, key):
    for root, _dirs, files in os.walk(video_dir):
        for name in (key + ".mp4", key + ".MP4", key + "_snippet.mp4"):
            if name in files:
                return os.path.join(root, name)
    return None


def evidence(npz):
    o = npz["obs"]
    if o.shape[1] == 5:     # t, det, gate, sign, corr  (California record)
        return dict(t=o[:, 0].astype(int), det=o[:, 1], gate=o[:, 2] > .5,
                    sign=o[:, 3] > .5, corr=o[:, 4] > .5), True
    return dict(t=o[:, 0].astype(int), det=np.zeros(len(o)), gate=o[:, 1] > .5,
                sign=o[:, 2] > .5, corr=o[:, 3] > .5), False   # t, gate, sign, corr


def per_second(t_arr, states, n_s):
    """Hold each computed state from its second until the next update."""
    out = ["outside"] * n_s
    cur = "outside"
    j = 0
    for t in range(n_s):
        while j < len(t_arr) and t_arr[j] <= t:
            cur = states[j]; j += 1
        out[t] = cur
    return out


def gt_per_second(rec, fps, n_s):
    out = ["outside"] * n_s
    for st in STATES:
        for a, b in rec.get(st, []):
            for t in range(int(a // fps), min(int(b // fps) + 1, n_s)):
                out[t] = st
    return out


def chip(canvas, y, h, label, state, width):
    cv2.rectangle(canvas, (0, y), (width, y + h), COLOR[state], -1)
    cv2.putText(canvas, f"{label}: {state.upper()}", (10, y + h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def timeline(canvas, y, h, seq, cursor, width, label):
    n = len(seq)
    for i, st in enumerate(seq):
        x0, x1 = int(i * width / n), int((i + 1) * width / n)
        canvas[y:y + h, x0:x1] = COLOR[st]
    x = int(cursor * width / n)
    canvas[y:y + h, max(0, x - 1):x + 2] = (255, 255, 255)
    cv2.putText(canvas, label, (6, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--extra", action="append", default=[],
                    help="DIR:LABEL of another model's record; its raw GATE answer is shown "
                         "as an extra row (e.g. ~/eval_cache/qwendrive_ood_gate:Qwen-Drive)")
    args = ap.parse_args()

    stream = os.path.expanduser(args.stream)
    z = np.load(os.path.join(stream, f"{args.key}.npz"), allow_pickle=True)
    texts = {r["t"]: r for r in json.load(open(os.path.join(stream, f"{args.key}.json")))}
    s, has_det = evidence(z)

    vpath = find_video(os.path.expanduser(args.video_dir), args.key)
    if not vpath:
        raise SystemExit(f"video not found for {args.key}")
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_s = int(n_frames / fps) + 1

    pred_casc = per_second(s["t"], C.run_cascade(s), n_s)
    pred_joint = None
    if has_det:
        P = json.load(open(os.path.expanduser("~/eval_cache/joint_params_full.json")))
        pred_joint = per_second(s["t"], C.make_joint(P)(s), n_s)
    gt = None
    if args.labels:
        L = json.load(open(os.path.expanduser(args.labels)))
        if args.key in L:
            gt = gt_per_second(L[args.key], fps, n_s)

    extras = []
    for spec in args.extra:
        d, label = spec.rsplit(":", 1)
        jp = os.path.join(os.path.expanduser(d), f"{args.key}.json")
        if os.path.exists(jp):
            extras.append((label, {r["t"]: r for r in json.load(open(jp))}))

    ow = 960
    oh = int(H * ow / W)
    chip_h, ev_h, tl_h = 34, 74, 16
    rows_top = 1 + (pred_joint is not None) + (gt is not None)
    top = rows_top * chip_h + ev_h + len(extras) * 26
    bottom = tl_h * (1 + (pred_joint is not None) + (gt is not None))
    os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    dst = os.path.join(os.path.expanduser(args.out), f"{args.key}.mp4")
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, top + oh + bottom))

    cur = texts.get(0, {"gate": "", "sign": "", "desc": ""})
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = min(int(fidx / fps), n_s - 1)
        if t in texts:
            cur = texts[t]
        canvas = np.zeros((top + oh + bottom, ow, 3), np.uint8)

        y = 0
        chip(canvas, y, chip_h, "C3E cascade", pred_casc[t], ow); y += chip_h
        if pred_joint is not None:
            chip(canvas, y, chip_h, "Joint estimator", pred_joint[t], ow); y += chip_h
        if gt is not None:
            chip(canvas, y, chip_h, "GROUND TRUTH (draft)", gt[t], ow); y += chip_h

        g_txt = (cur.get("gate") or "").replace("<|im_end|>", "").strip()
        yes = g_txt.lower().startswith("yes")
        cv2.rectangle(canvas, (0, y), (ow, y + ev_h), (0, 0, 150) if yes else (0, 100, 0), -1)
        cv2.putText(canvas, f"C3E GATE: {g_txt[:30]}", (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        sign = (cur.get("sign") or "").replace("\n", " ").strip()
        desc = (cur.get("desc") or "").replace("\n", " ").strip()
        cv2.putText(canvas, ("SIGN: " + sign)[:110], (10, y + 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1)
        cv2.putText(canvas, ("DESC: " + desc)[:110], (10, y + 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1)
        cv2.putText(canvas, f"t={t}s", (ow - 80, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 1)
        y += ev_h
        for label, ex in extras:
            r = ex.get(t) or ex.get(max([k for k in ex if k <= t], default=-1)) or {}
            txt = (r.get("gate") or "").replace("<|im_end|>", "").strip()
            yes2 = txt.lower().startswith("yes")
            cv2.rectangle(canvas, (0, y), (ow, y + 26), (0, 0, 150) if yes2 else (0, 100, 0), -1)
            cv2.putText(canvas, f"{label} GATE: {txt[:40]}", (10, y + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 26

        canvas[y:y + oh] = cv2.resize(frame, (ow, oh)); y += oh

        timeline(canvas, y, tl_h, pred_casc, t, ow, "cascade"); y += tl_h
        if pred_joint is not None:
            timeline(canvas, y, tl_h, pred_joint, t, ow, "joint"); y += tl_h
        if gt is not None:
            timeline(canvas, y, tl_h, gt, t, ow, "truth"); y += tl_h

        vw.write(canvas)
        fidx += 1
    cap.release(); vw.release()
    print(f"wrote {dst}  ({fidx / fps:.0f}s)")


if __name__ == "__main__":
    main()
