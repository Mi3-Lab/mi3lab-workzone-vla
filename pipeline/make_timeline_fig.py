#!/usr/bin/env python3
"""Qualitative timeline for the IV paper: ground truth and each system's state
over one validation video, with thumbnails at one frame per annotated state.

    python3 pipeline/make_timeline_fig.py boston_4d91f8efe4d44a4694ecb0e440c74dc4_000001_21120_snippet.mp4 \
        --out paper/iv2027/fig_timeline.pdf
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

H = os.path.expanduser
ST = ["outside", "approaching", "inside", "exiting"]
COL = {"outside": "#d9d9d9", "approaching": "#f4a582", "inside": "#ca0020", "exiting": "#92c5de"}
ROWS = [("Ground truth", None),
        ("Det.+CLIP, hand-tuned", "~/eval_cache/yoloclip"),
        ("Det.+CLIP, recalibrated", "~/eval_cache/detclip_cal_val"),
        ("C3E, 2B constants", "~/eval_cache/cosmos3edge"),
        ("C3E, recalibrated", "~/eval_cache/cosmos3edge_cal"),
        ("Joint (Det.+C3E)", "~/eval_cache/joint_full_val")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seqs = {}
    for name, cache in ROWS[1:]:
        z = np.load(H(f"{cache}/{args.video}.npz"), allow_pickle=True)
        seqs[name] = [str(x) for x in z["predicted"]]
        gt = [str(x) for x in z["gt"]]
    seqs["Ground truth"] = gt
    n = min(len(s) for s in seqs.values())
    fps = 30.0

    # one thumbnail at the middle of each annotated state's first interval
    picks = []
    for st in ST:
        idx = [i for i in range(n) if gt[i] == st]
        if idx:
            run = [idx[0]]
            for i in idx[1:]:
                if i != run[-1] + 1:
                    break
                run.append(i)
            picks.append((run[len(run) // 2], st))
    picks.sort()
    cap = cv2.VideoCapture(H(f"~/workzone/data/videos/{args.video}"))
    thumbs = []
    for f, st in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        thumbs.append((f, st, cv2.cvtColor(cv2.resize(im, (320, 180)), cv2.COLOR_BGR2RGB)))
    cap.release()

    fig = plt.figure(figsize=(3.45, 2.05))
    gs = fig.add_gridspec(2, len(thumbs), height_ratios=[0.9, 2.2], hspace=0.12, wspace=0.04,
                          left=0.32, right=0.99, top=0.99, bottom=0.17)
    for k, (f, st, im) in enumerate(thumbs):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(COL[st]); sp.set_linewidth(2)
        ax.set_title(f"{f / fps:.0f}s", fontsize=6, pad=1)
    ax = fig.add_subplot(gs[1, :])
    for r, (name, _) in enumerate(ROWS):
        s = seqs[name][:n]
        y = len(ROWS) - 1 - r
        start = 0
        for i in range(1, n + 1):
            if i == n or s[i] != s[start]:
                ax.add_patch(plt.Rectangle((start / fps, y + 0.12), (i - start) / fps, 0.76,
                                           color=COL[s[start]], lw=0))
                start = i
    for f, st, _ in thumbs:
        ax.axvline(f / fps, color="k", lw=0.4, ls=":")
    ax.set_xlim(0, n / fps); ax.set_ylim(0, len(ROWS))
    ax.set_yticks([len(ROWS) - 1 - r + 0.5 for r in range(len(ROWS))])
    ax.set_yticklabels([n_ for n_, _ in ROWS], fontsize=6)
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.set_xlabel("time (s)", fontsize=6, labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(handles=[Patch(color=COL[s], label=s) for s in ST], loc="lower left",
               ncol=4, fontsize=5.5, frameon=False, bbox_to_anchor=(0.0, -0.01), handlelength=1)
    fig.savefig(args.out, dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
