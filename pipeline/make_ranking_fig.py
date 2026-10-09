#!/usr/bin/env python3
"""Thesis figure for the IV paper: validation macro-F1 of each system under
three comparison regimes (all values from Table I / Section VI).

  inherited       every VLM runs the 2B's constants, the detector its published ones
  VLMs calibrated the usual comparison: the proposed systems tuned, the baseline as published
  all calibrated  every system recalibrated by the same procedure

    python3 pipeline/make_ranking_fig.py --out paper/iv2027/fig_ranking.pdf
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Times-metric serif (TrueType, embeds cleanly) to match the IEEE body text
for f in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSerif-*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": 0.6})

COLS = ["Constants\ninherited", "VLMs\ncalibrated", "All\ncalibrated"]
SYS = [  # name, values per column (None = not defined), colour, marker, line width
    ("Detector+CLIP", [0.470, 0.470, 0.551], "#1f3a5f", "s", 2.2),
    ("Joint (Det.+C3E)", [None, 0.546, 0.546], "#7a5195", "D", 1.2),
    ("C3E, 4B zero-shot", [0.290, 0.515, 0.515], "#b03a2e", "o", 1.2),
    ("2B fine-tuned", [0.453, 0.459, 0.459], "#d9883b", "^", 1.2),
]
LABEL_DY = {"Detector+CLIP": 4.5, "Joint (Det.+C3E)": -4.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(3.45, 1.75))

    # the usual comparison: proposed systems calibrated, baseline as published
    ax.axvspan(0.62, 1.38, color="#eef1f5", zorder=0, lw=0)
    ax.text(1.0, 0.585, "usual comparison", ha="center", va="top", fontsize=6.5, style="italic", color="#5a6675")

    for name, vals, col, mk, lw in SYS:
        xs = [i for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, color=col, lw=lw, zorder=3 if lw > 2 else 2, solid_capstyle="round")
        ax.scatter(xs, ys, color=col, marker=mk, s=16 if lw > 2 else 11, zorder=4, edgecolors="white", linewidths=0.4)
        ax.annotate(f"{name}  {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(6, LABEL_DY.get(name, 0)),
                    textcoords="offset points", va="center", fontsize=6.5, color=col,
                    fontweight="bold" if lw > 2 else "normal")

    # the detector's rank among the systems in each regime
    navy = SYS[0][2]
    ax.annotate("3rd", (1, 0.470), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6,
                color=navy, fontweight="bold")
    ax.annotate("1st", (2, 0.551), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6,
                color=navy, fontweight="bold")
    ax.annotate("0.290", (0, 0.290), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6,
                color=SYS[2][2])

    ax.set_xticks(range(3))
    ax.set_xticklabels(COLS, fontsize=7)
    ax.set_xlim(-0.25, 3.45)
    ax.set_ylim(0.27, 0.59)
    ax.set_yticks([0.3, 0.4, 0.5])
    ax.set_ylabel("Macro-F1 (validation)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5, length=2, width=0.6)
    ax.tick_params(axis="x", length=0, pad=3)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.15)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
