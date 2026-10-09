#!/usr/bin/env python3
"""Thesis figure for the IV paper: validation macro-F1 of each system under
three comparison regimes (all values from Table I / Section VI).

  inherited       every VLM runs the 2B's constants, the detector its published ones
  VLMs calibrated the usual comparison: the proposed systems tuned, the baseline as published
  all calibrated  every system recalibrated by the same procedure

    python3 pipeline/make_ranking_fig.py --out paper/iv2027/fig_ranking.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["Constants\ninherited", "VLMs\ncalibrated", "All\ncalibrated"]
SYS = [  # name, values per column (None = not defined), colour, marker
    ("Detector+CLIP", [0.470, 0.470, 0.551], "#1a1a1a", "s"),
    ("Joint (Det.+C3E)", [None, 0.546, 0.546], "#7b3294", "D"),
    ("C3E (4B, zero-shot)", [0.290, 0.515, 0.515], "#ca0020", "o"),
    ("2B (fine-tuned)", [0.453, 0.459, 0.459], "#f4a582", "^"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(3.45, 1.6))
    for name, vals, col, mk in SYS:
        xs = [i for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, color=col, marker=mk, ms=4, lw=1.6 if name.startswith("Detector") else 1.1)
        dy = {"Detector+CLIP": 5, "Joint (Det.+C3E)": -5}.get(name, 0)
        ax.annotate(f"{name}  {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(5, dy), textcoords="offset points",
                    va="center", fontsize=6, color=col)
    ax.annotate("0.470", (1, 0.470), xytext=(-4, -7), textcoords="offset points", ha="right", fontsize=5.5)
    ax.annotate("0.290", (0, 0.290), xytext=(5, 0), textcoords="offset points", va="center", fontsize=5.5,
                color="#ca0020")
    ax.set_xticks(range(3))
    ax.set_xticklabels(COLS, fontsize=6.5)
    ax.set_xlim(-0.15, 3.35)
    ax.set_ylim(0.27, 0.58)
    ax.set_ylabel("validation macro-F1", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
