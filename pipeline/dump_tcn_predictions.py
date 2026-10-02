#!/usr/bin/env python3
"""Write the TCN segmenter's per-frame predictions to a cache directory, so it
is scored by exactly the same code as every other system.

Usage:
  python3 dump_tcn_predictions.py --model ~/eval_cache/tcn_journal.pt \
      --eval ~/eval_cache/joint_dump_val --out ~/eval_cache/tcn_val
"""
import argparse
import os

import numpy as np
import torch

import tcn_segmenter as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--stages", type=int, default=3)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    model = T.MSTCN(n_hidden=args.hidden, n_stages=args.stages)
    model.load_state_dict(torch.load(os.path.expanduser(args.model), map_location="cpu"))
    model.eval()

    n = 0
    with torch.no_grad():
        for feats, _labels, name, fidx, n_frames, gt in T.load_streams(args.eval):
            grid = model(torch.from_numpy(feats.T[None]))[-1].argmax(dim=1)[0].numpy()
            pred = np.full(n_frames, "outside", dtype=object)
            for k, fi in enumerate(fidx):
                pred[max(0, fi):] = T.STATES[int(grid[k])]
            m = min(len(pred), len(gt))
            np.savez_compressed(os.path.join(out, name),
                                predicted=np.array([str(x) for x in pred[:m]]),
                                gt=np.array(gt[:m]))
            n += 1
    print(f"wrote {n} videos to {out}")


if __name__ == "__main__":
    main()
