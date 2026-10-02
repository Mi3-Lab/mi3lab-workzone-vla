#!/usr/bin/env python3
"""Causal multi-stage temporal convolutional segmenter over the two models'
evidence streams.

Assigning one of four states to every frame of a video is temporal action
segmentation, and because the alert has to be issued now, it is the streaming
(causal) variant of it -- a task the literature treats as distinct from the
offline one.  The Bayesian filter this replaces is weaker than that literature
in two specific ways:

  its emission is memoryless given the state, so it sees only the current
  observation, whereas a dilated causal stack sees tens of seconds of past;

  its transition is first-order Markov, so it cannot represent a pattern longer
  than one step.

And the failure mode we measured -- 135 false alarms/h from over-segmentation --
is exactly what MS-TCN's smoothing loss was introduced to fix.  We were fighting
it with a time constant we invented instead.

Two things here are not standard.  The stack is strictly CAUSAL (left-padded
convolutions, no future context), because a driver-facing alert cannot look
ahead; most reported action-segmentation numbers are bidirectional and are not
comparable to ours.  And the input is a MULTI-RATE, HETEROGENEOUS stream: the
detector contributes at ~41 Hz and the world model at ~1.4 Hz, so the world
model's channels carry the age of the answer alongside it and are held between
arrivals.  The segmentation literature we found assumes a uniform feature stream
from a single backbone.

Usage:
  python3 tcn_segmenter.py --train ~/eval_cache/joint_dump_cal2 --holdout 20
  python3 tcn_segmenter.py --train ... --eval ~/eval_cache/joint_dump_val
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATES = ["outside", "approaching", "inside", "exiting"]
IDX = {s: i for i, s in enumerate(STATES)}
GRID_HZ = 10.0          # regular grid the stack runs on
N_FEAT = 6


def load_streams(dump_dir):
    """Resample each recorded run onto a regular grid, with its labels."""
    out = []
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(dump_dir), "*.npz"))):
        z = np.load(p, allow_pickle=True)
        rows = z["obs"]
        gt = [str(x) for x in z["gt"]]
        if rows.size == 0:
            continue
        # rows: frame_idx, clock, dt, y_s, gate, sign, corr, age
        clocks = rows[:, 1]
        t_end = float(clocks[-1])
        grid = np.arange(0.0, t_end, 1.0 / GRID_HZ)
        if len(grid) < 8:
            continue
        j = np.searchsorted(clocks, grid, side="right") - 1
        j = np.clip(j, 0, len(rows) - 1)
        sel = rows[j]
        feats = np.stack([
            sel[:, 3],                                   # detector score
            sel[:, 4], sel[:, 5], sel[:, 6],             # gate, sign, corroboration
            np.clip(sel[:, 7], 0.0, 10.0) / 10.0,        # age of that answer
            np.ones(len(grid)),                          # bias
        ], axis=1).astype(np.float32)
        fi = np.clip(sel[:, 0].astype(int), 0, len(gt) - 1)
        labels = np.array([IDX.get(gt[k], 0) for k in fi], dtype=np.int64)
        out.append((feats, labels, os.path.basename(p), sel[:, 0].astype(int),
                    len(gt), gt))
    return out


class CausalStage(nn.Module):
    """One MS-TCN stage: dilated causal residual convolutions."""

    def __init__(self, n_in, n_hidden, n_classes, n_layers=8, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.inp = nn.Conv1d(n_in, n_hidden, 1)
        self.layers = nn.ModuleList()
        self.pads = []
        for i in range(n_layers):
            d = 2 ** i
            self.layers.append(nn.Conv1d(n_hidden, n_hidden, 3, dilation=d))
            self.pads.append(2 * d)          # left-pad only -> strictly causal
        self.res = nn.ModuleList(nn.Conv1d(n_hidden, n_hidden, 1) for _ in range(n_layers))
        self.out = nn.Conv1d(n_hidden, n_classes, 1)

    def forward(self, x):
        h = self.inp(x)
        for conv, res, pad in zip(self.layers, self.res, self.pads):
            y = self.drop(F.relu(conv(F.pad(h, (pad, 0)))))
            h = h + res(y)
        return self.out(h)


class MSTCN(nn.Module):
    def __init__(self, n_in=N_FEAT, n_hidden=32, n_classes=4, n_stages=3, n_layers=8,
                 dropout=0.2):
        super().__init__()
        self.first = CausalStage(n_in, n_hidden, n_classes, n_layers, dropout)
        self.refine = nn.ModuleList(
            CausalStage(n_classes, n_hidden, n_classes, n_layers, dropout)
            for _ in range(n_stages - 1))

    def forward(self, x):
        outs = [self.first(x)]
        for stage in self.refine:
            outs.append(stage(F.softmax(outs[-1], dim=1)))
        return outs


DET_CH = [0]              # detector score
SEM_CH = [1, 2, 3, 4]     # world-model gate, sign, corroboration, age


def modality_dropout(x, p=0.3, rng=None):
    """Zero one modality at random during training.

    The system fuses a detector that fails out of domain (44% at night, 40% in
    fog) with a world model that does not (100%, 80%).  Fitted on in-domain
    data, a fusion model never sees the regime where the detector is blind and
    the world model is reading a sign: that evidence combination covers 124 of
    377,288 calibration frames, and the joint estimator that trained on it
    scored 20% in fog against the world model's 80%.

    Randomly masking a whole modality manufactures that regime during training,
    so the model must learn to act on either stream alone rather than on their
    in-domain co-occurrence.  This is the standard remedy for fusion models
    being brittle to the absence of an expected input stream.
    """
    if rng is None or p <= 0:
        return x
    r = rng.random()
    if r < p / 2:
        x = x.clone(); x[:, DET_CH, :] = 0.0        # detector blind
    elif r < p:
        x = x.clone(); x[:, SEM_CH, :] = 0.0        # world model silent
    return x


def losses(outs, y, lam=0.15, tau=4.0):
    """Cross entropy plus MS-TCN's truncated MSE smoothing term.

    The smoothing term penalises frame-to-frame log-probability jumps, which is
    precisely the over-segmentation that produced our false-alarm rate.
    """
    total = 0.0
    for o in outs:
        total = total + F.cross_entropy(o.transpose(1, 2).reshape(-1, o.shape[1]), y.reshape(-1))
        lsm = F.log_softmax(o, dim=1)
        d = torch.clamp((lsm[:, :, 1:] - lsm[:, :, :-1]).abs(), max=tau) ** 2
        total = total + lam * d.mean()
    return total


def evaluate(model, streams, device):
    from paper_metrics import (EventLevelAccumulator, extract_events, _overlaps,
                               per_state_iou, frame_accuracy, macro_f1)
    model.eval()
    conf, ev = {}, EventLevelAccumulator(STATES)
    fa, frames = 0, 0
    with torch.no_grad():
        for feats, labels, _name, fidx, n_frames, gt in streams:
            x = torch.from_numpy(feats.T[None]).to(device)
            pred_grid = model(x)[-1].argmax(dim=1)[0].cpu().numpy()
            # expand grid predictions back onto the video's frame timeline
            pred = np.full(n_frames, "outside", dtype=object)
            for k, fi in enumerate(fidx):
                pred[max(0, fi):] = STATES[int(pred_grid[k])]
            # Score against the annotation itself, not the grid labels.
            # Rebuilding ground truth by forward-filling the 10 Hz grid would
            # quantise the interval boundaries onto the same timeline the
            # predictions live on, which flatters every boundary metric and
            # makes these numbers incomparable to the joint filter and the
            # cascade, both scored per annotated frame.
            ps = [str(x) for x in pred]
            gs = [str(x) for x in gt]
            frames += len(ps)
            for a, b in zip(gs, ps):
                conf[(a, b)] = conf.get((a, b), 0) + 1
            ev.add_video(ps, gs)
            pa = ["active" if s != "outside" else "outside" for s in ps]
            ga = ["active" if s != "outside" else "outside" for s in gs]
            G = extract_events(ga, "active")
            for e in extract_events(pa, "active"):
                if not any(_overlaps(e, g) for g in G):
                    fa += 1
    io = per_state_iou(conf, STATES)
    r = ev.result()
    # precision is undefined when the model predicts no INSIDE event at all,
    # which happens early in training; report it as zero rather than crash
    prec = r["inside"]["precision"]
    return dict(acc=frame_accuracy(conf, STATES), f1=macro_f1(conf, STATES),
                ap=io["approaching"], ins=io["inside"], ex=io["exiting"],
                p=0.0 if prec is None else prec,
                fa=fa / (frames / 30.0 / 3600.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--holdout", type=int, default=20,
                    help="videos held out of the training split for a sanity check")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--mod-dropout", type=float, default=0.3,
                    help="probability of masking a whole modality per training step")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--cpu", action="store_true",
                    help="train on CPU so a concurrent latency-honest run is not\n"
                         "disturbed; the model is small enough that this is cheap")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    streams = load_streams(args.train)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(streams))
    hold = [streams[i] for i in order[:args.holdout]]
    train = [streams[i] for i in order[args.holdout:]]
    print(f"{len(streams)} streams: {len(train)} train, {len(hold)} held out "
          f"| grid {GRID_HZ:.0f} Hz | device {device}")

    model = MSTCN(n_hidden=args.hidden, n_stages=args.stages,
                  dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"causal MS-TCN: {n_par} parameters, {args.stages} stages\n")

    # Early stopping on the held-out calibration videos.  Training loss keeps
    # falling long after held-out quality turns: at 120 epochs the loss reached
    # 0.44 while held-out macro-F1 fell from 0.607 to 0.547 and the false-alarm
    # rate doubled.  80 training streams do not support 100k parameters, so the
    # epoch is selected by held-out macro-F1 and the best state is kept.
    best = (-1.0, None, 0)
    patience = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        for feats, labels, *_ in train:
            x = torch.from_numpy(feats.T[None]).to(device)
            x = modality_dropout(x, args.mod_dropout, rng)
            y = torch.from_numpy(labels[None]).to(device)
            opt.zero_grad()
            loss = losses(model(x), y, lam=args.lam)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if ep % 5 == 0 or ep == 1:
            m = evaluate(model, hold, device)
            star = ""
            if m["f1"] > best[0]:
                best = (m["f1"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
                patience, star = 0, "  *"
            else:
                patience += 1
            print(f"  epoch {ep:3d}  loss {tot/len(train):7.4f}   held-out "
                  f"acc {m['acc']:5.1%}  macroF1 {m['f1']:.3f}  "
                  f"IoUexit {m['ex']:.3f}  precIN {m['p']:5.1%}  FA/h {m['fa']:5.1f}{star}")
            if patience >= args.patience:
                print(f"  early stop: no held-out gain for {patience} checks")
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"\n>>> best held-out macro-F1 {best[0]:.3f} at epoch {best[2]}")

    if args.save:
        torch.save(model.state_dict(), os.path.expanduser(args.save))
        print(f"\nmodel saved to {args.save}")

    if args.eval:
        ev_streams = load_streams(args.eval)
        m = evaluate(model, ev_streams, device)
        print(f"\n=== evaluation on {len(ev_streams)} videos ===")
        print(f"  acc {m['acc']:.1%}  macroF1 {m['f1']:.3f}  IoUappr {m['ap']:.3f}  "
              f"IoUin {m['ins']:.3f}  IoUexit {m['ex']:.3f}  precIN {m['p']:.1%}  "
              f"FA/h {m['fa']:.1f}")


if __name__ == "__main__":
    main()
