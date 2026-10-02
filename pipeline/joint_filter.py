#!/usr/bin/env python3
"""A joint state estimator over raw evidence from the detector and the world
model, with every parameter calibrated for the pair.

Six earlier attempts to combine these two models all failed, and they failed for
one reason: each grafted one model onto the other's temporal machinery.  The
ported detector counts cycles (MAX_APPROACH_DUR=150 cycles, min_out_frames=20)
tuned for a 43 ms loop; the world model's cascade votes over a window tuned for
its own ~700 ms stride.  Both were calibrated to run alone.  Reusing either as
the spine imports constants that are wrong for the combination, which is the
same inheritance error this paper documents at the threshold level -- one level
up.

So nothing is inherited here.  There is no EMA, no hand-set hysteresis, no
threshold, no cycle counter.  The estimator is:

  state space   outside / approaching / inside / exiting

  evidence      detector: work-zone object count, bucketed
                world model: gate answer, sign-keyword hit, description
                corroboration -- whichever answer is current, with its age

  emission      P(detector bucket, world-model evidence | state), JOINT rather
                than factorised, because the interaction is the signal: the
                world model's APPROACHING is trustworthy when the detector
                corroborates and mostly a false alarm when it does not (73% of
                such frames are OUTSIDE).  A factorised model cannot express
                that; it is exactly what six hand-written rules kept missing.

  transition    a per-SECOND rate matrix, scaled by the elapsed dt of each step,
                estimated from ground-truth transitions.  Hysteresis is then a
                learned self-transition probability rather than a constant in
                cycles, so the same model behaves identically if the loop rate
                changes.

  decision      argmax of the filtered posterior.

Everything above is fitted on the calibration split by counting, and reported on
validation.  Run with the workzone venv (numpy<2 for ultralytics).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

try:  # live runs need the detector; replaying recorded evidence does not
    from ultralytics import YOLO
    import evaluate_yolo_baseline as B
except ImportError:
    YOLO = B = None
from cascade_state_machine import has_corroboration, parse_gate
from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, SIGN_PROMPT, PersistentEngine
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                           TransitionToleranceAccumulator, print_paper_report)

SHM = "/dev/shm/joint_filter.jpg"
STATES = ["outside", "approaching", "inside", "exiting"]
C3E_SIZE = (736, 416)

DET_SCORE_EDGES = [0.1, 0.25, 0.4, 0.55, 0.75]   # the detector's own calibrated
                                             # score, bucketed -- NOT a raw
                                             # object count.  yolo_frame_score
                                             # weights cues by category (cones
                                             # 0.9, workers 0.8, signs 0.7 ...);
                                             # counting objects throws that away
                                             # and was measured to cost 16 points
                                             # of INSIDE precision.
AGE_BUCKETS = [1.5, 4.0]         # seconds since the world-model answer -> 3 buckets
EVIDENCE_TAU = 0.5               # seconds of observation worth one likelihood
                                 # application.  Swept on the calibration split
                                 # under the criterion used elsewhere in this
                                 # work -- maximise macro-F1 subject to a
                                 # false-alarm rate no worse than the deployed
                                 # detector's.  tau=0.25 scores 0.591 but at
                                 # 43.1 alarms/h; 0.5 scores 0.587 at 27.9.


def det_bucket(score):
    for i, e in enumerate(DET_SCORE_EDGES):
        if score < e:
            return i
    return len(DET_SCORE_EDGES)


def age_bucket(age_s):
    for i, e in enumerate(AGE_BUCKETS):
        if age_s < e:
            return i
    return len(AGE_BUCKETS)


def obs_index(det_b, gate, sign, corr, age_b):
    """Flatten the joint observation into one integer.

    The semantic part is ordered by evidence strength: no gate, gate only,
    transcribed sign -- times whether the description corroborated.
    """
    sem = (2 if sign else 1 if gate else 0) * 2 + (1 if corr else 0)   # 0..5
    n_age = len(AGE_BUCKETS) + 1
    return (det_b * 6 + sem) * n_age + age_b


N_OBS = (len(DET_SCORE_EDGES) + 1) * 6 * (len(AGE_BUCKETS) + 1)


class JointFilter:
    """Forward filter with a per-second rate matrix and a joint emission table."""

    def __init__(self, params):
        self.log_emit = np.log(np.asarray(params["emission"]) + 1e-9)   # [S, N_OBS]
        self.rate = np.asarray(params["rate"])                          # [S, S] per second
        self.prior = np.asarray(params["prior"])
        self.belief = self.prior.copy()

    def step(self, obs, dt):
        # transition over dt seconds: first-order expansion of expm(rate * dt),
        # which is exact enough for the millisecond steps this loop runs at
        T = np.eye(len(STATES)) + self.rate * dt
        T = np.clip(T, 1e-9, None)
        T /= T.sum(axis=1, keepdims=True)
        b = self.belief @ T
        # Evidence accrues at a RATE, not once per loop iteration.  Consecutive
        # frames 24 ms apart are near-duplicates, and a world-model answer
        # persists across ~30 of them; multiplying its likelihood once per step
        # treats 30 correlated looks as 30 independent observations, which makes
        # the posterior overconfident and drove the measured false-alarm rate to
        # 135/h.  Scaling the log-likelihood by dt/EVIDENCE_TAU makes the update
        # invariant to the loop rate, which is the same principle as expressing
        # the transition matrix per second.
        b = b * np.exp(self.log_emit[:, obs] * (dt / EVIDENCE_TAU))
        s = b.sum()
        self.belief = b / s if s > 0 else self.prior.copy()
        return STATES[int(np.argmax(self.belief))]


def run(model, engine, video_path, params=None, dump=None):
    """One execution.  With params, filters; with dump, records raw evidence."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predicted = np.full(n_frames, "outside", dtype=object)
    filt = JointFilter(params) if params else None

    frame_idx, sim_clock = -1, 0.0
    pending, free_at = None, 0.0
    gate, sign, corr, sem_time = False, False, False, -1e9
    cycles, polls = [], 0

    while True:
        target = int(sim_clock * fps)
        if target >= n_frames:
            break
        while frame_idx < target:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.retrieve()
        if not ok:
            break

        # world model, asynchronous, polled as fast as its own latency allows
        if pending is not None and sim_clock >= pending[1]:
            gate, sign, corr = pending[0]
            sem_time = pending[1]
            pending = None
        if pending is None and sim_clock >= free_at:
            t_v = time.monotonic()
            cv2.imwrite(SHM, cv2.resize(frame, C3E_SIZE, interpolation=cv2.INTER_AREA))
            g_txt, _ = engine.infer(SHM, GATE_PROMPT, 8)
            g = bool(parse_gate(g_txt or ""))
            s_txt, _ = engine.infer(SHM, SIGN_PROMPT, 15)
            s_hit = has_corroboration("", (s_txt or "").strip())
            d_txt, _ = engine.infer(SHM, DESC_PROMPT, 40)
            c_hit = has_corroboration(d_txt or "", "")
            lat = time.monotonic() - t_v
            pending = ((g, s_hit, c_hit), sim_clock + lat)
            free_at = sim_clock + lat
            polls += 1

        # detector, every cycle
        t0 = time.monotonic()
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]
        counts = {}
        for cid in res.boxes.cls.cpu().tolist():
            cat = B.get_cue_category(model.names[int(cid)])
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        y_s, _total = B.yolo_frame_score(counts)
        dt = time.monotonic() - t0
        cycles.append(dt * 1000.0)

        age = sim_clock - sem_time if sem_time > -1e8 else 1e9
        obs = obs_index(det_bucket(y_s), gate, sign, corr, age_bucket(age))
        if dump is not None:
            # raw components, not the encoded index: bucket edges and the
            # evidence time-constant can then be swept offline without paying
            # for the GPU pass again
            dump.append((frame_idx, sim_clock, dt, y_s,
                         float(gate), float(sign), float(corr), age))
        if filt is not None:
            predicted[max(0, frame_idx):] = filt.step(obs, dt)
        sim_clock += dt

    cap.release()
    return predicted, cycles, polls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    ap.add_argument("--annotations",
                    default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    ap.add_argument("--weights",
                    default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    ap.add_argument("--split-file", default="eval_split_full.json")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--engine-dir", default=os.path.expanduser("~/cosmos3edge-engines"))
    ap.add_argument("--params", default=None, help="fitted parameters (JSON) to filter with")
    ap.add_argument("--dump", default=None, help="record raw evidence for calibration")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    params = None
    if args.params:
        with open(os.path.expanduser(args.params)) as fh:
            params = json.load(fh)
        print(f"filtering with parameters fitted on "
              f"{params.get('n_calibration_videos', '?')} calibration videos")

    videos = json.load(open(args.split_file))[args.split][:args.limit]
    ann = json.load(open(args.annotations))
    model = YOLO(args.weights)
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        args.engine_dir, os.path.join(args.engine_dir, "visual"), temperature=0.0)
    for d in (args.cache_dir, args.dump):
        if d:
            os.makedirs(os.path.expanduser(d), exist_ok=True)

    ev, tm = EventLevelAccumulator(STATES), TimingOffsetAccumulator(STATES)
    tr = TransitionToleranceAccumulator()
    conf, cyc, polls_tot = {}, [], 0
    for i, v in enumerate(videos, 1):
        p = os.path.join(args.videos_dir, v)
        if not os.path.exists(p) or v not in ann:
            continue
        cap = cv2.VideoCapture(p); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        gt = [s.value if hasattr(s, "value") else str(s)
              for s in B.load_ground_truth(ann, v, nf)]
        dump = [] if args.dump else None
        t0 = time.monotonic()
        pred, cycles, polls = run(model, engine, p, params, dump)
        cyc.extend(cycles); polls_tot += polls
        if args.dump:
            np.savez_compressed(os.path.join(os.path.expanduser(args.dump), f"{v}.npz"),
                                obs=np.array(dump, dtype=np.float64),
                                gt=np.array(gt), n_frames=nf)
        if params:
            n = min(len(pred), len(gt))
            ps, gs = [str(pred[j]) for j in range(n)], [str(gt[j]) for j in range(n)]
            for a, b in zip(gs, ps):
                conf[(a, b)] = conf.get((a, b), 0) + 1
            ev.add_video(ps, gs); tm.add_video(ps, gs); tr.add_video(ps, gs)
            if args.cache_dir:
                np.savez_compressed(
                    os.path.join(os.path.expanduser(args.cache_dir), f"{v}.npz"),
                    predicted=np.array(ps), gt=np.array(gs))
        print(f"[{i}/{len(videos)}] {v}  det={len(cycles)} polls={polls} "
              f"({time.monotonic()-t0:.1f}s)", flush=True)

    engine.close()
    c = np.array(cyc)
    print(f"\n=== joint filter ({'filtering' if params else 'dumping evidence'}) ===")
    print(f"  detector cycle p50 {np.percentile(c,50):.0f} ms | {1000/np.mean(c):.1f} Hz")
    print(f"  world-model polls {polls_tot}")
    if params:
        print_paper_report(conf, ev, tm, tr, STATES)


if __name__ == "__main__":
    main()
