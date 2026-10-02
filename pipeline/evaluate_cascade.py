#!/usr/bin/env python3
"""Offline calibration/validation of the cascade against human ground truth.

Ground truth lives at ~/workzone/data/workzone_annotations_full.json: for each
video, frame-index intervals [start, end] (30fps) for each of the 4 states
(outside/approaching/inside/exiting) — hand-labeled by the ROADWork/CMU team,
same 4 states our WZState enum uses.

Runs the SAME cascade logic as live_cascade_demo.py (GATE+SIGN every cycle in
OUTSIDE, EGO in active mode, DESC/SIGN corroboration, fast_entry bypass) but
headless and driven by a SIMULATED camera clock instead of real wall-clock
sleep: each cycle's actual processing latency determines how many video
frames to skip forward (grab-without-decode), exactly mirroring the
camera-sim frame-dropping in the live demo, but running as fast as the
hardware allows (no sleeping) so a batch of videos can be evaluated quickly.

For every video frame, the last predicted state (held constant between
inference cycles, matching what a viewer watching the live overlay would
see) is compared against the ground-truth state at that frame to compute
per-frame accuracy, a confusion matrix, and per-transition detection lag.

Usage:
    python3 pipeline/evaluate_cascade.py --videos-dir ~/workzone/data/videos \\
        --annotations ~/workzone/data/workzone_annotations_full.json \\
        --limit 10
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from cascade_state_machine import CascadeStateMachine, WZState, has_corroboration, parse_gate
from live_cascade_demo import (DESC_PROMPT, EGO_PROMPT, GATE_PROMPT, SIGN_PROMPT,
                                PersistentEngine)
from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                            TransitionToleranceAccumulator, print_paper_report)

SHM_FRAME = "/dev/shm/eval_frame.jpg"
STATE_FROM_STR = {s.value: s for s in WZState}
STATES = ["outside", "approaching", "inside", "exiting"]


def load_ground_truth(annotations, video_name, n_frames):
    """Expands {"approaching": [[151,228]], ...} into a per-frame state array."""
    gt = np.full(n_frames, WZState.OUTSIDE, dtype=object)
    spans = annotations[video_name]
    for state_str, intervals in spans.items():
        state = STATE_FROM_STR[state_str]
        for start, end in intervals:
            gt[start:min(end + 1, n_frames)] = state
    return gt


BINARY_SIGN_PROMPT = ("Is there a temporary road work or traffic control sign visible in "
                       "this scene (e.g. 'ROAD WORK', 'DETOUR', 'LANE CLOSED')? Answer Yes or No.")


def _corr_no_filter(desc_txt, sign_txt):
    """Ablation variant of has_corroboration WITHOUT the qualifier lexicon
    (sidewalk/parked): any work-zone object mention in the object-location
    prefix corroborates."""
    import re as _re
    from cascade_state_machine import WZ_OBJECT_RE, SIGN_WZ_KEYWORDS
    if desc_txt:
        prefix = desc_txt.split("Scene:", 1)[0]
        if WZ_OBJECT_RE.search(prefix):
            return True
    if sign_txt and any(k in sign_txt.upper() for k in SIGN_WZ_KEYWORDS):
        return True
    return False


def run_cascade_on_video(engine, video_path, infer_width=480, desc_max_len=40,
                          desc_refresh_s=1.5, verbose=False,
                          no_sign=False, binary_sign=False, no_fast_entry=False,
                          no_qualifier_filter=False, unrestricted_bayes=False,
                          infer_size=None, csm_cfg=None):
    """Headless cascade run with a simulated (not real-time) camera clock.

    Ablation switches (reviewer P2): no_sign disables the SIGN channel in
    OUTSIDE; binary_sign replaces transcription with an equal-role yes/no
    sign question; no_fast_entry forces the voting door only;
    no_qualifier_filter drops the sidewalk/parked lexicon;
    unrestricted_bayes restores the original 4-state argmax.

    Returns (predicted_per_frame: np.ndarray[WZState], transitions: list of
    (frame_idx, from_state, to_state)).
    """
    corr = _corr_no_filter if no_qualifier_filter else has_corroboration
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # csm_cfg carries per-model calibrated thresholds (calibrate_cascade.py).
    # The published defaults were fixed on the calibration split for our 2B;
    # inheriting them for another model would measure the thresholds, not it.
    csm = CascadeStateMachine(restrict_active=not unrestricted_bayes,
                              **(csm_cfg or {}))
    predicted = np.full(n_frames, WZState.OUTSIDE, dtype=object)
    transitions = []

    frame_idx = -1
    last_desc_time = -999.0
    sim_clock = 0.0  # seconds of simulated video time elapsed

    while True:
        target_idx = int(sim_clock * fps)
        if target_idx >= n_frames:
            break
        while frame_idx < target_idx:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.retrieve()
        if not ok:
            break

        t0 = time.monotonic()
        if infer_size is not None:
            # Fixed input size: the Cosmos3-Edge visual tower is frozen at a
            # 26x46 patch grid (736x416), so proportional scaling won't do.
            frame = cv2.resize(frame, infer_size, interpolation=cv2.INTER_AREA)
        elif infer_width > 0 and frame.shape[1] > infer_width:
            h = int(frame.shape[0] * infer_width / frame.shape[1])
            frame = cv2.resize(frame, (infer_width, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(SHM_FRAME, frame)

        prev_state = csm.state
        gate_txt, _ = engine.infer(SHM_FRAME, GATE_PROMPT, 3)
        gate = parse_gate(gate_txt or "")

        desc_txt, sign_txt, ego_txt = "", "", ""
        candidate = bool(gate)
        fast_entry = False

        if csm.state == WZState.OUTSIDE:
            sign_detected = False
            if binary_sign:
                sg_txt, _ = engine.infer(SHM_FRAME, BINARY_SIGN_PROMPT, 3)
                sign_detected = bool(parse_gate(sg_txt or ""))
            elif not no_sign:
                sign_txt, _ = engine.infer(SHM_FRAME, SIGN_PROMPT, 15)
                sign_txt = sign_txt or ""
                sign_detected = corr("", sign_txt)
            candidate = bool(gate) or sign_detected
            if candidate:
                desc_txt, _ = engine.infer(SHM_FRAME, DESC_PROMPT, desc_max_len)
                desc_txt = desc_txt or ""
                last_desc_time = sim_clock
                if not no_fast_entry:
                    fast_entry = sign_detected or (bool(gate) and corr(desc_txt, ""))
        else:
            ego_txt, _ = engine.infer(SHM_FRAME, EGO_PROMPT, 8)
            ego_txt = ego_txt or ""
            if sim_clock - last_desc_time >= desc_refresh_s:
                desc_txt, _ = engine.infer(SHM_FRAME, DESC_PROMPT, desc_max_len)
                desc_txt = desc_txt or ""
                last_desc_time = sim_clock

        state = csm.update(candidate, ego_txt, desc_txt, sign_txt, fast_entry=fast_entry)
        if state != prev_state:
            transitions.append((frame_idx, prev_state, state))

        if verbose:
            print(f"  frame={frame_idx:4d} t={sim_clock:6.1f}s state={state.value:11s} "
                  f"gate={gate_txt!r} sign={sign_txt[:50]!r} ego={ego_txt[:30]!r} "
                  f"desc={desc_txt[:60]!r}", flush=True)

        # Hold this prediction constant until the next sample.
        predicted[max(0, frame_idx):] = state

        elapsed_s = time.monotonic() - t0
        sim_clock += elapsed_s  # camera keeps moving while we "think"

    cap.release()
    return predicted, transitions


def evaluate(predicted, ground_truth):
    n = min(len(predicted), len(ground_truth))
    correct = sum(predicted[i] == ground_truth[i] for i in range(n))
    confusion = {}
    for i in range(n):
        key = (ground_truth[i].value, predicted[i].value)
        confusion[key] = confusion.get(key, 0) + 1
    return correct / n, confusion


def transition_lag(transitions, ground_truth):
    """For each GT state onset, find the first frame our system predicted
    that state at or after it, and report the lag in frames (None if never)."""
    n = len(ground_truth)
    onsets = {}
    for i in range(1, n):
        if ground_truth[i] != ground_truth[i - 1]:
            onsets.setdefault(ground_truth[i], []).append(i)
    results = []
    for state, idxs in onsets.items():
        for onset in idxs:
            candidates = [f for f, _, to in transitions if to == state and f >= onset - 30]
            after = [f for f in candidates if f >= onset]
            if after:
                results.append((state.value, onset, min(after) - onset))
            else:
                results.append((state.value, onset, None))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    parser.add_argument("--annotations",
                         default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    parser.add_argument("--engine-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines", "llm"))
    # Default derives from --engine-dir so that pointing the reasoner at
    # another model cannot silently keep our 2B's visual tower: mismatched
    # towers raise no error, the model just answers fluently and almost
    # identically for every frame (measured: GATE='No.' on 86/86 cycles).
    parser.add_argument("--multimodal-engine-dir", default=None)
    parser.add_argument("--plugin-path",
                         default=os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"))
    parser.add_argument("--binary", default=os.path.expanduser(
        "~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"))
    parser.add_argument("--videos", nargs="*", help="Specific video filenames to evaluate")
    parser.add_argument("--split-file", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_split.json"),
        help="JSON with {'calibration': [...], 'validation': [...]} video lists")
    parser.add_argument("--split", choices=["calibration", "validation"],
                         help="Use the named list from --split-file (takes precedence "
                              "over --category/--limit)")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--category", choices=["all4", "outside_only", "any"], default="any")
    parser.add_argument("--verbose", action="store_true", help="Print per-cycle model outputs")
    parser.add_argument("--cache-dir", default=None,
                         help="If set, save each video's per-frame predicted/gt state arrays "
                              "here (.npz) so paper-exact metrics can be recomputed later "
                              "without re-running inference")
    parser.add_argument("--csm", default=None,
                        help="calibrated state-machine thresholds, "
                             "e.g. N=3,k_enter=3,k_exit=2,k_fading=1,use_ego=0")
    parser.add_argument("--infer-size", default=None,
                        help="Fixed WxH input, e.g. 736x416 for the Cosmos3-Edge "
                             "visual tower (frozen 26x46 patch grid).")
    parser.add_argument("--temperature", type=float, default=0.4,
                         help="Sampling temperature (0 = greedy/deterministic)")
    parser.add_argument("--no-sign", action="store_true")
    parser.add_argument("--binary-sign", action="store_true")
    parser.add_argument("--no-fast-entry", action="store_true")
    parser.add_argument("--no-qualifier-filter", action="store_true")
    parser.add_argument("--unrestricted-bayes", action="store_true")
    args = parser.parse_args()
    if args.multimodal_engine_dir is None:
        sibling = os.path.join(os.path.dirname(os.path.abspath(args.engine_dir)), "visual")
        nested = os.path.join(args.engine_dir, "visual")
        args.multimodal_engine_dir = nested if os.path.isdir(nested) else sibling
    print(f"reasoner: {args.engine_dir}\ntorre visual: {args.multimodal_engine_dir}")

    csm_cfg = None
    if args.csm:
        csm_cfg = {}
        for kv in args.csm.split(","):
            k, v = kv.split("=")
            csm_cfg[k.strip()] = bool(int(v)) if k.strip().startswith("use_") else int(v)
        print(f"state machine calibrada: {csm_cfg}")

    infer_size = None
    if args.infer_size:
        w, h = args.infer_size.lower().split('x')
        infer_size = (int(w), int(h))

    annotations = json.load(open(args.annotations))

    if args.videos:
        video_list = args.videos
    elif args.split:
        video_list = json.load(open(args.split_file))[args.split]
    else:
        if args.category == "all4":
            pool = [k for k, v in annotations.items()
                    if all(v.get(s) for s in ["outside", "approaching", "inside", "exiting"])]
        elif args.category == "outside_only":
            pool = [k for k, v in annotations.items()
                    if v.get("outside") and not any(v.get(s) for s in
                                                     ["approaching", "inside", "exiting"])]
        else:
            pool = list(annotations.keys())
        video_list = pool[:args.limit]

    print(f"Evaluating {len(video_list)} video(s)...")
    engine = PersistentEngine(args.binary, args.plugin_path, args.engine_dir,
                              args.multimodal_engine_dir, temperature=args.temperature)

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    all_confusion = {}
    total_correct, total_frames = 0, 0
    all_lags = []
    per_video_acc = []
    event_acc = EventLevelAccumulator(STATES)
    timing_acc = TimingOffsetAccumulator(STATES)
    transition_acc = TransitionToleranceAccumulator()

    for i, video_name in enumerate(video_list):
        video_path = os.path.join(args.videos_dir, video_name)
        if not os.path.exists(video_path):
            print(f"[{i + 1}/{len(video_list)}] SKIP (not found): {video_name}")
            continue
        t0 = time.monotonic()
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        gt = load_ground_truth(annotations, video_name, n_frames)

        predicted, transitions = run_cascade_on_video(
            engine, video_path, verbose=args.verbose,
            no_sign=args.no_sign, binary_sign=args.binary_sign,
            no_fast_entry=args.no_fast_entry,
            no_qualifier_filter=args.no_qualifier_filter,
            unrestricted_bayes=args.unrestricted_bayes,
            infer_size=infer_size, csm_cfg=csm_cfg)
        acc, confusion = evaluate(predicted, gt)
        lags = transition_lag(transitions, gt)
        elapsed = time.monotonic() - t0

        n = min(len(predicted), len(gt))
        pred_str = [predicted[j].value for j in range(n)]
        gt_str = [gt[j].value for j in range(n)]
        event_acc.add_video(pred_str, gt_str)
        timing_acc.add_video(pred_str, gt_str)
        transition_acc.add_video(pred_str, gt_str)
        if args.cache_dir:
            np.savez_compressed(os.path.join(args.cache_dir, f"{video_name}.npz"),
                                 predicted=np.array(pred_str), gt=np.array(gt_str))

        per_video_acc.append((video_name, acc))
        all_lags.extend(lags)
        for k, v in confusion.items():
            all_confusion[k] = all_confusion.get(k, 0) + v
        total_correct += acc * min(len(predicted), len(gt))
        total_frames += min(len(predicted), len(gt))

        print(f"[{i + 1}/{len(video_list)}] {video_name}  acc={acc:.1%}  "
              f"transitions={len(transitions)}  ({elapsed:.1f}s)")

    engine.close()

    print("\n=== RESUMO ===")
    print(f"Acuracia por-frame agregada: {total_correct / total_frames:.1%}  "
          f"({total_frames} frames em {len(per_video_acc)} videos)")

    print("\n--- Matriz de confusao (linha=verdade, coluna=previsto) ---")
    states = ["outside", "approaching", "inside", "exiting"]
    header = "GT\\PRED".ljust(14) + "".join(s[:8].rjust(10) for s in states)
    print(header)
    for gt_s in states:
        row = gt_s.ljust(14)
        for pred_s in states:
            row += str(all_confusion.get((gt_s, pred_s), 0)).rjust(10)
        print(row)

    print("\n--- Atraso de deteccao de transicao (frames a 30fps, None=nunca detectado) ---")
    by_state = {}
    for state, onset, lag in all_lags:
        by_state.setdefault(state, []).append(lag)
    for state, lags in by_state.items():
        found = [l for l in lags if l is not None]
        missed = len(lags) - len(found)
        if found:
            print(f"{state:14s} media={sum(found) / len(found):.1f}f "
                  f"({sum(found) / len(found) / 30:.2f}s)  min={min(found)}  max={max(found)}  "
                  f"perdidas={missed}/{len(lags)}")
        else:
            print(f"{state:14s} TODAS perdidas ({missed}/{len(lags)})")

    print_paper_report(all_confusion, event_acc, timing_acc, transition_acc, STATES)


if __name__ == "__main__":
    main()
